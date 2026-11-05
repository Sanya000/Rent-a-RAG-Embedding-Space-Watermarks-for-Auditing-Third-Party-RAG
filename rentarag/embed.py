"""Sentence encoders.

An :class:`Embedder` is any callable mapping a list of strings to a
``(n, dim)`` float32 array of L2-normalised rows, with a ``dim`` attribute.
:func:`get_embedder` builds one from a spec (any sentence-transformers model
id, ``"openai/text-embedding-3-small"``, ``"hashing"``) or wraps your own
function. Provider and auditor must use the same encoder: buckets and
directions are defined in its embedding space, and their dimension follows
the encoder automatically.
"""

from __future__ import annotations

import hashlib
import os
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

__all__ = [
    "Embedder",
    "SentenceTransformerEmbedder",
    "HashingEmbedder",
    "CallableEmbedder",
    "OpenAIEmbedder",
    "get_embedder",
    "DEFAULT_EMBED_MODEL",
]

DEFAULT_EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def __call__(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """``sentence-transformers`` encoder with unit-normalised float32 output.

    Defaults to ``all-mpnet-base-v2`` (d=768), the paper's encoder. Loaded
    models are shared across instances with the same ``(model_name, device)``.
    Encoding calls are identical to the paper code, so results match it
    bit-for-bit on the same hardware and library versions.
    """

    _models: ClassVar[dict[tuple[str, str], Any]] = {}
    _locks: ClassVar[dict[tuple[str, str], threading.Lock]] = {}

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL, device: str = "auto", batch_size: int = 32):
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"
        key = (model_name, device)
        if key not in self._models:
            from sentence_transformers import SentenceTransformer

            self._models[key] = SentenceTransformer(model_name, device=device)
            self._locks[key] = threading.Lock()
        self.model = self._models[key]
        self._lock = self._locks[key]
        self.model_name = model_name
        self.device = device
        self.batch_size = int(batch_size)
        self.dim = int(self.model.get_sentence_embedding_dimension())

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        with self._lock:  # one forward pass at a time when shared by worker threads
            X = self.model.encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        return X.astype(np.float32)

    def __repr__(self) -> str:
        return f"SentenceTransformerEmbedder({self.model_name!r}, dim={self.dim})"


class HashingEmbedder:
    """Deterministic bag-of-words embedder for tests and offline demos.

    Each lower-cased word maps to a fixed pseudo-random Gaussian vector; a text
    is the normalised sum of its word vectors. Texts sharing words are similar,
    which is enough to exercise alignment and the watermark search, but this is
    not a semantic encoder and gives no meaningful paraphrase robustness.
    """

    _TOKEN_RE = re.compile(r"\w+")

    def __repr__(self) -> str:
        return f"HashingEmbedder(dim={self.dim})"

    def __init__(self, dim: int = 256, seed: int = 0):
        self.dim = int(dim)
        self.seed = int(seed)
        self._vocab: dict[str, np.ndarray] = {}

    def _word(self, w: str) -> np.ndarray:
        v = self._vocab.get(w)
        if v is None:
            h = hashlib.sha256(f"{self.seed}:{w}".encode()).digest()
            v = np.random.default_rng(int.from_bytes(h[:8], "big")).standard_normal(self.dim)
            self._vocab[w] = v
        return v

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        X = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for w in self._TOKEN_RE.findall(t.lower()):
                X[i] += self._word(w)
            n = np.linalg.norm(X[i])
            if n > 0:
                X[i] /= n
        return X.astype(np.float32)


def _normalise(X: Any, n: int, dim: int | None = None) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 1:
        X = X[None, :]
    if X.ndim != 2 or X.shape[0] != n:
        raise ValueError(f"embedder returned shape {X.shape} for {n} texts; expected ({n}, dim)")
    if dim is not None and X.shape[1] != dim:
        raise ValueError(f"embedder returned dim {X.shape[1]}, expected {dim}")
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.where(norms > 0, norms, 1.0)).astype(np.float32)


class CallableEmbedder:
    """Adapt ``fn(texts) -> array-like (n, d)``.

    Rows are L2-normalised and cast to float32. ``dim`` is inferred by
    embedding one probe text when not given. ``batch_size`` splits large inputs.
    """

    def __init__(self, fn: Callable[[list[str]], Any], dim: int | None = None, *, batch_size: int | None = None):
        self.fn = fn
        self.batch_size = int(batch_size) if batch_size else None
        self.dim = int(dim) if dim else int(_normalise(fn(["dimension probe"]), 1).shape[1])

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        step = self.batch_size or len(texts)
        parts = [_normalise(self.fn(texts[i : i + step]), len(texts[i : i + step]), self.dim) for i in range(0, len(texts), step)]
        return np.concatenate(parts, axis=0)

    def __repr__(self) -> str:
        return f"CallableEmbedder({getattr(self.fn, '__name__', self.fn)!r}, dim={self.dim})"


_OPENAI_EMBED_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072, "text-embedding-ada-002": 1536}


class OpenAIEmbedder:
    """Embeddings API of OpenAI or any OpenAI-compatible server (Ollama, vLLM, Gemini, ...).

    Texts are sent in batches of ``batch_size``; rate limits and transient
    errors are retried with exponential backoff. ``dimensions`` requests
    shortened embeddings (text-embedding-3 models).
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        dimensions: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | tuple[str, ...] = "OPENAI_API_KEY",
        require_key: bool | None = None,
        batch_size: int = 256,
        max_retries: int = 5,
        timeout: float | None = 120.0,
        client: Any = None,
    ):
        from .llm import OpenAIChat

        self.model = model
        self.dimensions = int(dimensions) if dimensions else None
        self.batch_size = int(batch_size)
        self.max_retries = int(max_retries)
        if client is None:
            client = OpenAIChat(
                model, base_url=base_url, api_key=api_key, api_key_env=api_key_env, timeout=timeout, require_key=require_key
            ).client
        self.client = client
        self.dim = self.dimensions or _OPENAI_EMBED_DIMS.get(model) or int(self._embed(["dimension probe"]).shape[1])

    def _embed(self, batch: list[str]) -> np.ndarray:
        from .llm import call_with_retries

        kw: dict[str, Any] = {"model": self.model, "input": batch}
        if self.dimensions:
            kw["dimensions"] = self.dimensions
        r = call_with_retries(lambda: self.client.embeddings.create(**kw), max_retries=self.max_retries)
        rows = sorted(r.data, key=lambda d: getattr(d, "index", 0))
        return _normalise([d.embedding for d in rows], len(batch))

    def __call__(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.concatenate([self._embed(texts[i : i + self.batch_size]) for i in range(0, len(texts), self.batch_size)])

    def __repr__(self) -> str:
        return f"OpenAIEmbedder({self.model!r}, dim={self.dim})"


def embedder_name(emb: Any) -> str | None:
    """Short model name of an embedder (``None`` when unknown, e.g. a plain function)."""
    if isinstance(emb, HashingEmbedder):
        return "hashing"
    for attr in ("model_name", "model"):
        v = getattr(emb, attr, None)
        if isinstance(v, str) and v:
            return v
    return None


def same_embedder_name(a: str | None, b: str | None) -> bool:
    """``True`` unless both names are known and differ (``sentence-transformers/x`` equals ``x``)."""
    if not a or not b:
        return True
    return a.rstrip("/").rsplit("/", 1)[-1].lower() == b.rstrip("/").rsplit("/", 1)[-1].lower()


def get_embedder(spec: Any = None, **kwargs: Any) -> Embedder:
    """Build an :class:`Embedder` from a spec, an object or a function.

    * ``None`` / ``"default"``: ``sentence-transformers/all-mpnet-base-v2`` (paper).
    * any sentence-transformers / Hugging Face model id, optionally prefixed
      with ``st/`` or ``hf/`` (``"BAAI/bge-small-en-v1.5"``, ``"all-MiniLM-L6-v2"``).
    * ``"openai/<model>"``: OpenAI embeddings API (``OPENAI_API_KEY``; ``base_url=``
      for compatible servers); ``"ollama/<model>"``, ``"vllm/<model>"`` and
      ``"gemini/<model>"`` use the same client against those endpoints.
    * ``"hashing"`` / ``"hashing:<dim>"``: toy bag-of-words embedder (tests, demos).
    * an object with ``dim`` and ``__call__``: returned as is.
    * any other callable ``texts -> array``: wrapped in :class:`CallableEmbedder`
      (rows are L2-normalised automatically).
    """
    if spec is None or spec == "default":
        return SentenceTransformerEmbedder(DEFAULT_EMBED_MODEL, **kwargs)
    if not isinstance(spec, str):
        if callable(spec) and hasattr(spec, "dim") and not isinstance(spec, type):
            if kwargs:
                raise TypeError(f"unexpected options for an existing embedder: {sorted(kwargs)}")
            return spec
        if callable(spec):
            return CallableEmbedder(spec, **kwargs)
        raise TypeError(f"embedder spec must be a model id string or a callable, got {spec!r}")
    s = spec.strip()
    head, sep, rest = s.partition("/")
    head_l = head.lower()
    if head_l == "hashing" or s.lower().startswith("hashing:"):
        dim = s.partition(":")[2] or rest
        if dim:
            kwargs.setdefault("dim", int(dim))
        return HashingEmbedder(**kwargs)
    if sep and head_l in ("openai", "ollama", "vllm", "gemini"):
        from .llm import GEMINI_BASE_URL, OLLAMA_BASE_URL, VLLM_BASE_URL

        if head_l == "ollama":
            kwargs.setdefault("base_url", os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL))
            kwargs.setdefault("api_key_env", "OLLAMA_API_KEY")
        elif head_l == "vllm":
            kwargs.setdefault("base_url", os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL))
            kwargs.setdefault("api_key_env", "VLLM_API_KEY")
        elif head_l == "gemini":
            kwargs.setdefault("base_url", GEMINI_BASE_URL)
            kwargs.setdefault("api_key_env", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
            kwargs.setdefault("require_key", True)
        return OpenAIEmbedder(rest, **kwargs)
    model_id = rest if sep and head_l in ("st", "hf", "sentence-transformer") else s
    try:
        return SentenceTransformerEmbedder(model_id, **kwargs)
    except OSError as exc:
        import difflib

        prefixes = ("st", "hf", "openai", "ollama", "vllm", "gemini", "hashing")
        hint = ""
        if sep:
            close = difflib.get_close_matches(head_l, prefixes, n=1, cutoff=0.6)
            if close:
                hint = f" Did you mean '{close[0]}/{rest}'?"
        raise ValueError(
            f"cannot load encoder {spec!r} as a sentence-transformers model ({exc.__class__.__name__}).{hint} "
            "Valid specs: a sentence-transformers / Hugging Face model id (optionally st/ or hf/), "
            "openai/<model>, ollama/<model>, vllm/<model>, gemini/<model>, hashing[:dim]."
        ) from exc
