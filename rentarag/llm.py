"""Chat-model backends behind one interface.

A :class:`ChatModel` turns a prompt into text::

    llm.complete(prompt, temperature=0.7, max_tokens=700) -> str

:func:`get_llm` builds one from a spec string such as ``"openai/gpt-4o-mini"``,
``"anthropic/claude-haiku-5-5"``, ``"ollama/llama3.1"`` or
``"hf/Qwen/Qwen2.5-1.5B-Instruct"``. Paraphrasers (:mod:`rentarag.paraphrase`)
and audit-query generation (:mod:`rentarag.queries`) are built on top of it.

SDKs are imported lazily, so only the backend you use needs to be installed
(``pip install "rentarag[openai]"``, ``[anthropic]``, ``[hf]``, ``[litellm]``).
API keys are read from the environment or passed explicitly, never from files.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ChatModel",
    "LLMError",
    "MissingDependencyError",
    "MissingCredentialsError",
    "OpenAIChat",
    "AnthropicChat",
    "HFChat",
    "LiteLLMChat",
    "CallableChat",
    "ToyChat",
    "get_llm",
    "is_retryable",
    "call_with_retries",
    "LLM_PREFIXES",
    "GEMINI_BASE_URL",
    "OLLAMA_BASE_URL",
    "VLLM_BASE_URL",
]

log = logging.getLogger(__name__)

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
VLLM_BASE_URL = "http://localhost:8000/v1"

LLM_PREFIXES = ("openai", "anthropic", "gemini", "ollama", "vllm", "hf", "litellm", "toy")


class LLMError(RuntimeError):
    """Base class for backend construction and call errors."""


class MissingDependencyError(LLMError, ImportError):
    """An optional SDK needed by the requested backend is not installed."""


class MissingCredentialsError(LLMError):
    """No API key was found for a hosted backend."""


def _require(module: str, extra: str) -> Any:
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise MissingDependencyError(
            f"this backend needs the '{module}' package: pip install \"rentarag[{extra}]\""
        ) from exc


@runtime_checkable
class ChatModel(Protocol):
    """Anything with ``complete(prompt, *, temperature, max_tokens) -> str``."""

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str: ...


# ---------------------------------------------------------------------------
# retries
# ---------------------------------------------------------------------------

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 529}
_RETRYABLE_NAMES = {
    "RateLimitError",
    "APITimeoutError",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "OverloadedError",
    "Timeout",
    "TimeoutError",
    "ConnectTimeout",
    "ReadTimeout",
    "ConnectError",
}
_jitter = random.Random()


def is_retryable(exc: BaseException) -> bool:
    """True for rate limits, timeouts, connection problems and 5xx responses.

    Works across SDKs by looking at the HTTP status (``status_code`` / ``status``)
    and the exception class names (``RateLimitError``, ``APITimeoutError``, ...).
    Client errors such as 400/401/404 are not retried.
    """
    status = getattr(exc, "status_code", None)
    if status is None:
        status = getattr(exc, "status", None)
    if isinstance(status, int):
        return status in _RETRYABLE_STATUS or status >= 500
    names = {cls.__name__ for cls in type(exc).__mro__}
    if names & _RETRYABLE_NAMES:
        return True
    if "APIError" in names:  # SDK error without an HTTP status (dropped stream, ...)
        return True
    return isinstance(exc, (TimeoutError, ConnectionError))


def _retry_after(exc: BaseException) -> float | None:
    headers = getattr(getattr(exc, "response", None), "headers", None)
    try:
        value = headers.get("retry-after") if headers is not None else None
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def call_with_retries(
    fn: Callable[[], Any],
    *,
    max_retries: int = 5,
    backoff: float = 1.0,
    max_backoff: float = 60.0,
    sleep: Callable[[float], None] | None = None,
) -> Any:
    """Call ``fn()``; retry retryable errors with exponential backoff and jitter.

    Waits ``min(max_backoff, backoff * 2**attempt)`` seconds (times a random
    factor in [1, 1.25), or the server's ``retry-after`` when given) between
    attempts, for at most ``max_retries`` retries. Non-retryable errors and the
    last failure are re-raised.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as exc:
            if attempt >= max_retries or not is_retryable(exc):
                raise
            delay = _retry_after(exc)
            if delay is None:
                delay = min(max_backoff, backoff * (2**attempt)) * (1.0 + 0.25 * _jitter.random())
            log.info("retryable %s (attempt %d/%d), sleeping %.1fs", type(exc).__name__, attempt + 1, max_retries, delay)
            (sleep or time.sleep)(delay)
            attempt += 1


# ---------------------------------------------------------------------------
# backends
# ---------------------------------------------------------------------------


class OpenAIChat:
    """Chat Completions on OpenAI or any OpenAI-compatible server.

    Covers OpenAI, Gemini (``base_url=GEMINI_BASE_URL``), Ollama, vLLM,
    llama.cpp, LM Studio, TGI, Together, Groq, OpenRouter and other endpoints
    that speak the Chat Completions protocol. The key comes from ``api_key`` or
    the ``api_key_env`` environment variable(s); a server at a custom
    ``base_url`` needs none unless ``require_key=True``.
    SDK-level retries are disabled: callers retry through :func:`call_with_retries`.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        api_key_env: str | tuple[str, ...] = "OPENAI_API_KEY",
        timeout: float | None = 120.0,
        client: Any = None,
        extra_body: dict[str, Any] | None = None,
        require_key: bool | None = None,
    ):
        self.model = model
        self.extra_body = dict(extra_body or {})
        if client is None:
            openai = _require("openai", "openai")
            envs = (api_key_env,) if isinstance(api_key_env, str) else tuple(api_key_env)
            if base_url is None and "OPENAI_API_KEY" in envs:
                base_url = os.environ.get("OPENAI_BASE_URL")
            api_key = api_key or next((os.environ[e] for e in envs if os.environ.get(e)), None)
            if not api_key:
                if require_key or (require_key is None and base_url is None):
                    raise MissingCredentialsError(
                        f"{' or '.join(envs)} is not set. Export it, pass api_key=..., or set base_url to a local "
                        "OpenAI-compatible server."
                    )
                api_key = "EMPTY"
            client = openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0, timeout=timeout)
        self.client = client

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        kw: dict[str, Any] = {"model": self.model}
        if temperature is not None:
            kw["temperature"] = temperature
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        kw["messages"] = [{"role": "user", "content": prompt}]
        if self.extra_body:
            kw["extra_body"] = self.extra_body
        r = self.client.chat.completions.create(**kw)
        return r.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"OpenAIChat(model={self.model!r})"


class AnthropicChat:
    """Anthropic Messages API.

    Current Claude models fix their own sampling settings, so ``temperature`` is
    not sent unless ``send_temperature=True`` (for older models that accept
    it); repeated calls still return independent samples. ``max_tokens`` is
    raised to at least ``min_max_tokens`` because the budget is shared with the
    model's thinking. Extra ``create_kwargs`` (e.g.
    ``output_config={"effort": "low"}``) are passed to ``messages.create``.
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = 120.0,
        client: Any = None,
        send_temperature: bool = False,
        min_max_tokens: int = 4096,
        **create_kwargs: Any,
    ):
        self.model = model
        self.send_temperature = bool(send_temperature)
        self.min_max_tokens = int(min_max_tokens)
        self.create_kwargs = create_kwargs
        if client is None:
            anthropic = _require("anthropic", "anthropic")
            kw: dict[str, Any] = {"max_retries": 0, "timeout": timeout}
            if api_key:
                kw["api_key"] = api_key
            if base_url:
                kw["base_url"] = base_url
            try:
                client = anthropic.Anthropic(**kw)
            except Exception as exc:  # the SDK raises if it finds no credentials
                raise MissingCredentialsError(
                    f"could not create an Anthropic client ({exc}). Export ANTHROPIC_API_KEY or pass api_key=..."
                ) from exc
        self.client = client

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        kw: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max(int(max_tokens or 0), self.min_max_tokens),
            "messages": [{"role": "user", "content": prompt}],
            **self.create_kwargs,
        }
        if self.send_temperature and temperature is not None:
            kw["temperature"] = temperature
        r = self.client.messages.create(**kw)
        return "".join(getattr(b, "text", "") for b in r.content if getattr(b, "type", "text") == "text")

    def __repr__(self) -> str:
        return f"AnthropicChat(model={self.model!r})"


class HFChat:
    """Local Hugging Face ``transformers`` chat model.

    The prompt is wrapped in the model's chat template (when it has one) and
    sampled with ``temperature`` / ``top_p`` (greedy when ``temperature`` is 0).
    Calls are serialised with a lock, so one instance can be shared by worker
    threads. ``seed`` makes sampling reproducible for a fixed call order.
    """

    def __init__(
        self,
        model: str,
        *,
        device: str = "auto",
        dtype: str = "auto",
        top_p: float = 0.95,
        default_max_tokens: int = 512,
        seed: int | None = None,
        model_kwargs: dict[str, Any] | None = None,
        tokenizer: Any = None,
        model_obj: Any = None,
    ):
        torch = _require("torch", "hf")
        self.model_name = model
        self.top_p = float(top_p)
        self.default_max_tokens = int(default_max_tokens)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if tokenizer is None or model_obj is None:
            transformers = _require("transformers", "hf")
            tokenizer = tokenizer or transformers.AutoTokenizer.from_pretrained(model)
            if model_obj is None:
                kw = dict(model_kwargs or {})
                major_minor = tuple(int(x) for x in re.findall(r"\d+", transformers.__version__)[:2])
                kw.setdefault("dtype" if major_minor >= (4, 56) else "torch_dtype", dtype)
                model_obj = transformers.AutoModelForCausalLM.from_pretrained(model, **kw)
            model_obj = model_obj.to(device)
            model_obj.eval()
        self.tokenizer = tokenizer
        self.model = model_obj
        self._torch = torch
        self._lock = threading.Lock()
        if seed is not None:
            torch.manual_seed(int(seed))

    def _render(self, prompt: str) -> str:
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
            )
        return prompt

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        enc = self.tokenizer(self._render(prompt), return_tensors="pt", add_special_tokens=False)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        gen: dict[str, Any] = {"max_new_tokens": int(max_tokens or self.default_max_tokens)}
        if temperature is not None and temperature > 0:
            gen.update(do_sample=True, temperature=float(temperature), top_p=self.top_p)
        else:
            gen.update(do_sample=False)
        pad = self.tokenizer.pad_token_id
        gen["pad_token_id"] = pad if pad is not None else self.tokenizer.eos_token_id
        with self._lock, self._torch.no_grad():
            out = self.model.generate(**enc, **gen)
        new = out[0][enc["input_ids"].shape[1] :]
        return self.tokenizer.decode(new, skip_special_tokens=True)

    def __repr__(self) -> str:
        return f"HFChat(model={self.model_name!r}, device={self.device!r})"


class LiteLLMChat:
    """Any provider supported by `LiteLLM <https://docs.litellm.ai>`_ (Bedrock, Vertex, Azure, Mistral, ...)."""

    def __init__(self, model: str, **completion_kwargs: Any):
        self.model = model
        self.completion_kwargs = completion_kwargs
        self._litellm = _require("litellm", "litellm")

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        kw: dict[str, Any] = {"model": self.model, "messages": [{"role": "user", "content": prompt}], **self.completion_kwargs}
        if temperature is not None:
            kw["temperature"] = temperature
        if max_tokens is not None:
            kw["max_tokens"] = max_tokens
        r = self._litellm.completion(**kw)
        return r.choices[0].message.content or ""

    def __repr__(self) -> str:
        return f"LiteLLMChat(model={self.model!r})"


class CallableChat:
    """Adapt ``fn(prompt) -> str``; ``temperature``/``max_tokens`` are passed if ``fn`` accepts them."""

    def __init__(self, fn: Callable[..., Any]):
        self.fn = fn
        try:
            params = inspect.signature(fn).parameters
            var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
            self._passes = {n for n in ("temperature", "max_tokens") if var_kw or n in params}
        except (TypeError, ValueError):
            self._passes = set()

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        kw = {k: v for k, v in (("temperature", temperature), ("max_tokens", max_tokens)) if k in self._passes and v is not None}
        out = self.fn(prompt, **kw)
        return out if isinstance(out, str) else str(getattr(out, "content", out))

    def __repr__(self) -> str:
        return f"CallableChat({getattr(self.fn, '__name__', self.fn)!r})"


class ToyChat:
    """Offline stand-in LLM for tests and demos (no model, not fluent).

    Paraphrase prompts (ending in ``Sentence:\\n<s>``) get a numbered list of
    :class:`~rentarag.ToyParaphraser` edits; audit-query prompts (with a
    ``Document:`` block) get a question built from the document's first sentence.
    """

    def __init__(self, seed: int = 0):
        from .paraphrase import ToyParaphraser

        self._para = ToyParaphraser(seed=seed)

    def complete(self, prompt: str, *, temperature: float | None = None, max_tokens: int | None = None) -> str:
        if "\nSentence:\n" in prompt:
            sentence = prompt.rsplit("\nSentence:\n", 1)[1].strip()
            return "\n".join(f"{i}. {c}" for i, c in enumerate(self._para(sentence, 8), 1))
        m = re.search(r"Document:\n(.*?)(?:\n\nQuestion:|\Z)", prompt, flags=re.S)
        if m:
            words = re.findall(r"[A-Za-z0-9']+", m.group(1))[:14]
            return "What does the document report about " + " ".join(words) + "?"
        return prompt[:200]

    def __repr__(self) -> str:
        return "ToyChat()"


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def split_spec(spec: str) -> tuple[str, str]:
    """``"ollama/llama3.1"`` -> ``("ollama", "llama3.1")``; ``"hf/Qwen/Qwen2.5"`` -> ``("hf", "Qwen/Qwen2.5")``."""
    prefix, sep, rest = spec.strip().partition("/")
    if not sep:
        prefix, _, rest = spec.strip().partition(":")
    return prefix.lower(), rest


def get_llm(spec: str | ChatModel | Callable[..., Any], **kwargs: Any) -> ChatModel:
    """Build a :class:`ChatModel` from a spec string, an object or a callable.

    ============================== =======================================================
    spec                            backend
    ============================== =======================================================
    ``openai/<model>``              OpenAI (``OPENAI_API_KEY``); add ``base_url=`` for any
                                    OpenAI-compatible server
    ``anthropic/<model>``           Anthropic Messages API (``ANTHROPIC_API_KEY``)
    ``gemini/<model>``              Gemini via its OpenAI-compatible endpoint
                                    (``GEMINI_API_KEY`` or ``GOOGLE_API_KEY``)
    ``ollama/<model>``              local Ollama (``OLLAMA_BASE_URL``, default
                                    ``http://localhost:11434/v1``)
    ``vllm/<model>``                local vLLM server (``VLLM_BASE_URL``, default
                                    ``http://localhost:8000/v1``)
    ``hf/<model id>``               local ``transformers`` model with its chat template
    ``litellm/<model>``             anything LiteLLM supports
    ``toy``                         offline stand-in (tests, demos)
    ============================== =======================================================

    A :class:`ChatModel` instance is returned as is; any other callable is
    treated as ``fn(prompt) -> str``. ``kwargs`` go to the backend constructor.
    """
    if isinstance(spec, ChatModel) and not isinstance(spec, type):
        if kwargs:
            raise TypeError(f"unexpected options for an existing ChatModel: {sorted(kwargs)}")
        return spec
    if callable(spec) and not isinstance(spec, str):
        return CallableChat(spec)
    if not isinstance(spec, str) or not spec.strip():
        raise TypeError(f"LLM spec must be a string like 'openai/gpt-4o-mini' or a callable, got {spec!r}")
    prefix, model = split_spec(spec)
    if prefix == "toy":
        return ToyChat(seed=int(model or kwargs.pop("seed", 0)))
    if prefix not in LLM_PREFIXES + ("huggingface", "transformers"):
        hint = f" Did you mean 'openai/{spec}'?" if prefix.startswith(("gpt", "o1", "o3", "o4")) else ""
        raise ValueError(
            f"unknown LLM spec {spec!r}; use one of {', '.join(p + '/<model>' for p in LLM_PREFIXES[:-1])} or 'toy'.{hint}"
        )
    if not model:
        raise ValueError(f"LLM spec {spec!r} names no model; use e.g. '{prefix}/<model>'")
    if prefix == "openai":
        return OpenAIChat(model, **kwargs)
    if prefix == "gemini":
        kwargs.setdefault("base_url", GEMINI_BASE_URL)
        kwargs.setdefault("api_key_env", ("GEMINI_API_KEY", "GOOGLE_API_KEY"))
        kwargs.setdefault("require_key", True)
        return OpenAIChat(model, **kwargs)
    if prefix == "ollama":
        kwargs.setdefault("base_url", os.environ.get("OLLAMA_BASE_URL", OLLAMA_BASE_URL))
        kwargs.setdefault("api_key_env", "OLLAMA_API_KEY")
        return OpenAIChat(model, **kwargs)
    if prefix == "vllm":
        kwargs.setdefault("base_url", os.environ.get("VLLM_BASE_URL", VLLM_BASE_URL))
        kwargs.setdefault("api_key_env", "VLLM_API_KEY")
        return OpenAIChat(model, **kwargs)
    if prefix == "anthropic":
        return AnthropicChat(model, **kwargs)
    if prefix in ("hf", "huggingface", "transformers"):
        return HFChat(model, **kwargs)
    return LiteLLMChat(model, **kwargs)
