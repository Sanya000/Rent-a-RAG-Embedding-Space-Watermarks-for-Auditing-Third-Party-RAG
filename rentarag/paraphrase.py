"""Paraphrase candidate generators.

A :data:`Paraphraser` is any callable ``(sentence, n) -> list[str]`` returning
up to ``n`` distinct rewrites of ``sentence``. The watermarker only scores the
candidates, so any generator works. :func:`get_paraphraser` builds one from a
spec string (``"openai/gpt-4o-mini"``, ``"anthropic/claude-haiku-5-5"``,
``"ollama/llama3.1"``, ``"hf/Qwen/Qwen2.5-1.5B-Instruct"``, ``"toy"``, ...) or
wraps your own function.
"""

from __future__ import annotations

import inspect
import logging
import random
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any, Literal

from .llm import ChatModel, LLMError, OpenAIChat, call_with_retries, get_llm, split_spec

__all__ = [
    "Paraphraser",
    "ParaphraseError",
    "PROMPT_CORPUS",
    "PROMPT_K_VARIANTS",
    "PROMPTS",
    "parse_numbered_list",
    "LLMParaphraser",
    "OpenAIParaphraser",
    "CallableParaphraser",
    "ToyParaphraser",
    "get_paraphraser",
]

log = logging.getLogger(__name__)

Paraphraser = Callable[[str, int], list[str]]

PROMPT_CORPUS = (
    "Rewrite the sentence in a **diverse** set of paraphrases while preserving all facts.\n"
    "Produce 8 variants: 2 very light, 3 medium, 3 strong rewrites.\n"
    "Vary voice, ordering, connectives, nominalization, and tone.\n"
    "Return one variant per line, numbered.\n\nSentence:\n"
    "{sentence}"
)
"""Prompt that built the paper's primary DirBucket corpora.

It always asks for 8 variants. With the default call pattern (a call at
temperature 0.7, then one at 1.0 only if fewer than ``n`` unique variants were
returned) a round yields at most ~16 unique candidates.
"""

PROMPT_K_VARIANTS = (
    "Rewrite the sentence in a diverse set of paraphrases while preserving all facts.\n"
    "Produce {k} variants: {k_light} very light, {k_medium} medium, {k_strong} strong rewrites.\n"
    "Vary voice, ordering, connectives, nominalization, and tone.\n"
    "Return one variant per line, numbered.\n\nSentence:\n"
    "{sentence}"
)
"""``K``-parameterised prompt from the paper appendix (two calls per round, at 0.7 and 1.0)."""

PROMPTS: dict[str, tuple[str, dict[str, Any]]] = {
    "corpus": (PROMPT_CORPUS, {"stop_when_enough": True, "max_tokens": 700}),
    "k-variants": (PROMPT_K_VARIANTS, {"stop_when_enough": False, "max_tokens": 900}),
}
"""Named prompts with their call settings: ``"corpus"`` (default) and ``"k-variants"``."""


class ParaphraseError(LLMError):
    """Raised when a paraphrase backend fails and ``on_error="raise"``."""


def parse_numbered_list(s: str) -> list[str]:
    """Parse ``1. foo`` / ``- bar`` style lines into unique stripped items."""
    items = [re.sub(r"^\d+[\)\.\-]\s*", "", ln.strip(" -*\t")) for ln in s.strip().splitlines() if ln.strip()]
    out: list[str] = []
    seen: set[str] = set()
    for t in items:
        if t and t not in seen:
            out.append(t)
            seen.add(t)
    return out


def _format_prompt(template: str, sentence: str, n: int) -> str:
    k_third = max(1, n // 3)
    return template.format(
        sentence=sentence, n=n, k=n, k_light=k_third, k_medium=k_third, k_strong=n - 2 * k_third
    )


def resolve_prompt(prompt: str) -> tuple[str, dict[str, Any]]:
    """Map a prompt name or template to ``(template, default call settings)``.

    Templates must contain ``{sentence}`` and may use ``{n}`` (number of
    candidates requested) and ``{k_light}``/``{k_medium}``/``{k_strong}``.
    Literal braces must be doubled (``{{`` / ``}}``).
    """
    if prompt in PROMPTS:
        template, defaults = PROMPTS[prompt]
        return template, dict(defaults)
    for template, defaults in PROMPTS.values():
        if prompt == template:
            return template, dict(defaults)
    if "{sentence}" not in prompt:
        raise ValueError(f"prompt must be one of {sorted(PROMPTS)} or a template containing '{{sentence}}'")
    return prompt, {"stop_when_enough": True, "max_tokens": 700}


class _ErrorPolicy:
    on_error: Literal["raise", "warn"]
    failures: int

    def _fail(self, sentence: str, exc: BaseException) -> list[str]:
        self.failures += 1
        if self.on_error == "raise":
            raise ParaphraseError(f"paraphrase backend failed: {exc}") from exc
        log.warning("paraphrase failed (%d so far), sentence left unchanged: %s", self.failures, exc)
        return []


class LLMParaphraser(_ErrorPolicy):
    """Prompted paraphraser on top of any chat model.

    Each call formats the prompt for the sentence, queries the model once per
    temperature in ``temperatures`` (stopping after the first call when
    ``stop_when_enough`` and ``n`` unique candidates are already in hand),
    parses the numbered list and returns at most ``n`` unique candidates.

    Args:
        llm: a :class:`~rentarag.llm.ChatModel`, a spec string for
            :func:`~rentarag.get_llm` (``"anthropic/claude-haiku-5-5"``), or a
            function ``prompt -> str``.
        prompt: ``"corpus"`` (paper default), ``"k-variants"`` (paper appendix)
            or your own template with ``{sentence}`` and optionally ``{n}``.
        temperatures: one model call per entry.
        stop_when_enough: skip later calls once ``n`` candidates are collected
            (default from the prompt: True for ``"corpus"``, False for ``"k-variants"``).
        max_tokens: completion budget per call (default 700, or 900 for ``"k-variants"``).
        max_retries, backoff, max_backoff: retry policy for rate limits,
            timeouts and 5xx errors (exponential backoff with jitter).
        on_error: ``"raise"`` raises :class:`ParaphraseError`; ``"warn"`` logs,
            counts the failure in :attr:`failures` and returns no candidates
            (the sentence stays unchanged).
        **llm_kwargs: passed to :func:`~rentarag.get_llm` when ``llm`` is a spec.

    Instances are thread-safe and can be shared by worker threads.
    """

    def __init__(
        self,
        llm: ChatModel | str | Callable[..., Any],
        *,
        prompt: str = "corpus",
        temperatures: Sequence[float] = (0.7, 1.0),
        stop_when_enough: bool | None = None,
        max_tokens: int | None = None,
        max_retries: int = 5,
        backoff: float = 1.0,
        max_backoff: float = 60.0,
        on_error: Literal["raise", "warn"] = "raise",
        **llm_kwargs: Any,
    ):
        template, defaults = resolve_prompt(prompt)
        try:
            self.llm = get_llm(llm, **llm_kwargs)
        except LLMError as exc:
            raise ParaphraseError(str(exc)) from exc
        self.prompt = template
        self.temperatures = tuple(float(t) for t in temperatures)
        self.stop_when_enough = bool(defaults["stop_when_enough"] if stop_when_enough is None else stop_when_enough)
        self.max_tokens = int(defaults["max_tokens"] if max_tokens is None else max_tokens)
        self.max_retries = int(max_retries)
        self.backoff = float(backoff)
        self.max_backoff = float(max_backoff)
        if on_error not in ("raise", "warn"):
            raise ValueError("on_error must be 'raise' or 'warn'")
        self.on_error = on_error
        self.failures = 0
        self.calls = 0
        self._lock = threading.Lock()

    @property
    def model(self) -> str:
        return str(getattr(self.llm, "model", getattr(self.llm, "model_name", self.llm)))

    def _chat(self, prompt: str, temperature: float) -> str:
        with self._lock:
            self.calls += 1
        return call_with_retries(
            lambda: self.llm.complete(prompt, temperature=temperature, max_tokens=self.max_tokens),
            max_retries=self.max_retries,
            backoff=self.backoff,
            max_backoff=self.max_backoff,
        )

    def __call__(self, sentence: str, n: int) -> list[str]:
        prompt = _format_prompt(self.prompt, sentence, n)
        cands: list[str] = []
        try:
            for i, t in enumerate(self.temperatures):
                if i > 0 and self.stop_when_enough and n <= len(cands):
                    break
                seen = set(cands)
                cands += [c for c in parse_numbered_list(self._chat(prompt, t)) if c not in seen]
        except Exception as exc:  # noqa: BLE001 - any backend error is reported via the policy
            with self._lock:
                return self._fail(sentence, exc)
        return cands[:n]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(llm={self.llm!r})"


class OpenAIParaphraser(LLMParaphraser):
    """:class:`LLMParaphraser` on OpenAI or any OpenAI-compatible server.

    The defaults reproduce the paper's primary corpus build: ``gpt-4o-mini``,
    :data:`PROMPT_CORPUS`, one call at temperature 0.7 and a second at 1.0
    only if fewer than ``n`` unique candidates came back, ``max_tokens=700``,
    a single user message. Use :meth:`k_variants` for the appendix prompt.

    Point ``base_url`` at a local server (``http://localhost:8000/v1`` for
    ``vllm serve``, ``http://localhost:11434/v1`` for Ollama) to paraphrase
    without an API key. The key is taken from ``api_key`` or the
    ``OPENAI_API_KEY`` environment variable.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        prompt: str = PROMPT_CORPUS,
        temperatures: Sequence[float] = (0.7, 1.0),
        stop_when_enough: bool | None = None,
        max_tokens: int | None = None,
        max_retries: int = 5,
        backoff: float = 1.0,
        timeout: float | None = 120.0,
        on_error: Literal["raise", "warn"] = "raise",
        client: Any = None,
        **chat_kwargs: Any,
    ):
        try:
            llm = OpenAIChat(model, base_url=base_url, api_key=api_key, timeout=timeout, client=client, **chat_kwargs)
        except LLMError as exc:
            raise ParaphraseError(str(exc)) from exc
        super().__init__(
            llm,
            prompt=prompt,
            temperatures=temperatures,
            stop_when_enough=stop_when_enough,
            max_tokens=max_tokens,
            max_retries=max_retries,
            backoff=backoff,
            on_error=on_error,
        )

    @property
    def client(self) -> Any:
        return self.llm.client

    @classmethod
    def k_variants(cls, model: str = "gpt-4o-mini", **kwargs: Any) -> OpenAIParaphraser:
        """Appendix setting: :data:`PROMPT_K_VARIANTS`, always two calls, ``max_tokens=900``."""
        kwargs.setdefault("prompt", PROMPT_K_VARIANTS)
        return cls(model, **kwargs)


def _positional_arity(fn: Callable[..., Any]) -> int | None:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return None
    if any(p.kind is inspect.Parameter.VAR_POSITIONAL for p in params):
        return None
    return sum(p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params)


class CallableParaphraser(_ErrorPolicy):
    """Adapt your own paraphrase function.

    Accepted shapes: ``fn(sentence, n) -> list[str]``, ``fn(sentence) -> list[str]``
    and ``fn(sentence) -> str``. ``takes_n`` is detected from the signature when
    not given. A function that returns a single string is called repeatedly
    (up to ``n`` times, stopping early after 3 calls in a row add nothing new).
    Output is stripped, de-duplicated (order preserved) and truncated to ``n``.
    Exceptions follow the ``on_error`` policy of :class:`LLMParaphraser`.
    """

    def __init__(
        self,
        fn: Callable[..., Sequence[str] | str],
        *,
        takes_n: bool | None = None,
        on_error: Literal["raise", "warn"] = "raise",
    ):
        self.fn = fn
        if takes_n is None:
            arity = _positional_arity(fn)
            takes_n = arity is None or arity >= 2
        self.takes_n = takes_n
        self.on_error = on_error
        self.failures = 0

    def _once(self, sentence: str, n: int) -> Sequence[str] | str:
        return self.fn(sentence, n) if self.takes_n else self.fn(sentence)

    def __call__(self, sentence: str, n: int) -> list[str]:
        out: list[str] = []

        def add(items: Sequence[str] | str) -> int:
            before = len(out)
            for c in [items] if isinstance(items, str) else items:
                c = str(c).strip()
                if c and c not in out:
                    out.append(c)
            return len(out) - before

        try:
            raw = self._once(sentence, n)
            add(raw)
            if isinstance(raw, str):
                idle = 0 if out else 1
                for _ in range(n - 1):
                    if len(out) >= n or idle >= 3:
                        break
                    idle = 0 if add(self._once(sentence, n)) else idle + 1
        except Exception as exc:  # noqa: BLE001
            return self._fail(sentence, exc)
        return out[:n]

    def __repr__(self) -> str:
        return f"CallableParaphraser({getattr(self.fn, '__name__', self.fn)!r})"


_LLM_ONLY_OPTIONS = {"prompt", "temperatures", "stop_when_enough", "max_tokens", "max_retries", "backoff", "max_backoff", "on_error"}


def get_paraphraser(spec: Any = "openai/gpt-4o-mini", **kwargs: Any) -> Paraphraser:
    """Build a paraphraser from a spec string, an LLM, or a function.

    * ``"toy"`` / ``"toy:<seed>"``: offline :class:`ToyParaphraser` (not fluent; tests and demos).
    * ``"<backend>/<model>"`` with backend ``openai``, ``anthropic``, ``gemini``,
      ``ollama``, ``vllm``, ``hf`` or ``litellm``: an :class:`LLMParaphraser`
      (:class:`OpenAIParaphraser` for ``openai``). Paraphraser options
      (``prompt``, ``temperatures``, ``max_tokens``, ``stop_when_enough``,
      ``max_retries``, ``on_error``) and backend options (``base_url``,
      ``api_key``, ``device``, ...) can be mixed in ``kwargs``.
    * a :class:`~rentarag.llm.ChatModel` (``.complete(prompt, ...)``): wrapped
      in :class:`LLMParaphraser` with the paper prompt.
    * any other callable: your paraphrase function, wrapped in
      :class:`CallableParaphraser` (``fn(sentence, n) -> list[str]`` or
      ``fn(sentence) -> str | list[str]``). To use a plain ``prompt -> text``
      function with the paper prompt instead, pass ``LLMParaphraser(fn)``.
    * an existing paraphraser instance is returned unchanged.
    """
    if isinstance(spec, (LLMParaphraser, ToyParaphraser, CallableParaphraser)):
        if kwargs:
            raise TypeError(f"unexpected options for an existing paraphraser: {sorted(kwargs)}")
        return spec
    if isinstance(spec, ChatModel) and not isinstance(spec, type):
        return LLMParaphraser(spec, **kwargs)
    if callable(spec) and not isinstance(spec, str):
        return CallableParaphraser(spec, **kwargs)
    if not isinstance(spec, str) or not spec.strip():
        raise TypeError(f"paraphraser spec must be a string like 'openai/gpt-4o-mini' or a callable, got {spec!r}")
    prefix, model = split_spec(spec)
    if prefix == "toy":
        opts = {k: v for k, v in kwargs.items() if k not in _LLM_ONLY_OPTIONS}
        if model:
            opts.setdefault("seed", int(model))
        return ToyParaphraser(**opts)
    if prefix == "openai":
        if not model:
            raise ValueError("paraphraser spec 'openai/' names no model; use e.g. 'openai/gpt-4o-mini'")
        return OpenAIParaphraser(model, **kwargs)
    llm_kw = {k: v for k, v in kwargs.items() if k not in _LLM_ONLY_OPTIONS}
    para_kw = {k: v for k, v in kwargs.items() if k in _LLM_ONLY_OPTIONS}
    try:
        llm = get_llm(spec, **llm_kw)
    except LLMError as exc:
        raise ParaphraseError(str(exc)) from exc
    return LLMParaphraser(llm, **para_kw)


_FILLERS = ("In fact,", "Notably,", "Indeed,", "Importantly,", "As reported,", "Overall,", "In particular,")
_SWAPS = {
    "the": "this",
    "a": "one",
    "and": "as well as",
    "but": "yet",
    "also": "additionally",
    "big": "large",
    "large": "big",
    "said": "stated",
    "shows": "indicates",
    "used": "employed",
    "many": "numerous",
    "help": "assist",
    "important": "key",
    "about": "roughly",
}


class ToyParaphraser:
    """Deterministic rule-based "paraphraser" for tests and offline demos.

    Produces random filler insertions, word substitutions and drops. The output
    is NOT fluent text and preserves meaning only loosely; it exists so the
    full pipeline can run without any model or API key. Candidates depend on
    ``seed``, the input sentence and the number of previous calls, so results
    are reproducible for a fixed call order (``max_workers=1``).
    """

    def __init__(self, seed: int = 0, edits: int = 2):
        self.seed = int(seed)
        self.edits = int(edits)
        self.calls = 0
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return f"ToyParaphraser(seed={self.seed})"

    def _variant(self, words: list[str], rng: random.Random) -> str:
        w = list(words)
        for _ in range(rng.randint(1, self.edits)):
            op = rng.random()
            if op < 0.35:
                filler = rng.choice(_FILLERS)
                if rng.random() < 0.5:
                    w.insert(0, filler)
                else:
                    w.insert(rng.randrange(len(w) + 1), filler.rstrip(",").lower())
            elif op < 0.75:
                idx = [i for i, t in enumerate(w) if t.lower() in _SWAPS]
                if idx:
                    i = rng.choice(idx)
                    w[i] = _SWAPS[w[i].lower()]
            elif len(w) > 4:
                del w[rng.randrange(1, len(w) - 1)]
        return " ".join(w)

    def __call__(self, sentence: str, n: int) -> list[str]:
        with self._lock:
            call = self.calls
            self.calls += 1
        rng = random.Random(f"{self.seed}|{call}|{sentence}")
        words = sentence.split()
        if not words:
            return []
        out: list[str] = []
        for _ in range(4 * n):
            v = self._variant(words, rng)
            if v != sentence and v not in out:
                out.append(v)
            if len(out) >= n:
                break
        return out
