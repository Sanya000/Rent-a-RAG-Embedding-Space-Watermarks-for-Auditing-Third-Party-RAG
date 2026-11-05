"""Audit-query generation with the paper's prompt."""

from __future__ import annotations

import random
import re
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from .llm import ChatModel, call_with_retries, get_llm

__all__ = ["PROMPT_AUDIT_QUERY", "generate_audit_queries"]

PROMPT_AUDIT_QUERY = (
    "Read the document below and write ONE question whose answer must cite at least three specific details "
    "(names, dates, numbers, or short phrases) from the document. Avoid yes/no or vague prompts. "
    "Return ONLY the question.\n\nDocument:\n{doc}\n\nQuestion:"
)
"""Query-generation prompt of the paper's audit runner (``{doc}`` = first 1500 characters of a document)."""


def _source_texts(
    docs: Any,
    *,
    text_field: str | Sequence[str] = ("text", "text_wm"),
    id_field: str = "id",
    provider_field: str | None = None,
) -> list[str]:
    from .corpus import WatermarkedDoc, iter_documents

    if isinstance(docs, str):
        try:
            is_path = Path(docs).exists()
        except OSError:  # e.g. a long text is not a valid file name
            is_path = False
        if not is_path:
            docs = [docs]  # a single document text
    if isinstance(docs, Sequence) and not isinstance(docs, str) and all(isinstance(d, str) for d in docs):
        return list(docs)
    if isinstance(docs, Sequence) and docs and all(isinstance(d, WatermarkedDoc) for d in docs):
        return [d.original for d in docs]
    return [
        d.text
        for d in iter_documents(docs, text_field=text_field, id_field=id_field, provider_field=provider_field, default_provider="")
    ]


def _clean(q: str) -> str:
    lines = [ln.strip() for ln in q.strip().splitlines() if ln.strip()]
    q = lines[0] if lines else ""
    q = re.sub(r"^(question|q)\s*[:.\-]\s*", "", q, flags=re.I)
    return q.strip().strip('"').strip("'").strip()


def generate_audit_queries(
    docs: Any,
    llm: ChatModel | str | Callable[..., Any] = "openai/gpt-4o-mini",
    n: int | None = None,
    *,
    seed: int = 0,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    max_chars: int = 1500,
    prompt: str = PROMPT_AUDIT_QUERY,
    max_workers: int = 1,
    max_retries: int = 5,
    text_field: str | Sequence[str] = ("text", "text_wm"),
    id_field: str = "id",
    provider_field: str | None = None,
    **llm_kwargs: Any,
) -> list[str]:
    """Generate one audit question per sampled document.

    Questions ask for several specific details of a document, so an honest RAG
    answer has to reproduce its content, which is what the auditor aligns.

    Args:
        docs: documents (strings, a corpus path, dicts, a
            :class:`~rentarag.WatermarkedCorpus` -- its original texts are used, ...).
        llm: a spec for :func:`~rentarag.get_llm`, a :class:`~rentarag.llm.ChatModel`
            or a function ``prompt -> str``.
        n: number of questions; ``None`` = one per document in order. Otherwise
            documents are sampled without replacement (cycling when ``n`` exceeds them).
        seed: sampling seed.
        temperature, max_tokens: generation settings (paper: 0.2, no limit).
        max_chars: document prefix shown to the LLM (paper: 1500).
        prompt: template with ``{doc}``.
        max_workers: parallel LLM calls; the output order is fixed.
        text_field, id_field, provider_field: field names for dict rows and
            corpus files, as in :func:`~rentarag.watermark_corpus`.
        llm_kwargs: options for :func:`~rentarag.get_llm` (``base_url``, ``api_key``, ...).

    With ``n`` larger than the number of documents, documents are reused, so
    some questions repeat; diverse questions give more independent evidence.
    """
    texts = _source_texts(docs, text_field=text_field, id_field=id_field, provider_field=provider_field)
    if not texts:
        return []
    model = get_llm(llm, **llm_kwargs)
    if n is None:
        idx = list(range(len(texts)))
    else:
        rng = random.Random(seed)
        idx = []
        while len(idx) < n:
            perm = list(range(len(texts)))
            rng.shuffle(perm)
            idx += perm[: n - len(idx)]

    def one(i: int) -> str:
        p = prompt.format(doc=texts[i][:max_chars])
        out = call_with_retries(lambda: model.complete(p, temperature=temperature, max_tokens=max_tokens), max_retries=max_retries)
        return _clean(out)

    if max_workers <= 1:
        return [one(i) for i in idx]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, idx))
