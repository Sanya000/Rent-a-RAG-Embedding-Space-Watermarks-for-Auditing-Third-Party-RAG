"""Corpus-level API: load documents, watermark them for any number of providers.

Documents can come from a file (:func:`load_corpus`: JSONL, JSON, CSV/TSV or a
directory of ``.txt`` files) or from Python objects (dicts, ``(provider, text)``
pairs, ``{provider: [texts]}``). :func:`watermark_corpus` watermarks every
document with its provider's key, in parallel threads if asked, with
resumable caching, and returns the documents in input order.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import sys
import threading
from collections.abc import Hashable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_EXCEPTION, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .embed import Embedder, embedder_name, get_embedder
from .keys import ProviderKeys, as_provider_keys
from .paraphrase import Paraphraser, get_paraphraser
from .watermark import DirBucketWatermarker, WatermarkConfig, WatermarkResult

__all__ = [
    "Document",
    "WatermarkedDoc",
    "WatermarkedCorpus",
    "load_corpus",
    "iter_documents",
    "watermark_corpus",
]

_PROVIDER_FIELDS = ("provider", "provider_id")
_TEXT_SUFFIXES = (".txt", ".md")


@dataclass
class Document:
    """One source document: ``id``, owning ``provider`` and ``text``.

    ``meta`` keeps the other fields of the input row; ``provider_field`` is the
    field name the provider was read from (used when writing rows back).
    """

    id: Any
    provider: Hashable
    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    provider_field: str = "provider"


def _pick(row: Mapping[str, Any], names: Sequence[str]) -> str | None:
    return next((n for n in names if n in row), None)


def iter_documents(
    docs: Any,
    *,
    id_field: str = "id",
    provider_field: str | None = None,
    text_field: str | Sequence[str] = "text",
    default_provider: Hashable | None = None,
) -> Iterator[Document]:
    """Normalise any supported corpus shape into :class:`Document` objects.

    ``docs`` may be a path (see :func:`load_corpus`), a mapping
    ``{provider: [text or dict, ...]}``, or an iterable of :class:`Document`,
    :class:`WatermarkedDoc`, dicts, ``(provider, text)`` / ``(id, provider, text)``
    tuples or plain strings (with ``default_provider``). For dicts the
    provider is read from ``provider_field`` (default: ``provider``, then
    ``provider_id``) and the text from ``text_field`` (a name or a list of
    candidate names). Missing ids become ``doc-<index>``.
    """
    if isinstance(docs, (str, Path)):
        yield from load_corpus(
            docs, id_field=id_field, provider_field=provider_field, text_field=text_field, default_provider=default_provider
        )
        return
    text_fields = (text_field,) if isinstance(text_field, str) else tuple(text_field)
    prov_fields = (provider_field,) if provider_field else _PROVIDER_FIELDS

    def from_row(row: Mapping[str, Any], i: int, provider: Hashable | None) -> Document:
        tf = _pick(row, text_fields)
        if tf is None:
            raise ValueError(f"document {i} has no {' / '.join(map(repr, text_fields))} field (fields: {sorted(row)})")
        pf = _pick(row, prov_fields)
        if pf is not None:
            provider = row[pf]
        elif provider is None:
            provider = default_provider
        if provider is None:
            raise ValueError(
                f"document {i} has no {' / '.join(map(repr, prov_fields))} field; set the provider field or a default provider"
            )
        doc_id = row.get(id_field, f"doc-{i}")
        return Document(doc_id, provider, str(row[tf] or ""), dict(row), pf or "provider")

    if isinstance(docs, Mapping):
        i = 0
        for provider, items in docs.items():
            if isinstance(items, (str, bytes)):
                raise TypeError(f"docs[{provider!r}] must be a list of documents, not a single str")
            for j, item in enumerate(items):
                if isinstance(item, Mapping):
                    yield from_row(item, i, provider)
                else:
                    yield Document(f"{provider}-{j}", provider, str(item))
                i += 1
        return

    for i, item in enumerate(docs):
        if isinstance(item, Document):
            yield item
        elif isinstance(item, WatermarkedDoc):
            yield Document(item.id, item.provider, item.text, dict(item.meta), item.provider_field)
        elif isinstance(item, Mapping):
            yield from_row(item, i, None)
        elif isinstance(item, (tuple, list)) and len(item) in (2, 3):
            if len(item) == 2:
                yield Document(f"doc-{i}", item[0], str(item[1]))
            else:
                yield Document(item[0], item[1], str(item[2]))
        elif isinstance(item, str):
            if default_provider is None:
                raise ValueError(f"document {i} is a plain string; pass default_provider=... or (provider, text) pairs")
            yield Document(f"doc-{i}", default_provider, item)
        else:
            raise TypeError(f"unsupported document type {type(item).__name__} at index {i}")


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open(encoding="utf-8", newline="")


def _read_rows(path: Path) -> Any:
    name = path.name.lower().removesuffix(".gz")
    if name.endswith((".jsonl", ".ndjson")):
        rows = []
        with _open_text(path) as f:
            for lineno, line in enumerate(f, 1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as e:
                    raise ValueError(f"{path}:{lineno}: invalid JSON: {e.msg} (column {e.colno})") from e
        return rows
    if name.endswith(".json"):
        with _open_text(path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{e.lineno}: invalid JSON: {e.msg} (column {e.colno})") from e
        if isinstance(data, Mapping):
            for k in ("documents", "docs", "data", "rows"):
                if isinstance(data.get(k), list):
                    return data[k]
        return data
    if name.endswith((".csv", ".tsv")):
        csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
        with _open_text(path) as f:
            return list(csv.DictReader(f, delimiter="\t" if name.endswith(".tsv") else ","))
    raise ValueError(f"unsupported corpus format {path.suffix!r}: use .jsonl, .json, .csv, .tsv or a directory of .txt files")


def load_corpus(
    path: str | Path,
    *,
    id_field: str = "id",
    provider_field: str | None = None,
    text_field: str | Sequence[str] = "text",
    default_provider: Hashable | None = None,
) -> list[Document]:
    """Load documents from a file or directory.

    * ``.jsonl`` / ``.ndjson`` (optionally ``.gz``): one JSON object per line.
    * ``.json``: a list of objects, ``{"documents": [...]}``, or ``{provider: [texts]}``.
    * ``.csv`` / ``.tsv``: one row per document with a header.
    * a directory: every ``.txt``/``.md`` file below it; the provider is the
      first sub-directory (``corpus/acme/report.txt`` -> provider ``acme``;
      files at the top level use ``default_provider``), the id is the
      relative path without suffix.

    Field names are configurable; the provider comes from ``provider_field``
    (default: ``provider`` or ``provider_id``) or ``default_provider``.
    """
    p = Path(path)
    if p.is_dir():
        out: list[Document] = []
        files = sorted(f for f in p.rglob("*") if f.is_file() and f.suffix.lower() in _TEXT_SUFFIXES)
        for f in files:
            rel = f.relative_to(p)
            provider: Hashable | None = rel.parts[0] if len(rel.parts) > 1 else default_provider
            if provider is None:
                raise ValueError(f"{rel} is not inside a provider sub-directory; pass default_provider=...")
            doc_id = rel.with_suffix("").as_posix()
            out.append(Document(doc_id, provider, f.read_text(encoding="utf-8"), {"id": doc_id, "path": rel.as_posix()}))
        if not out:
            raise ValueError(f"no .txt or .md files found under {p}")
        return out
    if not p.is_file():
        raise FileNotFoundError(2, "corpus not found", str(p))
    return list(
        iter_documents(
            _read_rows(p), id_field=id_field, provider_field=provider_field, text_field=text_field, default_provider=default_provider
        )
    )


@dataclass
class WatermarkedDoc:
    """A watermarked document: ``text`` is the version to publish, ``original`` the input."""

    id: Any
    provider: Hashable
    text: str
    original: str
    stats: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)
    provider_field: str = "provider"
    result: WatermarkResult | None = None

    def to_row(self, *, out_field: str = "text_wm", text_field: str = "text", id_field: str = "id") -> dict[str, Any]:
        """JSON row: the input fields plus ``out_field`` and ``wm_*`` statistics."""
        row = dict(self.meta)
        row[id_field] = self.id
        row[self.provider_field] = self.provider
        row.setdefault(text_field, self.original)
        row[out_field] = self.text
        row.update({f"wm_{k}": v for k, v in self.stats.items()})
        return row

    @classmethod
    def from_row(cls, row: Mapping[str, Any], doc: Document, out_field: str = "text_wm") -> WatermarkedDoc:
        stats = {k[3:]: v for k, v in row.items() if k.startswith("wm_")}
        meta = {k: v for k, v in row.items() if k != out_field and not k.startswith("wm_")}
        return cls(doc.id, doc.provider, str(row[out_field]), doc.text, stats, meta, doc.provider_field)


class WatermarkedCorpus(list):
    """List of :class:`WatermarkedDoc` in input order, with helpers."""

    def __init__(self, docs: Iterable[WatermarkedDoc] = (), *, out_field: str = "text_wm", text_field: str = "text", id_field: str = "id"):
        super().__init__(docs)
        self.out_field = out_field
        self.text_field = text_field
        self.id_field = id_field

    def rows(self) -> list[dict[str, Any]]:
        return [d.to_row(out_field=self.out_field, text_field=self.text_field, id_field=self.id_field) for d in self]

    def save(self, path: str | Path) -> None:
        """Write one JSON row per document (``.jsonl``)."""
        _write_jsonl_atomic(Path(path), self.rows())

    def by_provider(self) -> dict[Hashable, list[str]]:
        """``{provider: [watermarked texts]}``."""
        out: dict[Hashable, list[str]] = {}
        for d in self:
            out.setdefault(d.provider, []).append(d.text)
        return out

    def summary(self) -> dict[Hashable, dict[str, float]]:
        """Per-provider totals: documents, sentences, changed fraction, mean gain."""
        acc: dict[Hashable, dict[str, float]] = {}
        for d in self:
            a = acc.setdefault(d.provider, {"documents": 0, "sentences": 0, "changed_sentences": 0, "gain_sum": 0.0})
            n = int(d.stats.get("n_sentences", 0))
            a["documents"] += 1
            a["sentences"] += n
            a["changed_sentences"] += int(d.stats.get("changed_sentences", 0))
            a["gain_sum"] += float(d.stats.get("gain_mean", 0.0)) * n
        return {
            p: {
                "documents": int(a["documents"]),
                "sentences": int(a["sentences"]),
                "changed_frac": a["changed_sentences"] / max(1, a["sentences"]),
                "gain_mean": a["gain_sum"] / max(1, a["sentences"]),
            }
            for p, a in acc.items()
        }


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def _read_cache(path: Path, id_field: str = "id") -> dict[str, dict[str, Any]]:
    """Rows already written to ``path`` keyed by ``id_field``; a partial last line (interrupted run) is dropped."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        cut = data.rfind(b"\n") + 1
        with path.open("r+b") as f:
            f.truncate(cut)
        data = data[:cut]
    rows: dict[str, dict[str, Any]] = {}
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and id_field in row:
            rows[str(row[id_field])] = row
    return rows


def watermark_corpus(
    docs: Any,
    keys: ProviderKeys | Mapping[Hashable, str] | Sequence[str] | str | Path,
    *,
    paraphraser: Paraphraser | str,
    embedder: Embedder | str | None = None,
    config: WatermarkConfig | str | Mapping[str, Any] | None = None,
    max_workers: int = 1,
    cache_path: str | Path | None = None,
    progress: bool = False,
    id_field: str = "id",
    provider_field: str | None = None,
    text_field: str = "text",
    out_field: str = "text_wm",
    default_provider: Hashable | None = None,
    selection_embedder: Embedder | str | None = None,
    keep_sentences: bool = False,
) -> WatermarkedCorpus:
    """Watermark every document with its provider's key.

    Args:
        docs: documents in any form accepted by :func:`iter_documents` (a path,
            dicts, ``(provider, text)`` pairs, ``{provider: [texts]}``, ...).
        keys: :class:`~rentarag.ProviderKeys`, a key file path, ``{provider: key_hex}``
            or a list of keys (provider ids ``0..n-1``). Provider ids in the
            documents may be names or indices (``"0"`` matches ``0``).
        paraphraser: a spec for :func:`~rentarag.get_paraphraser`
            (``"openai/gpt-4o-mini"``, ``"anthropic/claude-haiku-5-5"``,
            ``"ollama/llama3.1"``, ``"hf/Qwen/Qwen2.5-1.5B-Instruct"``, ``"toy"``) or a callable.
        embedder: encoder spec or object (default ``all-mpnet-base-v2``); the auditor must use the same.
            Its name and dimension are stored in each row (``wm_embedder``, ``wm_dim``)
            and checked by :meth:`~rentarag.Auditor.from_corpus`.
        config: :class:`~rentarag.WatermarkConfig`, preset name or dict (default: paper settings).
        max_workers: documents processed in parallel threads. API backends
            spend most of their time waiting on the network, so 8-32 workers
            give a near-linear speed-up up to the provider's rate limit.
            Results are returned in input order.
        cache_path: JSONL file to which each finished document is appended.
            Re-running with the same path skips documents already there (same
            id, provider and input text), so an interrupted run resumes. A
            paraphraser given as a spec string is only built when documents are
            left to watermark.
        progress: show a progress bar over documents.
        id_field, provider_field, text_field: input field names for dict rows.
        out_field: field name of the watermarked text in output rows.
        default_provider: provider for rows without a provider field.
        selection_embedder: encoder for MMR sentence selection (see ``config``).
        keep_sentences: keep per-sentence logs in ``WatermarkedDoc.result``
            (not stored in the cache).

    Returns:
        A :class:`WatermarkedCorpus` (list of :class:`WatermarkedDoc`).
    """
    pk = as_provider_keys(keys)
    emb = get_embedder(embedder)
    cfg = WatermarkConfig.coerce(config)
    documents = list(
        iter_documents(docs, id_field=id_field, provider_field=provider_field, text_field=text_field, default_provider=default_provider)
    )
    for d in documents:
        d.provider = pk.resolve(d.provider)
    if cache_path is not None:
        seen: set[str] = set()
        for d in documents:
            if str(d.id) in seen:
                raise ValueError(f"duplicate document id {d.id!r}; ids must be unique when cache_path is set")
            seen.add(str(d.id))

    results: list[WatermarkedDoc | None] = [None] * len(documents)
    cache = Path(cache_path) if cache_path is not None else None
    if cache is not None and cache.exists():
        cached = _read_cache(cache, id_field)
        for i, d in enumerate(documents):
            row = cached.get(str(d.id))
            if (
                row is not None
                and out_field in row
                and str(row.get(text_field, "")) == d.text
                and str(row.get(d.provider_field)) == str(d.provider)
            ):
                results[i] = WatermarkedDoc.from_row(row, d, out_field)

    todo = [i for i, r in enumerate(results) if r is None]
    marks: dict[Hashable, DirBucketWatermarker] = {}
    if todo:  # the paraphraser is only built (credentials checked) when there is work left
        para = get_paraphraser(paraphraser)
        sel = selection_embedder
        if sel is None and cfg.sentence_selection == "mmr" and cfg.max_sentences is not None:
            sel = get_embedder(cfg.selection_model) if cfg.selection_model else emb
        for i in todo:
            d = documents[i]
            if d.provider not in marks:
                marks[d.provider] = DirBucketWatermarker(pk[d.provider], para, emb, cfg, selection_embedder=sel)
    # recorded as wm_embedder / wm_dim so the auditor can check it uses the same encoder
    encoder: dict[str, Any] = {"dim": int(emb.dim)}
    name = embedder_name(emb)
    if name:
        encoder["embedder"] = name
    lock = threading.Lock()
    fout = None
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        fout = cache.open("a", encoding="utf-8")
    bar = None
    if progress:
        from tqdm import tqdm

        bar = tqdm(total=len(documents), initial=len(documents) - len(todo), desc="Watermarking", unit="doc")

    def run(i: int) -> None:
        d = documents[i]
        res = marks[d.provider].watermark(d.text, doc_id=str(d.id))
        stats = {**res.summary(), **encoder}
        wd = WatermarkedDoc(d.id, d.provider, res.text, d.text, stats, dict(d.meta), d.provider_field, res if keep_sentences else None)
        with lock:
            results[i] = wd
            if fout is not None:
                fout.write(json.dumps(wd.to_row(out_field=out_field, text_field=text_field, id_field=id_field), ensure_ascii=False) + "\n")
                fout.flush()
            if bar is not None:
                bar.update()

    try:
        if max_workers <= 1:
            for i in todo:
                run(i)
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures: list[Future] = [ex.submit(run, i) for i in todo]
                done, _ = wait(futures, return_when=FIRST_EXCEPTION)
                failed = [f for f in done if f.exception() is not None]
                if failed:
                    for f in futures:
                        f.cancel()
                    raise failed[0].exception()  # type: ignore[misc]
    finally:
        if fout is not None:
            fout.close()
        if bar is not None:
            bar.close()
    return WatermarkedCorpus(results, out_field=out_field, text_field=text_field, id_field=id_field)  # type: ignore[arg-type]
