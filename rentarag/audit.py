"""Auditor-side DirBucket detection (paper Sec. 3-4, Alg. ``dirbucket-detector``).

For every audited answer:

1. split it into within-sentence windows (``W`` words, stride ``s``);
2. align each window to every provider's *reference windows* (windows of the
   provider's watermarked documents) by cosine similarity; with competitive
   assignment a window goes to at most one provider, and only if its best
   cosine is ``>= sim_threshold`` and beats the runner-up provider by at least
   ``tie_margin``;
3. score each assigned window as ``<u_answer, v_k(b_ref)>`` where ``b_ref`` is
   the SimHash bucket of the *matched reference window* (reference-bucket
   anchoring: the bucket the provider watermarked, not the answer's own);
4. pool per-provider sums ``S_i`` and supports ``T_i`` over answers, compute
   ``Z_i = S_i / sqrt(T_i/d)``, gate providers with ``T_i < T_min``, and apply
   Benjamini-Hochberg across providers.

:class:`AlignedScorer` reproduces ``MultiDirBucketDetector.detect_windows_multi_aligned``
and :class:`Auditor` reproduces the pooling of the paper's audit runner.
"""

from __future__ import annotations

import math
import warnings
from collections import OrderedDict
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ._cfg import coerce_config, config_from_dict, config_to_dict
from .embed import Embedder, embedder_name, get_embedder, same_embedder_name
from .keys import DirectionManager, DirectionSource, OrthoDirectionManager, SimHashCoder, as_provider_keys
from .stats import gate_by_min_support, pooled_z
from .text import text_windows

__all__ = [
    "AuditConfig",
    "AUDIT_PRESETS",
    "AuditStep",
    "answer_text",
    "ProviderReference",
    "ProviderAnswerScore",
    "AnswerScore",
    "AlignedScorer",
    "QueryReferenceSelector",
    "ProviderResult",
    "AuditReport",
    "Auditor",
]

ProviderId = Hashable
Sources = Mapping[ProviderId, Sequence[str]]


@dataclass(frozen=True)
class AuditConfig:
    """Auditor hyperparameters; defaults are the paper's operating point.

    Attributes:
        win_words: window length ``W`` (words). Shorter windows align more
            easily after paraphrasing but each carries less signal.
        stride_words: window stride ``s`` (words).
        sim_threshold: minimum cosine for an answer window to align (``tau_sim``).
            Lower values gather more support from loosely reworded answers and
            admit more unrelated text.
        tie_margin: minimum best-minus-second-best cosine under competitive
            assignment (``tau_margin``); ``None`` disables the margin. Not
            applied when the runner-up cosine is negative or only one provider
            has references. Higher values drop windows that several providers explain.
        competitive: assign each window to at most one provider. If ``False``
            a window counts for every provider above ``sim_threshold``.
        min_support: support gate ``T_min``; providers with fewer pooled
            aligned windows are not tested. Lower values decide earlier on less
            evidence; the paper uses 25.
        alpha: BH false-discovery-rate level across providers.
        z_threshold: reference line for ``first_crossed_at`` (not the decision rule).
        bucket_bits, bucket_seed: SimHash parameters; must match the provider's.
        direction_scheme: ``"naive"`` (paper) or ``"ortho"`` (ablation).
        reference_selection: ``"all"`` uses every provider document as a
            reference for every answer; ``"query"`` picks references per audit
            query by retrieval similarity (the paper's audits). ``"query"``
            scales to large corpora and needs the query of each answer.
        query_embedder: encoder spec for ``"query"`` selection.
        query_sim_threshold: minimum query-document cosine for a reference.
        max_reference_docs: at most this many references per query (0 = no limit).
        max_reference_docs_per_provider: per-provider cap (0 = no limit).
    """

    win_words: int = 28
    stride_words: int = 14
    sim_threshold: float = 0.82
    tie_margin: float | None = 0.05
    competitive: bool = True
    min_support: int = 25
    alpha: float = 0.05
    z_threshold: float = 4.0
    bucket_bits: int = 96
    bucket_seed: int = 2025
    direction_scheme: str = "naive"
    reference_selection: str = "all"
    query_embedder: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_sim_threshold: float = 0.48
    max_reference_docs: int = 120
    max_reference_docs_per_provider: int = 0

    def __post_init__(self) -> None:
        if self.reference_selection not in ("all", "query"):
            raise ValueError("reference_selection must be 'all' or 'query'")
        if self.direction_scheme.lower() not in ("naive", "ortho"):
            raise ValueError(f"unknown direction_scheme {self.direction_scheme!r}")
        if self.win_words < 1 or not 0.0 < self.alpha < 1.0:
            raise ValueError("win_words must be >= 1 and alpha in (0, 1)")
        if self.stride_words < 1:
            raise ValueError("stride_words must be >= 1")
        if not -1.0 <= self.sim_threshold <= 1.0:
            raise ValueError("sim_threshold is a cosine and must be in [-1, 1]")
        if self.min_support < 0:
            raise ValueError("min_support must be >= 0")
        if self.tie_margin is not None and self.tie_margin < 0:
            raise ValueError("tie_margin must be >= 0 (or None to disable it)")
        if self.bucket_bits < 1:
            raise ValueError("bucket_bits must be >= 1")
        if not -1.0 <= self.query_sim_threshold <= 1.0:
            raise ValueError("query_sim_threshold is a cosine and must be in [-1, 1]")
        if self.max_reference_docs < 0 or self.max_reference_docs_per_provider < 0:
            raise ValueError("max_reference_docs and max_reference_docs_per_provider must be >= 0 (0 = no limit)")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> AuditConfig:
        """Build from a dict; an optional ``"preset"`` key is applied first, other keys override it."""
        return config_from_dict(cls, data, AUDIT_PRESETS)

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> AuditConfig:
        """A named preset (see :data:`AUDIT_PRESETS`) with optional overrides."""
        return cls.from_dict({"preset": name, **overrides})

    def replace(self, **changes: Any) -> AuditConfig:
        return replace(self, **changes)

    @classmethod
    def coerce(cls, value: Any) -> AuditConfig:
        """Accept ``None`` (defaults), a preset name, a dict or a config."""
        return coerce_config(cls, value, AUDIT_PRESETS)


AUDIT_PRESETS: dict[str, dict[str, Any]] = {
    "paper": {},
    "paper-query": {"reference_selection": "query"},
    "small-corpus": {"min_support": 10},
}
"""Audit presets.

* ``paper``: the paper's operating point (the defaults); every document is a reference.
* ``paper-query``: ``paper`` with per-query reference selection, as in the paper's audits.
* ``small-corpus``: ``min_support=10`` for corpora or answer sets too small to reach 25
  aligned windows per provider; decisions rest on less evidence.
"""


@dataclass
class ProviderReference:
    """A provider known to the auditor.

    ``texts`` are the provider's *watermarked* documents as published; they are
    the reference set answers are aligned against.
    """

    provider_id: ProviderId
    key_hex: str
    texts: Sequence[str] = ()

    def __post_init__(self) -> None:
        if isinstance(self.texts, (str, bytes)):
            raise TypeError(
                f"ProviderReference({self.provider_id!r}): texts must be a sequence of documents, "
                "not a single str (wrap it in a list)"
            )


@dataclass
class ProviderAnswerScore:
    support: int
    mean_proj: float
    z: float

    @property
    def sum_proj(self) -> float:
        return self.mean_proj * self.support


@dataclass
class AnswerScore:
    """Per-answer alignment and scoring result."""

    providers: dict[ProviderId, ProviderAnswerScore]
    windows_total: int
    assigned_total: int
    dropped: int
    best_cos_mean: float | None = None
    second_cos_mean: float | None = None

    @property
    def winner(self) -> ProviderId | None:
        """Provider with the highest per-answer z (first one on ties).

        ``None`` when no window of the answer was assigned to any provider
        (empty answer or no alignment), so "no evidence" is not reported as a win.
        """
        if not self.providers or self.assigned_total == 0:
            return None
        return max(self.providers, key=lambda p: self.providers[p].z)


class AlignedScorer:
    """Align one answer to provider references and compute anchored scores.

    Reference windows are embedded once per distinct reference text and cached
    (``cache_size`` entries); caching does not change results because the same
    window list is always embedded in one call, exactly as in the paper code.
    """

    def __init__(
        self,
        keys: Mapping[ProviderId, str],
        embedder: Embedder | str | None = None,
        config: AuditConfig | str | Mapping[str, Any] | None = None,
        *,
        cache_size: int = 64,
    ):
        self.config = config = AuditConfig.coerce(config)
        self.embedder = get_embedder(embedder)
        self.dim = int(self.embedder.dim)
        self.coder = SimHashCoder(self.dim, config.bucket_bits, config.bucket_seed)
        self.provider_ids: list[ProviderId] = list(keys)
        scheme = config.direction_scheme.lower()
        if scheme == "naive":
            self.directions: dict[ProviderId, DirectionSource] = {
                pid: DirectionManager(k, self.dim) for pid, k in keys.items()
            }
        elif scheme == "ortho":
            mgr = OrthoDirectionManager(keys, self.dim)
            self.directions = {pid: mgr.for_provider(pid) for pid in keys}
        else:
            raise ValueError(f"unknown direction_scheme {config.direction_scheme!r}")
        self.cache_size = int(cache_size)
        self._ref_cache: OrderedDict[str, tuple[np.ndarray, list[bytes]] | None] = OrderedDict()

    def _reference(self, merged: str) -> tuple[np.ndarray, list[bytes]] | None:
        if merged in self._ref_cache:
            self._ref_cache.move_to_end(merged)
            return self._ref_cache[merged]
        wins = text_windows(merged, self.config.win_words, self.config.stride_words)
        ref = None
        if wins:
            U = self.embedder(wins)
            ref = (U, [self.coder.code(U[j]) for j in range(U.shape[0])])
        if self.cache_size > 0:
            self._ref_cache[merged] = ref
            if len(self._ref_cache) > self.cache_size:
                self._ref_cache.popitem(last=False)
        return ref

    def _merged_sources(self, sources: Sources | Iterable[tuple[ProviderId, str]]) -> list[tuple[ProviderId, str]]:
        grouped: dict[ProviderId, list[str]] = {}
        items = sources.items() if isinstance(sources, Mapping) else ((p, [t]) for p, t in sources)
        for pid, docs in items:
            if pid not in self.directions:
                raise ValueError(f"reference for unknown provider {pid!r}")
            if isinstance(docs, (str, bytes)):
                raise TypeError(
                    f"sources[{pid!r}] must be a sequence of documents, not a single str (wrap it in a list)"
                )
            grouped.setdefault(pid, []).extend(str(d) for d in docs if str(d).strip())
        out = []
        for pid in self.provider_ids:
            merged = "\n\n".join(grouped.get(pid, []))
            if merged.strip():
                out.append((pid, merged))
        return out

    def score(self, answer: str, sources: Sources | Iterable[tuple[ProviderId, str]]) -> AnswerScore:
        """Score ``answer`` against per-provider reference documents.

        ``sources`` maps provider id to its reference documents (or is a list of
        ``(provider_id, text)`` pairs). A provider's documents are joined with
        blank lines before windowing, as in the paper's audit runner.
        """
        cfg = self.config
        ans_wins = text_windows(answer, cfg.win_words, cfg.stride_words)
        empty = {pid: ProviderAnswerScore(0, 0.0, 0.0) for pid in self.provider_ids}
        if not ans_wins:
            return AnswerScore(empty, 0, 0, 0)
        U_ans = self.embedder(ans_wins)

        refs: dict[ProviderId, tuple[np.ndarray, list[bytes]]] = {}
        for pid, merged in self._merged_sources(sources):
            ref = self._reference(merged)
            if ref is not None:
                refs[pid] = ref

        assigned: dict[ProviderId, list[tuple[int, int]]] = {pid: [] for pid in self.provider_ids}
        dropped = 0
        best_cos: list[float] = []
        second_cos: list[float] = []
        for i in range(U_ans.shape[0]):
            u = U_ans[i]
            best_by_pid: dict[ProviderId, tuple[float, int]] = {}
            for pid, (Uref, _) in refs.items():
                sims = Uref @ u
                j = int(np.argmax(sims))
                best_by_pid[pid] = (float(sims[j]), j)
            if not best_by_pid:
                dropped += 1
                continue
            if cfg.competitive:
                pid_star, (best, j_star) = max(best_by_pid.items(), key=lambda kv: kv[1][0])
                best_cos.append(best)
                if len(best_by_pid) > 1:
                    second = sorted((t[0] for t in best_by_pid.values()), reverse=True)[1]
                    second_cos.append(second)
                else:
                    second = -1.0
                if best < cfg.sim_threshold or (
                    cfg.tie_margin is not None and second >= 0 and (best - second) < cfg.tie_margin
                ):
                    dropped += 1
                    continue
                assigned[pid_star].append((i, j_star))
            else:
                hit = False
                for pid, (val, j_star) in best_by_pid.items():
                    if val >= cfg.sim_threshold:
                        assigned[pid].append((i, j_star))
                        best_cos.append(val)
                        hit = True
                if not hit:
                    dropped += 1

        out: dict[ProviderId, ProviderAnswerScore] = {}
        for pid in self.provider_ids:
            pairs = assigned[pid]
            if not pairs or pid not in refs:
                out[pid] = ProviderAnswerScore(0, 0.0, 0.0)
                continue
            codes = refs[pid][1]
            dirs = self.directions[pid]
            arr = np.array([float(np.dot(U_ans[i], dirs.direction(codes[j]))) for i, j in pairs], dtype=np.float32)
            mean = float(arr.mean())
            out[pid] = ProviderAnswerScore(int(arr.size), mean, mean * math.sqrt(self.dim * int(arr.size)))
        return AnswerScore(
            providers=out,
            windows_total=int(U_ans.shape[0]),
            assigned_total=sum(len(v) for v in assigned.values()),
            dropped=dropped,
            best_cos_mean=float(np.mean(best_cos)) if best_cos else None,
            second_cos_mean=float(np.mean(second_cos)) if (cfg.competitive and second_cos) else None,
        )


class QueryReferenceSelector:
    """Choose reference documents per query (the paper's ``global_corpus`` mode).

    The auditor does not see which documents the RAG system retrieved. Instead,
    every document whose retrieval embedding (first ``max_chars`` characters)
    has cosine ``>= sim_threshold`` with the audit query is used as a reference,
    most similar first, up to ``max_docs`` documents (and ``max_docs_per_provider``
    per provider; 0 = unlimited). Paper values: ``all-MiniLM-L6-v2``, 0.48, 120, 0.
    """

    def __init__(
        self,
        documents: Sequence[tuple[ProviderId, str]],
        embedder: Embedder | None = None,
        *,
        sim_threshold: float = 0.48,
        max_docs: int = 120,
        max_docs_per_provider: int = 0,
        max_chars: int = 1000,
        doc_embeddings: np.ndarray | None = None,
    ):
        self.documents = list(documents)
        self.embedder = get_embedder(embedder if embedder is not None else "sentence-transformers/all-MiniLM-L6-v2")
        self.sim_threshold = float(sim_threshold)
        self.max_docs = int(max_docs)
        self.max_docs_per_provider = int(max_docs_per_provider)
        if doc_embeddings is None:
            doc_embeddings = self.embedder([t[:max_chars] for _, t in self.documents])
        self.X = np.asarray(doc_embeddings)
        if self.X.shape[0] != len(self.documents):
            raise ValueError("doc_embeddings must have one row per document")

    @classmethod
    def from_providers(cls, providers: Sequence[ProviderReference], embedder: Embedder | None = None, **kwargs: Any) -> QueryReferenceSelector:
        for p in providers:
            if isinstance(p.texts, (str, bytes)):
                raise TypeError(
                    f"provider {p.provider_id!r}: texts must be a sequence of documents, not a single str"
                )
        return cls([(p.provider_id, t) for p in providers for t in p.texts], embedder, **kwargs)

    def select(self, query: str) -> dict[ProviderId, list[str]]:
        q = np.asarray(self.embedder([query])[0], dtype=np.float32)
        sims = self.X @ q
        chunks: dict[ProviderId, list[str]] = {}
        n_sel = 0
        for idx in np.argsort(-sims):
            if float(sims[int(idx)]) < self.sim_threshold:
                break
            if self.max_docs > 0 and n_sel >= self.max_docs:
                break
            pid, txt = self.documents[int(idx)]
            if self.max_docs_per_provider > 0 and len(chunks.get(pid, [])) >= self.max_docs_per_provider:
                continue
            txt = str(txt or "").strip()
            if not txt:
                continue
            chunks.setdefault(pid, []).append(txt)
            n_sel += 1
        return chunks


@dataclass
class ProviderResult:
    provider_id: ProviderId
    support: int
    sum_proj: float
    mean_proj: float
    z_raw: float
    z: float
    p_value: float
    q_value: float
    eligible: bool
    detected: bool
    first_crossed_at: int | None


@dataclass
class AuditStep:
    """One query of a live audit (:meth:`Auditor.audit`)."""

    step: int
    query: str
    answer: str
    support: dict[ProviderId, int]
    z: dict[ProviderId, float]
    detected: list[ProviderId]
    error: str | None = None


@dataclass
class AuditReport:
    """Pooled audit outcome. ``detected`` providers passed the support gate and BH.

    After :meth:`Auditor.audit`, ``trajectory`` holds one :class:`AuditStep`
    per query (pooled support, Z and detections after that answer).
    """

    n_answers: int
    dim: int
    config: dict[str, Any]
    providers: list[ProviderResult] = field(default_factory=list)
    trajectory: list[AuditStep] = field(default_factory=list)

    @property
    def detected(self) -> list[ProviderId]:
        return [p.provider_id for p in self.providers if p.detected]

    @property
    def first_detected_at(self) -> dict[ProviderId, int]:
        """Query step at which each provider was first detected during a live audit."""
        out: dict[ProviderId, int] = {}
        for st in self.trajectory:
            for pid in st.detected:
                out.setdefault(pid, st.step)
        return out

    def __getitem__(self, provider_id: ProviderId) -> ProviderResult:
        for p in self.providers:
            if p.provider_id == provider_id:
                return p
        raise KeyError(provider_id)

    def to_dict(self) -> dict[str, Any]:
        out = {
            "n_answers": self.n_answers,
            "dim": self.dim,
            "config": self.config,
            "detected": list(self.detected),
            "providers": [asdict(p) for p in self.providers],
        }
        if self.trajectory:
            out["trajectory"] = [
                {**asdict(st), "support": {str(k): v for k, v in st.support.items()}, "z": {str(k): v for k, v in st.z.items()}}
                for st in self.trajectory
            ]
        return out

    def __str__(self) -> str:
        w = max([10] + [len(str(p.provider_id)) for p in self.providers])
        head = f"{'provider':>{w}} {'T':>6} {'Z':>8} {'p':>10} {'q(BH)':>10} {'eligible':>9} {'DETECTED':>9}"
        rows = [
            f"{str(p.provider_id):>{w}} {p.support:>6d} {p.z_raw:>8.2f} {p.p_value:>10.2e} {p.q_value:>10.2e} "
            f"{str(p.eligible):>9} {('YES' if p.detected else '-'):>9}"
            for p in self.providers
        ]
        return "\n".join([f"answers audited: {self.n_answers}", head, *rows])


def answer_text(raw: Any) -> str:
    """Extract the answer text from what a RAG call returned.

    Handles plain strings, dicts with an ``answer`` / ``result`` / ``output`` /
    ``response`` / ``text`` / ``content`` field (LangChain chains, REST
    payloads), and objects with a ``content`` (chat messages), ``response``
    (LlamaIndex) or ``text`` attribute.
    """
    if isinstance(raw, str):
        return raw
    if isinstance(raw, Mapping):
        for k in ("answer", "result", "output", "response", "text", "content", "output_text"):
            if k in raw:
                return answer_text(raw[k])
        return str(raw)
    for attr in ("content", "response", "text"):
        v = getattr(raw, attr, None)
        if isinstance(v, str):
            return v
    return "" if raw is None else str(raw)


class Auditor:
    """Pool DirBucket evidence over audited answers and test every provider.

    Build it from watermarked documents and keys, then either feed answers you
    already have or let it query a live RAG system::

        auditor = Auditor.from_corpus(watermarked, keys)            # paper settings
        auditor.add_answers(answers)                                # offline answers
        report = auditor.audit(rag_chain.invoke, queries)           # or query a live system
        print(report)

    With a ``selector`` (e.g. :class:`QueryReferenceSelector`, or
    ``config.reference_selection="query"``) references are chosen per audit
    query, which must then be passed to :meth:`add_answer`; otherwise all
    provider documents are references for every answer.
    """

    def __init__(
        self,
        providers: Sequence[ProviderReference],
        config: AuditConfig | str | Mapping[str, Any] | None = None,
        embedder: Embedder | str | None = None,
        *,
        selector: QueryReferenceSelector | None = None,
    ):
        ids = [p.provider_id for p in providers]
        if len(set(ids)) != len(ids):
            raise ValueError("provider ids must be unique")
        self.providers = list(providers)
        self.config = config = AuditConfig.coerce(config)
        self.scorer = AlignedScorer({p.provider_id: p.key_hex for p in providers}, embedder, config)
        if selector is None and config.reference_selection == "query":
            selector = QueryReferenceSelector.from_providers(
                self.providers,
                config.query_embedder,
                sim_threshold=config.query_sim_threshold,
                max_docs=config.max_reference_docs,
                max_docs_per_provider=config.max_reference_docs_per_provider,
            )
        self.selector = selector
        self._static_sources = {p.provider_id: list(p.texts) for p in providers}
        self.reset()

    @classmethod
    def from_corpus(
        cls,
        corpus: Any,
        keys: Any,
        config: AuditConfig | str | Mapping[str, Any] | None = None,
        embedder: Embedder | str | None = None,
        *,
        selector: QueryReferenceSelector | None = None,
        text_field: str | Sequence[str] = ("text_wm", "text"),
        provider_field: str | None = None,
        id_field: str = "id",
    ) -> Auditor:
        """Auditor for every provider in ``keys`` with references from ``corpus``.

        ``corpus`` is the published (watermarked) corpus in any form accepted
        by :func:`~rentarag.iter_documents`: the result of
        :func:`~rentarag.watermark_corpus`, a path, dicts (text from
        ``text_wm``, else ``text``), ``(provider, text)`` pairs or
        ``{provider: [texts]}``. ``keys`` is a :class:`~rentarag.ProviderKeys`,
        a key file path, a mapping or a list. Providers without documents are
        still tested (they collect no support).

        Corpora written by :func:`~rentarag.watermark_corpus` record the
        encoder they were watermarked with (``wm_dim``, ``wm_embedder``). A
        different dimension raises ``ValueError``; a different model name
        emits a warning, since detection only works with the provider's encoder.
        """
        from .corpus import WatermarkedDoc, iter_documents

        pk = as_provider_keys(keys)
        emb = get_embedder(embedder)
        texts: dict[ProviderId, list[str]] = {pid: [] for pid in pk}
        dims: set[int] = set()
        names: set[str] = set()

        def note(dim: Any, name: Any) -> None:
            if dim not in (None, ""):
                dims.add(int(dim))
            if name:
                names.add(str(name))

        if isinstance(corpus, Sequence) and not isinstance(corpus, (str, bytes)):
            for item in corpus:
                if isinstance(item, WatermarkedDoc):
                    note(item.stats.get("dim"), item.stats.get("embedder"))
        for doc in iter_documents(corpus, id_field=id_field, provider_field=provider_field, text_field=text_field):
            texts[pk.resolve(doc.provider)].append(doc.text)
            note(doc.meta.get("wm_dim"), doc.meta.get("wm_embedder"))
        used = ", ".join(sorted(names)) or "unknown encoder"
        if dims and int(emb.dim) not in dims:
            raise ValueError(
                f"the corpus was watermarked with a {'/'.join(map(str, sorted(dims)))}-dimensional encoder ({used}), "
                f"but the audit encoder {embedder_name(emb) or emb!r} has dimension {emb.dim}; "
                "audit with the encoder used to watermark"
            )
        mine = embedder_name(emb)
        if any(not same_embedder_name(n, mine) for n in names):
            warnings.warn(
                f"the corpus was watermarked with {used} but the auditor uses {mine}; "
                "detection only works with the provider's encoder",
                stacklevel=2,
            )
        refs = [ProviderReference(pid, pk[pid], texts[pid]) for pid in pk]
        return cls(refs, config, emb, selector=selector)

    @property
    def dim(self) -> int:
        return self.scorer.dim

    def reset(self) -> None:
        """Discard all pooled evidence."""
        self.n_answers = 0
        self.support = {p.provider_id: 0 for p in self.providers}
        self.sum_proj = {p.provider_id: 0.0 for p in self.providers}
        self.first_crossed_at: dict[ProviderId, int | None] = {p.provider_id: None for p in self.providers}
        self.answers: list[AnswerScore] = []
        self.trajectory: list[AuditStep] = []

    def add_answer(self, answer: str, query: str | None = None, *, sources: Sources | None = None) -> AnswerScore:
        """Score one answer and add it to the pooled evidence.

        Reference documents are, in order of precedence: explicit ``sources``,
        the selector's choice for ``query``, or every provider's ``texts``.
        """
        if sources is None:
            if self.selector is not None:
                if query is None:
                    raise ValueError("this auditor selects references by query; pass query=...")
                sources = self.selector.select(query)
            else:
                sources = self._static_sources
        res = self.scorer.score(answer, sources)
        for pid, s in res.providers.items():
            self.support[pid] += s.support
            self.sum_proj[pid] += s.mean_proj * s.support
        self.n_answers += 1
        self.answers.append(res)
        z = self.pooled_z()
        for pid, zi in zip(self.support, z):
            if (
                self.first_crossed_at[pid] is None
                and self.support[pid] >= self.config.min_support
                and zi >= self.config.z_threshold
            ):
                self.first_crossed_at[pid] = self.n_answers
        return res

    def add_answers(self, answers: Iterable[Any], *, progress: bool = False) -> list[AnswerScore]:
        """Add many answers: strings, ``(query, answer)`` pairs or dicts with ``answer`` (and ``query``)."""
        items: Iterable[Any] = answers
        if progress:
            from tqdm import tqdm

            items = tqdm(answers, desc="Scoring answers", unit="answer")
        out = []
        for item in items:
            if isinstance(item, str):
                out.append(self.add_answer(item))
            elif isinstance(item, Mapping):
                out.append(self.add_answer(answer_text(item), query=item.get("query")))
            elif isinstance(item, (tuple, list)) and len(item) == 2:
                out.append(self.add_answer(answer_text(item[1]), query=item[0]))
            else:
                out.append(self.add_answer(answer_text(item)))
        return out

    def audit(
        self,
        ask: Callable[[str], Any],
        queries: Iterable[str],
        *,
        stop_when_detected: bool | Iterable[ProviderId] = True,
        max_queries: int | None = None,
        on_error: str = "raise",
        progress: bool = False,
        callback: Callable[[AuditStep], None] | None = None,
    ) -> AuditReport:
        """Query a live black-box RAG system and pool evidence until a decision.

        Args:
            ask: the system under audit, ``query -> answer``. Any callable works:
                ``chain.invoke`` (LangChain), ``query_engine.query`` (LlamaIndex),
                a function wrapping an HTTP endpoint. Its return value is turned
                into text with :func:`answer_text`.
            queries: audit queries, asked sequentially (see
                :func:`~rentarag.generate_audit_queries`).
            stop_when_detected: ``True`` stops at the first detected provider;
                a provider id (``"acme"``) or a collection of ids stops once
                all of them are detected; ``False`` asks every query.
            max_queries: query budget.
            on_error: ``"raise"`` propagates errors from ``ask``; ``"skip"``
                records them in the trajectory and continues.
            callback: called with each :class:`AuditStep`.

        Evidence accumulates on this auditor (call :meth:`reset` to start
        over). The returned report carries the per-query ``trajectory``.
        Stopping at the first detection is a sequential test; for a fixed-size
        test, set ``stop_when_detected=False`` and ``max_queries``.
        """
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        targets: set[ProviderId] | None = None
        if isinstance(stop_when_detected, (str, int)) and not isinstance(stop_when_detected, bool):
            stop_when_detected = [stop_when_detected]
        if not isinstance(stop_when_detected, bool):
            ids = {str(p.provider_id): p.provider_id for p in self.providers}
            targets = {ids.get(str(t), t) for t in stop_when_detected}
            unknown = targets - set(self.support)
            if unknown:
                raise ValueError(f"unknown provider(s) in stop_when_detected: {sorted(map(str, unknown))}")
        bar = None
        if progress:
            from tqdm import tqdm

            bar = tqdm(total=max_queries, desc="Auditing", unit="query")
        for k, query in enumerate(queries, 1):
            if max_queries is not None and k > max_queries:
                break
            query = str(query)
            try:
                answer = answer_text(ask(query))
            except Exception as exc:
                if on_error == "raise":
                    raise
                step = AuditStep(k, query, "", dict(self.support), self._z_by_pid(), self.report().detected, repr(exc))
            else:
                self.add_answer(answer, query=query)
                step = AuditStep(k, query, answer, dict(self.support), self._z_by_pid(), self.report().detected)
            self.trajectory.append(step)
            if bar is not None:
                bar.update()
                bar.set_postfix(detected=",".join(map(str, step.detected)) or "-")
            if callback is not None:
                callback(step)
            if stop_when_detected is True and step.detected:
                break
            if targets is not None and targets <= set(step.detected):
                break
        if bar is not None:
            bar.close()
        return self.report()

    def _z_by_pid(self) -> dict[ProviderId, float]:
        return dict(zip(self.support, self.pooled_z()))

    def pooled_z(self) -> list[float]:
        """Current raw pooled Z per provider (provider order)."""
        return pooled_z(list(self.sum_proj.values()), list(self.support.values()), self.dim)

    def report(self) -> AuditReport:
        """Support gate + BH over all providers on the evidence pooled so far."""
        ids = list(self.support)
        T = [self.support[p] for p in ids]
        z_raw = self.pooled_z()
        test = gate_by_min_support(z_raw, T, min_support=self.config.min_support, alpha=self.config.alpha)
        results = [
            ProviderResult(
                provider_id=pid,
                support=T[k],
                sum_proj=self.sum_proj[pid],
                mean_proj=(self.sum_proj[pid] / T[k]) if T[k] else 0.0,
                z_raw=z_raw[k],
                z=test.z[k],
                p_value=test.pvals[k],
                q_value=test.qvals[k],
                eligible=test.eligible[k],
                detected=test.significant[k],
                first_crossed_at=self.first_crossed_at[pid],
            )
            for k, pid in enumerate(ids)
        ]
        return AuditReport(self.n_answers, self.dim, asdict(self.config), results, list(self.trajectory))
