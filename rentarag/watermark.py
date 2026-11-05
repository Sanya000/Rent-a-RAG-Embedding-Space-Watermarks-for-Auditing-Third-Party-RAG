"""Provider-side DirBucket watermarking (paper Sec. 4, Alg. ``dirbucket-watermark``).

Each sentence is rewritten by hill climbing: in every round the paraphraser
proposes ``K`` candidates for the current sentence; a candidate is admissible
if its cosine similarity to the current sentence is at least ``min_sem_sim``
and its directional score ``<u, v_k(bucket(u))>`` exceeds the current one by at
least ``min_delta_proj``. The admissible candidate with the largest gain
replaces the current sentence. After ``rounds`` rounds the sentence carries a
small positive bias along the provider's keyed direction field.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from ._cfg import coerce_config, config_from_dict, config_to_dict
from .embed import Embedder, get_embedder
from .keys import DirectionManager, DirectionSource, SimHashCoder
from .paraphrase import Paraphraser, get_paraphraser
from .text import split_sentences

__all__ = [
    "WatermarkConfig",
    "WATERMARK_PRESETS",
    "CandidateLog",
    "SentenceLog",
    "WatermarkResult",
    "DirBucketWatermarker",
    "mmr_select",
    "coverage_keep",
]

_SELECTIONS = ("first", "mmr", "all")


@dataclass(frozen=True)
class WatermarkConfig:
    """Watermark hyperparameters; defaults are the values used in the paper.

    Hill climbing (paper Alg. 1):

    Attributes:
        bucket_bits: SimHash code length ``h``. More bits give finer buckets
            (directions change more often across the space); must match the auditor.
        bucket_seed: public seed of the SimHash hyperplanes; must match the auditor.
        n_candidates: paraphrase candidates requested per round (``K``). More
            candidates find larger gains per round at a higher LLM cost.
        rounds: hill-climbing rounds per sentence (``rho``). Each accepted round
            adds signal; each round costs one or two LLM calls per sentence and
            lets the text drift further from the original.
        min_sem_sim: minimum cosine between a candidate and the current
            sentence (``tau_sem``). Higher keeps rewrites closer to the source;
            lower admits stronger rewrites and a larger gain.
        min_delta_proj: minimum per-round gain in directional score
            (``delta_min``). Rejects rewrites that change wording without adding signal.
        force_accept_last_round: if no candidate was ever admissible, keep the
            semantically valid candidate with the highest score anyway (off in the paper).

    Preprocessing (applied by :meth:`DirBucketWatermarker.watermark` before
    hill climbing; all off by default):

    Attributes:
        max_chars: truncate each document to this many characters.
        max_sentences: keep at most this many sentences per document. Fewer
            sentences means fewer LLM calls and a shorter deployed document.
        sentence_selection: which sentences ``max_sentences`` keeps: ``"first"``,
            ``"mmr"`` (maximal marginal relevance: central yet diverse sentences,
            as in the paper's corpus build) or ``"all"`` (ignore ``max_sentences``).
        mmr_lambda: MMR relevance/diversity weight (1 = relevance only).
        selection_model: encoder spec for MMR (``None`` = the watermark encoder).
        coverage: fraction of sentences to watermark (selective watermarking).
            Sentences are picked by a keyed hash of the document id and sentence
            index; the others stay verbatim. Lower coverage leaves more text
            untouched and yields less evidence per answer.
    """

    bucket_bits: int = 96
    bucket_seed: int = 2025
    n_candidates: int = 24
    rounds: int = 6
    min_sem_sim: float = 0.70
    min_delta_proj: float = 5e-4
    force_accept_last_round: bool = False
    max_chars: int | None = None
    max_sentences: int | None = None
    sentence_selection: str = "first"
    mmr_lambda: float = 0.7
    selection_model: str | None = None
    coverage: float = 1.0

    def __post_init__(self) -> None:
        if self.n_candidates < 1 or self.rounds < 0 or self.bucket_bits < 1:
            raise ValueError("n_candidates and bucket_bits must be >= 1 and rounds >= 0")
        if self.sentence_selection not in _SELECTIONS:
            raise ValueError(f"sentence_selection must be one of {_SELECTIONS}")
        if not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be in [0, 1]")
        if self.max_sentences is not None and self.max_sentences < 1:
            raise ValueError("max_sentences must be >= 1 (or None for all)")
        if self.max_chars is not None and self.max_chars < 1:
            raise ValueError("max_chars must be >= 1 (or None for no limit)")
        if not -1.0 <= self.min_sem_sim <= 1.0:
            raise ValueError("min_sem_sim is a cosine and must be in [-1, 1]")
        if self.min_delta_proj < 0:
            raise ValueError("min_delta_proj must be >= 0")
        if not 0.0 <= self.mmr_lambda <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> WatermarkConfig:
        """Build from a dict; an optional ``"preset"`` key is applied first, other keys override it."""
        return config_from_dict(cls, data, WATERMARK_PRESETS)

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @classmethod
    def preset(cls, name: str, **overrides: Any) -> WatermarkConfig:
        """A named preset (see :data:`WATERMARK_PRESETS`) with optional overrides."""
        return cls.from_dict({"preset": name, **overrides})

    def replace(self, **changes: Any) -> WatermarkConfig:
        return replace(self, **changes)

    @classmethod
    def coerce(cls, value: Any) -> WatermarkConfig:
        """Accept ``None`` (defaults), a preset name, a dict or a config."""
        return coerce_config(cls, value, WATERMARK_PRESETS)


WATERMARK_PRESETS: dict[str, dict[str, Any]] = {
    "paper": {},
    "paper-corpus": {
        "max_chars": 4000,
        "max_sentences": 12,
        "sentence_selection": "mmr",
        "mmr_lambda": 0.7,
        "selection_model": "sentence-transformers/all-MiniLM-L6-v2",
    },
    "fast": {"n_candidates": 8, "rounds": 3},
    "strong": {"n_candidates": 32, "rounds": 10},
}
"""Watermark presets.

* ``paper``: the paper's hill-climbing settings (the defaults).
* ``paper-corpus``: ``paper`` plus the preprocessing of the paper's corpus build
  (4000 characters, 12 MMR-selected sentences, MiniLM for selection).
* ``fast``: 8 candidates x 3 rounds, about a quarter of the LLM calls; weaker signal per sentence.
* ``strong``: 32 candidates x 10 rounds; more signal per sentence, more LLM calls and more
  drift from the source. Pair it with ``prompt="k-variants"`` so the LLM is asked for 32 variants.
"""


def mmr_select(emb: np.ndarray, k: int, lambda_mult: float = 0.7) -> list[int]:
    """Maximal-marginal-relevance selection of ``k`` rows of unit-norm ``emb`` (sorted indices).

    Same procedure as the paper's corpus build: start from the sentence closest
    to the document centroid, then repeatedly add the sentence maximising
    ``lambda * sim(doc) - (1 - lambda) * max sim(selected)``.
    """
    n = emb.shape[0]
    if n <= k:
        return list(range(n))
    doc_vec = emb.mean(axis=0, keepdims=True)
    sims_doc = (emb @ doc_vec.T).ravel()
    first = int(np.argmax(sims_doc))
    selected = [first]
    cand = [i for i in range(n) if i != first]
    while len(selected) < k and cand:
        max_score, best = -1e9, None
        for c in cand:
            diversity = max(float(emb[c] @ emb[s]) for s in selected)
            score = lambda_mult * float(sims_doc[c]) - (1 - lambda_mult) * diversity
            if score > max_score:
                max_score, best = score, c
        selected.append(best)
        cand.remove(best)
    return sorted(selected)


def coverage_keep(doc_key: str, index: int, coverage: float) -> bool:
    """Keyed-hash decision whether sentence ``index`` of document ``doc_key`` is watermarked."""
    if coverage >= 1.0:
        return True
    h = hashlib.sha256(f"cov:{doc_key}:{index}".encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64 < coverage


@dataclass
class CandidateLog:
    text: str
    sem_sim: float
    proj: float
    delta: float
    bucket: str
    accepted: bool
    reject_reason: str | None


@dataclass
class SentenceLog:
    """Outcome of watermarking one sentence.

    ``gain`` is ``final_proj - base_proj``; ``sem_sim`` is the cosine between
    the original and final sentence embeddings (the acceptance test itself
    compares each candidate with the *current* sentence of that round).
    ``watermarked`` is False for sentences skipped by ``coverage < 1``.
    """

    index: int
    original: str
    text: str
    base_proj: float
    final_proj: float
    gain: float
    sem_sim: float
    changed: bool
    bucket_changed: bool
    rounds: list[list[CandidateLog]] = field(default_factory=list)
    watermarked: bool = True


@dataclass
class WatermarkResult:
    text: str
    sentences: list[SentenceLog]

    @property
    def n_sentences(self) -> int:
        return len(self.sentences)

    @property
    def changed_sentences(self) -> int:
        return sum(s.changed for s in self.sentences)

    @property
    def changed_frac(self) -> float:
        return self.changed_sentences / max(1, self.n_sentences)

    @property
    def bucket_flips(self) -> int:
        return sum(s.changed and s.bucket_changed for s in self.sentences)

    def _gains(self) -> np.ndarray:
        return np.array([s.gain for s in self.sentences], dtype=np.float32)

    @property
    def gain_mean(self) -> float:
        return float(self._gains().mean()) if self.sentences else 0.0

    @property
    def gain_median(self) -> float:
        return float(np.median(self._gains())) if self.sentences else 0.0

    @property
    def gain_p90(self) -> float:
        return float(np.percentile(self._gains(), 90)) if self.sentences else 0.0

    def summary(self) -> dict[str, Any]:
        """Aggregate statistics (same fields as the paper code's ``GenSummary``)."""
        return {
            "n_sentences": self.n_sentences,
            "changed_sentences": self.changed_sentences,
            "changed_frac": self.changed_frac,
            "gain_mean": self.gain_mean,
            "gain_median": self.gain_median,
            "gain_p90": self.gain_p90,
            "bucket_flips": self.bucket_flips,
        }

    def to_dict(self, with_candidates: bool = False) -> dict[str, Any]:
        sents = []
        for s in self.sentences:
            d = asdict(s)
            if not with_candidates:
                d.pop("rounds")
            sents.append(d)
        return {"text": self.text, "summary": self.summary(), "sentences": sents}


class DirBucketWatermarker:
    """Embed a provider's DirBucket watermark into text.

    Args:
        key_hex: the provider's secret key (hex). Ignored if ``directions`` is given.
        paraphraser: candidate generator ``(sentence, n) -> list[str]``, or a
            spec for :func:`~rentarag.get_paraphraser` (``"openai/gpt-4o-mini"``,
            ``"ollama/llama3.1"``, ``"toy"``, ...).
        embedder: sentence encoder or a spec for :func:`~rentarag.get_embedder`;
            defaults to ``all-mpnet-base-v2``. Must be the encoder the auditor uses.
        config: :class:`WatermarkConfig`, a preset name (``"paper"``, ``"fast"``,
            ...) or a dict; ``None`` = paper defaults.
        directions: custom direction source, e.g.
            ``OrthoDirectionManager(...).for_provider(pid)`` for the ortho ablation.
        selection_embedder: encoder for ``sentence_selection="mmr"`` (default:
            ``config.selection_model``, else ``embedder``).
        keep_candidates: store every scored candidate in the sentence logs.
        progress: show a progress bar over sentences.

    Given the same paraphraser outputs, results are identical to the paper's
    reference implementation (``strategy="batch"``).
    """

    def __init__(
        self,
        key_hex: str | None,
        paraphraser: Paraphraser | str,
        embedder: Embedder | str | None = None,
        config: WatermarkConfig | str | Mapping[str, Any] | None = None,
        *,
        directions: DirectionSource | None = None,
        selection_embedder: Embedder | str | None = None,
        keep_candidates: bool = False,
        progress: bool = False,
    ):
        self.embedder = get_embedder(embedder)
        self.config = config = WatermarkConfig.coerce(config)
        self.paraphraser = get_paraphraser(paraphraser) if isinstance(paraphraser, str) else paraphraser
        self.coder = SimHashCoder(self.embedder.dim, config.bucket_bits, config.bucket_seed)
        if directions is None:
            if key_hex is None:
                raise ValueError("either key_hex or directions is required")
            directions = DirectionManager(key_hex, self.embedder.dim)
        self.directions = directions
        self._selection_embedder = selection_embedder
        self.keep_candidates = keep_candidates
        self.progress = progress

    @property
    def selection_embedder(self) -> Embedder:
        se = self._selection_embedder
        if se is None:
            se = get_embedder(self.config.selection_model) if self.config.selection_model else self.embedder
        elif isinstance(se, str) or not hasattr(se, "dim"):
            se = get_embedder(se)
        self._selection_embedder = se
        return se

    def score(self, u: np.ndarray) -> tuple[float, bytes]:
        """Directional score of one embedding and its bucket code."""
        b = self.coder.code(u)
        return float(np.dot(u, self.directions.direction(b))), b

    def prepare(self, text: str) -> str:
        """Apply the config's preprocessing (``max_chars``, ``max_sentences``); identity by default."""
        cfg = self.config
        if cfg.max_chars is not None:
            text = text[: cfg.max_chars]
        if cfg.max_sentences is None or cfg.sentence_selection == "all":
            return text
        sents = split_sentences(text)
        k = min(cfg.max_sentences, len(sents))
        if cfg.sentence_selection == "first" or len(sents) <= k:
            keep = list(range(k))
        else:
            keep = mmr_select(self.selection_embedder(sents), k, cfg.mmr_lambda)
        return " ".join(sents[i] for i in keep)

    def watermark(self, text: str, *, doc_id: str | None = None, max_workers: int = 1) -> WatermarkResult:
        """Watermark ``text``; sentences are re-joined with single spaces.

        The config's preprocessing runs first (off by default). With
        ``coverage < 1`` only the sentences picked by a keyed hash of ``doc_id``
        (default: a hash of the text) and the sentence index are watermarked.
        ``max_workers > 1`` processes sentences in parallel threads (useful
        with API paraphrasers); the output order is unchanged.
        """
        src = self.prepare(text)
        sentences = split_sentences(src)
        if not sentences:
            return WatermarkResult(text=text, sentences=[])
        U0 = self.embedder(sentences)
        cov = self.config.coverage
        doc_key = doc_id if doc_id is not None else hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]

        def one(i: int) -> SentenceLog:
            if cov < 1.0 and not coverage_keep(str(doc_key), i, cov):
                return self._untouched(sentences[i], U0[i], i)
            return self.watermark_sentence(sentences[i], U0[i], index=i)

        bar = None
        if self.progress:
            from tqdm import tqdm

            bar = tqdm(total=len(sentences), desc="Watermarking sentences")
        logs: list[SentenceLog] = []
        if max_workers <= 1:
            for i in range(len(sentences)):
                logs.append(one(i))
                if bar is not None:
                    bar.update()
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                for log_ in ex.map(one, range(len(sentences))):
                    logs.append(log_)
                    if bar is not None:
                        bar.update()
        if bar is not None:
            bar.close()
        return WatermarkResult(text=" ".join(s.text for s in logs), sentences=logs)

    def _untouched(self, sentence: str, u: np.ndarray, index: int) -> SentenceLog:
        proj, _ = self.score(u)
        return SentenceLog(
            index=index,
            original=sentence,
            text=sentence,
            base_proj=proj,
            final_proj=proj,
            gain=0.0,
            sem_sim=float(np.dot(u, u)),
            changed=False,
            bucket_changed=False,
            watermarked=False,
        )

    def watermark_sentence(self, sentence: str, u: np.ndarray | None = None, *, index: int = 0) -> SentenceLog:
        """Hill-climb one sentence. ``u`` is its embedding (computed if omitted)."""
        cfg = self.config
        base_u = self.embedder([sentence])[0] if u is None else u
        base_proj, base_b = self.score(base_u)
        cur_text, cur_u, cur_b, cur_proj = sentence, base_u, base_b, base_proj
        accepted_any = False
        fallback: tuple[str, float, bytes] | None = None
        fallback_proj = -1e9
        round_logs: list[list[CandidateLog]] = []

        for _ in range(cfg.rounds):
            cands = self.paraphraser(cur_text, cfg.n_candidates)
            U = self.embedder(cands) if cands else np.zeros((0, self.embedder.dim), dtype=np.float32)
            best: tuple[str, float, bytes] | None = None
            best_gain = 0.0
            rlog: list[CandidateLog] = []
            for cand, cu in zip(cands, U):
                proj, b = self.score(cu)
                sim = float(np.dot(cur_u, cu))
                delta = proj - cur_proj
                if sim < cfg.min_sem_sim:
                    reason = "sem_sim_below"
                elif delta < cfg.min_delta_proj:
                    reason = "delta_too_small"
                else:
                    reason = None
                if self.keep_candidates:
                    rlog.append(CandidateLog(cand, sim, proj, delta, b.decode("ascii"), reason is None, reason))
                if reason is None and delta > best_gain:
                    best_gain = delta
                    best = (cand, proj, b)
                if sim >= cfg.min_sem_sim and proj > fallback_proj:
                    fallback_proj = proj
                    fallback = (cand, proj, b)
            round_logs.append(rlog)
            if best is not None:
                cur_text, cur_proj, cur_b = best
                cur_u = self.embedder([cur_text])[0]
                accepted_any = True

        if not accepted_any and cfg.force_accept_last_round and fallback is not None:
            cur_text, cur_proj, cur_b = fallback
            cur_u = self.embedder([cur_text])[0]
            accepted_any = True

        return SentenceLog(
            index=index,
            original=sentence,
            text=cur_text,
            base_proj=float(base_proj),
            final_proj=float(cur_proj),
            gain=float(cur_proj - base_proj),
            sem_sim=float(np.dot(base_u, cur_u)),
            changed=accepted_any,
            bucket_changed=cur_b != base_b,
            rounds=round_logs if self.keep_candidates else [],
        )
