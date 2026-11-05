"""Non-watermark attribution heuristics: semantic similarity, lexical overlap and NLI entailment."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from watermark_dirbucket import split_sentences


_TOK_RE = re.compile(r"[A-Za-z0-9']+")


def _tokens(text: str) -> List[str]:
    return _TOK_RE.findall((text or "").lower())


def _sent_units(text: str, *, min_words: int = 1) -> List[str]:
    out: List[str] = []
    for s in split_sentences(text or ""):
        if len(_tokens(s)) >= int(min_words):
            out.append(s.strip())
    return out


def _topm_mean(vals: Sequence[float], m: int) -> float:
    if not vals:
        return 0.0
    m_eff = max(1, min(int(m), len(vals)))
    arr = np.array(vals, dtype=np.float32)
    idx = np.argsort(-arr)[:m_eff]
    return float(arr[idx].mean())


@dataclass
class SimilarityBaselineParams:
    top_m: int = 3
    min_sent_words: int = 4


@dataclass
class OverlapBaselineParams:
    top_m: int = 3
    ngram_n: int = 3
    min_sent_words: int = 4
    idf_floor: float = 0.5


@dataclass
class EntailmentBaselineParams:
    nli_model: str = "cross-encoder/nli-deberta-v3-base"
    topk_passages_per_claim: int = 5
    claim_min_words: int = 4
    passage_min_words: int = 4
    max_claims: int = 8
    batch_size: int = 16
    device: str = "auto"  # auto|cpu|cuda


class LexicalOverlapScorer:
    """
    Rare n-gram overlap baseline with corpus-level IDF weighting.
    """

    def __init__(
        self,
        rows: List[Dict[str, Any]],
        *,
        source_field: str,
        ngram_n: int = 3,
        max_chars: int = 1200,
        idf_floor: float = 0.5,
    ) -> None:
        self.n = max(1, int(ngram_n))
        self.idf_floor = float(idf_floor)
        self.df: Dict[Tuple[str, ...], int] = {}
        self.num_docs = 0
        for row in rows:
            txt = str(row.get(source_field) or "")[:max_chars]
            toks = _tokens(txt)
            if not toks:
                continue
            grams = self._ngrams(toks, self.n)
            if not grams:
                continue
            self.num_docs += 1
            for g in set(grams):
                self.df[g] = self.df.get(g, 0) + 1

    @staticmethod
    def _ngrams(toks: List[str], n: int) -> List[Tuple[str, ...]]:
        if len(toks) < n:
            return []
        return [tuple(toks[i : i + n]) for i in range(0, len(toks) - n + 1)]

    def _idf(self, gram: Tuple[str, ...]) -> float:
        # smoothed IDF
        df = float(self.df.get(gram, 0))
        N = float(max(1, self.num_docs))
        val = math.log((N + 1.0) / (df + 1.0)) + 1.0
        return max(self.idf_floor, float(val))

    def score(self, answer: str, passage: str) -> float:
        a_toks = _tokens(answer)
        p_toks = _tokens(passage)
        a_grams = self._ngrams(a_toks, self.n)
        p_grams = set(self._ngrams(p_toks, self.n))
        if not a_grams or not p_grams:
            return 0.0
        num = 0.0
        den = 0.0
        for g in a_grams:
            w = self._idf(g)
            den += w
            if g in p_grams:
                num += w
        if den <= 0:
            return 0.0
        return float(num / den)


class EntailmentScorer:
    """
    Thin wrapper around an MNLI model; returns entailment probability.
    """

    def __init__(self, model_name: str, device: str = "auto") -> None:
        self.model_name = str(model_name)
        self.device_pref = str(device or "auto").lower()
        self._tok = None
        self._model = None
        self._device = None
        self._entail_idx = None

    def _ensure(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(self.model_name, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        if self.device_pref == "cuda":
            dev = torch.device("cuda")
        elif self.device_pref == "cpu":
            dev = torch.device("cpu")
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(dev)
        model.eval()
        id2label = getattr(model.config, "id2label", {}) or {}
        entail_idx = None
        for i, lab in id2label.items():
            if "entail" in str(lab).lower():
                entail_idx = int(i)
                break
        if entail_idx is None:
            # fallback for common MNLI orderings
            labels = [str(id2label.get(i, "")).lower() for i in range(len(id2label))]
            if "entailment" in labels:
                entail_idx = labels.index("entailment")
            else:
                # Common order: contradiction, neutral, entailment
                entail_idx = len(id2label) - 1 if id2label else 2

        self._tok = tok
        self._model = model
        self._device = dev
        self._entail_idx = int(entail_idx)

    @torch.no_grad()
    def entail_probs(self, pairs: Sequence[Tuple[str, str]], batch_size: int = 16) -> List[float]:
        self._ensure()
        assert self._tok is not None and self._model is not None and self._device is not None and self._entail_idx is not None
        out: List[float] = []
        if not pairs:
            return out
        bs = max(1, int(batch_size))
        for s in range(0, len(pairs), bs):
            batch = pairs[s : s + bs]
            premises = [p for p, _ in batch]
            hyps = [h for _, h in batch]
            enc = self._tok(
                premises,
                hyps,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            enc = {k: v.to(self._device) for k, v in enc.items()}
            logits = self._model(**enc).logits
            probs = torch.softmax(logits, dim=-1)[:, int(self._entail_idx)].detach().cpu().numpy()
            out.extend(float(x) for x in probs.tolist())
        return out


def score_similarity_by_provider(
    *,
    answer_text: str,
    sources: List[Tuple[int, str]],
    encoder,
    params: SimilarityBaselineParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Provider score = top-m mean cosine(answer, provider passages), where passages are source sentences.
    """
    Pids = sorted({int(pid) for pid, _ in sources})
    scores_by_pid: Dict[int, float] = {pid: 0.0 for pid in Pids}
    ans_u = encoder.encode([answer_text], normalize_embeddings=True, convert_to_numpy=True)[0]
    per_provider_passages: Dict[int, int] = {}
    top_sims_diag: Dict[str, float] = {}

    for pid, txt in sources:
        pid_i = int(pid)
        passages = _sent_units(txt, min_words=int(params.min_sent_words))
        per_provider_passages[pid_i] = len(passages)
        if not passages:
            scores_by_pid[pid_i] = 0.0
            continue
        U = encoder.encode(passages, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        sims = (U @ np.asarray(ans_u, dtype=np.float32)).astype(np.float32)
        score = _topm_mean([float(x) for x in sims.tolist()], int(params.top_m))
        scores_by_pid[pid_i] = float(score)
        top_sims_diag[str(pid_i)] = float(np.max(sims)) if sims.size else 0.0

    if not scores_by_pid:
        return np.zeros((0,), dtype=np.float32), {
            "units_total": 1,
            "unit_kind": "trial",
            "per_provider_passages": {},
            "top_similarity_max_by_provider": {},
        }

    pmax = max(scores_by_pid.keys()) + 1
    vec = np.zeros(pmax, dtype=np.float32)
    for pid, s in scores_by_pid.items():
        vec[int(pid)] = float(s)
    meta = {
        "units_total": 1,
        "unit_kind": "trial",
        "per_provider_passages": {str(k): int(v) for k, v in sorted(per_provider_passages.items())},
        "top_similarity_max_by_provider": top_sims_diag,
    }
    return vec, meta


def score_overlap_by_provider(
    *,
    answer_text: str,
    sources: List[Tuple[int, str]],
    scorer: LexicalOverlapScorer,
    params: OverlapBaselineParams,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    Pids = sorted({int(pid) for pid, _ in sources})
    scores_by_pid: Dict[int, float] = {pid: 0.0 for pid in Pids}
    per_provider_passages: Dict[int, int] = {}
    for pid, txt in sources:
        pid_i = int(pid)
        passages = _sent_units(txt, min_words=int(params.min_sent_words))
        per_provider_passages[pid_i] = len(passages)
        if not passages:
            scores_by_pid[pid_i] = 0.0
            continue
        vals = [float(scorer.score(answer_text, psg)) for psg in passages]
        scores_by_pid[pid_i] = _topm_mean(vals, int(params.top_m))

    if not scores_by_pid:
        return np.zeros((0,), dtype=np.float32), {
            "units_total": 1,
            "unit_kind": "trial",
            "per_provider_passages": {},
        }

    pmax = max(scores_by_pid.keys()) + 1
    vec = np.zeros(pmax, dtype=np.float32)
    for pid, s in scores_by_pid.items():
        vec[int(pid)] = float(s)
    meta = {
        "units_total": 1,
        "unit_kind": "trial",
        "per_provider_passages": {str(k): int(v) for k, v in sorted(per_provider_passages.items())},
    }
    return vec, meta


def score_entailment_by_provider(
    *,
    answer_text: str,
    sources: List[Tuple[int, str]],
    encoder,
    nli: EntailmentScorer,
    params: EntailmentBaselineParams,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Returns:
      provider_scores: (P,)
      unit_score_matrix: (U,P) claim-level entailment scores
    """
    claims = _sent_units(answer_text, min_words=int(params.claim_min_words))
    if int(params.max_claims) > 0:
        claims = claims[: int(params.max_claims)]
    pids = sorted({int(pid) for pid, _ in sources})
    if not claims or not pids:
        pmax = (max(pids) + 1) if pids else 0
        return (
            np.zeros((pmax,), dtype=np.float32),
            np.zeros((0, pmax), dtype=np.float32),
            {"unit_kind": "claim", "units_total": 0, "claims_total": 0, "per_provider_passages": {}},
        )

    claim_U = encoder.encode(claims, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)

    passages_by_pid: Dict[int, List[str]] = {}
    passU_by_pid: Dict[int, np.ndarray] = {}
    per_provider_passages: Dict[int, int] = {}
    for pid, txt in sources:
        pid_i = int(pid)
        passages = _sent_units(txt, min_words=int(params.passage_min_words))
        passages_by_pid[pid_i] = passages
        per_provider_passages[pid_i] = len(passages)
        if passages:
            passU_by_pid[pid_i] = encoder.encode(
                passages, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
            )
        else:
            passU_by_pid[pid_i] = np.zeros((0, claim_U.shape[1]), dtype=np.float32)

    pmax = max(pids) + 1
    unit_mat = np.zeros((len(claims), pmax), dtype=np.float32)
    topk = max(1, int(params.topk_passages_per_claim))
    all_pairs: List[Tuple[str, str]] = []
    pair_slots: List[Tuple[int, int, int]] = []  # (claim_idx, pid, local_rank)
    topk_by_claim_pid: Dict[Tuple[int, int], List[str]] = {}

    for ci, cu in enumerate(claim_U):
        cuv = np.asarray(cu, dtype=np.float32)
        for pid in pids:
            U = passU_by_pid.get(pid)
            passages = passages_by_pid.get(pid) or []
            if U is None or U.shape[0] == 0:
                continue
            sims = (U @ cuv).astype(np.float32)
            k = min(topk, len(passages))
            idx = np.argsort(-sims)[:k]
            cand_passages = [passages[int(j)] for j in idx.tolist()]
            topk_by_claim_pid[(ci, pid)] = cand_passages
            for r, psg in enumerate(cand_passages):
                all_pairs.append((psg, claims[ci]))  # premise=evidence, hypothesis=claim
                pair_slots.append((ci, pid, r))

    if all_pairs:
        entail_probs = nli.entail_probs(all_pairs, batch_size=int(params.batch_size))
        best_by_claim_pid: Dict[Tuple[int, int], float] = {}
        for prob, (ci, pid, _r) in zip(entail_probs, pair_slots):
            key = (ci, pid)
            if key not in best_by_claim_pid or float(prob) > best_by_claim_pid[key]:
                best_by_claim_pid[key] = float(prob)
        for (ci, pid), val in best_by_claim_pid.items():
            unit_mat[int(ci), int(pid)] = float(val)

    provider_scores = np.zeros((pmax,), dtype=np.float32)
    if unit_mat.shape[0] > 0:
        provider_scores = unit_mat.mean(axis=0).astype(np.float32)

    meta = {
        "unit_kind": "claim",
        "units_total": int(unit_mat.shape[0]),
        "claims_total": int(len(claims)),
        "per_provider_passages": {str(k): int(v) for k, v in sorted(per_provider_passages.items())},
        "topk_passages_per_claim": int(topk),
    }
    return provider_scores, unit_mat, meta

