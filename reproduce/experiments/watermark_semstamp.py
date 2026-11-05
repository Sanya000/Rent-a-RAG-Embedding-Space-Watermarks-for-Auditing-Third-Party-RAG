"""SemStamp-style sentence-level semantic watermark: clusterer, generation and aligned detection."""
from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

# Reuse the exact sentence splitter and paraphrase candidate generators already used
# in the DirBucket builder so the generation stack is comparable across baselines.
from watermark_dirbucket import paraphrase_contrastive, paraphrase_diverse_batch, split_sentences


# ---------------------------------------------------------------------------
# Core params / result containers
# ---------------------------------------------------------------------------


@dataclass
class SemStampParams:
    embed_model: str = "sentence-transformers/all-mpnet-base-v2"
    num_clusters: int = 256
    gamma: float = 0.5


@dataclass
class SemStampGenParams:
    rounds: int = 4
    k_batch: int = 20
    k_stage1: int = 16
    k_stage2: int = 8
    strategy: str = "batch"  # "batch" | "contrastive"
    min_sem_sim: float = 0.78
    cluster_margin: float = 0.02  # k-SemStamp delta (nearest-vs-second centroid gap)
    min_sent_words: int = 1
    keep_if_already_green: bool = True
    force_accept_last_round: bool = False
    openai_model: str = "gpt-4o-mini"


@dataclass
class SemStampDetectionResult:
    T: int
    greens: int
    z: float
    p_one_sided: float
    green_fraction: float


@dataclass
class SemStampGenSummary:
    n_sentences: int
    changed_sentences: int
    already_green_sentences: int
    green_after_sentences: int
    failed_to_green_sentences: int
    mean_sem_sim_accepted: float
    mean_cluster_gap_accepted: float
    min_cluster_gap_accepted: float
    latency_s: float


# ---------------------------------------------------------------------------
# Shared embedder + centroids
# ---------------------------------------------------------------------------


class _Embedder:
    _CACHE: Dict[Tuple[str, str], SentenceTransformer] = {}

    def __init__(self, model_name: str, device: str = "auto") -> None:
        if device == "auto":
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"
        key = (str(model_name), str(device))
        st = self._CACHE.get(key)
        if st is None:
            st = SentenceTransformer(model_name, device=device)
            self._CACHE[key] = st
        self.st = st
        self.model_name = str(model_name)
        self.device = str(device)

    def embed(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        if not texts:
            dim = int(self.st.get_sentence_embedding_dimension())
            return np.zeros((0, dim), dtype=np.float32)
        X = self.st.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=bool(show_progress_bar),
        )
        X = np.asarray(X, dtype=np.float32)
        # Safety: ensure L2 normalized for cosine math.
        n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        return X / n


class SemStampClusterer:
    def __init__(
        self,
        *,
        centroids: np.ndarray,
        embed_model: str,
        device: str = "auto",
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        C = np.asarray(centroids, dtype=np.float32)
        if C.ndim != 2 or C.shape[0] <= 1 or C.shape[1] <= 0:
            raise ValueError(f"Invalid centroids shape: {C.shape}")
        # Normalize centroids for cosine nearest-centroid.
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-12)
        self.centroids = C.astype(np.float32)
        self.embedder = _Embedder(embed_model, device=device)
        self.meta = dict(meta or {})

    @property
    def num_clusters(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def dim(self) -> int:
        return int(self.centroids.shape[1])

    def embed_texts(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        return self.embedder.embed(texts, show_progress_bar=show_progress_bar)

    def assign_embeddings(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        sims = X @ self.centroids.T  # cosine since all normalized
        return np.argmax(sims, axis=1).astype(np.int64)

    def assign_texts(self, texts: Sequence[str], *, show_progress_bar: bool = False) -> np.ndarray:
        X = self.embed_texts(texts, show_progress_bar=show_progress_bar)
        return self.assign_embeddings(X)

    def nearest_centroid_sims(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        sims = X @ self.centroids.T
        return np.max(sims, axis=1).astype(np.float32)

    def top2_centroid_sims(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        if X.shape[0] == 0:
            z0 = np.zeros((0,), dtype=np.int64)
            zf = np.zeros((0,), dtype=np.float32)
            return z0, zf, zf
        sims = X @ self.centroids.T
        best_idx = np.argmax(sims, axis=1).astype(np.int64)
        best_sim = sims[np.arange(sims.shape[0]), best_idx].astype(np.float32)
        if sims.shape[1] <= 1:
            second_sim = np.full_like(best_sim, -1.0, dtype=np.float32)
        else:
            tmp = sims.copy()
            tmp[np.arange(tmp.shape[0]), best_idx] = -np.inf
            second_sim = np.max(tmp, axis=1).astype(np.float32)
        return best_idx, best_sim, second_sim

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = dict(self.meta)
        meta.setdefault("embed_model", str(self.embedder.model_name))
        meta.setdefault("num_clusters", int(self.num_clusters))
        meta.setdefault("dim", int(self.dim))
        np.savez(
            path,
            centroids=self.centroids.astype(np.float32),
            meta_json=np.array([json.dumps(meta)], dtype=object),
        )

    @classmethod
    def load(cls, path: Path, *, device: str = "auto") -> "SemStampClusterer":
        dat = np.load(path, allow_pickle=True)
        C = np.asarray(dat["centroids"], dtype=np.float32)
        meta: Dict[str, Any] = {}
        if "meta_json" in dat:
            try:
                raw = dat["meta_json"]
                if isinstance(raw, np.ndarray) and raw.size > 0:
                    meta = json.loads(str(raw.reshape(-1)[0]))
            except Exception:
                meta = {}
        embed_model = str(meta.get("embed_model") or "sentence-transformers/all-mpnet-base-v2")
        return cls(centroids=C, embed_model=embed_model, device=device, meta=meta)

    @classmethod
    def train_kmeans(
        cls,
        *,
        sentences: Sequence[str],
        embed_model: str,
        num_clusters: int,
        random_state: int = 13,
        batch_size: int = 1024,
        n_init: int = 10,
        device: str = "auto",
    ) -> "SemStampClusterer":
        if len(sentences) < int(num_clusters):
            raise ValueError(
                f"Need at least num_clusters sentences to train centroids: "
                f"{len(sentences)} < {int(num_clusters)}"
            )
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as e:
            raise RuntimeError(
                "scikit-learn is required for SemStamp centroid training "
                "(MiniBatchKMeans). Install scikit-learn or precompute centroids."
            ) from e

        emb = _Embedder(embed_model, device=device)
        X = emb.embed(list(sentences), show_progress_bar=True)
        km = MiniBatchKMeans(
            n_clusters=int(num_clusters),
            random_state=int(random_state),
            batch_size=int(batch_size),
            n_init=int(n_init),
            max_no_improvement=20,
            verbose=0,
        )
        km.fit(X)
        centroids = np.asarray(km.cluster_centers_, dtype=np.float32)
        meta = {
            "embed_model": str(embed_model),
            "num_clusters": int(num_clusters),
            "train_sentences": int(len(sentences)),
            "random_state": int(random_state),
            "batch_size": int(batch_size),
            "n_init": int(n_init),
        }
        return cls(centroids=centroids, embed_model=embed_model, device=device, meta=meta)


# ---------------------------------------------------------------------------
# k-SemStamp-style keyed valid-cluster sets (sequence-conditioned)
# ---------------------------------------------------------------------------

# Cache per (key, prev_cluster, K, valid_count) -> set(valid cluster ids). K is small
# (e.g., 256), so an exact pseudo-random subset is cheap and avoids Bernoulli drift.
_VALID_SET_CACHE: Dict[Tuple[str, int, int, int], set[int]] = {}


def _key_bytes_from_hexish(key_hex: str) -> bytes:
    try:
        return bytes.fromhex(key_hex)
    except Exception:
        return str(key_hex).encode("utf-8", errors="ignore")


def _num_valid_clusters(num_clusters: int, gamma: float) -> int:
    K = int(num_clusters)
    if K <= 0:
        return 0
    g = float(gamma)
    if g <= 0.0:
        return 0
    if g >= 1.0:
        return K
    # Exact subset size (k-SemStamp-style keyed partition), rounded to nearest int.
    m = int(round(g * K))
    return max(1, min(K - 1, m)) if 0 < g < 1 else max(0, min(K, m))


def _valid_cluster_set(
    *,
    prev_cluster_id: int,
    key_hex: str,
    num_clusters: int,
    gamma: float,
) -> set[int]:
    K = int(num_clusters)
    m = _num_valid_clusters(K, gamma)
    if m <= 0:
        return set()
    if m >= K:
        return set(range(K))
    cache_key = (str(key_hex), int(prev_cluster_id), K, int(m))
    cached = _VALID_SET_CACHE.get(cache_key)
    if cached is not None:
        return cached

    key_bytes = _key_bytes_from_hexish(key_hex)
    scored: List[Tuple[int, int]] = []
    for cid in range(K):
        payload = (
            b"ksemstamp:prev_cluster:"
            + int(prev_cluster_id).to_bytes(4, "little", signed=True)
            + b":next_cluster:"
            + int(cid).to_bytes(4, "little", signed=False)
        )
        h = hashlib.sha256(key_bytes + b"|" + payload).digest()
        scored.append((int.from_bytes(h[:8], "big", signed=False), cid))
    scored.sort(key=lambda t: t[0])
    out = {cid for _, cid in scored[:m]}
    _VALID_SET_CACHE[cache_key] = out
    return out


def _cluster_is_valid_given_prev(
    *,
    cluster_id: int,
    prev_cluster_id: int,
    key_hex: str,
    num_clusters: int,
    gamma: float,
) -> bool:
    return int(cluster_id) in _valid_cluster_set(
        prev_cluster_id=int(prev_cluster_id),
        key_hex=str(key_hex),
        num_clusters=int(num_clusters),
        gamma=float(gamma),
    )


def _z_and_p(T: int, greens: int, gamma: float) -> Tuple[float, float]:
    if int(T) <= 0:
        return (0.0, 1.0)
    g = float(gamma)
    mu = g * int(T)
    var = g * (1.0 - g) * int(T) + 1e-12
    z = (float(greens) - mu) / math.sqrt(var)
    try:
        from scipy.stats import norm

        p = float(norm.sf(z))
    except Exception:
        p = float(0.5 * math.erfc(z / math.sqrt(2.0)))
    return (float(z), float(p))


# ---------------------------------------------------------------------------
# Detection (answer-only and aligned multi-source)
# ---------------------------------------------------------------------------


def _sentence_units(text: str, *, min_words: int = 1) -> List[str]:
    sents = [s.strip() for s in split_sentences(text or "") if s and s.strip()]
    if min_words <= 1:
        return sents
    out: List[str] = []
    for s in sents:
        if len(s.split()) >= int(min_words):
            out.append(s)
    return out


def detect_semstamp_text(
    *,
    text: str,
    key_hex: str,
    clusterer: SemStampClusterer,
    params: SemStampParams,
    min_sent_words: int = 1,
) -> SemStampDetectionResult:
    sents = _sentence_units(text, min_words=min_sent_words)
    if len(sents) <= 1:
        return SemStampDetectionResult(T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0)
    cids = clusterer.assign_texts(sents)
    greens = 0
    T = 0
    cids_l = cids.tolist()
    for i in range(1, len(cids_l)):
        prev_cid = int(cids_l[i - 1])
        cur_cid = int(cids_l[i])
        T += 1
        if _cluster_is_valid_given_prev(
            cluster_id=cur_cid,
            prev_cluster_id=prev_cid,
            key_hex=key_hex,
            num_clusters=clusterer.num_clusters,
            gamma=params.gamma,
        ):
            greens += 1
    z, p = _z_and_p(T=T, greens=greens, gamma=float(params.gamma))
    gf = float(greens / T) if T > 0 else 0.0
    return SemStampDetectionResult(T=T, greens=greens, z=z, p_one_sided=p, green_fraction=gf)


def detect_semstamp_multi_source_aligned(
    *,
    answer_text: str,
    sources: Iterable[Tuple[int, str]],
    provider_keys_by_pid: Dict[int, str],
    clusterer: SemStampClusterer,
    params: SemStampParams,
    sim_threshold: float = 0.82,
    tie_margin: float = 0.05,
    competitive: bool = True,
    min_sent_words: int = 1,
) -> Tuple[Dict[int, SemStampDetectionResult], Dict[str, Any]]:
    """
    Fair k-SemStamp adaptation for multi-provider RAG auditing:
      - align answer sentences to source sentences using semantic similarity
      - optionally competitive assignment (one provider per answer sentence)
      - for each provider, run k-SemStamp-style sequence-conditioned detection on the
        subsequence of answer sentences assigned to that provider (in answer order)

    This keeps the SemStamp-family sequence semantics in the *answer* (where detection
    is performed), while using source alignment only for provider assignment.
    """
    pids = sorted(int(pid) for pid in provider_keys_by_pid.keys())
    ans_sents = _sentence_units(answer_text, min_words=min_sent_words)

    source_sents: List[str] = []
    source_pid_of_idx: List[int] = []
    for pid, txt in sources:
        pid_i = int(pid)
        if pid_i not in provider_keys_by_pid:
            continue
        doc_sents = _sentence_units(txt, min_words=min_sent_words)
        for s in doc_sents:
            source_sents.append(s)
            source_pid_of_idx.append(pid_i)

    if len(ans_sents) == 0 or len(source_sents) == 0:
        zeros = {
            int(pid): SemStampDetectionResult(T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0)
            for pid in pids
        }
        meta = {
            "mode": "aligned",
            "assignment_mode": ("competitive" if competitive else "non-competitive"),
            "windows_total": int(len(ans_sents)),
            "assigned_total": 0,
            "dropped_below_thresh_windows": int(len(ans_sents)),
            "best_cos_stats": {"mean": None},
            "second_best_cos_stats": {"mean": None},
            "assigned_tokens_by_pid": {int(pid): 0 for pid in pids},
        }
        return zeros, meta

    A = clusterer.embed_texts(ans_sents)
    S = clusterer.embed_texts(source_sents)
    sims = A @ S.T  # cosine

    # Precompute answer sentence clusters once.
    ans_cids = clusterer.assign_embeddings(A)
    assigned_ans_by_pid: Dict[int, List[int]] = {int(pid): [] for pid in pids}
    best_cos_list: List[float] = []
    second_best_list: List[float] = []
    assigned_total = 0
    dropped = 0

    # Group source indices by provider for provider-level best matching and tie-margin logic.
    src_idxs_by_pid: Dict[int, List[int]] = {int(pid): [] for pid in pids}
    for j, pid in enumerate(source_pid_of_idx):
        if int(pid) in src_idxs_by_pid:
            src_idxs_by_pid[int(pid)].append(int(j))

    for ai in range(A.shape[0]):
        row = sims[ai]
        if row.size == 0:
            dropped += 1
            continue

        # Provider-level max cosine.
        per_pid_best: List[Tuple[int, float, int]] = []
        for pid in pids:
            idxs = src_idxs_by_pid.get(int(pid), [])
            if not idxs:
                continue
            idxs_np = np.asarray(idxs, dtype=np.int64)
            local_argmax = int(np.argmax(row[idxs_np]))
            best_j = int(idxs_np[local_argmax])
            best = float(row[best_j])
            per_pid_best.append((int(pid), best, best_j))
        if not per_pid_best:
            dropped += 1
            continue

        per_pid_best.sort(key=lambda t: t[1], reverse=True)
        best_pid, best_cos, best_src_j = per_pid_best[0]
        second_cos = float(per_pid_best[1][1]) if len(per_pid_best) >= 2 else float("-inf")
        best_cos_list.append(float(best_cos))
        if second_cos != float("-inf"):
            second_best_list.append(float(second_cos))

        if float(best_cos) < float(sim_threshold):
            dropped += 1
            continue

        if competitive:
            if len(per_pid_best) >= 2 and float(best_cos - second_cos) < float(tie_margin):
                dropped += 1
                continue
            assigned_ans_by_pid[int(best_pid)].append(int(ai))
            assigned_total += 1
        else:
            any_assigned = False
            for pid, c, src_j in per_pid_best:
                if float(c) >= float(sim_threshold):
                    assigned_ans_by_pid[int(pid)].append(int(ai))
                    assigned_total += 1
                    any_assigned = True
            if not any_assigned:
                dropped += 1

    results: Dict[int, SemStampDetectionResult] = {}
    for pid in pids:
        idxs = assigned_ans_by_pid.get(int(pid), [])
        # k-SemStamp-style detection scores transitions, so the first assigned sentence
        # is context bootstrap and does not contribute an event.
        if len(idxs) <= 1:
            results[int(pid)] = SemStampDetectionResult(
                T=0, greens=0, z=0.0, p_one_sided=1.0, green_fraction=0.0
            )
            continue
        T = int(len(idxs) - 1)
        greens = 0
        for k in range(1, len(idxs)):
            prev_ai = int(idxs[k - 1])
            ai = int(idxs[k])
            prev_c = int(ans_cids[prev_ai])
            cid = int(ans_cids[ai])
            if _cluster_is_valid_given_prev(
                cluster_id=cid,
                prev_cluster_id=int(prev_c),
                key_hex=provider_keys_by_pid[int(pid)],
                num_clusters=clusterer.num_clusters,
                gamma=params.gamma,
            ):
                greens += 1
        z, p = _z_and_p(T=T, greens=int(greens), gamma=float(params.gamma))
        results[int(pid)] = SemStampDetectionResult(
            T=T,
            greens=int(greens),
            z=float(z),
            p_one_sided=float(p),
            green_fraction=float(greens / T) if T > 0 else 0.0,
        )

    meta = {
        "mode": "aligned",
        "assignment_mode": ("competitive" if competitive else "non-competitive"),
        "sequence_context_mode": "answer_assigned_subsequence",
        "windows_total": int(len(ans_sents)),
        "assigned_total": int(assigned_total),
        "dropped_below_thresh_windows": int(dropped),
        "best_cos_stats": {"mean": (float(np.mean(best_cos_list)) if best_cos_list else None)},
        "second_best_cos_stats": {"mean": (float(np.mean(second_best_list)) if second_best_list else None)},
        "assigned_tokens_by_pid": {int(pid): int(len(assigned_ans_by_pid.get(int(pid), []))) for pid in pids},
    }
    return results, meta


# ---------------------------------------------------------------------------
# Watermark generation (offline corpus build)
# ---------------------------------------------------------------------------


class SemStampWatermarker:
    def __init__(
        self,
        *,
        provider_key_hex: str,
        clusterer: SemStampClusterer,
        params: SemStampParams,
        gen_params: SemStampGenParams,
    ) -> None:
        self.provider_key_hex = str(provider_key_hex)
        self.clusterer = clusterer
        self.params = params
        self.gen = gen_params

    def _score_candidates(
        self,
        *,
        base_u: np.ndarray,
        prev_cluster_id: int,
        candidates: Sequence[str],
    ) -> List[Tuple[str, float, int, bool, float]]:
        if not candidates:
            return []
        U = self.clusterer.embed_texts(candidates)
        cids, best_centroid_sims, second_centroid_sims = self.clusterer.top2_centroid_sims(U)
        out: List[Tuple[str, float, int, bool, float]] = []
        for txt, u, cid, s1, s2 in zip(
            candidates,
            U,
            cids.tolist(),
            best_centroid_sims.tolist(),
            second_centroid_sims.tolist(),
        ):
            sim = float(np.dot(base_u, u))
            is_valid = _cluster_is_valid_given_prev(
                cluster_id=int(cid),
                prev_cluster_id=int(prev_cluster_id),
                key_hex=self.provider_key_hex,
                num_clusters=self.clusterer.num_clusters,
                gamma=self.params.gamma,
            )
            out.append((str(txt), sim, int(cid), bool(is_valid), float(s1 - s2)))
        # Highest semantic similarity first.
        out.sort(key=lambda t: (t[1], int(t[3])), reverse=True)
        return out

    def watermark(
        self,
        text: str,
        *,
        timing: bool = False,
        timing_doc_id: Optional[str] = None,
        timing_every: int = 1,
    ) -> Tuple[str, SemStampGenSummary]:
        t0 = time.perf_counter()
        sents = _sentence_units(text, min_words=max(1, int(self.gen.min_sent_words)))
        if not sents:
            return (
                text,
                SemStampGenSummary(
                    n_sentences=0,
                    changed_sentences=0,
                    already_green_sentences=0,
                    green_after_sentences=0,
                    failed_to_green_sentences=0,
                    mean_sem_sim_accepted=0.0,
                    mean_cluster_gap_accepted=0.0,
                    min_cluster_gap_accepted=0.0,
                    latency_s=0.0,
                ),
            )

        base_U = self.clusterer.embed_texts(sents)
        base_cids = self.clusterer.assign_embeddings(base_U)

        out_sents: List[str] = []
        changed = 0
        already_green = 0
        green_after = 0
        failed_to_green = 0
        accepted_sims: List[float] = []
        accepted_cluster_gaps: List[float] = []
        prev_out_cid: Optional[int] = None

        tag = f"doc_id={str(timing_doc_id)[:24]}" if timing_doc_id else "doc_id=?"
        if timing:
            print(f"[SEMSTAMP-TIME] start {tag} sents={len(sents)}", flush=True)

        for i, (sent, base_u, base_cid) in enumerate(zip(sents, base_U, base_cids.tolist())):
            t_sent0 = time.perf_counter() if timing else 0.0
            sent = str(sent).strip()
            if not sent:
                out_sents.append(sent)
                continue

            # k-SemStamp-style sequence conditioning: sentence i is constrained by the
            # previous *accepted output* sentence cluster. For the first sentence there is
            # no predecessor context, so we keep it unchanged as context bootstrap.
            if prev_out_cid is None:
                _, s1, s2 = self.clusterer.top2_centroid_sims(np.asarray(base_u[None, :], dtype=np.float32))
                gap0 = float(s1[0] - s2[0]) if len(s1) else 0.0
                out_sents.append(sent)
                prev_out_cid = int(base_cid)
                already_green += 1
                green_after += 1
                accepted_sims.append(1.0)
                accepted_cluster_gaps.append(gap0)
                if timing and (i % max(1, int(timing_every)) == 0):
                    dt = time.perf_counter() - t_sent0
                    print(
                        f"[SEMSTAMP-TIME] {tag} sent={i+1}/{len(sents)} bootstrap=1 total={dt:.2f}s",
                        flush=True,
                    )
                continue

            base_valid = _cluster_is_valid_given_prev(
                cluster_id=int(base_cid),
                prev_cluster_id=int(prev_out_cid),
                key_hex=self.provider_key_hex,
                num_clusters=self.clusterer.num_clusters,
                gamma=self.params.gamma,
            )
            _, s1b, s2b = self.clusterer.top2_centroid_sims(np.asarray(base_u[None, :], dtype=np.float32))
            base_cluster_gap = float(s1b[0] - s2b[0]) if len(s1b) else 0.0
            if (
                base_valid
                and bool(self.gen.keep_if_already_green)
                and float(base_cluster_gap) >= float(self.gen.cluster_margin)
            ):
                already_green += 1
                green_after += 1
                out_sents.append(sent)
                prev_out_cid = int(base_cid)
                accepted_sims.append(1.0)
                accepted_cluster_gaps.append(float(base_cluster_gap))
                if timing and (i % max(1, int(timing_every)) == 0):
                    dt = time.perf_counter() - t_sent0
                    print(
                        f"[SEMSTAMP-TIME] {tag} sent={i+1}/{len(sents)} kept_valid=1 total={dt:.2f}s",
                        flush=True,
                    )
                continue

            chosen_text = sent
            chosen_green = False
            chosen_sim = 1.0
            chosen_cid = int(base_cid)
            chosen_cluster_gap = float(base_cluster_gap)
            fallback_best: Optional[Tuple[str, float, int, bool]] = None

            cur_text = sent
            for _round in range(max(1, int(self.gen.rounds))):
                if str(self.gen.strategy).lower() == "contrastive":
                    prelim = paraphrase_diverse_batch(
                        cur_text,
                        model=self.gen.openai_model,
                        n_total=max(2, int(self.gen.k_stage1)),
                    )
                    scored_pre = self._score_candidates(
                        base_u=base_u, prev_cluster_id=int(prev_out_cid), candidates=prelim
                    )
                    pos = [t[0] for t in scored_pre[:3]]
                    neg = [t[0] for t in scored_pre[-3:]] if len(scored_pre) >= 6 else []
                    cands = paraphrase_contrastive(
                        cur_text,
                        positives=pos,
                        negatives=neg,
                        model=self.gen.openai_model,
                        n_total=max(2, int(self.gen.k_stage2)),
                    )
                else:
                    cands = paraphrase_diverse_batch(
                        cur_text,
                        model=self.gen.openai_model,
                        n_total=max(2, int(self.gen.k_batch)),
                    )

                # Keep unique, non-empty, and not identical-only if possible.
                uniq: List[str] = []
                seen = set()
                for c in cands:
                    c2 = str(c or "").strip()
                    if not c2 or c2 in seen:
                        continue
                    seen.add(c2)
                    uniq.append(c2)
                if not uniq:
                    continue

                scored = self._score_candidates(
                    base_u=base_u, prev_cluster_id=int(prev_out_cid), candidates=uniq
                )
                if scored and (fallback_best is None or scored[0][1] > fallback_best[1]):
                    fallback_best = scored[0][:4]

                green_ok = [
                    row
                    for row in scored
                    if row[3]
                    and float(row[1]) >= float(self.gen.min_sem_sim)
                    and float(row[4]) >= float(self.gen.cluster_margin)
                ]
                if green_ok:
                    best = green_ok[0]
                    chosen_text = best[0]
                    chosen_sim = float(best[1])
                    chosen_cid = int(best[2])
                    chosen_cluster_gap = float(best[4])
                    chosen_green = True
                    break

            if not chosen_green and self.gen.force_accept_last_round and fallback_best is not None:
                if float(fallback_best[1]) >= float(self.gen.min_sem_sim):
                    chosen_text = fallback_best[0]
                    chosen_sim = float(fallback_best[1])
                    chosen_cid = int(fallback_best[2])
                    chosen_green = bool(fallback_best[3])

            if chosen_green:
                green_after += 1
            else:
                failed_to_green += 1
            if chosen_text != sent:
                changed += 1
            accepted_sims.append(float(chosen_sim))
            accepted_cluster_gaps.append(float(chosen_cluster_gap))
            out_sents.append(chosen_text)
            prev_out_cid = int(chosen_cid)

            if timing and (i % max(1, int(timing_every)) == 0):
                dt = time.perf_counter() - t_sent0
                print(
                    f"[SEMSTAMP-TIME] {tag} sent={i+1}/{len(sents)} green={int(chosen_green)} "
                    f"changed={int(chosen_text != sent)} sim={chosen_sim:.3f} "
                    f"gap={chosen_cluster_gap:.3f} total={dt:.2f}s",
                    flush=True,
                )

        text_out = " ".join(s.strip() for s in out_sents if s and s.strip()).strip()
        if not text_out:
            text_out = text
        latency = time.perf_counter() - t0
        summary = SemStampGenSummary(
            n_sentences=int(len(sents)),
            changed_sentences=int(changed),
            already_green_sentences=int(already_green),
            green_after_sentences=int(green_after),
            failed_to_green_sentences=int(failed_to_green),
            mean_sem_sim_accepted=float(np.mean(accepted_sims)) if accepted_sims else 0.0,
            mean_cluster_gap_accepted=(
                float(np.mean(accepted_cluster_gaps)) if accepted_cluster_gaps else 0.0
            ),
            min_cluster_gap_accepted=(
                float(np.min(accepted_cluster_gaps)) if accepted_cluster_gaps else 0.0
            ),
            latency_s=float(latency),
        )
        if timing:
            print(
                f"[SEMSTAMP-TIME] done {tag} total={latency:.2f}s "
                f"sents={len(sents)} changed={changed} green_after={green_after}/{len(sents)}",
                flush=True,
            )
        return text_out, summary
