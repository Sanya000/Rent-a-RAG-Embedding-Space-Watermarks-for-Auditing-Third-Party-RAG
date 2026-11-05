# experiments/watermark_dirbucket.py
# ------------------------------------------------------------
# Directional bucket watermarking (provider-side, uncooperative RAG setting)
# - Embedding: all-mpnet-base-v2 (D=768)
# - Buckets: SimHash over sentence/window embeddings
# - Secret directions per bucket: v_b = PRF(provider_key, bucket_code)
# - Generator strategies: "batch" or "contrastive"
# - Detector:
#     * Sentence mode
#     * Window mode with ANCHOR control:
#         - anchor="sentence" (default): windows project onto PARENT SENTENCE bucket (recommended)
#         - anchor="window"  : legacy behavior, project onto window's own bucket (can fail)
#     * Multi-provider aligned window mode (DEFAULT for RAG auditing):
#         - Align each answer window to provider-doc sentences; drop if below cosine threshold
#         - Optional competitive assignment (each window contributes to at most one provider)
#     * Optional neighbor smoothing on buckets at detection (Hamming radius=1, combine=max/mean)
# - Last-resort acceptance (optional)
# ------------------------------------------------------------
from __future__ import annotations
import json, math, hmac, hashlib, re, time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any

import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

try:
    import openai
    _HAS_OPENAI = True
except Exception:
    _HAS_OPENAI = False

EMBED_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# ---------- sentence splitter ----------
_SENT_RE = re.compile(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|!)\s+')

def split_sentences(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []
    parts = _SENT_RE.split(text)
    out, buf = [], ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if len(p) < 40 and buf:
            buf += " " + p
        else:
            if buf:
                out.append(buf)
            buf = p
    if buf: out.append(buf)
    return out

def _hexcode_hamming_neighbors(hex_bytes: bytes, radius: int) -> List[bytes]:
    """
    Return hex-encoded (ascii bytes) SimHash codes within Hamming radius r of `hex_bytes`.
    Includes the center (r=0) implicitly handled by caller.
    Only supports radius 1 or 2 to avoid blow-ups.
    """
    if radius <= 0:
        return []
    if radius > 2:
        radius = 2  # guard; 96C2 ~ 4560 neighbors is still OK

    data = bytearray.fromhex(hex_bytes.decode("ascii"))
    nbits = len(data) * 8
    neighbors: List[bytes] = []

    # r = 1
    for i in range(nbits):
        bidx, off = divmod(i, 8)
        nb = bytearray(data)
        nb[bidx] ^= (1 << off)
        neighbors.append(bytes(nb).hex().encode("ascii"))

    if radius >= 2:
        # r = 2
        for i in range(nbits):
            bidx1, off1 = divmod(i, 8)
            for j in range(i + 1, nbits):
                bidx2, off2 = divmod(j, 8)
                nb = bytearray(data)
                nb[bidx1] ^= (1 << off1)
                nb[bidx2] ^= (1 << off2)
                neighbors.append(bytes(nb).hex().encode("ascii"))

    return neighbors

# ---------- in-sentence word windows ----------
def _windows_within_sentence(sent: str, win_words: int, stride_words: int) -> List[str]:
    toks = sent.strip().split()
    n = len(toks)
    if n == 0:
        return []
    if n <= win_words:
        return [" ".join(toks)]
    out = []
    for s in range(0, n - win_words + 1, max(1, stride_words)):
        out.append(" ".join(toks[s:s+win_words]))
    # tail coverage
    last_start = n - win_words
    tail = " ".join(toks[last_start:last_start+win_words])
    if out[-1] != tail:
        out.append(tail)
    return out

# ---------- SimHash bucketizer ----------
class SimHashCoder:
    def __init__(self, dim: int, m_bits: int, seed: int):
        self.dim = int(dim)
        self.m = int(m_bits)
        rng = np.random.RandomState(seed)
        self.R = rng.normal(0.0, 1.0, size=(self.m, self.dim)).astype(np.float32)
    def code(self, u: np.ndarray) -> bytes:
        proj = (self.R @ u.astype(np.float32))
        bits = (proj >= 0.0)
        pad = (-len(bits)) % 8
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=bool)])
        bits = bits.reshape(-1, 8)
        byte_vals = (bits * (1 << np.arange(8))).sum(axis=1).astype(np.uint8)
        return bytearray(byte_vals).hex().encode("ascii")

# ---------- Secret direction manager ----------
def _hmac_sha256(key: bytes, payload: bytes) -> bytes:
    return hmac.new(key, payload, hashlib.sha256).digest()

class DirectionManager:
    def __init__(self, provider_key_hex: str, dim: int):
        self.key = bytes.fromhex(provider_key_hex)
        self.dim = int(dim)
        self.cache: Dict[bytes, np.ndarray] = {}
    def direction(self, bucket_code: bytes) -> np.ndarray:
        if bucket_code in self.cache:
            return self.cache[bucket_code]
        seed_bytes = _hmac_sha256(self.key, bucket_code)
        seed_int = int.from_bytes(seed_bytes[:8], "big", signed=False) & 0x7fffffff
        rng = np.random.RandomState(seed_int)
        v = rng.normal(0.0, 1.0, size=(self.dim,)).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        self.cache[bucket_code] = v
        return v

class OrthoDirectionManager:
    """Coordinate per-bucket *orthonormal* directions across providers.

    Given provider keys (one per provider), we first derive a deterministic base
    direction per (provider, bucket) via HMAC(key, bucket_code). We then
    orthogonalize these P base vectors in R^D (P <= D) using a QR decomposition.

    This is the "Option A" scheme: for each bucket, providers' secret directions
    are as close to orthogonal as possible, which should reduce cross-provider
    accidental alignment and false positives.

    Notes:
      - Deterministic given (keys, bucket_code).
      - QR sign ambiguity is fixed by aligning each orthogonal vector with its
        corresponding base vector (positive dot).
      - If P > D, full orthogonality is impossible; we raise a ValueError.
    """

    def __init__(self, provider_keys_by_pid: Dict[int, str], dim: int):
        self.dim = int(dim)
        self.keys_by_pid: Dict[int, bytes] = {int(pid): bytes.fromhex(khex)
                                              for pid, khex in provider_keys_by_pid.items()}
        self.pids: List[int] = sorted(self.keys_by_pid.keys())
        self.pid_to_idx: Dict[int, int] = {pid: i for i, pid in enumerate(self.pids)}
        self.P = len(self.pids)
        if self.P > self.dim:
            raise ValueError(f"Cannot orthogonalize P={self.P} provider directions in dim={self.dim}. Use P<=D or fall back to naive.")
        self._cache: Dict[bytes, np.ndarray] = {}

    def _base_vec(self, pid: int, bucket_code: bytes) -> np.ndarray:
        key = self.keys_by_pid[int(pid)]
        seed_bytes = _hmac_sha256(key, b"dirbucket:base:" + bucket_code)
        seed_int = int.from_bytes(seed_bytes[:8], "big", signed=False) & 0x7fffffff
        rng = np.random.RandomState(seed_int)
        v = rng.normal(0.0, 1.0, size=(self.dim,)).astype(np.float32)
        v /= (np.linalg.norm(v) + 1e-12)
        return v

    def _compute_bucket(self, bucket_code: bytes) -> np.ndarray:
        M = np.stack([self._base_vec(pid, bucket_code) for pid in self.pids], axis=0).astype(np.float32)  # (P,D)
        A = M.T.astype(np.float64, copy=False)  # (D,P)
        Q, _R = np.linalg.qr(A)                # Q: (D,P)
        V = Q[:, : self.P].T.astype(np.float32)  # (P,D)

        dots = (V * M).sum(axis=1)
        signs = np.where(dots >= 0.0, 1.0, -1.0).astype(np.float32)
        V = V * signs[:, None]

        norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
        V = V / norms
        return V.astype(np.float32)

    def direction(self, pid: int, bucket_code: bytes) -> np.ndarray:
        pid = int(pid)
        if bucket_code not in self._cache:
            self._cache[bucket_code] = self._compute_bucket(bucket_code)
        return self._cache[bucket_code][self.pid_to_idx[pid]]

# ---------- Embedder ----------
class Embedder:
    _CACHE = {}  # (model_name, device) -> SentenceTransformer

    def __init__(self, model_name: str = EMBED_MODEL_NAME, device: str = "auto"):
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        key = (model_name, device)
        st = self._CACHE.get(key)
        if st is None:
            st = SentenceTransformer(model_name, device=device)
            self._CACHE[key] = st
        self.st = st

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.st.get_sentence_embedding_dimension()), dtype=np.float32)
        X = self.st.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return X.astype(np.float32)

# ---------- Paraphrasers ----------
def _ensure_openai():
    if not _HAS_OPENAI:
        raise RuntimeError("openai package not available.")

def _chat_once(model: str, prompt: str, temperature: float = 0.8, max_tokens: int = 512) -> str:
    client = openai.OpenAI()
    r = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[{"role":"user","content":prompt}]
    )
    return r.choices[0].message.content or ""

def _parse_numbered_list(s: str) -> List[str]:
    lines = [l.strip(" -*\t") for l in s.strip().splitlines() if l.strip()]
    out = []
    for l in lines:
        out.append(re.sub(r"^\d+[\)\.\-]\s*", "", l))
    uniq, seen = [], set()
    for t in out:
        if t and t not in seen:
            uniq.append(t); seen.add(t)
    return uniq

def paraphrase_diverse_batch(base_sentence: str, model: str = "gpt-4o-mini",
                             n_total: int = 16, t1: float = 0.7, t2: float = 1.0) -> List[str]:
    try:
        prompt = (
            "Rewrite the sentence in a **diverse** set of paraphrases while preserving all facts.\n"
            "Produce 8 variants: 2 very light, 3 medium, 3 strong rewrites.\n"
            "Vary voice, ordering, connectives, nominalization, and tone.\n"
            "Return one variant per line, numbered.\n\nSentence:\n"
            f"{base_sentence}"
        )
        out1 = _chat_once(model, prompt, temperature=t1, max_tokens=700)
        cand1 = _parse_numbered_list(out1)
        if n_total <= len(cand1): return cand1[:n_total]
        out2 = _chat_once(model, prompt, temperature=t2, max_tokens=700)
        cand2 = _parse_numbered_list(out2)
        return (cand1 + [c for c in cand2 if c not in set(cand1)])[:n_total]
    except Exception:
        return [base_sentence]

def paraphrase_contrastive(base_sentence: str, positives: List[str], negatives: List[str],
                           model: str = "gpt-4o-mini", n_total: int = 8) -> List[str]:
    try:
        pos_blk = "\n".join([f"- {p}" for p in positives]) if positives else "(none)"
        neg_blk = "\n".join([f"- {n}" for n in negatives]) if negatives else "(none)"
        prompt = (
            "Rewrite the sentence while preserving every factual detail.\n"
            "Make it **stylistically closer** to set (A) and **less like** set (B).\n"
            "Stay within ±20% length.\n"
            f"\n(A) positive style anchors:\n{pos_blk}\n(B) negative style anchors:\n{neg_blk}\n"
            "Produce 8 variants as a numbered list."
        )
        out = _chat_once(model, prompt, temperature=0.7, max_tokens=700)
        return _parse_numbered_list(out)[:n_total]
    except Exception:
        return [base_sentence]

# ---------- Watermark generator ----------
@dataclass
class GenParams:
    bucket_bits: int = 96
    bucket_seed: int = 2025
    embed_model: str = EMBED_MODEL_NAME
    min_sem_sim: float = 0.72
    min_delta_proj: float = 1e-3
    rounds: int = 3
    k_batch: int = 16
    k_stage1: int = 16
    k_stage2: int = 8
    strategy: str = "batch"
    openai_model: str = "gpt-4o-mini"

@dataclass
class GenLogSentence:
    sent_idx: int
    base_bucket_hex: str
    base_proj: float
    rounds: List[List[Dict[str, Any]]]
    final_text: str
    final_bucket_hex: str
    final_proj: float
    total_gain: float

@dataclass
class GenSummary:
    n_sentences: int
    changed_sentences: int
    changed_frac: float
    gain_mean: float
    gain_median: float
    gain_p90: float
    bucket_flips: int
    params: Dict[str, Any]

class DirBucketWatermarker:
    def __init__(self, provider_key_hex: str, gen_params: GenParams, *,
                 direction_scheme: str = "naive",
                 provider_id: Optional[int] = None,
                 all_provider_keys: Optional[List[str]] = None,
                 shared_ortho_mgr: Optional[OrthoDirectionManager] = None,
                 force_accept_last_round: bool = False):
        self.embedder = Embedder(gen_params.embed_model)
        D = self.embedder.st.get_sentence_embedding_dimension()
        self.bucket = SimHashCoder(dim=D, m_bits=gen_params.bucket_bits, seed=gen_params.bucket_seed)

        self.direction_scheme = str(direction_scheme or "naive").lower()
        if self.direction_scheme not in {"naive", "ortho"}:
            raise ValueError(f"Unknown direction_scheme={direction_scheme}. Use 'naive' or 'ortho'.")

        self.provider_id = int(provider_id) if provider_id is not None else None
        self.all_provider_keys = list(all_provider_keys) if all_provider_keys is not None else None

        if self.direction_scheme == "naive":
            self.dir_mgr = DirectionManager(provider_key_hex, dim=D)
            self.ortho_mgr = None
        else:
            if self.provider_id is None or self.all_provider_keys is None:
                raise ValueError("direction_scheme='ortho' requires provider_id and all_provider_keys.")
            self.ortho_mgr = shared_ortho_mgr or OrthoDirectionManager(
                {i: self.all_provider_keys[i] for i in range(len(self.all_provider_keys))}, dim=D
            )
            self.dir_mgr = None
        self.p = gen_params
        self.force_accept_last_round = bool(force_accept_last_round)

    def _direction(self, bucket_code: bytes) -> np.ndarray:
        if self.direction_scheme == "naive":
            return self.dir_mgr.direction(bucket_code)  # type: ignore
        return self.ortho_mgr.direction(self.provider_id, bucket_code)  # type: ignore

    def _proj(self, u: np.ndarray, bucket_code: bytes) -> float:
        v = self._direction(bucket_code)
        return float(np.dot(u, v))

    def _sem_sim(self, u0: np.ndarray, u1: np.ndarray) -> float:
        return float(np.dot(u0, u1))

    def _cand_table(self, base_text: str, base_u: np.ndarray, base_bucket: bytes,
                    candidates: List[str], cur_proj: Optional[float] = None) -> List[Tuple[str, float, float, float, bytes]]:
        out = []
        if not candidates:
            return out
        U = self.embedder.embed(candidates)
        best_ref = self._proj(base_u, base_bucket) if cur_proj is None else cur_proj
        for i, cand in enumerate(candidates):
            u = U[i]
            b = self.bucket.code(u)
            proj = self._proj(u, b)
            sim = self._sem_sim(base_u, u)
            delta = proj - best_ref
            out.append((cand, sim, proj, delta, b))
        return out

    def watermark(self, text: str, debug_jsonl_path: Optional[str] = None, *,
                  timing: bool = False,
                  timing_doc_id: Optional[str] = None,
                  timing_every: int = 1) -> Tuple[str, GenSummary]:
        t_doc0 = time.perf_counter() if timing else 0.0
        doc_tag = f"doc_id={str(timing_doc_id)[:24]}" if timing_doc_id else "doc_id=?"
        timing_every = max(1, int(timing_every or 1))

        sentences = split_sentences(text)
        if not sentences:
            return text, GenSummary(0,0,0.0,0.0,0.0,0.0,0, vars(self.p))

        logs: List[GenLogSentence] = []
        changed = 0
        gains: List[float] = []
        flips = 0

        t_embed0 = time.perf_counter() if timing else 0.0
        U0 = self.embedder.embed(sentences)
        t_embed_init = (time.perf_counter() - t_embed0) if timing else 0.0
        if timing:
            print(f"[WM-TIME] start {doc_tag} sents={len(sentences)} embed_init={t_embed_init:.2f}s", flush=True)

        if debug_jsonl_path:
            with open(debug_jsonl_path, "w", encoding="utf-8") as f:
                pass

        for i, base_sent in enumerate(tqdm(sentences, desc="Watermarking sentences")):
            t_sent0 = time.perf_counter() if timing else 0.0
            t_llm = 0.0
            t_score = 0.0
            t_embed = 0.0
            n_cands = 0
            base_u = U0[i]
            base_b = self.bucket.code(base_u)
            base_proj = self._proj(base_u, base_b)

            cur_text = base_sent
            cur_u = base_u
            cur_b = base_b
            cur_proj = base_proj

            round_logs: List[List[Dict[str, Any]]] = []
            accepted_any = False

            fallback_best = None
            fallback_best_proj = -1e9

            for _ in range(self.p.rounds):
                if self.p.strategy == "batch":
                    t0 = time.perf_counter() if timing else 0.0
                    cands = paraphrase_diverse_batch(cur_text, model=self.p.openai_model,
                                                     n_total=self.p.k_batch)
                    if timing:
                        t_llm += (time.perf_counter() - t0)
                else:
                    t0 = time.perf_counter() if timing else 0.0
                    prelim = paraphrase_diverse_batch(cur_text, model=self.p.openai_model,
                                                      n_total=self.p.k_stage1)
                    if timing:
                        t_llm += (time.perf_counter() - t0)
                    prelim_tab = self._cand_table(cur_text, cur_u, cur_b, prelim, cur_proj=cur_proj)
                    prelim_tab.sort(key=lambda x: x[2], reverse=True)
                    pos_anchors = [t[0] for t in prelim_tab[:3]]
                    neg_anchors = [t[0] for t in prelim_tab[-3:]] if len(prelim_tab) >= 6 else []
                    t1 = time.perf_counter() if timing else 0.0
                    cands = paraphrase_contrastive(cur_text, pos_anchors, neg_anchors,
                                                   model=self.p.openai_model, n_total=self.p.k_stage2)
                    if timing:
                        t_llm += (time.perf_counter() - t1)

                t2 = time.perf_counter() if timing else 0.0
                table = self._cand_table(cur_text, cur_u, cur_b, cands, cur_proj=cur_proj)
                if timing:
                    t_score += (time.perf_counter() - t2)
                    n_cands += len(table)

                rlog: List[Dict[str, Any]] = []
                best_this_round_gain = 0.0
                best_row = None

                for (cand, sim, proj, delta, b) in table:
                    accept = False
                    reason = None
                    if sim < self.p.min_sem_sim:
                        reason = "sem_sim_below"
                    elif (proj - cur_proj) < self.p.min_delta_proj:
                        reason = "delta_too_small"
                    else:
                        accept = True

                    rlog.append({
                        "candidate": cand[:240],
                        "sem_sim": round(sim, 4),
                        "proj": round(proj, 5),
                        "delta": round(proj - cur_proj, 5),
                        "bucket_hex": (b.decode("ascii")[:16] if isinstance(b, (bytes, bytearray)) else str(b)[:16]),
                        "accepted": accept,
                        "reject_reason": reason
                    })

                    if accept and (proj - cur_proj) > best_this_round_gain:
                        best_this_round_gain = (proj - cur_proj)
                        best_row = (cand, sim, proj, delta, b)

                    if sim >= self.p.min_sem_sim and proj > fallback_best_proj:
                        fallback_best_proj = proj
                        fallback_best = (cand, sim, proj, delta, b)

                round_logs.append(rlog)

                if best_row is not None:
                    cand, sim, proj, delta, b = best_row
                    cur_text = cand
                    cur_b = b
                    t3 = time.perf_counter() if timing else 0.0
                    cur_u = self.embedder.embed([cur_text])[0]
                    if timing:
                        t_embed += (time.perf_counter() - t3)
                    cur_proj = proj
                    accepted_any = True
                else:
                    continue

            if (not accepted_any) and (self.force_accept_last_round) and (fallback_best is not None):
                cand, sim, proj, delta, b = fallback_best
                cur_text = cand
                cur_b = b
                t4 = time.perf_counter() if timing else 0.0
                cur_u = self.embedder.embed([cur_text])[0]
                if timing:
                    t_embed += (time.perf_counter() - t4)
                cur_proj = proj
                accepted_any = True

            total_gain = cur_proj - base_proj
            if accepted_any:
                changed += 1
                if cur_b != base_b:
                    flips += 1
            gains.append(float(total_gain))

            logs.append(GenLogSentence(
                sent_idx=i,
                base_bucket_hex=(base_b.decode("ascii")[:16] if isinstance(base_b, (bytes, bytearray)) else str(base_b)[:16]),
                base_proj=float(base_proj),
                rounds=round_logs,
                final_text=cur_text,
                final_bucket_hex=(cur_b.decode("ascii")[:16] if isinstance(cur_b, (bytes, bytearray)) else str(cur_b)[:16]),
                final_proj=float(cur_proj),
                total_gain=float(total_gain)
            ))

            if debug_jsonl_path:
                with open(debug_jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({
                        "sent_idx": i,
                        "base_bucket_hex": logs[-1].base_bucket_hex,
                        "base_proj": logs[-1].base_proj,
                        "rounds": logs[-1].rounds,
                        "final_text": logs[-1].final_text,
                        "final_bucket_hex": logs[-1].final_bucket_hex,
                        "final_proj": logs[-1].final_proj,
                        "total_gain": logs[-1].total_gain
                    }, ensure_ascii=False) + "\n")

            sentences[i] = cur_text
            if timing and ((i + 1) % timing_every == 0 or (i + 1) == len(sentences)):
                t_sent = time.perf_counter() - t_sent0
                print(
                    f"[WM-TIME] {doc_tag} sent={i+1}/{len(sentences)} "
                    f"total={t_sent:.2f}s llm={t_llm:.2f}s score={t_score:.2f}s "
                    f"embed={t_embed:.2f}s cands={n_cands} accepted={int(accepted_any)}",
                    flush=True,
                )

        gains_arr = np.array(gains, dtype=np.float32) if gains else np.zeros((0,), dtype=np.float32)
        summary = GenSummary(
            n_sentences=len(sentences),
            changed_sentences=int(changed),
            changed_frac=(changed / max(1, len(sentences))),
            gain_mean=float(gains_arr.mean()) if gains else 0.0,
            gain_median=float(np.median(gains_arr)) if gains else 0.0,
            gain_p90=float(np.percentile(gains_arr, 90)) if gains else 0.0,
            bucket_flips=int(flips),
            params=vars(self.p)
        )
        if timing:
            t_total = time.perf_counter() - t_doc0
            avg = t_total / max(1, len(sentences))
            print(
                f"[WM-TIME] done {doc_tag} total={t_total:.2f}s avg_per_sent={avg:.2f}s "
                f"sents={len(sentences)}",
                flush=True,
            )
        return " ".join(sentences), summary

# ---------- Detector ----------
@dataclass
class DetectSummary:
    T: int
    D: int
    mean_proj: float
    z: float
    p_one_sided: float
    proj_mean: float
    proj_median: float
    proj_p90: float
    extras: Optional[Dict[str, Any]] = None

def norm_sf(z: float) -> float:
    try:
        from scipy.stats import norm
        return float(norm.sf(z))
    except Exception:
        from math import erfc, sqrt
        return 0.5 * erfc(z / math.sqrt(2.0))

class DirBucketDetector:
    def __init__(self, provider_key_hex: str, bucket_bits: int = 96, bucket_seed: int = 2025,
                 embed_model: str = EMBED_MODEL_NAME):
        self.embedder = Embedder(embed_model)
        self.bucket = SimHashCoder(
            dim=self.embedder.st.get_sentence_embedding_dimension(),
            m_bits=bucket_bits, seed=bucket_seed
        )
        self.dir_mgr = DirectionManager(provider_key_hex,
                                        dim=self.embedder.st.get_sentence_embedding_dimension())

    def _summarize(self, projs: np.ndarray, extras: Optional[Dict[str, Any]] = None) -> DetectSummary:
        D = self.embedder.st.get_sentence_embedding_dimension()
        T = int(projs.size)
        mean_proj = float(projs.mean()) if T > 0 else 0.0
        z = mean_proj * math.sqrt(D * max(1, T))
        p = norm_sf(z)
        return DetectSummary(
            T=T, D=D, mean_proj=mean_proj, z=z, p_one_sided=p,
            proj_mean=float(projs.mean()) if T > 0 else 0.0,
            proj_median=float(np.median(projs)) if T > 0 else 0.0,
            proj_p90=float(np.percentile(projs, 90)) if T > 0 else 0.0,
            extras=extras
        )

    def detect(self, text: str) -> DetectSummary:
        sents = split_sentences(text)
        if not sents:
            return self._summarize(np.array([], dtype=np.float32))
        U = self.embedder.embed(sents)
        projs = []
        for i in range(len(sents)):
            b = self.bucket.code(U[i])
            v = self.dir_mgr.direction(b)
            projs.append(float(np.dot(U[i], v)))
        return self._summarize(np.array(projs, dtype=np.float32))

    def detect_windows(self, text: str, win_words: int = 28, stride_words: int = 14,
                       anchor: str = "sentence") -> DetectSummary:
        """
        Window detection. If anchor="sentence" (default), each window is projected onto
        the PARENT SENTENCE bucket direction (matches generation). If anchor="window",
        windows use their own buckets (legacy, can fail).
        """
        sents = split_sentences(text)
        if not sents:
            return self._summarize(np.array([], dtype=np.float32))

        # sentence embeddings and buckets
        U_s = self.embedder.embed(sents)
        b_s = [self.bucket.code(U_s[i]) for i in range(len(sents))]
        v_s = [self.dir_mgr.direction(b_s[i]) for i in range(len(sents))]

        # build windows per sentence
        win_texts: List[str] = []
        parent_idx: List[int] = []  # which sentence each window came from
        per_sent_counts: List[int] = []

        for i, sent in enumerate(sents):
            ws = _windows_within_sentence(sent, win_words=win_words, stride_words=stride_words)
            per_sent_counts.append(len(ws))
            win_texts.extend(ws)
            parent_idx.extend([i] * len(ws))

        if not win_texts:
            return self._summarize(np.array([], dtype=np.float32))

        U_w = self.embedder.embed(win_texts)

        projs_parent = []
        projs_window = []
        bucket_matches = 0

        if anchor == "sentence":
            # project onto parent sentence direction
            for k in range(len(win_texts)):
                i = parent_idx[k]
                projs_parent.append(float(np.dot(U_w[k], v_s[i])))
            projs_parent = np.array(projs_parent, dtype=np.float32)

            # diagnostics: compare to window-bucket projections
            for k in range(len(win_texts)):
                b_w = self.bucket.code(U_w[k])
                if b_w == b_s[parent_idx[k]]:
                    bucket_matches += 1
                v_w = self.dir_mgr.direction(b_w)
                projs_window.append(float(np.dot(U_w[k], v_w)))
            projs_window = np.array(projs_window, dtype=np.float32)

            extras = {
                "mode": "windows",
                "anchor": "sentence",
                "n_sentences": len(sents),
                "windows_total": int(len(win_texts)),
                "windows_per_sentence": per_sent_counts,
                "bucket_match_rate": (bucket_matches / max(1, len(win_texts))),
                "mean_proj_parent": float(projs_parent.mean()) if projs_parent.size else 0.0,
                "mean_proj_window": float(projs_window.mean()) if projs_window.size else 0.0,
                "median_proj_parent": float(np.median(projs_parent)) if projs_parent.size else 0.0,
                "median_proj_window": float(np.median(projs_window)) if projs_window.size else 0.0,
            }
            return self._summarize(projs_parent, extras=extras)

        else:  # anchor == "window" (legacy)
            for k in range(len(win_texts)):
                b_w = self.bucket.code(U_w[k])
                v_w = self.dir_mgr.direction(b_w)
                projs_window.append(float(np.dot(U_w[k], v_w)))
            projs_window = np.array(projs_window, dtype=np.float32)
            extras = {
                "mode": "windows",
                "anchor": "window",
                "n_sentences": len(sents),
                "windows_total": int(len(win_texts)),
                "windows_per_sentence": per_sent_counts,
            }
            return self._summarize(projs_window, extras=extras)


# ---------- Helper: neighbor smoothing (radius=1) ----------
def _hexascii_to_bytes(code: bytes) -> bytearray:
    s = code.decode("ascii") if isinstance(code, (bytes, bytearray)) else str(code)
    return bytearray(bytes.fromhex(s))

def _bytes_to_hexascii(b: bytearray) -> bytes:
    return b.hex().encode("ascii")

def _neighbor_codes_radius1(code: bytes, m_bits: int) -> List[bytes]:
    """
    Return [code] + all Hamming-1 neighbors (LSB-first per byte).
    """
    base = _hexascii_to_bytes(code)
    out = [code]
    nbytes = (m_bits + 7) // 8
    for i in range(m_bits):
        bi = i // 8
        bit = i % 8
        if bi >= nbytes:  # safety
            break
        b = bytearray(base)  # copy
        b[bi] ^= (1 << bit)
        out.append(_bytes_to_hexascii(b))
    return out


class MultiDirBucketDetector:
    """
    Multi-provider directional-bucket detector for uncooperative RAG answers.

    Two modes:
      1) Unaligned (legacy): detect_windows_multi() -> projects every window to every provider.
      2) Aligned (recommended): detect_windows_multi_aligned(..., sources=...)
         - Assign each answer window to the most similar provider's source windows (if above threshold),
           optionally with competitive tie-breaking.
         - Then compute projections only for the assigned provider(s).
         - Optional bucket neighbor smoothing in projection space (Hamming neighbors of SimHash code).
    """
    def __init__(self, provider_keys_by_pid: Dict[int, str] | List[str],
                 bucket_bits: int = 96, bucket_seed: int = 2025,
                 embed_model: str = EMBED_MODEL_NAME,
                 direction_scheme: str = "naive"):
        self.embedder = Embedder(embed_model)
        D = self.embedder.st.get_sentence_embedding_dimension()
        self.bucket = SimHashCoder(dim=D, m_bits=bucket_bits, seed=bucket_seed)
        if isinstance(provider_keys_by_pid, (list, tuple)):
            provider_keys_by_pid = {i: provider_keys_by_pid[i] for i in range(len(provider_keys_by_pid))}
        self.provider_keys_by_pid = {int(pid): str(khex) for pid, khex in provider_keys_by_pid.items()}

        self.direction_scheme = str(direction_scheme or "naive").lower()
        if self.direction_scheme not in {"naive", "ortho"}:
            raise ValueError(f"Unknown direction_scheme={direction_scheme}. Use 'naive' or 'ortho'.")

        if self.direction_scheme == "naive":
            self.dir_by_pid = {pid: DirectionManager(khex, dim=D) for pid, khex in self.provider_keys_by_pid.items()}
            self.ortho_mgr = None
        else:
            self.dir_by_pid = {}
            self.ortho_mgr = OrthoDirectionManager(self.provider_keys_by_pid, dim=D)

    def _direction(self, pid: int, bucket_code: bytes) -> np.ndarray:
        pid = int(pid)
        if self.direction_scheme == "naive":
            return self.dir_by_pid[pid].direction(bucket_code)
        return self.ortho_mgr.direction(pid, bucket_code)  # type: ignore

    def _z(self, mean_proj: float, T: int) -> Tuple[float, float]:
        D = self.embedder.st.get_sentence_embedding_dimension()
        z = mean_proj * math.sqrt(D * max(1, T))
        try:
            from scipy.stats import norm
            p = float(norm.sf(z))
        except Exception:
            from math import erfc
            p = 0.5 * erfc(z / math.sqrt(2.0))
        return z, p

    def detect_windows_multi(self, text: str, win_words: int = 28, stride_words: int = 14) -> Dict[int, DetectSummary]:
        # Legacy unaligned mode retained for backward-compat
        sents = split_sentences(text)
        wins = []
        for s in sents:
            wins.extend(_windows_within_sentence(s, win_words=win_words, stride_words=stride_words))
        if not wins:
            out = {}
            for pid in self.provider_keys_by_pid.keys():
                out[pid] = DetectSummary(
                    T=0, D=self.embedder.st.get_sentence_embedding_dimension(),
                    mean_proj=0.0, z=0.0, p_one_sided=1.0,
                    proj_mean=0.0, proj_median=0.0, proj_p90=0.0,
                    extras={"mode": "multi-windows", "windows_total": 0, "windows_per_sentence": []}
                )
            return out

        U = self.embedder.embed(wins)
        projs_by_pid = {pid: [] for pid in self.provider_keys_by_pid.keys()}
        for i in range(U.shape[0]):
            b = self.bucket.code(U[i])
            for pid in self.provider_keys_by_pid.keys():
                v = self._direction(pid, b)
                projs_by_pid[pid].append(float(np.dot(U[i], v)))

        out: Dict[int, DetectSummary] = {}
        for pid, arr in projs_by_pid.items():
            arr = np.array(arr, dtype=np.float32)
            T = int(arr.size)
            mean_proj = float(arr.mean()) if T else 0.0
            z, p = self._z(mean_proj, T)
            out[pid] = DetectSummary(
                T=T, D=self.embedder.st.get_sentence_embedding_dimension(),
                mean_proj=mean_proj, z=z, p_one_sided=p,
                proj_mean=float(arr.mean()) if T else 0.0,
                proj_median=float(np.median(arr)) if T else 0.0,
                proj_p90=float(np.percentile(arr, 90)) if T else 0.0,
                extras={"mode": "multi-windows-unaligned", "windows_total": int(U.shape[0])}
            )
        return out

    def detect_windows_multi_aligned(
        self,
        text: str,
        *,
        sources: List[Tuple[int, str]],             # list of (pid, provider_text)
        sim_threshold: float = 0.84,
        competitive: bool = False,
        tie_margin: Optional[float] = None,
        neighbor_smoothing: bool = False,
        ham_radius: Optional[int] = None,
        smooth_agg: Optional[str] = "max",          # {"max","mean","median"}
        win_words: int = 28,
        stride_words: int = 14,
    ) -> Tuple[Dict[int, DetectSummary], Dict[str, Any]]:
        """
        1) Build answer windows (within-sentence).
        2) Build reference windows for each provider in `sources`.
        3) For each answer window, compute best cosine to each provider's reference; assign if >= threshold.
           - If competitive=True: keep only the single best provider; optionally require (best - second_best) >= tie_margin.
           - Else: a window can be assigned to multiple providers (one per provider above threshold).
        4) For each assigned (window, pid), compute projection <u_ans, v_pid(b_ref)>, where b_ref is the SimHash bucket
           of the **best-matching reference window** for that provider (this matches the provider's watermark bucket).
           If neighbor smoothing is enabled, smooth around b_ref.
        """
        # ------------- Answer windows -------------
        sents_ans = split_sentences(text)
        ans_wins: List[str] = []
        for s in sents_ans:
            ans_wins.extend(_windows_within_sentence(s, win_words=win_words, stride_words=stride_words))
        if not ans_wins:
            empty = {
                pid: DetectSummary(
                    T=0, D=self.embedder.st.get_sentence_embedding_dimension(),
                    mean_proj=0.0, z=0.0, p_one_sided=1.0,
                    proj_mean=0.0, proj_median=0.0, proj_p90=0.0,
                    extras={"mode": "multi-windows-aligned", "windows_total": 0}
                ) for pid in self.provider_keys_by_pid.keys()
            }
            meta = {
                "windows_total": 0,
                "competitive": competitive,
                "assigned_total": 0,
                "dropped_below_thresh_windows": 0,
                "sim_threshold": sim_threshold,
                "tie_margin": (tie_margin if competitive else None),
                "neighbor_smoothing": neighbor_smoothing,
                "ham_radius": (ham_radius if neighbor_smoothing else None),
                "smooth_agg": (smooth_agg if neighbor_smoothing else None),
                "bucket_anchor": "ref_bucket",
                "assignment_mode": ("competitive" if competitive else "non-competitive"),
                "per_provider_assigned": {str(pid): 0 for pid in self.provider_keys_by_pid.keys()},
                "best_cos_stats": {"mean": None, "p90": None},
                "second_best_cos_stats": {"mean": None, "p90": None},
            }
            return empty, meta

        U_ans = self.embedder.embed(ans_wins)

        # ------------- Reference windows per provider -------------
        refs_by_pid: Dict[int, List[str]] = {}
        for pid, doc in sources:
            ws = []
            for s in split_sentences(doc):
                ws.extend(_windows_within_sentence(s, win_words=win_words, stride_words=stride_words))
            if ws:
                refs_by_pid[int(pid)] = ws

        U_refs_by_pid: Dict[int, np.ndarray] = {}
        b_refs_by_pid: Dict[int, List[bytes]] = {}
        for pid, ws in refs_by_pid.items():
            U_refs_by_pid[pid] = self.embedder.embed(ws)  # unit-normalized rows
            # Precompute each reference window's bucket (used for direction lookup)
            b_refs_by_pid[pid] = [self.bucket.code(U_refs_by_pid[pid][j]) for j in range(U_refs_by_pid[pid].shape[0])]

        # ------------- Alignment -------------
        N = U_ans.shape[0]
        assigned_idx_by_pid: Dict[int, List[int]] = {pid: [] for pid in self.provider_keys_by_pid.keys()}
        # track the ref-index chosen for each (pid, answer_i)
        ref_idx_for_assignment: Dict[int, Dict[int, int]] = {pid: {} for pid in self.provider_keys_by_pid.keys()}
        dropped = 0
        best_cos_assigned: List[float] = []
        second_bests_for_assigned: List[float] = []

        for i in range(N):
            u = U_ans[i]
            best_by_pid: Dict[int, Tuple[float, int]] = {}  # pid -> (best_cos, best_ref_idx)
            for pid, Uref in U_refs_by_pid.items():
                if Uref.size == 0:
                    continue
                sims = (Uref @ u)  # (R,)
                j_star = int(np.argmax(sims))
                best_by_pid[pid] = (float(sims[j_star]), j_star)

            if not best_by_pid:
                dropped += 1
                continue

            if competitive:
                pid_star, (best, j_star) = max(best_by_pid.items(), key=lambda kv: kv[1][0])
                # diagnostics
                best_cos_assigned.append(best)
                if len(best_by_pid) > 1:
                    second = sorted([t[0] for t in best_by_pid.values()], reverse=True)[1]
                    second_bests_for_assigned.append(second)
                else:
                    second = -1.0
                # gates
                if best < sim_threshold:
                    dropped += 1
                    continue
                if tie_margin is not None and second >= 0 and (best - second) < tie_margin:
                    dropped += 1
                    continue
                assigned_idx_by_pid[pid_star].append(i)
                ref_idx_for_assignment[pid_star][i] = j_star
            else:
                any_assigned = False
                for pid, (val, j_star) in best_by_pid.items():
                    if val >= sim_threshold:
                        assigned_idx_by_pid[pid].append(i)
                        ref_idx_for_assignment[pid][i] = j_star
                        best_cos_assigned.append(val)
                        any_assigned = True
                if not any_assigned:
                    dropped += 1

        # ------------- Projections using REF BUCKET (with optional Hamming smoothing) -------------
        def _agg(vals: List[float], mode: str) -> float:
            if not vals:
                return 0.0
            if mode == "max":
                return max(vals)
            if mode == "median":
                return float(np.median(vals))
            return float(np.mean(vals))

        projs_by_pid: Dict[int, List[float]] = {pid: [] for pid in self.provider_keys_by_pid.keys()}

        for pid, idxs in assigned_idx_by_pid.items():
            if not idxs:
                continue
            dm = self.dir_by_pid.get(pid)
            Uref = U_refs_by_pid.get(pid)
            bref_list = b_refs_by_pid.get(pid)
            if Uref is None or bref_list is None:
                continue

            for i in idxs:
                u = U_ans[i]
                j_star = ref_idx_for_assignment[pid][i]
                b_ref = bref_list[j_star]  # <-- anchor to provider's REF bucket

                # --- projection for (pid, b_ref) ---
                if neighbor_smoothing and ham_radius and ham_radius > 0:
                    codes = [b_ref] + list(_hexcode_hamming_neighbors(b_ref, ham_radius))
                    vals = [float(np.dot(u, self._direction(pid, c))) for c in codes]
                    projs_by_pid[pid].append(_agg(vals, smooth_agg or "max"))
                else:
                    v = self._direction(pid, b_ref)
                    projs_by_pid[pid].append(float(np.dot(u, v)))

        # ------------- Summaries -------------
        res_by_pid: Dict[int, DetectSummary] = {}
        for pid, arr in projs_by_pid.items():
            arr = np.array(arr, dtype=np.float32)
            T = int(arr.size)
            mean_proj = float(arr.mean()) if T else 0.0
            z, p = self._z(mean_proj, T)
            res_by_pid[pid] = DetectSummary(
                T=T,
                D=self.embedder.st.get_sentence_embedding_dimension(),
                mean_proj=mean_proj,
                z=z,
                p_one_sided=p,
                proj_mean=float(arr.mean()) if T else 0.0,
                proj_median=float(np.median(arr)) if T else 0.0,
                proj_p90=float(np.percentile(arr, 90)) if T else 0.0,
                extras={"mode": "multi-windows-aligned"}
            )

        meta = {
            "windows_total": int(N),
            "competitive": competitive,
            "assigned_total": int(sum(len(v) for v in assigned_idx_by_pid.values())),
            "dropped_below_thresh_windows": int(dropped),
            "sim_threshold": float(sim_threshold),
            "tie_margin": (float(tie_margin) if (competitive and tie_margin is not None) else None),
            "neighbor_smoothing": bool(neighbor_smoothing),
            "ham_radius": (int(ham_radius) if neighbor_smoothing and ham_radius is not None else None),
            "smooth_agg": (smooth_agg if neighbor_smoothing else None),
            "bucket_anchor": "ref_bucket",
            "assignment_mode": ("competitive" if competitive else "non-competitive"),
            "per_provider_assigned": {str(pid): int(len(idxs)) for pid, idxs in assigned_idx_by_pid.items()},
            "best_cos_stats": {
                "mean": (float(np.mean(best_cos_assigned)) if best_cos_assigned else None),
                "p90": (float(np.percentile(best_cos_assigned, 90)) if best_cos_assigned else None),
            },
            "second_best_cos_stats": {
                "mean": (float(np.mean(second_bests_for_assigned)) if (competitive and second_bests_for_assigned) else None),
                "p90": (float(np.percentile(second_bests_for_assigned, 90)) if (competitive and second_bests_for_assigned) else None),
            },
        }
        return res_by_pid, meta



