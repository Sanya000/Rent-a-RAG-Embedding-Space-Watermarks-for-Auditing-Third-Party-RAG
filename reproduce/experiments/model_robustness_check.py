#!/usr/bin/env python3
"""Model robustness checks (paper App. app:model-robustness, Table tab:model-robustness).

Part A  alternative embedding encoder: re-watermarks a FARAD subset (provider =
        sha256("13:" + doc_id) mod 6, keys from np.random seed 43) in the
        encoder's own space at rho=1, K=24, then audits with the same encoder
        (forced target inclusion, T_min=25). Paper rows: all-MiniLM-L6-v2
        (default) and sentence-transformers/gtr-t5-base
        (``--part_a_embed ... --out_dir results/model_robustness_gtr``).
Part B  alternative API answer generator (gpt-3.5-turbo) on the primary corpus
        and primary questions/retrieved docs (seed-01 trial logs), mpnet audit.
        Output dir: ``part_b_gpt4o``.
Part C  local open-weight generator (Qwen2.5-1.5B-Instruct via transformers),
        otherwise as Part B. Needs no API key.

Defaults: 4 targets, 80 trials (Part A) and 40 trials (Parts B/C). The
"mpnet (primary)" row of the table is the ``original`` condition of
post_answer_laundering_attack.py on seeds 1-3.
Reference sources are passed per retrieved document (the detector keeps the
last document per provider); ``--merge_sources_per_provider`` merges them like
the primary harness.
"""
from __future__ import annotations

import argparse, asyncio, json, math, os, random, sys, time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as scipy_norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from watermark_dirbucket import (
    DirectionManager, SimHashCoder, MultiDirBucketDetector,
    _windows_within_sentence, split_sentences,
)

# ── shared constants ─────────────────────────────────────────────────────────
BUCKET_BITS  = 96
BUCKET_SEED  = 2025
WIN_WORDS    = 28
STRIDE_WORDS = 14
SIM_THRESH   = 0.82
TIE_MARGIN   = 0.05
T_MIN        = 25
ALPHA        = 0.05
TOPK         = 5
MIN_UNQ_PROV = 3
K_GEN        = 24

# Part A settings — can be overridden by CLI (--part_a_ndocs, --part_a_rho, --part_a_trials)
PART_A_EMBED  = "sentence-transformers/all-MiniLM-L6-v2"   # alternative (d=384)
PART_A_RHO    = 1   # ρ=1 for overnight full-corpus run (fast, 73% gain)
PART_A_NDOCS  = 400 # full corpus — 67 docs/provider × 6 providers (proper scale)
PART_A_NPROV  = 6
PART_A_TRIALS = 80  # match paper's 80 trials
PART_A_NTARG  = 4

# Part B settings (API model — gpt-3.5-turbo: different family from the primary gpt-4o-mini)
PART_B_EMBED  = "sentence-transformers/all-mpnet-base-v2"   # paper model (d=768)
PART_B_MODEL  = "gpt-3.5-turbo"   # older GPT family, clearly distinct from gpt-4o-mini
PART_B_TRIALS = 40
PART_B_NTARG  = 4

# Part C settings (local open-source LLM — Qwen2.5-1.5B-Instruct)
PART_C_EMBED  = "sentence-transformers/all-mpnet-base-v2"   # paper model (d=768)
PART_C_HF_ID  = "Qwen/Qwen2.5-1.5B-Instruct"               # cached at ~/.cache/huggingface
PART_C_TRIALS = 40
PART_C_NTARG  = 4


# ── async helpers ─────────────────────────────────────────────────────────────
async def _achat(client, prompt: str, model: str, sem: asyncio.Semaphore,
                 temperature: float = 0.7, max_tokens: int = 900) -> str:
    async with sem:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )
                return (r.choices[0].message.content or "").strip()
            except Exception:
                if attempt == 2: return ""
                await asyncio.sleep(2 * (attempt + 1))
    return ""


def _parse_list(s: str) -> List[str]:
    import re
    lines = [l.strip(" -*\t") for l in s.strip().splitlines() if l.strip()]
    out, seen = [], set()
    for l in lines:
        t = re.sub(r"^\d+[\)\.\-]\s*", "", l)
        if t and t not in seen:
            out.append(t); seen.add(t)
    return out


# ── watermarking (generic, works for any embedding model) ────────────────────
async def wm_sentence(sentence: str, key_hex: str, enc: SentenceTransformer,
                       sh: SimHashCoder, client, model: str,
                       sem: asyncio.Semaphore, rho: int = 3) -> str:
    dm = DirectionManager(key_hex, dim=enc.get_sentence_embedding_dimension())
    K3 = max(1, K_GEN // 3)
    prompt = (
        f"Rewrite the sentence in a diverse set of paraphrases while preserving all facts.\n"
        f"Produce {K_GEN} variants: {K3} very light, {K3} medium, {K_GEN-2*K3} strong rewrites.\n"
        "Vary voice, ordering, connectives, nominalization, and tone.\n"
        "Return one variant per line, numbered.\n\nSentence:\n" + sentence
    )
    cur_text, cur_u = sentence, enc.encode([sentence], normalize_embeddings=True,
                                           convert_to_numpy=True)[0].astype(np.float32)
    cur_b = sh.code(cur_u)
    cur_proj = float(np.dot(cur_u, dm.direction(cur_b)))

    for _ in range(rho):
        r1 = await _achat(client, prompt, model, sem, temperature=0.7)
        cands = _parse_list(r1)
        r2 = await _achat(client, prompt, model, sem, temperature=1.0)
        seen = set(cands); cands += [c for c in _parse_list(r2) if c not in seen]
        cands = cands[:K_GEN]
        if not cands: continue
        U = enc.encode(cands, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
        best_gain, best_cand = 0.0, None
        for i, cand in enumerate(cands):
            u = U[i]; b = sh.code(u)
            proj = float(np.dot(u, dm.direction(b)))
            sim  = float(np.dot(cur_u, u))
            if sim >= 0.72 and (proj - cur_proj) >= 5e-4 and (proj - cur_proj) > best_gain:
                best_gain = proj - cur_proj
                best_cand = (cand, u, b, proj)
        if best_cand:
            cur_text, cur_u, cur_b, cur_proj = best_cand
    return cur_text


async def wm_doc(doc_id, src_text, key_hex, enc, sh, client, model, sem, rho=3) -> str:
    sents = split_sentences(src_text)
    if not sents: return src_text
    wm_sents = await asyncio.gather(*[
        wm_sentence(s, key_hex, enc, sh, client, model, sem, rho) for s in sents
    ])
    return " ".join(wm_sents)


# ── stats helpers ─────────────────────────────────────────────────────────────
def _norm_sf(z: float) -> float:
    try: return float(scipy_norm.sf(z))
    except: from math import erfc; return 0.5 * erfc(z / math.sqrt(2.0))

def _bh(pvals: List[float], alpha: float) -> List[bool]:
    m = len(pvals)
    if not m: return []
    idx = np.argsort(pvals)
    p_s = np.array(pvals)[idx]
    sig = p_s <= (np.arange(1, m+1) / m) * alpha
    k = int(np.max(np.where(sig)[0]) + 1) if np.any(sig) else 0
    out = np.zeros(m, dtype=bool)
    if k > 0: out[idx[:k]] = True
    return out.tolist()

def _pooled_z(T: float, S: float, D: int) -> float:
    return (S / T) * math.sqrt(D * T) if T > 0 else 0.0

def _provider_id(doc_id: str, m: int, seed: int = 13) -> int:
    import hashlib
    h = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest(), 16)
    return h % m

def _make_keys(m: int, seed: int = 42) -> List[str]:
    rng = np.random.RandomState(seed)
    return [rng.bytes(32).hex() for _ in range(m)]

def _merge_by_provider(sources: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    by_pid: Dict[int, List[str]] = {}
    for pid, txt in sources:
        by_pid.setdefault(int(pid), []).append(txt)
    return [(pid, "\n\n".join(by_pid[pid])) for pid in sorted(by_pid)]

QUERY_PROMPT = (
    "Read the document below and write ONE question whose answer must use at least "
    "three specific details (names, dates, numbers, or short phrases) from the document. "
    "Avoid yes/no or vague prompts. Return ONLY the question.\n\nDocument:\n{doc}"
)
ANSWER_PROMPT = (
    "You are a RAG assistant. Answer using ONLY the provided documents.\n"
    "Synthesize across multiple docs. Use your OWN WORDS; avoid quoting exact spans.\n\n"
    "Here are retrieved documents:\n{docs}\n\nQuestion: {question}"
)


# ═══════════════════════════════════════════════════════════════════════════════
# PART A: Alternative embedding model (MiniLM, d=384)
# ═══════════════════════════════════════════════════════════════════════════════
async def run_part_a(args, client, sem):
    PART_A_EMBED = args.part_a_embed
    PART_A_RHO = args.part_a_rho
    PART_A_NDOCS = args.part_a_ndocs
    PART_A_TRIALS = args.part_a_trials
    PART_A_NTARG = args.part_a_ntargets
    print("\n" + "="*60)
    print(f"PART A: Embedding model = {PART_A_EMBED}")
    print("="*60, flush=True)
    out_dir = Path(args.out_dir) / args.part_a_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = out_dir / "corpus.jsonl"
    keys_path   = out_dir / "keys.json"

    # Load FARAD manifest
    manifest = [json.loads(l) for l in Path(args.manifest).open("r", encoding="utf-8")]

    m = PART_A_NPROV
    keys = _make_keys(m, seed=43)
    if not keys_path.exists():
        keys_path.write_text(json.dumps(keys))

    # Stratified doc selection
    per_pid: Dict[int, List] = {p: [] for p in range(m)}
    for row in manifest:
        pid = _provider_id(str(row["id"]), m, seed=13)
        per_pid[pid].append(row)
    rng = np.random.RandomState(args.seed)
    selected = []
    n_per_pid = PART_A_NDOCS // m
    for p in range(m):
        pool = per_pid[p][:]
        rng.shuffle(pool)
        selected.extend(pool[:n_per_pid])
    print(f"  Selected {len(selected)} docs ({n_per_pid}/provider)", flush=True)

    # Load MiniLM encoder
    print(f"  Loading {PART_A_EMBED}...", flush=True)
    enc  = SentenceTransformer(PART_A_EMBED)
    D    = enc.get_sentence_embedding_dimension()
    sh   = SimHashCoder(dim=D, m_bits=BUCKET_BITS, seed=BUCKET_SEED)

    # Build (or load) watermarked corpus
    done_ids: set = set()
    corpus_rows: List[Dict] = []
    if corpus_path.exists():
        for line in corpus_path.open("r", encoding="utf-8"):
            try:
                row = json.loads(line); done_ids.add(row["doc_id"]); corpus_rows.append(row)
            except: pass

    todo = [r for r in selected if str(r["id"]) not in done_ids]
    if todo:
        print(f"  Watermarking {len(todo)} docs with {PART_A_EMBED} + ρ={PART_A_RHO}...", flush=True)
        t0 = time.perf_counter()
        with corpus_path.open("a", encoding="utf-8") as f:
            for row in tqdm(todo, desc="  WM (Part A)"):
                pid = _provider_id(str(row["id"]), m, seed=13)
                wm  = await wm_doc(str(row["id"]), row["text_src"], keys[pid],
                                   enc, sh, client, args.wm_model, sem, rho=PART_A_RHO)
                rec = {"doc_id": str(row["id"]), "provider_id": pid,
                       "text_src": row["text_src"], "text_wm": wm}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                corpus_rows.append(rec)
        print(f"  Watermarking done in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print("  Corpus already built.", flush=True)

    # Build retrieval index (MiniLM, same model)
    wm_texts = [(r["text_wm"] or "")[:1000] for r in corpus_rows]
    X_ret = enc.encode(wm_texts, normalize_embeddings=True, convert_to_numpy=True,
                       batch_size=256, show_progress_bar=False).astype(np.float32)

    # Detector (MiniLM, d=384)
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys[i] for i in range(m)},
        bucket_bits=BUCKET_BITS, bucket_seed=BUCKET_SEED,
        embed_model=PART_A_EMBED, direction_scheme="naive",
    )

    targets = list(range(min(PART_A_NTARG, m)))
    results: Dict[str, Any] = {}
    for target_pid in targets:
        tdocs = [r for r in corpus_rows if r["provider_id"] == target_pid]
        if not tdocs: continue
        rng2 = np.random.RandomState(args.seed + target_pid * 10)
        rng2.shuffle(tdocs)

        pooled_T = np.zeros(m); pooled_S = np.zeros(m)
        q2bh: Optional[int] = None

        for trial_idx in tqdm(range(PART_A_TRIALS), desc=f"  A pid={target_pid}", leave=False):
            tdoc = tdocs[trial_idx % len(tdocs)]
            q = await _achat(client, QUERY_PROMPT.format(doc=tdoc["text_src"][:600]),
                             args.wm_model, sem, temperature=0.2, max_tokens=80)
            if not q: continue
            q_vec = enc.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
            sims = X_ret @ q_vec; order = np.argsort(-sims)
            # Force target doc first, add up to TOPK total
            picked = [next(i for i, r in enumerate(corpus_rows) if r["doc_id"] == tdoc["doc_id"])]
            have = {corpus_rows[picked[0]]["provider_id"]}
            for i in order:
                if len(picked) >= TOPK: break
                if int(i) == picked[0]: continue
                picked.append(int(i)); have.add(corpus_rows[int(i)]["provider_id"])
            sources = [(corpus_rows[i]["provider_id"], corpus_rows[i]["text_wm"])
                       for i in picked if corpus_rows[i]["text_wm"]]
            if args.merge_sources_per_provider:
                sources = _merge_by_provider(sources)

            doc_block = "\n\n".join(f"[Doc {k+1}] {corpus_rows[i]['text_wm'][:1500]}"
                                    for k, i in enumerate(picked) if corpus_rows[i]["text_wm"])
            answer = await _achat(client, ANSWER_PROMPT.format(docs=doc_block, question=q),
                                  args.wm_model, sem, temperature=0.55, max_tokens=300)
            if not answer: continue

            res, _ = det.detect_windows_multi_aligned(
                answer, sources=sources, sim_threshold=SIM_THRESH, competitive=True,
                tie_margin=TIE_MARGIN, win_words=WIN_WORDS, stride_words=STRIDE_WORDS,
            )
            for pid in range(m):
                if pid in res:
                    T_new = int(res[pid].T); S_new = float(res[pid].mean_proj) * T_new
                else:
                    T_new = 0; S_new = 0.0
                pooled_T[pid] += T_new; pooled_S[pid] += S_new

            if q2bh is None:
                z_raw = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(m)]
                pvals = [(_norm_sf(z_raw[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(m)]
                if _bh(pvals, ALPHA)[target_pid] and pooled_T[target_pid] >= T_MIN:
                    q2bh = trial_idx + 1

        z_final = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(m)]
        pvals_f = [(_norm_sf(z_final[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(m)]
        bh_f = _bh(pvals_f, ALPHA)
        non_pids = [p for p in range(m) if p != target_pid]
        results[f"pid{target_pid}"] = {
            "target_z": float(z_final[target_pid] if pooled_T[target_pid] >= T_MIN else 0.0),
            "detected": bool(bh_f[target_pid]),
            "nt_z_max": float(max(z_final[p] if pooled_T[p] >= T_MIN else 0.0 for p in non_pids)),
            "any_nt_bh": bool(any(bh_f[p] for p in non_pids)),
            "support": int(pooled_T[target_pid]),
            "q2bh": q2bh,
        }
        print(f"  A pid={target_pid}: Z={results[f'pid{target_pid}']['target_z']:.2f}  "
              f"det={results[f'pid{target_pid}']['detected']}  "
              f"NT_Z={results[f'pid{target_pid}']['nt_z_max']:.2f}  "
              f"q2bh={q2bh}", flush=True)

    summary_a = {
        "embed_model": PART_A_EMBED, "D": D, "rho": PART_A_RHO,
        "n_docs": len(corpus_rows), "n_providers": m, "T_min": T_MIN,
        "n_trials": PART_A_TRIALS, "n_targets": len(targets),
        "median_target_z": float(np.median([v["target_z"] for v in results.values()])),
        "detection_rate": float(np.mean([v["detected"] for v in results.values()])),
        "max_nt_z": float(np.max([v["nt_z_max"] for v in results.values()])),
        "fpr": float(np.mean([v["any_nt_bh"] for v in results.values()])),
        "median_q2bh": (float(np.median([v["q2bh"] for v in results.values() if v["q2bh"]])) if any(v["q2bh"] for v in results.values()) else None),
        "per_target": results,
    }
    (out_dir / "summary_a.json").write_text(json.dumps(summary_a, indent=2))
    print(f"\n  Part A summary: Z={summary_a['median_target_z']:.2f}  "
          f"det={summary_a['detection_rate']:.2f}  FPR={summary_a['fpr']:.2f}  "
          f"q2bh={summary_a['median_q2bh']}", flush=True)
    return summary_a


# ═══════════════════════════════════════════════════════════════════════════════
# PART B: Alternative answer generation model (gpt-4o)
# ═══════════════════════════════════════════════════════════════════════════════
async def run_part_b(args, client, sem):
    PART_B_MODEL = args.part_b_model
    PART_B_TRIALS = args.bc_trials
    PART_B_NTARG = args.bc_targets
    print("\n" + "="*60)
    print(f"PART B: Answer generation model = {PART_B_MODEL}")
    print("="*60, flush=True)
    out_dir = Path(args.out_dir) / "part_b_gpt4o"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load the paper's watermarked corpus
    corpus_map: Dict[str, Dict] = {}
    for line in Path(args.wm_corpus).open("r", encoding="utf-8"):
        try:
            row = json.loads(line); corpus_map[str(row["id"])] = row
        except: pass
    keys_hex = json.loads(Path(args.wm_keys).read_text())["keys_hex"]
    n_providers = len(keys_hex)
    print(f"  Loaded {len(corpus_map)} watermarked docs, {n_providers} providers", flush=True)

    # Load mpnet detector (paper's model)
    print(f"  Loading {PART_B_EMBED}...", flush=True)
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys_hex[i] for i in range(n_providers)},
        bucket_bits=BUCKET_BITS, bucket_seed=BUCKET_SEED,
        embed_model=PART_B_EMBED, direction_scheme="naive",
    )
    D = det.embedder.st.get_sentence_embedding_dimension()

    # Load MiniLM for query retrieval (same as paper's retrieval)
    ret_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    corpus_rows = list(corpus_map.values())
    wm_texts = [(r["text_wm"] or "")[:1000] for r in corpus_rows]
    X_ret = ret_enc.encode(wm_texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=256, show_progress_bar=False).astype(np.float32)

    # Load existing trial logs (questions + retrieved_docs from primary setting)
    targets = list(range(PART_B_NTARG))
    trial_logs_dir = Path(args.trial_logs_dir)
    results: Dict[str, Any] = {}

    for target_pid in targets:
        log_path = trial_logs_dir / f"rag_multi_t{target_pid}_seed{args.trial_seed:02d}.trials.jsonl"
        if not log_path.exists():
            print(f"  [SKIP] {log_path}", flush=True); continue
        trials = sorted(
            [json.loads(l) for l in log_path.open("r", encoding="utf-8")],
            key=lambda r: int(r.get("trial_idx", 0))
        )[:PART_B_TRIALS]

        pooled_T = np.zeros(n_providers); pooled_S = np.zeros(n_providers)
        q2bh: Optional[int] = None

        for trial_idx, trial in enumerate(tqdm(trials, desc=f"  B pid={target_pid}", leave=False)):
            question = trial["question"]
            retrieved_ids = [d["doc_id"] for d in trial.get("retrieved_docs", [])]
            sources = [(corpus_map[did]["provider_id"], corpus_map[did]["text_wm"])
                       for did in retrieved_ids if did in corpus_map and corpus_map[did]["text_wm"]]
            if not sources: continue
            if args.merge_sources_per_provider:
                sources = _merge_by_provider(sources)

            doc_block = "\n\n".join(
                f"[Doc {k+1}] {corpus_map[did]['text_wm'][:1500]}"
                for k, did in enumerate(retrieved_ids) if did in corpus_map
            )
            # Generate answer with the ALTERNATIVE model (gpt-4o)
            answer = await _achat(client, ANSWER_PROMPT.format(docs=doc_block, question=question),
                                  PART_B_MODEL, sem, temperature=0.55, max_tokens=300)
            if not answer: continue

            # Detect with paper's embedding model (mpnet)
            res, _ = det.detect_windows_multi_aligned(
                answer, sources=sources, sim_threshold=SIM_THRESH, competitive=True,
                tie_margin=TIE_MARGIN, win_words=WIN_WORDS, stride_words=STRIDE_WORDS,
            )
            for pid in range(n_providers):
                if pid in res:
                    T_new = int(res[pid].T); S_new = float(res[pid].mean_proj) * T_new
                else:
                    T_new = 0; S_new = 0.0
                pooled_T[pid] += T_new; pooled_S[pid] += S_new

            if q2bh is None:
                z_raw = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]
                pvals = [(_norm_sf(z_raw[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(n_providers)]
                if _bh(pvals, ALPHA)[target_pid] and pooled_T[target_pid] >= T_MIN:
                    q2bh = trial_idx + 1

        z_final = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]
        pvals_f = [(_norm_sf(z_final[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(n_providers)]
        bh_f = _bh(pvals_f, ALPHA)
        non_pids = [p for p in range(n_providers) if p != target_pid]
        results[f"pid{target_pid}"] = {
            "target_z": float(z_final[target_pid] if pooled_T[target_pid] >= T_MIN else 0.0),
            "detected": bool(bh_f[target_pid]),
            "nt_z_max": float(max(z_final[p] if pooled_T[p] >= T_MIN else 0.0 for p in non_pids)),
            "any_nt_bh": bool(any(bh_f[p] for p in non_pids)),
            "support": int(pooled_T[target_pid]),
            "q2bh": q2bh,
        }
        print(f"  B pid={target_pid}: Z={results[f'pid{target_pid}']['target_z']:.2f}  "
              f"det={results[f'pid{target_pid}']['detected']}  q2bh={q2bh}", flush=True)

    summary_b = {
        "answer_model": PART_B_MODEL, "embed_model": PART_B_EMBED, "D": D,
        "n_trials": PART_B_TRIALS, "n_targets": len(targets), "T_min": T_MIN,
        "median_target_z": float(np.median([v["target_z"] for v in results.values()])) if results else float("nan"),
        "detection_rate": float(np.mean([v["detected"] for v in results.values()])) if results else float("nan"),
        "max_nt_z": float(np.max([v["nt_z_max"] for v in results.values()])) if results else 0.0,
        "fpr": float(np.mean([v["any_nt_bh"] for v in results.values()])) if results else float("nan"),
        "median_q2bh": (float(np.median([v["q2bh"] for v in results.values() if v["q2bh"]])) if any(v["q2bh"] for v in results.values()) else None),
        "per_target": results,
    }
    (out_dir / "summary_b.json").write_text(json.dumps(summary_b, indent=2))
    print(f"\n  Part B summary: Z={summary_b['median_target_z']:.2f}  "
          f"det={summary_b['detection_rate']:.2f}  FPR={summary_b['fpr']:.2f}  "
          f"q2bh={summary_b['median_q2bh']}", flush=True)
    return summary_b


# ═══════════════════════════════════════════════════════════════════════════════
# PART C: Local open-source LLM (Qwen2.5-1.5B-Instruct)
# ═══════════════════════════════════════════════════════════════════════════════
def _load_qwen(hf_id: str):
    """Load Qwen2.5-1.5B-Instruct from HF cache for local GPU inference.
    Does NOT use device_map='auto' (requires accelerate); uses .to(device) instead."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  Loading local model {hf_id}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)
    has_cuda = torch.cuda.is_available()
    device = "cuda" if has_cuda else "cpu"
    dtype  = torch.float16 if has_cuda else torch.float32
    print(f"  Device: {device}, dtype: {dtype}", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=dtype, trust_remote_code=True,
        # device_map="auto" intentionally omitted — requires accelerate package
    )
    model = model.to(device)
    model.eval()
    print(f"  Loaded Qwen on {device}.", flush=True)
    return tokenizer, model


def _qwen_generate(tokenizer, model, prompt: str, max_new_tokens: int = 280) -> str:
    """Generate a single response from the local Qwen model."""
    import torch
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=True, temperature=0.6, top_p=0.9, pad_token_id=tokenizer.eos_token_id,
        )
    # Strip the input tokens from output
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


async def run_part_c(args):
    PART_C_HF_ID = args.part_c_hf_id
    PART_C_TRIALS = args.bc_trials
    PART_C_NTARG = args.bc_targets
    print("\n" + "="*60)
    print(f"PART C: Local LLM = {PART_C_HF_ID} (Qwen2.5-1.5B-Instruct)")
    print("="*60, flush=True)
    out_dir = Path(args.out_dir) / "part_c_qwen"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load paper's watermarked corpus
    corpus_map: Dict[str, Dict] = {}
    for line in Path(args.wm_corpus).open("r", encoding="utf-8"):
        try:
            row = json.loads(line); corpus_map[str(row["id"])] = row
        except: pass
    keys_hex = json.loads(Path(args.wm_keys).read_text())["keys_hex"]
    n_providers = len(keys_hex)

    # Load mpnet detector (paper's model)
    print(f"  Loading {PART_C_EMBED}...", flush=True)
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys_hex[i] for i in range(n_providers)},
        bucket_bits=BUCKET_BITS, bucket_seed=BUCKET_SEED,
        embed_model=PART_C_EMBED, direction_scheme="naive",
    )
    D = det.embedder.st.get_sentence_embedding_dimension()

    # Load Qwen2.5-1.5B locally
    tokenizer, qwen_model = _load_qwen(PART_C_HF_ID)

    targets = list(range(PART_C_NTARG))
    trial_logs_dir = Path(args.trial_logs_dir)
    results: Dict[str, Any] = {}

    for target_pid in targets:
        log_path = trial_logs_dir / f"rag_multi_t{target_pid}_seed{args.trial_seed:02d}.trials.jsonl"
        if not log_path.exists():
            print(f"  [SKIP] {log_path}", flush=True); continue
        trials = sorted(
            [json.loads(l) for l in log_path.open("r", encoding="utf-8")],
            key=lambda r: int(r.get("trial_idx", 0))
        )[:PART_C_TRIALS]

        pooled_T = np.zeros(n_providers); pooled_S = np.zeros(n_providers)
        q2bh: Optional[int] = None

        for trial_idx, trial in enumerate(tqdm(trials, desc=f"  C pid={target_pid}", leave=False)):
            question = trial["question"]
            retrieved_ids = [d["doc_id"] for d in trial.get("retrieved_docs", [])]
            sources = [(corpus_map[did]["provider_id"], corpus_map[did]["text_wm"])
                       for did in retrieved_ids if did in corpus_map and corpus_map[did]["text_wm"]]
            if not sources: continue
            if args.merge_sources_per_provider:
                sources = _merge_by_provider(sources)

            doc_block = "\n\n".join(
                f"[Doc {k+1}] {corpus_map[did]['text_wm'][:1500]}"
                for k, did in enumerate(retrieved_ids) if did in corpus_map
            )
            # Generate with local Qwen (synchronous — GPU inference)
            prompt = ANSWER_PROMPT.format(docs=doc_block, question=question)
            answer = _qwen_generate(tokenizer, qwen_model, prompt, max_new_tokens=280)
            if not answer: continue

            res, _ = det.detect_windows_multi_aligned(
                answer, sources=sources, sim_threshold=SIM_THRESH, competitive=True,
                tie_margin=TIE_MARGIN, win_words=WIN_WORDS, stride_words=STRIDE_WORDS,
            )
            for pid in range(n_providers):
                if pid in res:
                    T_new = int(res[pid].T); S_new = float(res[pid].mean_proj) * T_new
                else:
                    T_new = 0; S_new = 0.0
                pooled_T[pid] += T_new; pooled_S[pid] += S_new

            if q2bh is None:
                z_raw = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]
                pvals = [(_norm_sf(z_raw[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(n_providers)]
                if _bh(pvals, ALPHA)[target_pid] and pooled_T[target_pid] >= T_MIN:
                    q2bh = trial_idx + 1

        z_final = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]
        pvals_f = [(_norm_sf(z_final[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(n_providers)]
        bh_f = _bh(pvals_f, ALPHA)
        non_pids = [p for p in range(n_providers) if p != target_pid]
        results[f"pid{target_pid}"] = {
            "target_z": float(z_final[target_pid] if pooled_T[target_pid] >= T_MIN else 0.0),
            "detected": bool(bh_f[target_pid]),
            "nt_z_max": float(max(z_final[p] if pooled_T[p] >= T_MIN else 0.0 for p in non_pids)),
            "any_nt_bh": bool(any(bh_f[p] for p in non_pids)),
            "support": int(pooled_T[target_pid]),
            "q2bh": q2bh,
        }
        print(f"  C pid={target_pid}: Z={results[f'pid{target_pid}']['target_z']:.2f}  "
              f"det={results[f'pid{target_pid}']['detected']}  q2bh={q2bh}", flush=True)

    summary_c = {
        "answer_model": PART_C_HF_ID, "embed_model": PART_C_EMBED, "D": D,
        "n_trials": PART_C_TRIALS, "n_targets": len(targets), "T_min": T_MIN,
        "median_target_z": float(np.median([v["target_z"] for v in results.values()])) if results else float("nan"),
        "detection_rate": float(np.mean([v["detected"] for v in results.values()])) if results else float("nan"),
        "max_nt_z": float(np.max([v["nt_z_max"] for v in results.values()])) if results else 0.0,
        "fpr": float(np.mean([v["any_nt_bh"] for v in results.values()])) if results else float("nan"),
        "median_q2bh": (float(np.median([v["q2bh"] for v in results.values() if v["q2bh"]])) if any(v["q2bh"] for v in results.values()) else None),
        "per_target": results,
    }
    (out_dir / "summary_c.json").write_text(json.dumps(summary_c, indent=2))
    print(f"\n  Part C summary: Z={summary_c['median_target_z']:.2f}  "
          f"det={summary_c['detection_rate']:.2f}  FPR={summary_c['fpr']:.2f}  "
          f"q2bh={summary_c['median_q2bh']}", flush=True)
    return summary_c


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Model robustness checks (Parts A, B, C).")
    ap.add_argument("--manifest",       type=str, default="work/farad_shared_manifest_p6_d400_seed13.jsonl")
    ap.add_argument("--wm_corpus",      type=str, default="work/farad_dirbucket_naive/corpus.jsonl")
    ap.add_argument("--wm_keys",        type=str, default="work/farad_dirbucket_naive/provider_keys.json")
    ap.add_argument("--trial_logs_dir", type=str, default="results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive")
    ap.add_argument("--out_dir",        type=str, default="results/model_robustness")
    ap.add_argument("--wm_model",       type=str, default="gpt-4o-mini", help="Model for watermark paraphrasing.")
    ap.add_argument("--concurrency",    type=int, default=15)
    ap.add_argument("--seed",           type=int, default=42)
    ap.add_argument("--parts",          type=str, default="A,B,C", help="Comma-sep subset of A,B,C")
    ap.add_argument("--part_a_embed",   type=str, default=PART_A_EMBED)
    ap.add_argument("--part_a_rho",     type=int, default=PART_A_RHO)
    ap.add_argument("--part_a_ndocs",   type=int, default=PART_A_NDOCS)
    ap.add_argument("--part_a_trials",  type=int, default=PART_A_TRIALS)
    ap.add_argument("--part_a_ntargets", type=int, default=PART_A_NTARG)
    ap.add_argument("--part_a_subdir",  type=str, default="part_a_minilm",
                    help="Part A output sub-directory (the paper GTR run also used part_a_minilm).")
    ap.add_argument("--part_b_model",   type=str, default=PART_B_MODEL)
    ap.add_argument("--part_c_hf_id",   type=str, default=PART_C_HF_ID)
    ap.add_argument("--bc_trials",      type=int, default=PART_B_TRIALS)
    ap.add_argument("--bc_targets",     type=int, default=PART_B_NTARG)
    ap.add_argument("--trial_seed",     type=int, default=1,
                    help="Primary-run seed whose trial logs feed Parts B and C.")
    ap.add_argument("--merge_sources_per_provider", action="store_true",
                    help="Merge reference docs per provider (primary-harness behaviour).")
    return ap.parse_args()


async def _main(args: argparse.Namespace) -> None:
    parts = [p.strip().upper() for p in args.parts.split(",")]
    unknown = sorted(set(parts) - {"A", "B", "C"})
    if unknown:
        raise SystemExit(f"[FATAL] unknown parts {unknown} (Part D was removed: not in the paper)")
    client = None
    if {"A", "B"} & set(parts):
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("[FATAL] OPENAI_API_KEY is not set (needed by Parts A and B).")
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
    sem = asyncio.Semaphore(args.concurrency)
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}
    if "A" in parts:
        results["part_a"] = await run_part_a(args, client, sem)
    if "B" in parts:
        results["part_b"] = await run_part_b(args, client, sem)
    if "C" in parts:
        results["part_c"] = await run_part_c(args)

    print("\n" + "="*80)
    print("MODEL ROBUSTNESS SUMMARY")
    print("="*80)
    print(f"{'Component':<32} {'Model':>36} {'Z':>7} {'Det':>6} {'FPR':>6} {'Q2BH':>6}")
    print("-"*80)
    for key, label, model in (("part_a", "A: embedding encoder", args.part_a_embed),
                              ("part_b", "B: API answer generator", args.part_b_model),
                              ("part_c", "C: local answer generator", args.part_c_hf_id)):
        if key in results:
            r = results[key]
            print(f"{label:<32} {model:>36} {r['median_target_z']:>7.2f} "
                  f"{r['detection_rate']:>6.2f} {r['fpr']:>6.2f} {str(r['median_q2bh']):>6}")

    out_json = Path(args.out_dir) / "robustness_results.json"
    merged = json.loads(out_json.read_text()) if out_json.exists() else {}
    merged.update(results)  # runs with different --parts accumulate
    out_json.write_text(json.dumps(merged, indent=2))
    print(f"\n[WRITE] {args.out_dir}/robustness_results.json")


def main():
    args = _parse_args()
    asyncio.run(_main(args))


if __name__ == "__main__":
    main()
