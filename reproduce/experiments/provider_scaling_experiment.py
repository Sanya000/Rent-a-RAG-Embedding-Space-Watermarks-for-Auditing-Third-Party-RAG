#!/usr/bin/env python3
"""Marketplace-size scaling harness (paper App. app:marketplace-robustness, Table tab:marketplace-scaling).

Builds an m-provider mini corpus from the FARAD manifest (provider =
sha256("13:" + doc_id) mod m, ``--docs_per_provider`` docs each, keys from
np.random seed 42) watermarked at rho=1, K=24 with an inline hill-climber, then
audits ``--n_targets`` targets for ``--n_trials`` queries each with forced
target-document inclusion (controlled-exposure harness). Output:
``<out_dir>/m<m>/{corpus.jsonl, provider_keys.json, <summary_name>}``.

Shipped runs (one audit seed, 4-5 targets): m=12 with ``--T_min 0``
(auto = max(5, round(25*6/m)) = 12) and m=20 with ``--T_min 25``.
``--audit_seed`` re-audits an existing corpus with a different target-document
order and query sample, giving more than one audit per target; see
reproduce/scripts/26_provider_scaling.sh for the 30-audit design.

Reference sources are passed per retrieved document (the detector then keeps
the last document per provider); ``--merge_sources_per_provider`` merges them
per provider like the primary harness. Requires OPENAI_API_KEY.
"""
from __future__ import annotations

import argparse, asyncio, hashlib, json, math, os, sys, time
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


# ── constants matching the paper ──────────────────────────────────────────────
EMBED_AUDIT    = "sentence-transformers/all-mpnet-base-v2"   # for detection
EMBED_RETRIEVAL = "sentence-transformers/all-MiniLM-L6-v2"  # for retrieval
BUCKET_BITS    = 96
BUCKET_SEED    = 2025
WIN_WORDS      = 28
STRIDE_WORDS   = 14
SIM_THRESHOLD  = 0.82
TIE_MARGIN     = 0.05
ALPHA          = 0.05
TOPK           = 5
MIN_UNIQUE_PROV = 3
# watermarking (ρ=1 for speed; signal ≈73% of paper's ρ=6, Z_est≈8.6)
RHO   = 1
K_GEN = 24


# ── async LLM helpers ─────────────────────────────────────────────────────────
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


# ── provider assignment ───────────────────────────────────────────────────────
def _provider_id(doc_id: str, m: int, seed: int = 13) -> int:
    h = int(hashlib.sha256(f"{seed}:{doc_id}".encode()).hexdigest(), 16)
    return h % m


def _make_keys(m: int, seed: int = 42) -> List[str]:
    rng = np.random.RandomState(seed)
    return [rng.bytes(32).hex() for _ in range(m)]


# ── async watermarking ────────────────────────────────────────────────────────
async def watermark_sentence_async(
    sentence: str, key_hex: str, audit_enc: SentenceTransformer,
    simhash: SimHashCoder, client, model: str, sem: asyncio.Semaphore,
) -> str:
    dm = DirectionManager(key_hex, dim=audit_enc.get_sentence_embedding_dimension())
    K3 = max(1, K_GEN // 3)
    prompt = (
        f"Rewrite the sentence in a diverse set of paraphrases while preserving all facts.\n"
        f"Produce {K_GEN} variants: {K3} very light, {K3} medium, {K_GEN-2*K3} strong rewrites.\n"
        "Vary voice, ordering, connectives, nominalization, and tone.\n"
        "Return one variant per line, numbered.\n\nSentence:\n" + sentence
    )
    # two calls at different temperatures to get K_GEN candidates
    r1 = await _achat(client, prompt, model, sem, temperature=0.7)
    cands1 = _parse_list(r1)
    r2 = await _achat(client, prompt, model, sem, temperature=1.0)
    cands2 = _parse_list(r2)
    seen = set(cands1); candidates = cands1 + [c for c in cands2 if c not in seen]
    candidates = candidates[:K_GEN]

    if not candidates:
        return sentence

    u_base = audit_enc.encode([sentence], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    b_base = simhash.code(u_base)
    base_proj = float(np.dot(u_base, dm.direction(b_base)))
    cur_proj = base_proj

    best_text = sentence
    U_cands = audit_enc.encode(candidates, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)
    for i, cand in enumerate(candidates):
        u_c = U_cands[i]
        b_c = simhash.code(u_c)
        proj_c = float(np.dot(u_c, dm.direction(b_c)))
        sim_c  = float(np.dot(u_base, u_c))
        if sim_c >= 0.72 and (proj_c - cur_proj) >= 5e-4:
            cur_proj = proj_c
            best_text = cand
    return best_text


async def watermark_doc_async(
    doc_id: str, source_text: str, key_hex: str,
    audit_enc: SentenceTransformer, simhash: SimHashCoder,
    client, model: str, sem: asyncio.Semaphore,
) -> str:
    sentences = split_sentences(source_text)
    if not sentences:
        return source_text
    wm_tasks = [
        watermark_sentence_async(s, key_hex, audit_enc, simhash, client, model, sem)
        for s in sentences
    ]
    wm_sentences = await asyncio.gather(*wm_tasks)
    return " ".join(wm_sentences)


# ── RAG helpers ───────────────────────────────────────────────────────────────
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


def _topk_retrieve(q_vec: np.ndarray, X_ret: np.ndarray, rows: List[Dict],
                   target_doc_id: str, topk: int, min_unique: int) -> List[int]:
    sims = X_ret @ q_vec
    order = np.argsort(-sims)
    id2idx = {r["doc_id"]: i for i, r in enumerate(rows)}
    target_idx = id2idx.get(target_doc_id)
    picked = [target_idx] if target_idx is not None else []
    have_pids = {rows[target_idx]["provider_id"]} if target_idx is not None else set()
    for i in order:
        if len(picked) >= topk: break
        if target_idx is not None and int(i) == target_idx: continue
        pid = rows[int(i)]["provider_id"]
        if len(have_pids) < min_unique and pid in have_pids: continue
        picked.append(int(i))
        have_pids.add(pid)
    return picked[:topk]


# ── detection helpers (mirrors run_dirbucket_quick_experiments.py) ─────────────
def _norm_sf(z: float) -> float:
    try: return float(scipy_norm.sf(z))
    except Exception:
        from math import erfc
        return 0.5 * erfc(z / math.sqrt(2.0))

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

def _pooled_z(T: float, S: float, D: int = 768) -> float:
    if T <= 0: return 0.0
    return (S / T) * math.sqrt(D * T)


# ── main experiment ───────────────────────────────────────────────────────────
def _merge_by_provider(sources: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    by_pid: Dict[int, List[str]] = {}
    for pid, txt in sources:
        by_pid.setdefault(int(pid), []).append(txt)
    return [(pid, "\n\n".join(by_pid[pid])) for pid in sorted(by_pid)]


async def run_experiment(args: argparse.Namespace) -> None:
    from openai import AsyncOpenAI
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[FATAL] OPENAI_API_KEY is not set.")
    client = AsyncOpenAI()
    audit_seed = args.seed if args.audit_seed is None else args.audit_seed
    sem = asyncio.Semaphore(args.concurrency)

    out_dir = Path(args.out_dir) / f"m{args.m_providers}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] m={args.m_providers} providers, {args.docs_per_provider} docs/provider, "
          f"ρ={RHO}, K={K_GEN}", flush=True)

    # ── load FARAD manifest ──
    print("[LOAD] Manifest...", flush=True)
    manifest_rows = [json.loads(l) for l in Path(args.manifest).open("r", encoding="utf-8")]

    # Assign docs to m providers via hash
    m = int(args.m_providers)
    keys = _make_keys(m, seed=42)
    doc_info: Dict[str, Dict] = {}
    for row in manifest_rows:
        pid = _provider_id(str(row["id"]), m, seed=13)
        doc_info[str(row["id"])] = {"pid": pid, "text_src": row["text_src"]}

    # Stratified sample: args.docs_per_provider per provider
    per_pid: Dict[int, List[str]] = {p: [] for p in range(m)}
    for did, info in doc_info.items():
        per_pid[info["pid"]].append(did)
    rng = np.random.RandomState(args.seed)
    selected_ids: List[str] = []
    for p in range(m):
        pool = per_pid[p]
        rng.shuffle(pool)
        selected_ids.extend(pool[:args.docs_per_provider])
    print(f"  Selected {len(selected_ids)} docs across {m} providers", flush=True)

    # ── async watermarking (resume-safe: appends incrementally) ──
    corpus_path = out_dir / "corpus.jsonl"
    keys_path   = out_dir / "provider_keys.json"

    # Always save keys first so they survive a partial run
    if not keys_path.exists():
        keys_path.write_text(json.dumps({"keys_hex": keys, "m": m}, indent=2))

    # Load already-done doc IDs from a partial corpus
    done_ids: set = set()
    corpus_rows: List[Dict] = []
    if corpus_path.exists():
        for line in corpus_path.open("r", encoding="utf-8"):
            try:
                row = json.loads(line)
                done_ids.add(row["doc_id"])
                corpus_rows.append(row)
            except Exception:
                pass

    todo_ids = [did for did in selected_ids if did not in done_ids]
    print(f"[WM] {len(done_ids)} docs already done, {len(todo_ids)} remaining.", flush=True)

    print("[WM] Loading audit encoder...", flush=True)
    audit_enc = SentenceTransformer(EMBED_AUDIT)
    simhash = SimHashCoder(dim=audit_enc.get_sentence_embedding_dimension(),
                           m_bits=BUCKET_BITS, seed=BUCKET_SEED)

    if todo_ids:
        print(f"[WM] Watermarking {len(todo_ids)} docs async (ρ={RHO})...", flush=True)
        t0 = time.perf_counter()
        with corpus_path.open("a", encoding="utf-8") as corp_f:
            for did in tqdm(todo_ids, desc="Watermarking docs"):
                info = doc_info[did]
                wm_text = await watermark_doc_async(
                    did, info["text_src"], keys[info["pid"]],
                    audit_enc, simhash, client, args.model, sem,
                )
                row = {"doc_id": did, "provider_id": info["pid"],
                       "text_src": info["text_src"], "text_wm": wm_text}
                corp_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                corp_f.flush()
                corpus_rows.append(row)
        print(f"[WM] Done in {time.perf_counter()-t0:.1f}s", flush=True)
    else:
        print("[SKIP] All docs already watermarked.", flush=True)

    # Ensure corpus_rows is in selected_ids order for consistent retrieval
    id2row = {r["doc_id"]: r for r in corpus_rows}
    corpus_rows = [id2row[did] for did in selected_ids if did in id2row]
    print(f"[WM] Corpus ready: {len(corpus_rows)} docs.", flush=True)

    # ── build retrieval index ──
    print("[INDEX] Building retrieval index (all-MiniLM-L6-v2)...", flush=True)
    ret_enc = SentenceTransformer(EMBED_RETRIEVAL)
    wm_texts = [(r["text_wm"] or "")[:1000] for r in corpus_rows]
    X_ret = ret_enc.encode(wm_texts, normalize_embeddings=True, convert_to_numpy=True,
                           batch_size=256, show_progress_bar=False).astype(np.float32)

    # ── detector ──
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys[i] for i in range(m)},
        bucket_bits=BUCKET_BITS, bucket_seed=BUCKET_SEED,
        embed_model=EMBED_AUDIT, direction_scheme="naive",
    )

    # ── audit experiment ──
    targets = list(range(min(args.n_targets, m)))
    n_trials = args.n_trials
    D = det.embedder.st.get_sentence_embedding_dimension()
    # Proportional T_min: scale down as providers increase to preserve statistical power
    T_MIN = args.T_min if args.T_min > 0 else max(5, round(25 * 6 / m))
    print(f"[AUDIT] T_min={T_MIN} (auto-scaled for m={m}), doc_max_chars={args.doc_max_chars}", flush=True)

    all_results: Dict[str, Any] = {}

    for target_pid in targets:
        # Get docs belonging to this provider
        target_docs = [r for r in corpus_rows if r["provider_id"] == target_pid]
        if not target_docs:
            print(f"[SKIP] Provider {target_pid} has no docs", flush=True)
            continue
        rng2 = np.random.RandomState(audit_seed + target_pid)
        rng2.shuffle(target_docs)

        # Pooling state
        pooled_T = np.zeros(m, dtype=np.float64)
        pooled_S = np.zeros(m, dtype=np.float64)
        q2bh: Optional[int] = None

        print(f"[AUDIT] target_pid={target_pid} ({n_trials} trials)...", flush=True)
        for trial_idx in tqdm(range(n_trials), desc=f"  pid={target_pid}", leave=False):
            # Select target doc (cycle through)
            tdoc = target_docs[trial_idx % len(target_docs)]

            # Generate query from target doc
            q_prompt = QUERY_PROMPT.format(doc=(tdoc["text_src"] or "")[:600])
            question = await _achat(client, q_prompt, args.model, sem, temperature=0.2, max_tokens=80)
            if not question: continue

            # Retrieve
            q_vec = ret_enc.encode([question], normalize_embeddings=True,
                                   convert_to_numpy=True)[0].astype(np.float32)
            idxs = _topk_retrieve(q_vec, X_ret, corpus_rows, tdoc["doc_id"],
                                   TOPK, MIN_UNIQUE_PROV)

            # Build reference sources (global corpus = all retrieved docs' wm text)
            sources = [(corpus_rows[i]["provider_id"], corpus_rows[i]["text_wm"])
                       for i in idxs if corpus_rows[i]["text_wm"]]
            if args.merge_sources_per_provider:
                sources = _merge_by_provider(sources)

            # Generate answer (ownwords style)
            doc_block = "\n\n".join(
                f"[Doc {k+1}] {corpus_rows[i]['text_wm'][:args.doc_max_chars]}"
                for k, i in enumerate(idxs) if corpus_rows[i]["text_wm"]
            )
            ans_prompt = ANSWER_PROMPT.format(docs=doc_block, question=question)
            answer = await _achat(client, ans_prompt, args.model, sem,
                                  temperature=0.55, max_tokens=300)
            if not answer: continue

            # Detect
            res, _ = det.detect_windows_multi_aligned(
                answer, sources=sources,
                sim_threshold=SIM_THRESHOLD, competitive=True,
                tie_margin=TIE_MARGIN, win_words=WIN_WORDS, stride_words=STRIDE_WORDS,
            )

            # Accumulate
            for pid in range(m):
                if pid in res:
                    ds = res[pid]
                    T_new = int(ds.T); S_new = float(ds.mean_proj) * T_new
                else:
                    T_new = 0; S_new = 0.0
                pooled_T[pid] += T_new
                pooled_S[pid] += S_new

            # Check BH
            if q2bh is None:
                z_raw = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(m)]
                pvals = [(_norm_sf(z_raw[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(m)]
                bh_sig = _bh(pvals, ALPHA)
                if pooled_T[target_pid] >= T_MIN and bh_sig[target_pid]:
                    q2bh = trial_idx + 1

        # Final stats
        z_final = [_pooled_z(pooled_T[p], pooled_S[p], D) for p in range(m)]
        pvals_final = [(_norm_sf(z_final[p]) if pooled_T[p] >= T_MIN else 1.0) for p in range(m)]
        bh_final = _bh(pvals_final, ALPHA)

        non_pids = [p for p in range(m) if p != target_pid]
        nt_z_max = max(z_final[p] if pooled_T[p] >= T_MIN else 0.0 for p in non_pids)
        any_nt_bh = any(bh_final[p] for p in non_pids)

        result = {
            "target_pid": target_pid,
            "final_target_z": float(z_final[target_pid] if pooled_T[target_pid] >= T_MIN else 0.0),
            "final_nontarget_z_max": float(nt_z_max),
            "bh_detected": bool(bh_final[target_pid]),
            "any_nontarget_bh": bool(any_nt_bh),
            "final_support": int(pooled_T[target_pid]),
            "queries_to_bh": q2bh,
        }
        all_results[f"t{target_pid}"] = result
        print(f"  → target Z={result['final_target_z']:.2f}  NT Z_max={result['final_nontarget_z_max']:.2f}  "
              f"detected={result['bh_detected']}  NT_FP={result['any_nontarget_bh']}  "
              f"q2bh={q2bh}", flush=True)

    # Summary
    tzs = [r["final_target_z"] for r in all_results.values()]
    nts = [r["final_nontarget_z_max"] for r in all_results.values()]
    dets = [r["bh_detected"] for r in all_results.values()]
    fps  = [r["any_nontarget_bh"] for r in all_results.values()]
    q2ds = [r["queries_to_bh"] for r in all_results.values() if r["queries_to_bh"] is not None]
    summary = {
        "m": m,
        "T_min": T_MIN,
        "doc_max_chars": args.doc_max_chars,
        "docs_per_provider": args.docs_per_provider,
        "n_trials": n_trials,
        "n_targets": len(targets),
        "rho": RHO,
        "audit_seed": audit_seed,
        "merge_sources_per_provider": bool(args.merge_sources_per_provider),
        "mean_target_z": float(np.mean(tzs)) if tzs else float("nan"),
        "median_target_z": float(np.median(tzs)) if tzs else float("nan"),
        "detection_rate": float(np.mean(dets)) if dets else float("nan"),
        "max_nontarget_z": float(np.max(nts)) if nts else 0.0,
        "fpr": float(np.mean(fps)) if fps else 0.0,
        "median_q2bh": float(np.median(q2ds)) if q2ds else None,
        "per_target": all_results,
    }

    out_json = out_dir / args.summary_name
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n[WRITE] {out_json}")
    print(f"[SUMMARY] m={m}: target_Z={summary['median_target_z']:.2f}  "
          f"det={summary['detection_rate']:.2f}  NT_Z={summary['max_nontarget_z']:.2f}  "
          f"FPR={summary['fpr']:.2f}  q2bh={summary['median_q2bh']}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  type=str, default="work/farad_shared_manifest_p6_d400_seed13.jsonl")
    ap.add_argument("--out_dir",   type=str, default="results/provider_scaling")
    ap.add_argument("--m_providers",     type=int, default=12)
    ap.add_argument("--docs_per_provider", type=int, default=20)
    ap.add_argument("--n_trials",  type=int, default=80,
                    help="Audit queries per target. Paper uses 80; scale up with m.")
    ap.add_argument("--n_targets", type=int, default=4)
    ap.add_argument("--T_min",     type=int, default=0,
                    help="Min support gate. 0 = auto-scale: 25 * (6/m_providers). "
                         "Paper uses T_min=25 for m=6; use ~12 for m=12.")
    ap.add_argument("--doc_max_chars", type=int, default=1500,
                    help="Max chars per retrieved doc in answer prompt. 400=quick/sparse, 1500=better alignment.")
    ap.add_argument("--model",     type=str, default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--audit_seed", type=int, default=None,
                    help="Seed for target-document order in the audit (default: --seed).")
    ap.add_argument("--summary_name", type=str, default="summary.json",
                    help="Summary file name inside <out_dir>/m<m>/ (avoid overwriting runs).")
    ap.add_argument("--merge_sources_per_provider", action="store_true",
                    help="Merge retrieved reference docs per provider (primary-harness behaviour).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    asyncio.run(run_experiment(args))


if __name__ == "__main__":
    main()
