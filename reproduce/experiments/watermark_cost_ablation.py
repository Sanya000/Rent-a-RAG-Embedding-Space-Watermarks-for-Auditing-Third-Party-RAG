#!/usr/bin/env python3
"""Watermark cost-strength ablation (paper App. app:cost-strength, Table tab:cost-strength).

Phase 1 (API): for a provider-stratified sample of manifest documents, every
source sentence gets rho=6 rounds of K=24 paraphrase candidates (two batched
gpt-4o-mini calls per round). All candidates with their directional score and
semantic similarity are logged to ``<out_dir>/candidates_debug.jsonl``
(resumable per document).

Phase 2 (offline): every (K, rho) in {6,12,24} x {1,3,6} is replayed from the
log by hill-climbing over the first K candidates of the first rho rounds
(``simulate_krho``). This is an approximation: smaller settings reuse the
candidates generated along the full-run trajectory.

Phase 3: "Target Z" is ``z_anchor x gain(K, rho) / gain(24, 6)`` with
``z_anchor = 11.85`` (median target Z of the primary audit on seeds 1-3, see
detector_sensitivity_sweep.py).

Paper run: ``--n_docs 54 --seed 42``. ``--simulate_only`` re-runs phases 2-3
from an existing candidates log without any API call.

Usage:
  python experiments/watermark_cost_ablation.py \
      --manifest work/farad_shared_manifest_p6_d400_seed13.jsonl \
      --work_dir work/farad_dirbucket_naive --out_dir results/watermark_cost_ablation \
      --n_docs 54 --seed 42 --model gpt-4o-mini --z_anchor 11.85 [--simulate_only]
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
    DirectionManager, SimHashCoder, split_sentences,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
K_VALUES   = [6, 12, 24]       # candidates per round to simulate
RHO_VALUES = [1,  3,  6]       # rounds to simulate
TAU_SEM    = 0.72               # paper's semantic similarity gate
DELTA_MIN  = 5e-4               # paper's minimum directional gain gate
K_GEN      = 24                 # candidates actually generated per round
RHO_GEN    = 6                  # rounds actually generated

# Anchor Z from full-corpus sensitivity sweep (W=28, τ_sim=0.82, T_min=25)
Z_FULL_CORPUS = 11.85


# ---------------------------------------------------------------------------
# Async GPT helpers
# ---------------------------------------------------------------------------
async def _achat(client: Any, prompt: str, model: str, temperature: float,
                 semaphore: asyncio.Semaphore) -> str:
    async with semaphore:
        for attempt in range(3):
            try:
                r = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=900,
                    temperature=temperature,
                )
                return (r.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    return ""
                await asyncio.sleep(2.0 * (attempt + 1))
    return ""


def _parse_list(s: str) -> List[str]:
    import re
    lines = [l.strip(" -*\t") for l in s.strip().splitlines() if l.strip()]
    out = []
    for l in lines:
        out.append(re.sub(r"^\d+[\)\.\-]\s*", "", l))
    seen = set()
    return [t for t in out if t and t not in seen and not seen.add(t)]  # type: ignore


async def _generate_candidates_for_sentence(
    client: Any, sentence: str, K: int, rho: int,
    model: str, semaphore: asyncio.Semaphore,
) -> List[List[Dict[str, Any]]]:
    """Generate K candidates for rho rounds. Returns list-of-rounds of raw text candidates."""
    K_third = max(1, K // 3)
    prompt = (
        f"Rewrite the sentence in a diverse set of paraphrases while preserving all facts.\n"
        f"Produce {K} variants: {K_third} very light, {K_third} medium, {K-2*K_third} strong rewrites.\n"
        "Vary voice, ordering, connectives, nominalization, and tone.\n"
        "Return one variant per line, numbered.\n\nSentence:\n" + sentence
    )
    all_rounds: List[List[str]] = []
    for _ in range(rho):
        t1_resp = await _achat(client, prompt, model, temperature=0.7, semaphore=semaphore)
        t2_resp = await _achat(client, prompt, model, temperature=1.0, semaphore=semaphore)
        cands1 = _parse_list(t1_resp)
        cands2 = _parse_list(t2_resp)
        seen = set(cands1)
        combined = cands1 + [c for c in cands2 if c not in seen]
        all_rounds.append(combined[:K])
    return all_rounds


# ---------------------------------------------------------------------------
# Watermark scoring helpers
# ---------------------------------------------------------------------------
class _Scorer:
    def __init__(self, key_hex: str, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.enc = SentenceTransformer(model_name)
        D = self.enc.get_sentence_embedding_dimension()
        self.simhash = SimHashCoder(dim=D, m_bits=96, seed=2025)
        self.dm = DirectionManager(key_hex, dim=D)

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.enc.encode(texts, normalize_embeddings=True, convert_to_numpy=True).astype(np.float32)

    def score(self, embedding: np.ndarray) -> Tuple[float, bytes]:
        b = self.simhash.code(embedding)
        v = self.dm.direction(b)
        return float(np.dot(embedding, v)), b

    def sem_sim(self, u0: np.ndarray, u1: np.ndarray) -> float:
        return float(np.dot(u0, u1))


# ---------------------------------------------------------------------------
# Phase 2: simulate (K_sim, rho_sim) from saved candidate data
# ---------------------------------------------------------------------------
def simulate_krho(
    sent_log: Dict[str, Any],
    K_sim: int,
    rho_sim: int,
    tau_sem: float = TAU_SEM,
    delta_min: float = DELTA_MIN,
) -> Tuple[float, bool, str]:
    """
    Replay the hill-climbing using first K_sim candidates from each of
    the first rho_sim rounds. Returns (total_gain, changed, accepted_text).
    """
    base_proj = sent_log["base_proj"]
    cur_proj = base_proj
    accepted_text = sent_log["base_text"]

    for r_idx in range(min(rho_sim, len(sent_log["rounds"]))):
        round_cands = sent_log["rounds"][r_idx][:K_sim]
        best_gain = 0.0
        best_cand = None
        for c in round_cands:
            if c["sem_sim"] < tau_sem:
                continue
            gain = c["proj"] - cur_proj
            if gain < delta_min:
                continue
            if gain > best_gain:
                best_gain = gain
                best_cand = c
        if best_cand is not None:
            cur_proj = best_cand["proj"]
            accepted_text = best_cand["text"]

    return cur_proj - base_proj, (accepted_text != sent_log["base_text"]), accepted_text


# ---------------------------------------------------------------------------
# Phase 1: async watermarking of one document
# ---------------------------------------------------------------------------
async def watermark_doc_async(
    doc_id: str,
    source_text: str,
    provider_id: int,
    key_hex: str,
    scorer: _Scorer,
    client: Any,
    model: str,
    semaphore: asyncio.Semaphore,
    K: int = K_GEN,
    rho: int = RHO_GEN,
) -> List[Dict[str, Any]]:
    """
    Watermark one document's sentences, saving full candidate data.
    Returns list of per-sentence logs.
    """
    sentences = split_sentences(source_text)
    if not sentences:
        return []

    # Embed all base sentences in one batch
    U_base = scorer.embed(sentences)

    # Generate candidates for all sentences concurrently (within this doc)
    tasks = [
        _generate_candidates_for_sentence(client, sent, K, rho, model, semaphore)
        for sent in sentences
    ]
    all_cand_rounds = await asyncio.gather(*tasks)

    sent_logs: List[Dict[str, Any]] = []
    for i, (sent, cand_rounds_texts) in enumerate(zip(sentences, all_cand_rounds)):
        u_base = U_base[i]
        base_proj, _ = scorer.score(u_base)

        rounds_data: List[List[Dict[str, Any]]] = []
        for r_texts in cand_rounds_texts:
            if not r_texts:
                rounds_data.append([])
                continue
            U_cands = scorer.embed(r_texts)
            round_cands = []
            for j, ctext in enumerate(r_texts):
                u_c = U_cands[j]
                proj_c, _ = scorer.score(u_c)
                sim_c = scorer.sem_sim(u_base, u_c)
                round_cands.append({
                    "text": ctext,
                    "proj": float(proj_c),
                    "sem_sim": float(sim_c),
                    "delta": float(proj_c - base_proj),  # delta from original
                })
            rounds_data.append(round_cands)

        sent_logs.append({
            "doc_id": doc_id,
            "sent_idx": i,
            "base_text": sent,
            "base_proj": float(base_proj),
            "rounds": rounds_data,
        })
    return sent_logs


# ---------------------------------------------------------------------------
# Load manifest
# ---------------------------------------------------------------------------
def load_manifest(path: Path, n_docs: int, seed: int) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rng = np.random.RandomState(seed)
    # Stratified sample: n_docs / n_providers per provider
    by_pid: Dict[int, List] = {}
    for r in rows:
        pid = int(r.get("provider_id", 0))
        by_pid.setdefault(pid, []).append(r)
    pids = sorted(by_pid.keys())
    per_pid = max(1, n_docs // len(pids))
    selected = []
    for pid in pids:
        pool = by_pid[pid]
        rng.shuffle(pool)
        selected.extend(pool[:per_pid])
    # top-up to n_docs if needed
    all_remaining = [r for r in rows if r not in selected]
    rng.shuffle(all_remaining)
    selected.extend(all_remaining[:max(0, n_docs - len(selected))])
    return selected[:n_docs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest",  type=Path, default=Path("work/farad_shared_manifest_p6_d400_seed13.jsonl"))
    ap.add_argument("--work_dir",  type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--out_dir",   type=Path, default=Path("results/watermark_cost_ablation"))
    ap.add_argument("--n_docs",    type=int,  default=50)
    ap.add_argument("--seed",      type=int,  default=42)
    ap.add_argument("--model",     type=str,  default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--z_anchor",  type=float, default=Z_FULL_CORPUS,
                    help="Full-corpus detection Z to use for extrapolation (from sensitivity sweep).")
    ap.add_argument("--simulate_only", action="store_true",
                    help="Skip Phase 1 (no API calls); simulate from an existing candidates_debug.jsonl.")
    return ap.parse_args()


async def _run_async(args: argparse.Namespace) -> None:
    semaphore = asyncio.Semaphore(args.concurrency)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = args.out_dir / "candidates_debug.jsonl"
    results_path = args.out_dir / "ablation_results.json"

    # Load provider keys
    meta = json.loads((args.work_dir / "provider_keys.json").read_text())
    keys_hex = list(meta["keys_hex"])

    # Load docs
    print(f"[LOAD] Manifest: {args.manifest}")
    docs = load_manifest(args.manifest, args.n_docs, args.seed)
    print(f"  Selected {len(docs)} docs across providers: "
          f"{dict(sorted({int(d['provider_id']): sum(1 for x in docs if int(x['provider_id'])==int(d['provider_id'])) for d in docs}.items()))}")

    # One scorer per provider key (all-mpnet-base-v2 + HMAC directions)
    # Share the embedding model; only direction manager differs per provider
    print("[LOAD] Embedding model (all-mpnet-base-v2)...")
    shared_enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
    D = shared_enc.get_sentence_embedding_dimension()
    sh = SimHashCoder(dim=D, m_bits=96, seed=2025)

    # Phase 1: generate candidates
    print(f"\n[PHASE 1] Generating K={K_GEN} candidates x rho={RHO_GEN} rounds for {len(docs)} docs...")
    t0 = time.perf_counter()

    # Open debug log (resume-safe: check already-done doc_ids)
    done_ids: set = set()
    if debug_path.exists():
        with debug_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    done_ids.add(rec.get("doc_id"))
                except Exception:
                    pass
        print(f"  Resuming: {len(done_ids)} doc_ids already done.")

    todo_docs = [d for d in docs if str(d["id"]) not in done_ids]
    print(f"  {len(todo_docs)} docs to process.")
    if args.simulate_only:
        if not debug_path.exists():
            raise SystemExit(f"[FATAL] --simulate_only needs {debug_path}")
        if todo_docs:
            print(f"  [WARN] --simulate_only: {len(todo_docs)} sampled docs are not in the log; skipping them.")
        todo_docs = []
    client = None
    if todo_docs:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit("[FATAL] OPENAI_API_KEY is not set (needed for Phase 1); "
                             "use --simulate_only to re-run the offline phases.")
        from openai import AsyncOpenAI
        client = AsyncOpenAI()

    # Build scorers lazily per provider
    _scorer_cache: Dict[int, _Scorer] = {}
    def get_scorer(pid: int) -> _Scorer:
        if pid not in _scorer_cache:
            sc = _Scorer.__new__(_Scorer)
            sc.enc = shared_enc
            sc.simhash = sh
            sc.dm = DirectionManager(keys_hex[pid], dim=D)
            _scorer_cache[pid] = sc
        return _scorer_cache[pid]

    n_total_sents = sum(len(split_sentences(d["text_src"])) for d in todo_docs)
    print(f"  Total sentences to watermark: {n_total_sents}")
    print(f"  Estimated API calls: {n_total_sents * RHO_GEN * 2} "
          f"(~${n_total_sents * RHO_GEN * 2 * 0.0003:.2f} at gpt-4o-mini pricing)")

    if todo_docs:
        with debug_path.open("a", encoding="utf-8") as dbf:
            for doc in tqdm(todo_docs, desc="Docs"):
                pid = int(doc["provider_id"])
                scorer = get_scorer(pid)
                sent_logs = await watermark_doc_async(
                    doc_id=str(doc["id"]),
                    source_text=str(doc["text_src"]),
                    provider_id=pid,
                    key_hex=keys_hex[pid],
                    scorer=scorer,
                    client=client,
                    model=args.model,
                    semaphore=semaphore,
                )
                # Save per-doc as one line
                rec = {"doc_id": str(doc["id"]), "provider_id": pid, "sentences": sent_logs}
                dbf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                dbf.flush()

    elapsed = time.perf_counter() - t0
    print(f"[PHASE 1] Done in {elapsed:.1f}s")

    # Phase 2: load debug log and simulate all (K, ρ) combinations
    print("\n[PHASE 2] Simulating all (K, ρ) combinations from saved candidates...")
    all_sent_logs: List[Dict[str, Any]] = []
    with debug_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                for sl in rec.get("sentences", []):
                    all_sent_logs.append(sl)
            except Exception:
                pass

    print(f"  Total sentences available: {len(all_sent_logs)}")

    # Reference gain at (K=24, rho=6) — the maximum
    gains_ref: List[float] = []
    for sl in all_sent_logs:
        g, _, _ = simulate_krho(sl, K_GEN, RHO_GEN)
        gains_ref.append(g)
    mean_gain_ref = float(np.mean(gains_ref))
    print(f"  Reference gain (K=24, ρ=6): mean={mean_gain_ref:.5f}")

    # Now simulate all 9 combinations
    grid_results: Dict[str, Any] = {}
    all_base_texts = [sl["base_text"] for sl in all_sent_logs]
    U_base_all = shared_enc.encode(all_base_texts, normalize_embeddings=True,
                                    convert_to_numpy=True, batch_size=256).astype(np.float32)

    for K_sim in K_VALUES:
        for rho_sim in RHO_VALUES:
            gains, changed_flags, accepted_texts = [], [], []
            for sl in all_sent_logs:
                g, ch, at = simulate_krho(sl, K_sim, rho_sim)
                gains.append(g)
                changed_flags.append(ch)
                accepted_texts.append(at)

            # Utility: cosine similarity between original and accepted text
            U_accepted = shared_enc.encode(accepted_texts, normalize_embeddings=True,
                                           convert_to_numpy=True, batch_size=256).astype(np.float32)
            sem_sims = (U_base_all * U_accepted).sum(axis=1)  # element-wise dot

            gains_arr = np.array(gains)
            gain_ratio = float(gains_arr.mean()) / max(mean_gain_ref, 1e-9)
            z_est = args.z_anchor * gain_ratio

            # Relative API cost: rho_sim rounds, 2 calls each (K doesn't affect call count)
            rel_api_calls = (rho_sim * 2) / (RHO_GEN * 2)  # relative to (rho=6)

            key = f"K{K_sim}_rho{rho_sim}"
            grid_results[key] = {
                "K": K_sim,
                "rho": rho_sim,
                "mean_gain": float(gains_arr.mean()),
                "median_gain": float(np.median(gains_arr)),
                "gain_ratio": float(gain_ratio),
                "changed_frac": float(np.mean(changed_flags)),
                "sem_sim_mean": float(sem_sims.mean()),
                "sem_sim_min":  float(sem_sims.min()),
                "z_estimated":  float(z_est),
                "rel_api_calls": float(rel_api_calls),
                "api_calls_per_sentence": int(rho_sim * 2),
            }

    # Report
    print("\n" + "=" * 70)
    print("(K, ρ) ABLATION RESULTS")
    print("=" * 70)
    print(f"{'K':>4} {'ρ':>3} | {'Rel. cost':>10} | {'Mean gain':>10} | {'Gain ratio':>11} | "
          f"{'Changed%':>9} | {'Sem sim':>8} | {'Est. Z':>7}")
    print("-" * 70)
    for rho_sim in RHO_VALUES:
        for K_sim in K_VALUES:
            r = grid_results[f"K{K_sim}_rho{rho_sim}"]
            is_paper = (K_sim == K_GEN and rho_sim == RHO_GEN)
            marker = "  *" if is_paper else "   "
            print(f"{K_sim:>4} {rho_sim:>3}{marker} | {r['rel_api_calls']:>9.2f}x | "
                  f"{r['mean_gain']:>10.5f} | {r['gain_ratio']:>10.1%} | "
                  f"{r['changed_frac']:>8.1%} | {r['sem_sim_mean']:>8.4f} | {r['z_estimated']:>7.2f}")

    print("\n* = paper setting (K=24, ρ=6)")

    # Save results
    output = {
        "n_docs": len(docs),
        "n_sentences": len(all_sent_logs),
        "k_generated": K_GEN,
        "rho_generated": RHO_GEN,
        "mean_gain_reference": mean_gain_ref,
        "z_anchor": args.z_anchor,
        "grid": grid_results,
    }
    results_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\n[WRITE] {results_path}")

    # Build markdown table
    _write_note(output, args.out_dir)


def _write_note(results: Dict[str, Any], out_dir: Path) -> None:
    grid = results["grid"]
    n_sents = results["n_sentences"]
    n_docs  = results["n_docs"]
    z_anch  = results["z_anchor"]

    rows = []
    for rho_sim in RHO_VALUES:
        for K_sim in K_VALUES:
            r = grid[f"K{K_sim}_rho{rho_sim}"]
            is_paper = (K_sim == K_GEN and rho_sim == RHO_GEN)
            rows.append({
                "K": f"**{K_sim}**" if is_paper else str(K_sim),
                "ρ": f"**{rho_sim}**" if is_paper else str(rho_sim),
                "API calls/sent": f"**{r['api_calls_per_sentence']}**" if is_paper else str(r["api_calls_per_sentence"]),
                "Relative cost": f"**1.00×**" if is_paper else f"{r['rel_api_calls']:.2f}×",
                "Mean gain": f"{r['mean_gain']:.5f}",
                "Gain ratio": f"**100%**" if is_paper else f"{r['gain_ratio']:.0%}",
                "Changed%": f"{r['changed_frac']:.0%}",
                "Sem. sim.": f"{r['sem_sim_mean']:.4f}",
                "Est. Z": f"**{r['z_estimated']:.2f}**" if is_paper else f"{r['z_estimated']:.2f}",
            })

    cols = list(rows[0].keys())
    header = "| " + " | ".join(cols) + " |"
    sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
    body   = "\n".join("| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows)
    table_md = "\n".join([header, sep, body])

    md = f"""# DirBucket — Watermark Cost / Strength Ablation

## Setup

- **Source docs**: {n_docs} documents sampled from the FARAD manifest (stratified by provider)
- **Sentences**: {n_sents} total sentences analysed
- **Candidate generation**: K={K_GEN} candidates per round, ρ={RHO_GEN} rounds (paper setting), one run
- **Simulation**: all 9 (K, ρ) combinations derived from the same saved candidates
- **Detection Z anchor**: {z_anch:.2f} (from sensitivity sweep: W=28, τ_sim=0.82, T_min=25, 18 target-seed pairs)
- **Cost note**: K candidates are generated in a SINGLE API call per round (batch prompt). Changing K does NOT change the number of API calls; only ρ (the number of rounds) determines the call count.

## Results Table

*(Paper setting in **bold**. Cost is relative to paper setting (K=24, ρ=6). Z is extrapolated from the full-corpus anchor.)*

{table_md}

---

*Generated by `experiments/watermark_cost_ablation.py`*
"""
    md_path = out_dir / "ABLATION_NOTE.md"
    md_path.write_text(md, encoding="utf-8")
    print(f"[WRITE] {md_path}")


def main() -> None:
    args = _parse_args()
    asyncio.run(_run_async(args))


if __name__ == "__main__":
    main()
