#!/usr/bin/env python3
"""Post-answer laundering attack (paper Table tab:answer-variation, Sec. 5.5).

Loads the primary rag_multi trial logs, applies an LLM post-processing
transform to every logged answer, re-runs the aligned DirBucket detector on
the transformed text and reports pooled target Z / BH detection / FPR.

Conditions: ``original`` (no transform), ``neutral``, ``neutral2`` (neutral
applied to the cached neutral answer), ``aggressive``, ``compress50``,
``compress25``. Prompts are in ``ATTACK_PROMPTS`` (paper App. A.4.3).

Settings
--------
* Transforms: ``--openai_model gpt-4o-mini`` (default) at temperature 0.3.
* Seeds 1-3 and seeds 4-5 are run into two separate out dirs
  (``results/laundering_attack`` and ``results/laundering_attack_seeds45``);
  the 30-audit table pools both (see ``paper_appendix_tables.py answer-variation``).
* Transforms are cached under ``<cache_dir>/<condition>/<key>.txt`` (default
  ``<out_dir>/answer_cache``).
  With the default ``--cache_key question`` the key is sha256(question)[:16],
  so trials that share a question share one transform (214/1440 trials for
  seeds 1-3); ``--cache_key trial`` keys on question+answer instead.
* Reference sources are passed to ``detect_windows_multi_aligned`` one tuple
  per document. That method keeps only the last document per provider, so this
  script audits against a smaller reference set than the primary harness (which
  merges all selected documents of a provider) and reports a lower Z scale
  (median 8.42 vs 11.85 for the primary audit on seeds 1-3).
  ``--merge_sources_per_provider`` reproduces the primary-harness reference set.

Re-running detection from a complete answer cache needs no API key.
Outputs: ``<out_dir>/laundering_results.json`` (merged per condition and pair
across invocations) and ``<out_dir>/laundering_summary.md``.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as scipy_norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Locate watermark_dirbucket relative to this file
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from watermark_dirbucket import (
    MultiDirBucketDetector,
    _windows_within_sentence,
    split_sentences,
)


# ---------------------------------------------------------------------------
# Attack prompts
# ---------------------------------------------------------------------------
ATTACK_PROMPTS: Dict[str, str] = {
    "neutral": (
        "Rewrite the following text in different wording while preserving every "
        "fact and detail. Do not add or remove information. "
        "Return only the rewritten text with no preamble.\n\n{answer}"
    ),
    "aggressive": (
        "Rewrite the following text completely from scratch. Remove all "
        "distinctive phrasing and sentence structure. Use plain, generic wording "
        "throughout. Preserve all facts and details exactly. "
        "Return only the rewritten text with no preamble.\n\n{answer}"
    ),
    "compress50": (
        "Summarise the following text to approximately half its length. "
        "Preserve the key facts and details. "
        "Return only the summary with no preamble.\n\n{answer}"
    ),
    "compress25": (
        "Summarise the following text very concisely to about one quarter of its "
        "original length. Keep only the most essential facts. "
        "Return only the summary with no preamble.\n\n{answer}"
    ),
    # Two-pass: neutral applied to already-neutral answers (iterative laundering)
    # The {answer} field is filled with the neutral-paraphrased answer at run time.
    "neutral2": (
        "Rewrite the following text in different wording while preserving every "
        "fact and detail. Do not add or remove information. "
        "Return only the rewritten text with no preamble.\n\n{answer}"
    ),
}


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def cache_key(question: str, answer: str, mode: str = "question") -> str:
    """Answer-cache file stem. ``question`` is the (lossy) paper behaviour."""
    if mode == "question":
        return _hash(question)
    if mode == "trial":
        return _hash(question + "\x1f" + answer)
    raise ValueError(f"unknown cache key mode: {mode}")


def merge_sources_per_provider(sources: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """Join all documents of a provider into one source, as the primary harness does."""
    by_pid: Dict[int, List[str]] = {}
    for pid, txt in sources:
        by_pid.setdefault(int(pid), []).append(txt)
    return [(pid, "\n\n".join(by_pid[pid])) for pid in sorted(by_pid)]


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus(work_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows = load_jsonl(work_dir / "corpus.jsonl")
    meta = json.loads((work_dir / "provider_keys.json").read_text())
    return rows, list(meta["keys_hex"])


# ---------------------------------------------------------------------------
# Reference-set reconstruction (mirrors run_dirbucket_quick_experiments.py)
# ---------------------------------------------------------------------------
def build_reference_sources(
    question: str,
    rows: List[Dict[str, Any]],
    corpus_emb: np.ndarray,        # (N, 384) - all-MiniLM-L6-v2 embeddings
    retrieval_enc: SentenceTransformer,
    *,
    sim_threshold: float = 0.48,
    max_docs: int = 120,
    source_field: str = "text_wm",
) -> List[Tuple[int, str]]:
    """
    Reconstruct the global-corpus reference set the auditor would build
    for this question.  Returns list of (provider_id, source_text).
    Matches global_corpus/query mode used in the main experiments.
    """
    q_vec = retrieval_enc.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    sims = corpus_emb @ q_vec          # (N,)
    order = np.argsort(-sims)

    sources: List[Tuple[int, str]] = []
    for idx in order:
        if float(sims[int(idx)]) < sim_threshold:
            break
        if len(sources) >= max_docs:
            break
        row = rows[int(idx)]
        txt = str(row.get(source_field) or "").strip()
        if txt:
            sources.append((int(row["provider_id"]), txt))
    return sources


# ---------------------------------------------------------------------------
# Per-trial detection
# ---------------------------------------------------------------------------
def detect_trial(
    answer: str,
    sources: List[Tuple[int, str]],
    det: MultiDirBucketDetector,
    *,
    sim_threshold: float = 0.82,
    tie_margin: float = 0.05,
    win_words: int = 28,
    stride_words: int = 14,
    n_providers: int = 6,
) -> Dict[int, Dict[str, float]]:
    """
    Run the aligned DirBucket detector.
    Returns {pid: {"T": int, "sum_proj": float, "z": float}} for all providers.
    Empty providers get T=0, sum_proj=0.
    """
    if not sources:
        return {
            pid: {"T": 0, "sum_proj": 0.0, "z": 0.0}
            for pid in range(n_providers)
        }

    res_by_pid, _ = det.detect_windows_multi_aligned(
        answer,
        sources=sources,
        sim_threshold=sim_threshold,
        competitive=True,
        tie_margin=tie_margin,
        win_words=win_words,
        stride_words=stride_words,
    )

    out: Dict[int, Dict[str, float]] = {}
    for pid in range(n_providers):
        if pid in res_by_pid:
            ds = res_by_pid[pid]
            T = int(ds.T)
            sp = float(ds.mean_proj) * T  # recover sum_proj
            z = float(ds.z)
        else:
            T, sp, z = 0, 0.0, 0.0
        out[pid] = {"T": T, "sum_proj": sp, "z": z}
    return out


# ---------------------------------------------------------------------------
# Pooling and BH correction (mirrors main experiment logic)
# ---------------------------------------------------------------------------
def _norm_sf(z: float) -> float:
    try:
        return float(scipy_norm.sf(z))
    except Exception:
        from math import erfc
        return 0.5 * erfc(z / math.sqrt(2.0))


def _bh_adjust(pvals: List[float], alpha: float = 0.05) -> List[bool]:
    m = len(pvals)
    if m == 0:
        return []
    idx = np.argsort(pvals)
    p_sorted = np.array(pvals, dtype=float)[idx]
    sig_sorted = p_sorted <= (np.arange(1, m + 1) / m) * alpha
    k = int(np.max(np.where(sig_sorted)[0]) + 1) if np.any(sig_sorted) else 0
    sig = np.zeros(m, dtype=bool)
    if k > 0:
        sig[idx[:k]] = True
    return [bool(x) for x in sig.tolist()]


def pooled_z_from_sums(
    pooled_T: float,
    pooled_sum_proj: float,
    D: int = 768,
) -> float:
    T = int(pooled_T)
    if T <= 0:
        return 0.0
    mean_proj = pooled_sum_proj / T
    return mean_proj * math.sqrt(D * T)


def analyse_condition(
    per_trial_results: List[Dict[str, Any]],
    target_provider: int,
    n_providers: int = 6,
    T_min: int = 25,
    alpha: float = 0.05,
    D: int = 768,
) -> Dict[str, Any]:
    """
    Given per-trial detection results for one (target, seed) pair,
    compute the same summary statistics as the main paper.

    Returns:
      final_target_z       pooled Z for target provider at end of audit
      final_nontarget_z    max pooled Z for non-target providers at end
      queries_to_bh        number of trials until first BH-significant
                           detection for target (None if never detected)
      final_support        pooled T for target provider at end
      bh_detected          whether target was BH-detected at end
    """
    # Running sums
    pooled_T = np.zeros(n_providers, dtype=np.float64)
    pooled_S = np.zeros(n_providers, dtype=np.float64)

    queries_to_bh: Optional[int] = None

    for q_idx, trial in enumerate(per_trial_results, start=1):
        detect = trial["detect"]        # {pid_str -> {T, sum_proj, z}}
        for pid_str, stats in detect.items():
            pid = int(pid_str)
            pooled_T[pid] += float(stats["T"])
            pooled_S[pid] += float(stats["sum_proj"])

        # Compute current pooled Z for all providers
        z_raw = [pooled_z_from_sums(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]

        # Apply T_min gate + BH
        pvals_gated = []
        eligible = []
        z_gated = []
        for p in range(n_providers):
            ok = int(pooled_T[p]) >= T_min
            eligible.append(ok)
            if ok:
                z_gated.append(z_raw[p])
                pvals_gated.append(_norm_sf(z_raw[p]))
            else:
                z_gated.append(0.0)
                pvals_gated.append(1.0)

        bh_sig = _bh_adjust(pvals_gated, alpha=alpha)

        if (
            queries_to_bh is None
            and eligible[target_provider]
            and bh_sig[target_provider]
        ):
            queries_to_bh = q_idx

    # Final state
    final_z = [pooled_z_from_sums(pooled_T[p], pooled_S[p], D) for p in range(n_providers)]
    pvals_final = []
    z_final_gated = []
    for p in range(n_providers):
        ok = int(pooled_T[p]) >= T_min
        if ok:
            z_final_gated.append(final_z[p])
            pvals_final.append(_norm_sf(final_z[p]))
        else:
            z_final_gated.append(0.0)
            pvals_final.append(1.0)

    bh_final = _bh_adjust(pvals_final, alpha=alpha)

    non_target_pids = [p for p in range(n_providers) if p != target_provider]
    max_nontarget_z = max((z_final_gated[p] for p in non_target_pids), default=0.0)

    return {
        "final_target_z": float(z_final_gated[target_provider]),
        "final_nontarget_z": float(max_nontarget_z),
        "queries_to_bh": queries_to_bh,
        "final_support": int(pooled_T[target_provider]),
        "bh_detected": bool(bh_final[target_provider]),
        "eligible_at_end": bool(int(pooled_T[target_provider]) >= T_min),
    }


# ---------------------------------------------------------------------------
# Async GPT transformation
# ---------------------------------------------------------------------------
async def _transform_one(
    client: Any,
    answer: str,
    condition: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.3,
    semaphore: Optional[asyncio.Semaphore] = None,
) -> str:
    prompt = ATTACK_PROMPTS[condition].format(answer=answer)
    async with (semaphore or asyncio.Semaphore(1)):
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1024,
                    temperature=temperature,
                )
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if attempt == 2:
                    print(f"[WARN] GPT call failed after 3 attempts: {e}")
                    return answer  # fallback: return original
                await asyncio.sleep(2.0 * (attempt + 1))
    return answer


async def transform_answers_async(
    answers: List[str],
    question_hashes: List[str],
    condition: str,
    cache_dir: Path,
    model: str = "gpt-4o-mini",
    concurrency: int = 20,
) -> List[str]:
    """Transform a list of answers for a given attack condition, with caching."""
    sem = asyncio.Semaphore(concurrency)
    cond_dir = cache_dir / condition
    cond_dir.mkdir(parents=True, exist_ok=True)

    results: List[str] = [""] * len(answers)
    n_missing = len({h for h in question_hashes if not (cond_dir / f"{h}.txt").exists()})
    client = None
    if n_missing:
        if not os.environ.get("OPENAI_API_KEY"):
            raise SystemExit(
                f"[FATAL] {n_missing} '{condition}' transforms are not cached in {cond_dir} "
                "and OPENAI_API_KEY is not set."
            )
        from openai import AsyncOpenAI
        client = AsyncOpenAI()
        print(f"[INFO] {n_missing} uncached '{condition}' transforms will be generated with {model}")

    async def _process(i: int) -> None:
        cache_file = cond_dir / f"{question_hashes[i]}.txt"
        if cache_file.exists():
            results[i] = cache_file.read_text(encoding="utf-8")
            return
        transformed = await _transform_one(
            client, answers[i], condition, model=model, semaphore=sem
        )
        cache_file.write_text(transformed, encoding="utf-8")
        results[i] = transformed

    tasks = [_process(i) for i in range(len(answers))]
    for f in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"Transforming [{condition}]",
    ):
        await f

    return results


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument(
        "--trial_logs_dir",
        type=Path,
        default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"),
    )
    ap.add_argument("--seeds", type=str, default="1,2,3", help="Comma-separated seeds.")
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5")
    ap.add_argument(
        "--conditions",
        type=str,
        default="original,neutral,aggressive,compress50,compress25",
    )
    ap.add_argument("--out_dir", type=Path, default=Path("results/laundering_attack"))
    ap.add_argument("--cache_dir", type=Path, default=None,
                    help="Answer cache (default: <out_dir>/answer_cache). Point it at a "
                         "downloaded cache to re-score without API calls.")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--concurrency", type=int, default=20)
    ap.add_argument("--sim_threshold", type=float, default=0.82)
    ap.add_argument("--tie_margin", type=float, default=0.05)
    ap.add_argument("--T_min", type=int, default=25)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--global_sim_threshold", type=float, default=0.48)
    ap.add_argument("--global_max_docs", type=int, default=120)
    ap.add_argument("--win_words", type=int, default=28)
    ap.add_argument("--stride_words", type=int, default=14)
    ap.add_argument("--n_providers", type=int, default=6)
    ap.add_argument("--cache_key", choices=["question", "trial"], default="question",
                    help="Answer-cache key: 'question' (paper, lossy) or 'trial' (question+answer).")
    ap.add_argument("--merge_sources_per_provider", action="store_true",
                    help="Merge reference docs per provider like the primary harness "
                         "(not used for the paper table).")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = args.cache_dir or args.out_dir / "answer_cache"

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]

    print(f"[INFO] Seeds: {seeds}, Targets: {targets}, Conditions: {conditions}")
    print(f"[INFO] Work dir: {args.work_dir}")
    print(f"[INFO] Trial logs: {args.trial_logs_dir}")

    # ------------------------------------------------------------------
    # Load corpus + keys
    # ------------------------------------------------------------------
    print("[LOAD] corpus...")
    corpus_rows, keys_hex = load_corpus(args.work_dir)
    corpus_emb = np.load(args.work_dir / "emb_text_wm.npz")["X"].astype(np.float32)
    assert len(corpus_rows) == corpus_emb.shape[0], "corpus / embedding size mismatch"

    # ------------------------------------------------------------------
    # Load retrieval encoder (for reference-set reconstruction)
    # ------------------------------------------------------------------
    print("[LOAD] retrieval encoder (all-MiniLM-L6-v2)...")
    retrieval_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # Load detector
    # ------------------------------------------------------------------
    print("[LOAD] DirBucket detector (all-mpnet-base-v2)...")
    det = MultiDirBucketDetector(
        provider_keys_by_pid={i: keys_hex[i] for i in range(len(keys_hex))},
        bucket_bits=96,
        bucket_seed=2025,
        embed_model="sentence-transformers/all-mpnet-base-v2",
        direction_scheme="naive",
    )
    n_prov = args.n_providers
    D = det.embedder.st.get_sentence_embedding_dimension()

    # ------------------------------------------------------------------
    # Load trial logs: collect all trials indexed by (target, seed)
    # ------------------------------------------------------------------
    print("[LOAD] trial logs...")
    # trials_by_ts[(target, seed)] = list of trial dicts in order
    trials_by_ts: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for target in targets:
        for seed in seeds:
            path = args.trial_logs_dir / f"rag_multi_t{target}_seed{seed:02d}.trials.jsonl"
            if not path.exists():
                print(f"[WARN] Missing: {path}")
                continue
            rows = load_jsonl(path)
            # Sort by trial_idx to ensure correct order
            rows.sort(key=lambda r: int(r.get("trial_idx", 0)))
            trials_by_ts[(target, seed)] = rows
            print(f"  t={target} seed={seed}: {len(rows)} trials loaded")

    if not trials_by_ts:
        raise SystemExit("No trial logs found. Check --trial_logs_dir and --seeds/--targets.")

    # ------------------------------------------------------------------
    # Pre-compute reference sources for each unique question
    # ------------------------------------------------------------------
    print("[PREP] computing reference sources for each unique question...")
    all_questions = set()
    for rows in trials_by_ts.values():
        for r in rows:
            all_questions.add(r["question"])

    sources_cache: Dict[str, List[Tuple[int, str]]] = {}
    for q in tqdm(sorted(all_questions), desc="Reference sets"):
        sources_cache[q] = build_reference_sources(
            q, corpus_rows, corpus_emb, retrieval_enc,
            sim_threshold=args.global_sim_threshold,
            max_docs=args.global_max_docs,
        )
        if args.merge_sources_per_provider:
            sources_cache[q] = merge_sources_per_provider(sources_cache[q])

    print(f"  {len(sources_cache)} unique questions, reference set sizes: "
          f"min={min(len(v) for v in sources_cache.values())}, "
          f"max={max(len(v) for v in sources_cache.values())}")

    # ------------------------------------------------------------------
    # For "original" condition: re-run detector on untransformed answers
    # For other conditions: transform then detect
    # ------------------------------------------------------------------

    # Structure: results_by_condition[condition][(target, seed)] = analyse_condition(...)
    results_by_condition: Dict[str, Dict[Tuple[int, int], Dict[str, Any]]] = {}

    for condition in conditions:
        print(f"\n{'='*60}")
        print(f"[CONDITION] {condition}")
        print(f"{'='*60}")

        # ------------------------------------------------------------------
        # Step 1: collect (answer, question, question_hash) for all trials
        # For neutral2 (two-pass), use the cached neutral answers as source.
        # ------------------------------------------------------------------
        all_answers: List[str] = []
        all_q_hashes: List[str] = []
        trial_index: List[Tuple[Tuple[int, int], int]] = []  # (ts_key, row_idx)

        neutral_cache_dir = cache_dir / "neutral"
        for ts_key, rows in sorted(trials_by_ts.items()):
            for row_idx, r in enumerate(rows):
                q_hash = cache_key(r["question"], r["answer"], args.cache_key)
                if condition == "neutral2":
                    # Use the neutral-paraphrased answer as input for second pass
                    neutral_cache_file = neutral_cache_dir / f"{q_hash}.txt"
                    if neutral_cache_file.exists():
                        all_answers.append(neutral_cache_file.read_text(encoding="utf-8"))
                    else:
                        # Neutral cache missing — fall back to original (with a warning)
                        print(f"[WARN] neutral cache missing for {q_hash}, using original")
                        all_answers.append(r["answer"])
                else:
                    all_answers.append(r["answer"])
                all_q_hashes.append(q_hash)
                trial_index.append((ts_key, row_idx))

        # ------------------------------------------------------------------
        # Step 2: transform answers (or keep original)
        # neutral2 uses the "neutral" prompt but on already-neutral answers.
        # ------------------------------------------------------------------
        if condition == "original":
            transformed_answers = all_answers
        else:
            # neutral2: uses same prompt as neutral but caches separately under "neutral2/".
            # Input is already the neutral-paraphrased answers from Step 1.
            transformed_answers = asyncio.run(
                transform_answers_async(
                    all_answers,
                    all_q_hashes,
                    condition,           # "neutral2" → uses ATTACK_PROMPTS["neutral2"]
                    cache_dir,           # caches to cache_dir/neutral2/ automatically
                    model=args.openai_model,
                    concurrency=args.concurrency,
                )
            )

        # ------------------------------------------------------------------
        # Step 3: run detector on each (transformed) answer
        # ------------------------------------------------------------------
        print(f"[DETECT] Running detector for {len(transformed_answers)} answers...")

        # per_trial_detect[(ts_key, row_idx)] = {pid_str -> {T, sum_proj, z}}
        per_trial_detect: Dict[Tuple[Tuple[int, int], int], Dict[str, Any]] = {}

        for flat_i, (answer_text, (ts_key, row_idx)) in enumerate(
            tqdm(
                zip(transformed_answers, trial_index),
                total=len(transformed_answers),
                desc=f"Detect [{condition}]",
            )
        ):
            q = trials_by_ts[ts_key][row_idx]["question"]
            sources = sources_cache[q]
            detect_res = detect_trial(
                answer_text,
                sources,
                det,
                sim_threshold=args.sim_threshold,
                tie_margin=args.tie_margin,
                win_words=args.win_words,
                stride_words=args.stride_words,
                n_providers=n_prov,
            )
            # Serialise keys as strings for JSON compatibility
            per_trial_detect[(ts_key, row_idx)] = {
                str(pid): v for pid, v in detect_res.items()
            }

        # ------------------------------------------------------------------
        # Step 4: pool and analyse per (target, seed)
        # ------------------------------------------------------------------
        print(f"[POOL] Pooling and analysing...")
        cond_results: Dict[Tuple[int, int], Dict[str, Any]] = {}

        for ts_key, rows in sorted(trials_by_ts.items()):
            target, seed = ts_key
            per_trial = [
                {"detect": per_trial_detect[(ts_key, row_idx)]}
                for row_idx in range(len(rows))
            ]
            analysis = analyse_condition(
                per_trial,
                target_provider=target,
                n_providers=n_prov,
                T_min=args.T_min,
                alpha=args.alpha,
                D=D,
            )
            cond_results[ts_key] = analysis

        results_by_condition[condition] = cond_results

        # Quick per-condition summary
        target_zs = [v["final_target_z"] for v in cond_results.values()]
        nontarget_zs = [v["final_nontarget_z"] for v in cond_results.values()]
        supports = [v["final_support"] for v in cond_results.values()]
        det_rates = [v["bh_detected"] for v in cond_results.values()]
        q2d = [v["queries_to_bh"] for v in cond_results.values() if v["queries_to_bh"] is not None]

        print(f"  Target Z:        mean={np.mean(target_zs):.2f}, median={np.median(target_zs):.2f}")
        print(f"  Non-target Z:    mean={np.mean(nontarget_zs):.2f}, max={np.max(nontarget_zs):.2f}")
        print(f"  BH det rate:     {sum(det_rates)}/{len(det_rates)} = {np.mean(det_rates):.2f}")
        print(f"  Median support:  {np.median(supports):.0f}")
        print(f"  Q to BH (det):   median={np.median(q2d):.0f}" if q2d else "  Q to BH: no detections")

    # ------------------------------------------------------------------
    # Save full results JSON
    # ------------------------------------------------------------------
    # Merge into an existing results file so per-condition / per-seed invocations
    # accumulate instead of overwriting each other.
    out_json = args.out_dir / "laundering_results.json"
    serialisable: Dict[str, Any] = (
        json.loads(out_json.read_text(encoding="utf-8")) if out_json.exists() else {}
    )
    for cond, cond_res in results_by_condition.items():
        serialisable.setdefault(cond, {}).update(
            {f"t{t}_s{s}": v for (t, s), v in cond_res.items()}
        )

    out_json.write_text(json.dumps(serialisable, indent=2), encoding="utf-8")
    print(f"\n[WRITE] {out_json}")

    # ------------------------------------------------------------------
    # Build markdown summary
    # ------------------------------------------------------------------
    cond_label = {
        "original": "Original (ownwords)",
        "neutral": "Neutral paraphrase",
        "neutral2": "Neutral x2",
        "aggressive": "Aggressive paraphrase",
        "compress50": "Compression 50%",
        "compress25": "Compression 25%",
    }

    rows_md: List[Dict[str, Any]] = []
    for cond in conditions:
        cond_res = results_by_condition.get(cond, {})
        if not cond_res:
            continue
        target_zs = [v["final_target_z"] for v in cond_res.values()]
        nontarget_zs = [v["final_nontarget_z"] for v in cond_res.values()]
        supports = [v["final_support"] for v in cond_res.values()]
        det_rates = [v["bh_detected"] for v in cond_res.values()]
        q2d = [v["queries_to_bh"] for v in cond_res.values() if v["queries_to_bh"] is not None]

        rows_md.append({
            "Condition": cond_label.get(cond, cond),
            "Target pooled Z (median)": f"{np.median(target_zs):.2f}",
            "BH detection rate": f"{np.mean(det_rates):.2f} ({sum(det_rates)}/{len(det_rates)})",
            "Median queries to BH": f"{int(np.median(q2d))}" if q2d else "—",
            "Max non-target Z": f"{np.max(nontarget_zs):.2f}",
            "Median support": f"{int(np.median(supports))}",
        })

    # Markdown table
    if rows_md:
        cols = list(rows_md[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        body   = "\n".join(
            "| " + " | ".join(str(r[c]) for c in cols) + " |"
            for r in rows_md
        )
        table_md = "\n".join([header, sep, body])
    else:
        table_md = "(no results)"

    # Count trials and seeds used
    n_total_trials = sum(len(v) for v in trials_by_ts.values())
    n_run_seeds = len(seeds)
    n_run_targets = len(targets)

    md = f"""# DirBucket — Post-Answer Laundering Attack Results

## Setup

- **Seeds**: {seeds} ({n_run_seeds} seed(s))
- **Targets**: {targets} ({n_run_targets} providers)
- **Total (target, seed) pairs**: {len(trials_by_ts)}
- **Trials per pair**: 80
- **Total answers evaluated per condition**: {n_total_trials}
- **Attack model**: {args.openai_model} (temperature 0.3)
- **Corpus**: {args.work_dir}
- **Detector params**: sim_threshold={args.sim_threshold}, tie_margin={args.tie_margin}, T_min={args.T_min}, alpha={args.alpha}

## Results

{table_md}

*Reported over all (target, seed) pairs. "Median queries to BH" is median number of
audited answers before first BH-significant detection, computed over pairs where detection
occurred. "—" means no pairs reached BH detection.*

## Condition Definitions

| Condition | GPT-4o-mini prompt (paraphrase of) |
|-----------|-------------------------------------|
| Neutral paraphrase | "Rewrite in different wording while preserving every fact." |
| Aggressive paraphrase | "Rewrite completely from scratch; remove distinctive phrasing; preserve all facts." |
| Compression 50% | "Summarise to ~50% length; keep key facts." |
| Compression 25% | "Summarise to ~25% length; keep only essentials." |

## Interpretation

See discussion below.

---
*Generated by `experiments/post_answer_laundering_attack.py`*
"""

    out_md = args.out_dir / "laundering_summary.md"
    out_md.write_text(md, encoding="utf-8")
    print(f"[WRITE] {out_md}")

    # Also print the table to stdout
    print("\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)
    print(table_md)
    print("=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
