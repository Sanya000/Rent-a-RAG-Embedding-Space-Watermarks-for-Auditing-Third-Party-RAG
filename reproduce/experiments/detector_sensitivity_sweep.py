#!/usr/bin/env python3
"""Detector hyperparameter sensitivity sweep (paper App. app:sensitivity, Table tab:sensitivity).

Re-audits the stored primary rag_multi answers (no new generation, no API) over
a grid of auditor parameters: window length W (stride W//2), competitive
alignment threshold tau_sim, margin tau_margin and support gate T_min.

Phase 1 (once per W) embeds every reference window of the corpus and every
answer window and stores per-window best cosine and directional score per
provider. Phase 2 applies the thresholds, pools evidence, gates by T_min and
runs BH; it takes seconds per combination.

Grids (``--grid``):
  base      W {20,28,40} x tau_sim {.78..86} x tau_margin {.02,.05,.08,.10}
            x T_min {10,25,50} = 180 combos -> sensitivity_results.json
  extended  W {28} x 12 tau_sim in [0.50, 0.92] x 9 tau_margin in [0, 0.30]
            x T_min {10,25,50} = 324 combos -> sensitivity_extended.json
            (the "324 parameter combinations" of the paper)

Reference documents are concatenated per provider, exactly as the primary
harness does, so the operating point (W=28, tau_sim=0.82, tau_margin=0.05,
T_min=25) reproduces the primary audit's median target Z on seeds 1-3 (11.85).

Paper command (seeds 1-3, all targets):
  python experiments/detector_sensitivity_sweep.py --grid base,extended \
      --seeds 1,2,3 --targets 0,1,2,3,4,5 --out_dir results/sensitivity_sweep
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import norm as scipy_norm
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from watermark_dirbucket import (
    DirectionManager,
    MultiDirBucketDetector,
    SimHashCoder,
    _windows_within_sentence,
    split_sentences,
)

# ---------------------------------------------------------------------------
# Fixed sweep grid
# ---------------------------------------------------------------------------
TAU_SIM_VALUES    = [0.78, 0.80, 0.82, 0.84, 0.86]
TAU_MARGIN_VALUES = [0.02, 0.05, 0.08, 0.10]
T_MIN_VALUES      = [10, 25, 50]
W_VALUES          = [20, 28, 40]

GRIDS: Dict[str, Dict[str, Any]] = {
    "base": dict(
        W=W_VALUES, tau_sim=TAU_SIM_VALUES, tau_margin=TAU_MARGIN_VALUES,
        T_min=T_MIN_VALUES, out="sensitivity_results.json",
    ),
    "extended": dict(
        W=[28],
        tau_sim=[0.50, 0.60, 0.70, 0.76, 0.78, 0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92],
        tau_margin=[0.00, 0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30],
        T_min=[10, 25, 50], out="sensitivity_extended.json",
    ),
}

# Paper defaults
DEFAULT_TAU_SIM    = 0.82
DEFAULT_TAU_MARGIN = 0.05
DEFAULT_T_MIN      = 25
DEFAULT_W          = 28
DEFAULT_ALPHA      = 0.05
N_PROVIDERS        = 6
D_AUDIT            = 768   # all-mpnet-base-v2 embedding dim


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_corpus_and_keys(work_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    rows = load_jsonl(work_dir / "corpus.jsonl")
    meta = json.loads((work_dir / "provider_keys.json").read_text())
    return rows, list(meta["keys_hex"])


# ---------------------------------------------------------------------------
# Reference-set selection (returns doc IDs, not texts)
# ---------------------------------------------------------------------------
def select_reference_doc_ids(
    question: str,
    corpus_rows: List[Dict[str, Any]],
    corpus_emb: np.ndarray,          # (N, 384) from all-MiniLM-L6-v2
    retrieval_enc: SentenceTransformer,
    *,
    sim_threshold: float = 0.48,
    max_docs: int = 120,
) -> List[Tuple[int, str]]:
    """Return list of (provider_id, doc_id) for the query-selected reference set."""
    q_vec = retrieval_enc.encode(
        [question], normalize_embeddings=True, convert_to_numpy=True
    )[0].astype(np.float32)
    sims = corpus_emb @ q_vec
    order = np.argsort(-sims)
    result: List[Tuple[int, str]] = []
    for idx in order:
        if float(sims[int(idx)]) < sim_threshold:
            break
        if len(result) >= max_docs:
            break
        row = corpus_rows[int(idx)]
        result.append((int(row["provider_id"]), str(row["id"])))
    return result


# ---------------------------------------------------------------------------
# Stats helpers (mirror run_dirbucket_quick_experiments.py)
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


def _pooled_z(pooled_T: float, pooled_S: float, D: int = D_AUDIT) -> float:
    T = int(pooled_T)
    if T <= 0:
        return 0.0
    return (pooled_S / T) * math.sqrt(D * T)


# ---------------------------------------------------------------------------
# Phase 1: precompute per-window alignment stats for a given W
# ---------------------------------------------------------------------------
def precompute_for_W(
    W: int,
    trials_by_ts: Dict[Tuple[int, int], List[Dict[str, Any]]],
    corpus_rows: List[Dict[str, Any]],
    ref_doc_ids_by_question: Dict[str, List[Tuple[int, str]]],
    audit_enc: SentenceTransformer,
    simhash: SimHashCoder,
    dir_managers: Dict[int, DirectionManager],  # pid -> DirectionManager
    *,
    source_field: str = "text_wm",
    batch_size: int = 512,
) -> Dict[Tuple[int, int], List[Dict[str, Any]]]:
    """
    For each (target, seed) pair and each of their 80 trials, compute:
      best_cos   : np.ndarray (n_wins, N_PROVIDERS) — NaN for providers with no ref docs
      dir_score  : np.ndarray (n_wins, N_PROVIDERS) — NaN for providers with no ref docs
    where n_wins = number of answer windows for this trial at the given W.

    best_cos[w, pid]  = max cosine of answer window w vs all reference windows of pid
    dir_score[w, pid] = directional score of answer window w using the SimHash bucket
                        of its best-matching reference window for pid

    Returns a dict mapping (target, seed) -> list of trial dicts with these arrays.
    """
    stride = W // 2
    doc_id_to_row: Dict[str, Dict[str, Any]] = {str(r["id"]): r for r in corpus_rows}

    # ------------------------------------------------------------------
    # Step 1: Pre-embed ALL corpus doc reference windows (cached by doc_id)
    # ------------------------------------------------------------------
    print(f"  [W={W}] Embedding corpus reference windows (stride={stride})...", flush=True)
    t0 = time.perf_counter()

    # Collect all windows across all docs (for batch embedding)
    all_ref_texts: List[str] = []
    doc_win_offsets: Dict[str, Tuple[int, int]] = {}  # doc_id -> (start, end) in all_ref_texts

    for row in corpus_rows:
        doc_id = str(row["id"])
        text = str(row.get(source_field) or "").strip()
        wins: List[str] = []
        for sent in split_sentences(text):
            wins.extend(_windows_within_sentence(sent, W, stride))
        if wins:
            start = len(all_ref_texts)
            all_ref_texts.extend(wins)
            doc_win_offsets[doc_id] = (start, start + len(wins))
        else:
            doc_win_offsets[doc_id] = (0, 0)  # empty

    # Batch embed
    if all_ref_texts:
        all_ref_embs = audit_enc.encode(
            all_ref_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
    else:
        all_ref_embs = np.zeros((0, D_AUDIT), dtype=np.float32)

    # Pre-compute bucket codes for all reference windows
    all_ref_buckets: List[bytes] = [simhash.code(all_ref_embs[i]) for i in range(len(all_ref_texts))]

    print(
        f"  [W={W}] {len(all_ref_texts)} ref windows embedded in {time.perf_counter()-t0:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 2: Collect all answer windows across all trials (batch embed)
    # ------------------------------------------------------------------
    print(f"  [W={W}] Embedding answer windows...", flush=True)
    t1 = time.perf_counter()

    # Flat list of answer windows with back-pointers
    flat_ans_texts: List[str] = []
    # trial_win_map[(ts_key, trial_idx)] = (start, end) in flat_ans_texts
    trial_win_map: Dict[Tuple[Tuple[int, int], int], Tuple[int, int]] = {}

    for ts_key, trials in sorted(trials_by_ts.items()):
        for t_idx, trial in enumerate(trials):
            wins: List[str] = []
            for sent in split_sentences(trial["answer"]):
                wins.extend(_windows_within_sentence(sent, W, stride))
            start = len(flat_ans_texts)
            flat_ans_texts.extend(wins)
            trial_win_map[(ts_key, t_idx)] = (start, start + len(wins))

    if flat_ans_texts:
        flat_ans_embs = audit_enc.encode(
            flat_ans_texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=batch_size,
            show_progress_bar=False,
        ).astype(np.float32)
    else:
        flat_ans_embs = np.zeros((0, D_AUDIT), dtype=np.float32)

    print(
        f"  [W={W}] {len(flat_ans_texts)} ans windows embedded in {time.perf_counter()-t1:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Step 3: For each trial, compute per-window alignment stats
    # ------------------------------------------------------------------
    print(f"  [W={W}] Computing per-window stats...", flush=True)
    t2 = time.perf_counter()

    result: Dict[Tuple[int, int], List[Dict[str, Any]]] = {
        ts_key: [] for ts_key in trials_by_ts.keys()
    }

    for ts_key, trials in sorted(trials_by_ts.items()):
        for t_idx, trial in enumerate(trials):
            question = trial["question"]
            ref_doc_ids = ref_doc_ids_by_question[question]  # list of (pid, doc_id)

            # Build per-provider reference embedding arrays
            # pid -> (U_ref: ndarray (n, D), buckets: list[bytes])
            refs_by_pid: Dict[int, Tuple[np.ndarray, List[bytes]]] = {}
            for pid, doc_id in ref_doc_ids:
                s, e = doc_win_offsets.get(doc_id, (0, 0))
                if e <= s:
                    continue
                U_ref = all_ref_embs[s:e]
                bks = all_ref_buckets[s:e]
                if pid not in refs_by_pid:
                    refs_by_pid[pid] = (U_ref, bks)
                else:
                    # Multiple docs for same provider: concatenate
                    U_prev, b_prev = refs_by_pid[pid]
                    refs_by_pid[pid] = (
                        np.concatenate([U_prev, U_ref], axis=0),
                        b_prev + bks,
                    )

            # Get answer window embeddings
            a_start, a_end = trial_win_map[(ts_key, t_idx)]
            n_wins = a_end - a_start
            if n_wins == 0:
                result[ts_key].append(
                    {
                        "n_wins": 0,
                        "best_cos": np.full((0, N_PROVIDERS), np.nan, dtype=np.float32),
                        "dir_score": np.full((0, N_PROVIDERS), np.nan, dtype=np.float32),
                    }
                )
                continue

            U_ans = flat_ans_embs[a_start:a_end]  # (n_wins, D)

            # For each answer window and each provider: compute best cosine + dir score
            best_cos = np.full((n_wins, N_PROVIDERS), np.nan, dtype=np.float32)
            dir_score = np.full((n_wins, N_PROVIDERS), np.nan, dtype=np.float32)

            for pid, (U_ref, bks) in refs_by_pid.items():
                if U_ref.shape[0] == 0:
                    continue
                # cosine matrix: (n_wins, n_ref_wins)
                sims = U_ans @ U_ref.T  # (n_wins, n_ref)
                best_j = np.argmax(sims, axis=1)  # (n_wins,)
                best_c = sims[np.arange(n_wins), best_j]  # (n_wins,)

                # directional score using the bucket of the best-matching reference window
                dm = dir_managers[pid]
                dir_sc = np.array(
                    [
                        float(np.dot(U_ans[w], dm.direction(bks[int(best_j[w])])))
                        for w in range(n_wins)
                    ],
                    dtype=np.float32,
                )

                best_cos[:, pid] = best_c
                dir_score[:, pid] = dir_sc

            result[ts_key].append(
                {
                    "n_wins": n_wins,
                    "best_cos": best_cos,
                    "dir_score": dir_score,
                }
            )

    print(
        f"  [W={W}] Per-window stats done in {time.perf_counter()-t2:.1f}s",
        flush=True,
    )
    return result


# ---------------------------------------------------------------------------
# Phase 2: sweep a single (tau_sim, tau_margin, T_min) on precomputed stats
# ---------------------------------------------------------------------------
def sweep_params(
    precomp: Dict[Tuple[int, int], List[Dict[str, Any]]],
    tau_sim: float,
    tau_margin: float,
    T_min: int,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """
    Apply competitive assignment with (tau_sim, tau_margin) to precomputed
    per-window stats, pool across trials, run BH, return aggregate metrics.
    """
    all_target_z: List[float] = []
    all_nontarget_z_max: List[float] = []
    all_support: List[int] = []
    all_detected: List[bool] = []
    all_nontarget_fpr: List[bool] = []   # any non-target BH-sig at end?
    all_q2bh: List[Optional[int]] = []

    for (target, seed), trials in sorted(precomp.items()):
        pooled_T = np.zeros(N_PROVIDERS, dtype=np.float64)
        pooled_S = np.zeros(N_PROVIDERS, dtype=np.float64)
        q2bh: Optional[int] = None

        for q_idx, trial in enumerate(trials, start=1):
            n_wins = trial["n_wins"]
            if n_wins == 0:
                continue
            best_cos = trial["best_cos"]   # (n_wins, N_PROVIDERS)
            dir_sc   = trial["dir_score"]  # (n_wins, N_PROVIDERS)

            for w in range(n_wins):
                cos_w = best_cos[w]       # (N_PROVIDERS,) with NaN for missing
                valid = ~np.isnan(cos_w)  # which providers have ref docs
                if not np.any(valid):
                    continue

                # Winner: provider with max cosine (among those with ref docs)
                valid_cos = cos_w.copy()
                valid_cos[~valid] = -2.0  # below any real cosine
                pid_star = int(np.argmax(valid_cos))
                best = float(cos_w[pid_star])  # NaN check via valid mask

                if best < tau_sim:
                    continue

                # Second best among providers with ref docs
                n_valid = int(np.sum(valid))
                if n_valid >= 2:
                    top2 = np.partition(cos_w[valid], -2)[-2:]  # two largest
                    second = float(np.sort(top2)[0])            # second largest
                    if (best - second) < tau_margin:
                        continue

                # Assign this window to pid_star
                pooled_T[pid_star] += 1.0
                pooled_S[pid_star] += float(dir_sc[w, pid_star])

            # After this trial: check BH
            if q2bh is None:
                z_raw = [_pooled_z(pooled_T[p], pooled_S[p]) for p in range(N_PROVIDERS)]
                pvals_g: List[float] = []
                z_g: List[float] = []
                for p in range(N_PROVIDERS):
                    ok = int(pooled_T[p]) >= T_min
                    if ok:
                        z_g.append(z_raw[p])
                        pvals_g.append(_norm_sf(z_raw[p]))
                    else:
                        z_g.append(0.0)
                        pvals_g.append(1.0)
                bh = _bh_adjust(pvals_g, alpha)
                if int(pooled_T[target]) >= T_min and bh[target]:
                    q2bh = q_idx

        # Final state
        z_raw_final = [_pooled_z(pooled_T[p], pooled_S[p]) for p in range(N_PROVIDERS)]
        pvals_final: List[float] = []
        z_gated: List[float] = []
        for p in range(N_PROVIDERS):
            ok = int(pooled_T[p]) >= T_min
            if ok:
                z_gated.append(z_raw_final[p])
                pvals_final.append(_norm_sf(z_raw_final[p]))
            else:
                z_gated.append(0.0)
                pvals_final.append(1.0)
        bh_final = _bh_adjust(pvals_final, alpha)

        target_z = float(z_gated[target])
        non_pids = [p for p in range(N_PROVIDERS) if p != target]
        max_nt_z = float(max(z_gated[p] for p in non_pids))
        any_nt_det = any(bh_final[p] for p in non_pids)

        all_target_z.append(target_z)
        all_nontarget_z_max.append(max_nt_z)
        all_support.append(int(pooled_T[target]))
        all_detected.append(bool(bh_final[target]))
        all_nontarget_fpr.append(bool(any_nt_det))
        all_q2bh.append(q2bh)

    q2bh_det = [q for q in all_q2bh if q is not None]
    return {
        "target_z_median": float(np.median(all_target_z)),
        "target_z_mean":   float(np.mean(all_target_z)),
        "target_z_min":    float(np.min(all_target_z)),
        "detection_rate":  float(np.mean(all_detected)),
        "n_detected":      int(sum(all_detected)),
        "n_total":         int(len(all_detected)),
        "max_nontarget_z": float(np.max(all_nontarget_z_max)),
        "mean_nontarget_z":float(np.mean(all_nontarget_z_max)),
        "fpr":             float(np.mean(all_nontarget_fpr)),
        "n_fpr":           int(sum(all_nontarget_fpr)),
        "median_support":  float(np.median(all_support)),
        "min_support":     int(np.min(all_support)),
        "median_q2bh":     float(np.median(q2bh_det)) if q2bh_det else None,
        "max_q2bh":        int(max(q2bh_det)) if q2bh_det else None,
    }


# ---------------------------------------------------------------------------
# Markdown table builder
# ---------------------------------------------------------------------------
def _md_row(cells: List[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def build_marginal_table(
    results: Dict[str, Any],
    sweep_dim: str,          # "tau_sim" | "tau_margin" | "T_min" | "W"
    sweep_vals: List[Any],
    default_val: Any,
    fixed: Dict[str, Any],
) -> str:
    """Build a markdown table for a single swept parameter."""
    cols = ["Setting", "Target Z (median)", "BH detection rate", "Median queries to BH", "Max non-target Z / FPR", "Median support"]
    lines = [_md_row(cols), _md_row(["---"] * len(cols))]

    for val in sweep_vals:
        params = dict(fixed)
        params[sweep_dim] = val
        key = _result_key(**params)
        r = results.get(key)
        if r is None:
            continue

        marker = "**" if val == default_val else ""
        setting = f"{marker}{val}{marker}"
        target_z = f"{r['target_z_median']:.2f}"
        det_rate  = f"{r['detection_rate']:.2f} ({r['n_detected']}/{r['n_total']})"
        q2bh_str  = f"{int(r['median_q2bh'])}" if r['median_q2bh'] is not None else "—"
        nt_str    = f"{r['max_nontarget_z']:.2f} / {r['fpr']:.2f}"
        supp_str  = f"{int(r['median_support'])}"
        lines.append(_md_row([setting, target_z, det_rate, q2bh_str, nt_str, supp_str]))

    return "\n".join(lines)


def _result_key(W: int, tau_sim: float, tau_margin: float, T_min: int) -> str:
    return f"W{W}_ts{tau_sim:.2f}_tm{tau_margin:.2f}_Tmin{T_min}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work_dir", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument(
        "--trial_logs_dir",
        type=Path,
        default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"),
    )
    ap.add_argument("--seeds",   type=str, default="1,2,3")
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--out_dir", type=Path, default=Path("results/sensitivity_sweep"))
    ap.add_argument("--global_sim_threshold", type=float, default=0.48)
    ap.add_argument("--global_max_docs",      type=int,   default=120)
    ap.add_argument("--batch_size",           type=int,   default=512)
    ap.add_argument("--alpha",                type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--grid", type=str, default="base",
                    help=f"Comma-separated grids to sweep: {sorted(GRIDS)}.")
    return ap.parse_args(argv)


def run_grid(
    precomp_by_W: Dict[int, Dict[Tuple[int, int], List[Dict[str, Any]]]],
    grid: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    """Sweep every (W, tau_sim, tau_margin, T_min) combination of ``grid``."""
    results: Dict[str, Any] = {}
    for W in grid["W"]:
        for tau_sim in grid["tau_sim"]:
            for tau_margin in grid["tau_margin"]:
                for T_min in grid["T_min"]:
                    r = sweep_params(precomp_by_W[W], tau_sim=tau_sim,
                                     tau_margin=tau_margin, T_min=T_min, alpha=alpha)
                    r.update({"W": W, "tau_sim": tau_sim, "tau_margin": tau_margin, "T_min": T_min})
                    results[_result_key(W=W, tau_sim=tau_sim, tau_margin=tau_margin, T_min=T_min)] = r
    return results


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid_names = [g.strip() for g in args.grid.split(",") if g.strip()]
    unknown = [g for g in grid_names if g not in GRIDS]
    if unknown:
        raise SystemExit(f"unknown --grid {unknown}; choose from {sorted(GRIDS)}")

    seeds   = [int(s) for s in args.seeds.split(",")   if s.strip()]
    targets = [int(t) for t in args.targets.split(",") if t.strip()]
    print(f"[INFO] Seeds={seeds}  Targets={targets}")
    print(f"[INFO] Work dir: {args.work_dir}")
    for g in grid_names:
        gr = GRIDS[g]
        n = len(gr["W"]) * len(gr["tau_sim"]) * len(gr["tau_margin"]) * len(gr["T_min"])
        print(f"[INFO] grid '{g}': W={gr['W']} tau_sim={gr['tau_sim']} "
              f"tau_margin={gr['tau_margin']} T_min={gr['T_min']} ({n} combos)")

    # ------------------------------------------------------------------
    # Load corpus, keys, embeddings
    # ------------------------------------------------------------------
    print("[LOAD] corpus + keys...")
    corpus_rows, keys_hex = load_corpus_and_keys(args.work_dir)
    corpus_emb = np.load(args.work_dir / "emb_text_wm.npz")["X"].astype(np.float32)
    assert len(corpus_rows) == corpus_emb.shape[0]

    # SimHash and DirectionManagers
    simhash = SimHashCoder(dim=D_AUDIT, m_bits=96, seed=2025)
    dir_managers: Dict[int, DirectionManager] = {
        i: DirectionManager(keys_hex[i], dim=D_AUDIT) for i in range(len(keys_hex))
    }

    # ------------------------------------------------------------------
    # Load retrieval encoder (for reference set selection)
    # ------------------------------------------------------------------
    print("[LOAD] retrieval encoder (all-MiniLM-L6-v2)...")
    retrieval_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # ------------------------------------------------------------------
    # Load audit encoder (all-mpnet-base-v2)
    # ------------------------------------------------------------------
    print("[LOAD] audit encoder (all-mpnet-base-v2)...")
    audit_enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    # ------------------------------------------------------------------
    # Load trial logs
    # ------------------------------------------------------------------
    print("[LOAD] trial logs...")
    trials_by_ts: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for target in targets:
        for seed in seeds:
            path = (
                args.trial_logs_dir
                / f"rag_multi_t{target}_seed{seed:02d}.trials.jsonl"
            )
            if not path.exists():
                print(f"  [WARN] Missing: {path}")
                continue
            rows = load_jsonl(path)
            rows.sort(key=lambda r: int(r.get("trial_idx", 0)))
            trials_by_ts[(target, seed)] = rows
    print(f"  Loaded {len(trials_by_ts)} (target, seed) pairs")

    # ------------------------------------------------------------------
    # Pre-compute reference doc IDs for each unique question
    # ------------------------------------------------------------------
    print("[PREP] Reference set selection for unique questions...")
    unique_questions = set()
    for rows in trials_by_ts.values():
        for r in rows:
            unique_questions.add(r["question"])

    ref_doc_ids_by_q: Dict[str, List[Tuple[int, str]]] = {}
    for q in tqdm(sorted(unique_questions), desc="Ref sets"):
        ref_doc_ids_by_q[q] = select_reference_doc_ids(
            q, corpus_rows, corpus_emb, retrieval_enc,
            sim_threshold=args.global_sim_threshold,
            max_docs=args.global_max_docs,
        )
    print(f"  {len(ref_doc_ids_by_q)} unique questions processed")

    # ------------------------------------------------------------------
    # Phase 1: Precompute per-window stats for each W needed by any grid
    # ------------------------------------------------------------------
    W_needed = sorted({W for g in grid_names for W in GRIDS[g]["W"]})
    precomp_by_W: Dict[int, Dict[Tuple[int, int], List[Dict[str, Any]]]] = {}
    for W in W_needed:
        print(f"\n[PRECOMPUTE] W={W} (stride={W//2})")
        precomp_by_W[W] = precompute_for_W(
            W=W,
            trials_by_ts=trials_by_ts,
            corpus_rows=corpus_rows,
            ref_doc_ids_by_question=ref_doc_ids_by_q,
            audit_enc=audit_enc,
            simhash=simhash,
            dir_managers=dir_managers,
            source_field="text_wm",
            batch_size=args.batch_size,
        )

    # ------------------------------------------------------------------
    # Phase 2: Sweep each grid
    # ------------------------------------------------------------------
    for g in grid_names:
        print(f"\n[SWEEP] grid '{g}'...")
        all_results = run_grid(precomp_by_W, GRIDS[g], args.alpha)
        out_json = args.out_dir / GRIDS[g]["out"]
        out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"[WRITE] {out_json} ({len(all_results)} combos)")
        if g == "base":
            _write_base_note(all_results, args.out_dir, seeds, targets, len(trials_by_ts))
        else:
            _print_marginals(all_results, GRIDS[g])
    print("\nDone.")


def _print_marginals(all_results: Dict[str, Any], grid: Dict[str, Any]) -> None:
    """Print one-at-a-time marginal tables around the paper operating point."""
    fixed_all = dict(W=DEFAULT_W, tau_sim=DEFAULT_TAU_SIM,
                     tau_margin=DEFAULT_TAU_MARGIN, T_min=DEFAULT_T_MIN)
    for dim in ("tau_sim", "tau_margin", "T_min", "W"):
        if len(grid[dim]) < 2:
            continue
        print(f"\n{'='*60}\n {dim} sweep\n{'='*60}")
        print(build_marginal_table(
            all_results, dim, grid[dim], fixed_all[dim],
            fixed={k: v for k, v in fixed_all.items() if k != dim},
        ))


def _write_base_note(
    all_results: Dict[str, Any], out_dir: Path, seeds: List[int], targets: List[int], n_pairs: int,
) -> None:
    default_key = _result_key(
        W=DEFAULT_W, tau_sim=DEFAULT_TAU_SIM,
        tau_margin=DEFAULT_TAU_MARGIN, T_min=DEFAULT_T_MIN,
    )
    fixed_all = dict(W=DEFAULT_W, tau_sim=DEFAULT_TAU_SIM,
                     tau_margin=DEFAULT_TAU_MARGIN, T_min=DEFAULT_T_MIN)
    tables = {
        dim: build_marginal_table(
            all_results, dim, vals, fixed_all[dim],
            fixed={k: v for k, v in fixed_all.items() if k != dim},
        )
        for dim, vals in (("tau_sim", TAU_SIM_VALUES), ("tau_margin", TAU_MARGIN_VALUES),
                          ("T_min", T_MIN_VALUES), ("W", W_VALUES))
    }
    for dim, tbl in tables.items():
        print(f"\n{'='*60}\n {dim} sweep\n{'='*60}")
        print(tbl)
    note_md = _build_note(
        all_results=all_results,
        default_r=all_results[default_key],
        table_tsim=tables["tau_sim"],
        table_tmargin=tables["tau_margin"],
        table_tmin=tables["T_min"],
        table_W=tables["W"],
        seeds=seeds,
        targets=targets,
        n_pairs=n_pairs,
    )
    out_md = out_dir / "SENSITIVITY_NOTE.md"
    out_md.write_text(note_md, encoding="utf-8")
    print(f"[WRITE] {out_md}")


def _build_note(
    all_results: Dict[str, Any],
    default_r: Dict[str, Any],
    table_tsim: str,
    table_tmargin: str,
    table_tmin: str,
    table_W: str,
    seeds: List[int],
    targets: List[int],
    n_pairs: int,
) -> str:
    n_combos = len(all_results)

    # Compute stability metrics from the full sweep
    det_rates = [r["detection_rate"] for r in all_results.values()]
    nt_z_vals = [r["max_nontarget_z"] for r in all_results.values()]
    fpr_vals  = [r["fpr"] for r in all_results.values()]
    tz_vals   = [r["target_z_median"] for r in all_results.values()]

    return f"""# DirBucket — Detector Hyperparameter Sensitivity Analysis

## Setup

- **Seeds**: {seeds} ({len(seeds)} seeds), **Targets**: {targets} ({len(targets)} providers)
- **Total (target, seed) pairs evaluated**: {n_pairs}
- **Trials per pair**: 80 (existing rag_multi trial logs, no new generation)
- **Total parameter combinations**: {n_combos}
  (W ∈ {{20, 28, 40}} × τ_sim ∈ {{0.78, 0.80, 0.82, 0.84, 0.86}} × τ_margin ∈ {{0.02, 0.05, 0.08, 0.10}} × T_min ∈ {{10, 25, 50}})
- **Paper default**: W=28, τ_sim=0.82, τ_margin=0.05, T_min=25
- **BH α**: {DEFAULT_ALPHA}

**Paper default setting result (sanity check):**
Median target Z = {default_r['target_z_median']:.2f}, BH detection = {default_r['n_detected']}/{default_r['n_total']}, Max non-target Z = {default_r['max_nontarget_z']:.2f}, Median support = {int(default_r['median_support'])}

---

## Results Tables

*(Paper default value in each table is shown in **bold**.)*

### Table 1: τ_sim sweep (W=28, τ_margin=0.05, T_min=25)

{table_tsim}

### Table 2: τ_margin sweep (W=28, τ_sim=0.82, T_min=25)

{table_tmargin}

### Table 3: T_min sweep (W=28, τ_sim=0.82, τ_margin=0.05)

{table_tmin}

### Table 4: W sweep (τ_sim=0.82, τ_margin=0.05, T_min=25)

{table_W}

---

## Summary Statistics Across All {n_combos} Combinations

| Metric | Min | Median | Max |
|---|---|---|---|
| BH detection rate | {min(det_rates):.2f} | {float(np.median(det_rates)):.2f} | {max(det_rates):.2f} |
| Max non-target Z | {min(nt_z_vals):.2f} | {float(np.median(nt_z_vals)):.2f} | {max(nt_z_vals):.2f} |
| Non-target FPR | {min(fpr_vals):.3f} | {float(np.median(fpr_vals)):.3f} | {max(fpr_vals):.3f} |
| Target Z (median) | {min(tz_vals):.2f} | {float(np.median(tz_vals)):.2f} | {max(tz_vals):.2f} |

---

## Interpretation

*(See paper App. app:sensitivity for the interpretation.)*

---

*Generated by `experiments/detector_sensitivity_sweep.py`*
"""


if __name__ == "__main__":
    main()
