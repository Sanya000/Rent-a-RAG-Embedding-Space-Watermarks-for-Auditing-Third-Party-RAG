#!/usr/bin/env python3
"""Dense T_min sweep at the paper operating point (App. app:sensitivity, optional figure).

W=28, tau_sim=0.82, tau_margin=0.05 fixed; 26 T_min values in [1, 150] on the
seeds 1-3 primary answers (18 audits), using the precompute and sweep of
detector_sensitivity_sweep.py (phase 1 once, phase 2 per T_min). The 30-audit
rows of tab:tmin-crossbenchmark come from e6_tmin_crossbenchmark.py.

Output: results/sensitivity_sweep/tmin_sweep_dense.json
"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Reuse infrastructure from the existing sweep script
from detector_sensitivity_sweep import (
    load_corpus_and_keys, load_jsonl, select_reference_doc_ids,
    precompute_for_W, sweep_params,
    D_AUDIT, DEFAULT_ALPHA, N_PROVIDERS,
)
from watermark_dirbucket import DirectionManager, SimHashCoder

# Dense T_min grid (covers the full range from almost no gate to very conservative)
T_MIN_VALUES = [
    1, 2, 3, 5, 7, 8, 10, 12, 14, 16, 18, 20, 22, 25, 28,
    30, 35, 40, 45, 50, 60, 70, 80, 100, 125, 150
]
W_FIXED          = 28
TAU_SIM_FIXED    = 0.82
TAU_MARGIN_FIXED = 0.05


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--trial_logs_dir", type=Path,
                    default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"))
    ap.add_argument("--seeds",   type=str, default="1,2,3")
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--global_sim_threshold", type=float, default=0.48)
    ap.add_argument("--global_max_docs",      type=int,   default=120)
    ap.add_argument("--batch_size",           type=int,   default=512)
    ap.add_argument("--alpha",                type=float, default=DEFAULT_ALPHA)
    ap.add_argument("--out",  type=Path,
                    default=Path("results/sensitivity_sweep/tmin_sweep_dense.json"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    seeds   = [int(s) for s in args.seeds.split(",") if s.strip()]
    targets = [int(t) for t in args.targets.split(",") if t.strip()]

    print(f"[INFO] seeds={seeds} targets={targets}")
    print(f"[INFO] W={W_FIXED} ts={TAU_SIM_FIXED} tm={TAU_MARGIN_FIXED}")
    print(f"[INFO] T_min grid ({len(T_MIN_VALUES)} values): {T_MIN_VALUES}")

    # Load corpus + embeddings
    print("[LOAD] corpus + keys ...")
    corpus_rows, keys_hex = load_corpus_and_keys(args.work_dir)
    corpus_emb = np.load(args.work_dir / "emb_text_wm.npz")["X"].astype(np.float32)

    simhash = SimHashCoder(dim=D_AUDIT, m_bits=96, seed=2025)
    dir_managers = {i: DirectionManager(keys_hex[i], dim=D_AUDIT) for i in range(len(keys_hex))}

    print("[LOAD] retrieval encoder ...")
    retrieval_enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("[LOAD] audit encoder ...")
    audit_enc = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    print("[LOAD] trial logs ...")
    trials_by_ts: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for target in targets:
        for seed in seeds:
            path = args.trial_logs_dir / f"rag_multi_t{target}_seed{seed:02d}.trials.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            rows.sort(key=lambda r: int(r.get("trial_idx", 0)))
            trials_by_ts[(target, seed)] = rows
    print(f"  {len(trials_by_ts)} (target, seed) pairs loaded")

    print("[PREP] Reference set selection ...")
    unique_questions = {r["question"] for rows in trials_by_ts.values() for r in rows}
    ref_doc_ids_by_q = {}
    for q in tqdm(sorted(unique_questions), desc="Ref sets"):
        ref_doc_ids_by_q[q] = select_reference_doc_ids(
            q, corpus_rows, corpus_emb, retrieval_enc,
            sim_threshold=args.global_sim_threshold,
            max_docs=args.global_max_docs,
        )

    print(f"\n[PRECOMPUTE] W={W_FIXED} (single pass) ...")
    precomp = precompute_for_W(
        W=W_FIXED,
        trials_by_ts=trials_by_ts,
        corpus_rows=corpus_rows,
        ref_doc_ids_by_question=ref_doc_ids_by_q,
        audit_enc=audit_enc,
        simhash=simhash,
        dir_managers=dir_managers,
        source_field="text_wm",
        batch_size=args.batch_size,
    )

    print(f"\n[SWEEP] {len(T_MIN_VALUES)} T_min values (Phase 2, fast) ...")
    results = {}
    for T_min in T_MIN_VALUES:
        r = sweep_params(
            precomp=precomp,
            tau_sim=TAU_SIM_FIXED,
            tau_margin=TAU_MARGIN_FIXED,
            T_min=T_min,
            alpha=args.alpha,
        )
        r["T_min"] = T_min
        results[str(T_min)] = r
        print(f"  T_min={T_min:3d}: det={r['detection_rate']:.2f} "
              f"FPR={r['fpr']:.2f} Z={r['target_z_median']:.2f} "
              f"q2bh={r.get('median_q2bh', '—')}", flush=True)

    args.out.write_text(json.dumps(results, indent=2))
    print(f"\n[WRITE] {args.out}")


if __name__ == "__main__":
    main()
