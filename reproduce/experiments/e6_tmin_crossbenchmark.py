"""E6: cross-benchmark T_min calibration (paper Table tab:tmin-crossbenchmark).

Pure post-processing of stored run JSONs (no trial logs, no API). The support
gate is applied post hoc to the final pooled statistics, so the complete
decision rule (gate at T_min, then BH at alpha=0.05) is re-applied over a T_min
grid to every ``rag_multi_t*_seed*.json`` of the primary FARAD benchmark and of
the real-domain benchmark (E1), including the wrong-key null stored in each run.

Paper command:
  python experiments/e6_tmin_crossbenchmark.py \
      --multidomain_dir results/multidomain_full/dirbucket_naive \
      --out results/e6_tmin/summary_cameraready.json

These are also the defaults.

``bh_sig`` tracks the BH threshold with ``None`` (not ``0.0``), so providers whose
p-value underflows to exactly 0 (Z above ~38) are reported significant.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

TMIN_GRID = [5, 10, 15, 20, 25, 30, 40, 50]
ALPHA = 0.05

BENCHES = {
    "primary_farad": Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"),
    "multidomain": Path("results/multidomain_full/dirbucket_naive"),
}


def norm_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def bh_sig(pvals: List[float], alpha: float) -> List[bool]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    thresh = None
    for rank, i in enumerate(order, 1):
        if pvals[i] <= alpha * rank / m:
            thresh = pvals[i]
    return [thresh is not None and p <= thresh for p in pvals]


def decide(z: List[float], T: List[int], tmin: int) -> List[bool]:
    p = [norm_sf(zi) if Ti >= tmin else 1.0 for zi, Ti in zip(z, T)]
    return bh_sig(p, ALPHA)


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--primary_dir", type=Path, default=None,
                    help="Override the primary FARAD run dir.")
    ap.add_argument("--multidomain_dir", type=Path, default=None,
                    help="Override the multidomain run dir; defaults to the 5-seed "
                         "results/multidomain_full/dirbucket_naive.")
    ap.add_argument("--out", type=Path, default=Path("results/e6_tmin/summary_cameraready.json"))
    args = ap.parse_args()
    benches = dict(BENCHES)
    if args.primary_dir is not None:
        benches["primary_farad"] = args.primary_dir
    if args.multidomain_dir is not None:
        benches["multidomain"] = args.multidomain_dir

    out: Dict[str, Any] = {"tmin_grid": TMIN_GRID, "alpha": ALPHA, "benchmarks": {}}
    for bench, run_dir in benches.items():
        runs = []
        files = sorted(run_dir.glob("rag_multi_t*_seed*.json"))
        if not files:
            raise SystemExit(f"[FATAL] no rag_multi_t*_seed*.json in {run_dir}")
        for f in files:
            d = json.loads(f.read_text())
            t = int(d["target_provider"])
            wk = d.get("wrongkey") or {}
            runs.append({
                "target": t,
                "z": [float(x) for x in d["z_raw"]],
                "T": [int(x) for x in d["pooled_T"]],
                "wk_z": [float(x) for x in (wk.get("z_raw") or [])],
                "wk_T": [int(x) for x in (wk.get("T") or wk.get("pooled_T") or [])],
            })
        rows = {}
        for tmin in TMIN_GRID:
            det = nt = wk_any = 0
            wk_n = 0
            for r in runs:
                sig = decide(r["z"], r["T"], tmin)
                det += sig[r["target"]]
                nt += any(s for i, s in enumerate(sig) if i != r["target"])
                if r["wk_z"] and r["wk_T"]:
                    wk_n += 1
                    wk_any += any(decide(r["wk_z"], r["wk_T"], tmin))
            rows[str(tmin)] = {
                "target_detection_rate": round(det / len(runs), 3),
                "any_nontarget_rate": round(nt / len(runs), 3),
                "wrongkey_any_sig_rate": (round(wk_any / wk_n, 3) if wk_n else None),
                "n_runs": len(runs), "n_wrongkey_runs": wk_n,
            }
        out["benchmarks"][bench] = rows
        print(f"== {bench} ({len(runs)} runs)")
        print(f"{'Tmin':>5} {'det':>6} {'nonTgt':>7} {'wkNull':>7}")
        for tmin in TMIN_GRID:
            r = rows[str(tmin)]
            wkv = r["wrongkey_any_sig_rate"]
            print(f"{tmin:>5} {r['target_detection_rate']:>6} "
                  f"{r['any_nontarget_rate']:>7} {wkv if wkv is not None else '—':>7}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[DONE] {args.out}")


if __name__ == "__main__":
    main()
