"""E1 real-domain benchmark analysis (paper App. app:multidomain, Table tab:multidomain).

Reads the per-run JSONs (``rag_multi_t*_seed*.json``) of the multi-domain
benchmark and reports, at whatever seed scale the directory holds:
  - per-domain median target Z and BH detection count (``per_domain_bh_detection``)
  - overall BH detection and FPR (audits with any BH-significant non-target)
  - within-domain SIBLING non-target gated Z (the same-domain provider that
    could steal attribution) vs cross-domain non-target gated Z
  - median queries to BH detection and to the Z=4 reference

Provider layout (``build_manifest_multidomain.py``): 0,1 = clinical; 2,3 = cti;
4,5 = legal. Sibling = the other provider of the same domain.

Paper command:
  python experiments/cr_e1_domain_analysis.py \
      --run_dir results/multidomain_full/dirbucket_naive \
      --out results/multidomain_full/e1_domain_analysis.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Any, Dict, List

DOMAIN = {0: "clinical", 1: "clinical", 2: "cti", 3: "cti", 4: "legal", 5: "legal"}
SIBLING = {0: 1, 1: 0, 2: 3, 3: 2, 4: 5, 5: 4}
TMIN = 25


def gated(z: List[float], T: List[int]) -> List[float]:
    return [float(z[i]) if T[i] >= TMIN else 0.0 for i in range(len(z))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    files = sorted(args.run_dir.glob("rag_multi_t*_seed*.json"))
    if not files:
        raise SystemExit(f"no rag_multi_t*_seed*.json in {args.run_dir}")

    per_domain_targetZ: Dict[str, List[float]] = {}
    per_domain_det: Dict[str, List[bool]] = {}
    sib_nt, cross_nt = [], []
    n_runs = det = fpr = 0
    q_bh, q_raw = [], []

    for f in files:
        d = json.loads(f.read_text())
        t = int(d["target_provider"])
        z = [float(x) for x in d["z_raw"]]
        T = [int(x) for x in d["pooled_T"]]
        bh = d.get("bh_sig") or [False] * len(z)
        zg = gated(z, T)
        n_runs += 1
        det += bool(bh[t])
        # FPR: any non-target BH-significant
        if any(bh[p] for p in range(len(z)) if p != t):
            fpr += 1
        per_domain_targetZ.setdefault(DOMAIN[t], []).append(z[t])
        per_domain_det.setdefault(DOMAIN[t], []).append(bool(bh[t]))
        for p in range(len(z)):
            if p == t:
                continue
            (sib_nt if p == SIBLING[t] else cross_nt).append(zg[p])
        ca = d.get("crossed_at")
        cr = d.get("crossed_at_raw")
        if isinstance(ca, list) and ca[t] is not None:
            q_bh.append(int(ca[t]))
        if isinstance(cr, list) and cr[t] is not None:
            q_raw.append(int(cr[t]))

    report: Dict[str, Any] = {
        "run_dir": str(args.run_dir),
        "n_runs": n_runs,
        "bh_detection": f"{det}/{n_runs}",
        "fpr_runs_with_any_nontarget_bh": f"{fpr}/{n_runs}",
        "per_domain_median_target_Z": {
            k: round(median(v), 2) for k, v in sorted(per_domain_targetZ.items())},
        "median_target_Z_all": round(median(z for v in per_domain_targetZ.values() for z in v), 2),
        "per_domain_bh_detection": {
            k: f"{sum(v)}/{len(v)}" for k, v in sorted(per_domain_det.items())},
        "within_domain_sibling_nontarget_gated_Z": {
            "max": round(max(sib_nt), 4) if sib_nt else None,
            "mean": round(sum(sib_nt) / len(sib_nt), 4) if sib_nt else None,
            "n": len(sib_nt)},
        "cross_domain_nontarget_gated_Z": {
            "max": round(max(cross_nt), 4) if cross_nt else None,
            "mean": round(sum(cross_nt) / len(cross_nt), 4) if cross_nt else None,
            "n": len(cross_nt)},
        "median_queries_to_bh": (median(q_bh) if q_bh else None),
        "n_reached_bh_within_budget": len(q_bh),
        "median_queries_to_Z4": (median(q_raw) if q_raw else None),
    }
    print(json.dumps(report, indent=2))
    out = args.out or (args.run_dir.parent / "e1_domain_analysis.json")
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[WRITE] {out}")


if __name__ == "__main__":
    main()
