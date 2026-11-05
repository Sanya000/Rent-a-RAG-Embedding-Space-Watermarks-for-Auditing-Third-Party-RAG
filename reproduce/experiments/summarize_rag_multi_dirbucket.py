#!/usr/bin/env python3
"""
Summarize DirBucket rag_multi outputs.

This is intended for reproduction workflows:
- Point to a folder containing rag_multi_t{target}_seed{seed}.json files.
- Get a compact aggregate summary.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Dict, List, Optional, Tuple

RUN_RE = re.compile(r"rag_multi_t(?P<t>\d+)_seed(?P<s>\d+)\.json$")


def _safe_mean(xs: List[float]) -> Optional[float]:
    return float(statistics.mean(xs)) if xs else None


def _safe_median(xs: List[float]) -> Optional[float]:
    return float(statistics.median(xs)) if xs else None


def _parse_run_file(path: Path) -> Optional[Tuple[int, int]]:
    m = RUN_RE.match(path.name)
    if not m:
        return None
    return int(m.group("t")), int(m.group("s"))


def compute_summary(results_dir: Path) -> Dict[str, object]:
    files = sorted(p for p in results_dir.glob("rag_multi_t*_seed*.json") if _parse_run_file(p) is not None)

    per_run: List[Dict[str, object]] = []
    winners: List[float] = []
    bh_target_flags: List[float] = []
    bh_any_nt_flags: List[float] = []
    cross_bh_qs: List[float] = []
    cross_raw_qs: List[float] = []
    t_targets: List[float] = []
    z_targets: List[float] = []
    assign_rates: List[float] = []
    selected_src_docs: List[float] = []

    seeds_seen = set()
    targets_seen = set()
    trials_values = set()

    for fp in files:
        parsed = _parse_run_file(fp)
        if parsed is None:
            continue
        target_from_name, seed_from_name = parsed
        seeds_seen.add(seed_from_name)
        targets_seen.add(target_from_name)

        run = json.loads(fp.read_text(encoding="utf-8"))
        tp = int(run.get("target_provider", target_from_name))
        trials = int(run.get("trials", 0))
        trials_values.add(trials)

        winner = float(run.get("winner_fraction_target", 0.0))
        winners.append(winner)

        bh_sig = run.get("bh_sig") or []
        bh_target = bool(tp < len(bh_sig) and bh_sig[tp])
        bh_nt = any(bool(v) for i, v in enumerate(bh_sig) if i != tp)
        bh_target_flags.append(1.0 if bh_target else 0.0)
        bh_any_nt_flags.append(1.0 if bh_nt else 0.0)

        crossed = run.get("crossed_at") or []
        crossed_raw = run.get("crossed_at_raw") or []
        q_bh = crossed[tp] if tp < len(crossed) else None
        q_raw = crossed_raw[tp] if tp < len(crossed_raw) else None
        if isinstance(q_bh, int):
            cross_bh_qs.append(float(q_bh))
        if isinstance(q_raw, int):
            cross_raw_qs.append(float(q_raw))

        pooled_t = run.get("pooled_T") or []
        pooled_z = run.get("z_raw") or []
        t_target = float(pooled_t[tp]) if tp < len(pooled_t) else 0.0
        z_target = float(pooled_z[tp]) if tp < len(pooled_z) else 0.0
        t_targets.append(t_target)
        z_targets.append(z_target)

        align = run.get("align") or {}
        ar = align.get("assign_rate")
        if isinstance(ar, (int, float)):
            assign_rates.append(float(ar))
        sds = align.get("source_docs_selected_mean")
        if isinstance(sds, (int, float)):
            selected_src_docs.append(float(sds))

        per_run.append(
            {
                "file": fp.name,
                "target_provider": tp,
                "seed": seed_from_name,
                "trials": trials,
                "winner_fraction_target": winner,
                "bh_target": bh_target,
                "bh_any_nontarget": bh_nt,
                "crossed_at_target_bh": q_bh,
                "crossed_at_target_raw": q_raw,
                "pooled_T_target": t_target,
                "z_raw_target": z_target,
                "align_assign_rate": ar,
            }
        )

    summary: Dict[str, object] = {
        "n_runs": len(per_run),
        "seeds": sorted(seeds_seen),
        "targets": sorted(targets_seen),
        "trials_values": sorted(trials_values),
        "metrics": {
            "winner_fraction_target_mean": _safe_mean(winners),
            "winner_fraction_target_median": _safe_median(winners),
            "bh_target_rate": _safe_mean(bh_target_flags),
            "bh_any_nontarget_rate": _safe_mean(bh_any_nt_flags),
            "crossed_at_bh_rate": (len(cross_bh_qs) / len(per_run) if per_run else None),
            "crossed_at_bh_median": _safe_median(cross_bh_qs),
            "crossed_at_raw_rate": (len(cross_raw_qs) / len(per_run) if per_run else None),
            "crossed_at_raw_median": _safe_median(cross_raw_qs),
            "pooled_T_target_mean": _safe_mean(t_targets),
            "z_raw_target_mean": _safe_mean(z_targets),
            "align_assign_rate_mean": _safe_mean(assign_rates),
            "align_source_docs_selected_mean": _safe_mean(selected_src_docs),
        },
        "per_run": per_run,
    }
    return summary


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Summarize DirBucket rag_multi outputs.")
    ap.add_argument("--results_dir", type=Path, required=True, help="Directory containing rag_multi_t*_seed*.json.")
    ap.add_argument("--out_json", type=Path, default=None, help="Where to write summary JSON.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results_dir not found: {results_dir}")

    summary = compute_summary(results_dir)
    print(f"[SUMMARY] runs={summary.get('n_runs')} seeds={summary.get('seeds')} targets={summary.get('targets')}")
    print(json.dumps(summary.get("metrics", {}), indent=2))

    if args.out_json is not None:
        out_path = args.out_json.expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()
