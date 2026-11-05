"""E3: pool the staleness re-audit over the 5 primary seeds.

Kept for command compatibility with the paper run (cr_e3_staleness_bycond);
equivalent to ``staleness_reaudit.py --seeds 01,02,03,04,05``. Writes
<out_dir>/reaudit_<tag>_allseeds.json and <out_dir>/staleness_full_summary.json,
resuming per (condition, seed).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from staleness_reaudit import run_multiseed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--conditions", type=str,
                    default="f00:work/farad_dirbucket_naive,"
                            "f10:work/staleness/f10,f25:work/staleness/f25,"
                            "f50:work/staleness/f50")
    ap.add_argument("--trial_logs_dir", type=Path,
                    default=Path("results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive"))
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--seeds", type=str, default="01,02,03,04,05")
    ap.add_argument("--max_trials", type=int, default=80)
    ap.add_argument("--out_dir", type=Path, default=Path("results/staleness_full"))
    args = ap.parse_args()

    run_multiseed(
        conditions=[c.split(":", 1) for c in args.conditions.split(",")],
        trial_logs_dir=args.trial_logs_dir,
        targets=[int(t) for t in args.targets.split(",")],
        seed_tags=[s.strip() for s in args.seeds.split(",") if s.strip()],
        max_trials=args.max_trials,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
