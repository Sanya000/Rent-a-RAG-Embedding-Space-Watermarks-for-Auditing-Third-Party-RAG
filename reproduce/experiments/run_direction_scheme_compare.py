"""Driver for the naive-vs-orthogonalised direction ablation (tab:ortho-ablation).

Writes <out_dir>/{naive,ortho}/{sanity_coop,sanity_unrel}_t{t}_seed{SS}.json,
rag_multi_seed{SS}.json and style_sweep_seed{SS}.json. See reproduce/scripts/15_direction_ablation.sh.
"""
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import List

RUNNER = Path(__file__).resolve().parent / "run_dirbucket_quick_experiments.py"


def _run(cmd: List[str]) -> None:
    print("[RUN ] " + " ".join(cmd))
    subprocess.run(cmd, check=True)

def _maybe_run(out_path: Path, cmd: List[str], *, overwrite: bool = False) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] {out_path.as_posix()}")
        return
    _run(cmd)

def main() -> None:
    ap = argparse.ArgumentParser(description="Run the 3 paper experiments under two direction schemes (naive vs ortho).")
    ap.add_argument("--work_dir_naive", type=Path, required=True)
    ap.add_argument("--work_dir_ortho", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("results/direction_compare"))
    ap.add_argument("--schemes", nargs="+", choices=["naive", "ortho"], default=["naive", "ortho"],
                    help="Which direction schemes to run.")
    ap.add_argument("--overwrite_schemes", nargs="+", choices=["naive", "ortho"], default=[],
                    help="Re-run only the listed schemes even if their output JSON exists.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5])
    ap.add_argument("--target_provider", type=int, default=0)
    ap.add_argument("--providers", type=int, default=6)
    ap.add_argument("--trials_sanity", type=int, default=80)
    ap.add_argument("--trials_rag", type=int, default=80)
    ap.add_argument("--trials_style", type=int, default=30)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_unique_providers", type=int, default=3)
    ap.add_argument("--force_cross_provider", action="store_true", default=True)
    ap.add_argument("--sim_threshold", type=float, default=0.90,
                    help="Cosine threshold for aligned assignment.")
    ap.add_argument("--competitive", action=argparse.BooleanOptionalAction, default=True,
                    help="Enable competitive aligned assignment in detector.")
    ap.add_argument("--tie_margin", type=float, default=0.05,
                    help="Tie margin for competitive assignment.")
    ap.add_argument("--min_pooled_T", type=int, default=10,
                    help="Minimum pooled assigned windows required before counting z-threshold crossing.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-run experiments even if output JSON already exists.")
    ap.add_argument("--audit_mode", choices=["answer_only", "aligned"], default=None,
                    help="Forwarded to the runner when set (runner default: answer_only).")
    ap.add_argument("--aligned_source_mode", choices=["oracle", "global_corpus"], default=None,
                    help="Forwarded to the runner when set (runner default: oracle).")
    args = ap.parse_args()
    passthrough: List[str] = []
    if args.audit_mode is not None:
        passthrough += ["--audit_mode", args.audit_mode]
    if args.aligned_source_mode is not None:
        passthrough += ["--aligned_source_mode", args.aligned_source_mode]

    schemes = {"naive": args.work_dir_naive, "ortho": args.work_dir_ortho}
    run_schemes = [s for s in ["naive", "ortho"] if s in set(args.schemes)]
    overwrite_schemes = set(args.overwrite_schemes)

    for scheme in run_schemes:
        work_dir = schemes[scheme]
        scheme_overwrite = bool(args.overwrite or (scheme in overwrite_schemes))
        scheme_out = args.out_dir / scheme
        scheme_out.mkdir(parents=True, exist_ok=True)
        for seed in args.seeds:
            # 1) Sanity: coop+wrongkey and unrelated null, for each target provider
            for t in range(args.providers):
                out = scheme_out / f"sanity_coop_t{t}_seed{seed:02d}.json"
                cmd = [
                    sys.executable, str(RUNNER),
                    "--exp", "sanity_single",
                    "--work_dir", str(work_dir),
                    "--trials", str(args.trials_sanity),
                    "--target_provider", str(t),
                    "--direction_scheme", scheme,
                    "--answer_style", "qa",
                    "--do_wrongkey",
                    "--sim_threshold", str(args.sim_threshold),
                    "--tie_margin", str(args.tie_margin),
                    "--min_pooled_T", str(args.min_pooled_T),
                    "--seed", str(seed),
                    "--out_json", str(out),
                ]
                if args.competitive:
                    cmd.append("--competitive")
                _maybe_run(out, cmd + passthrough, overwrite=scheme_overwrite)

                out = scheme_out / f"sanity_unrel_t{t}_seed{seed:02d}.json"
                cmd = [
                    sys.executable, str(RUNNER),
                    "--exp", "sanity_single",
                    "--work_dir", str(work_dir),
                    "--trials", str(args.trials_sanity),
                    "--target_provider", str(t),
                    "--direction_scheme", scheme,
                    "--answer_style", "qa",
                    "--null_unrelated",
                    "--sim_threshold", str(args.sim_threshold),
                    "--tie_margin", str(args.tie_margin),
                    "--min_pooled_T", str(args.min_pooled_T),
                    "--seed", str(seed),
                    "--out_json", str(out),
                ]
                if args.competitive:
                    cmd.append("--competitive")
                _maybe_run(out, cmd + passthrough, overwrite=scheme_overwrite)

            # 2) Multi-provider RAG (own-words)
            out = scheme_out / f"rag_multi_seed{seed:02d}.json"
            cmd = [
                sys.executable, str(RUNNER),
                "--exp", "rag_multi",
                "--work_dir", str(work_dir),
                "--trials", str(args.trials_rag),
                "--target_provider", str(args.target_provider),
                "--direction_scheme", scheme,
                "--answer_style", "ownwords",
                "--topk", str(args.topk),
                "--min_unique_providers", str(args.min_unique_providers),
                "--sim_threshold", str(args.sim_threshold),
                "--tie_margin", str(args.tie_margin),
                "--min_pooled_T", str(args.min_pooled_T),
            ]
            if args.force_cross_provider:
                cmd.append("--force_cross_provider")
            if args.competitive:
                cmd.append("--competitive")
            cmd += ["--do_wrongkey", "--log_traj", "--seed", str(seed), "--out_json", str(out)]
            _maybe_run(out, cmd + passthrough, overwrite=scheme_overwrite)

            # 3) Style sweep
            out = scheme_out / f"style_sweep_seed{seed:02d}.json"
            cmd = [
                sys.executable, str(RUNNER),
                "--exp", "style_sweep",
                "--work_dir", str(work_dir),
                "--trials", str(args.trials_style),
                "--target_provider", str(args.target_provider),
                "--direction_scheme", scheme,
                "--topk", str(args.topk),
                "--min_unique_providers", str(args.min_unique_providers),
                "--sim_threshold", str(args.sim_threshold),
                "--tie_margin", str(args.tie_margin),
                "--min_pooled_T", str(args.min_pooled_T),
            ]
            if args.force_cross_provider:
                cmd.append("--force_cross_provider")
            if args.competitive:
                cmd.append("--competitive")
            cmd += ["--do_wrongkey", "--seed", str(seed), "--out_json", str(out)]
            _maybe_run(out, cmd + passthrough, overwrite=scheme_overwrite)

    print(f"[DONE] Results in: {args.out_dir}")

if __name__ == "__main__":
    main()
