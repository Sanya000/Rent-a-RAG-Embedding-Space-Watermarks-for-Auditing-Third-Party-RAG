#!/usr/bin/env python3
"""
Run the paper-style DirBucket rag_multi audit matrix (no baselines).

Reproduces the *main DirBucket rag_multi* configuration of
  results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive
(global_corpus alignment, query floor 0.48, tau_sim 0.82, tie margin 0.05, T_min 25), and is the
entry point reused by the E1 (multi-domain) and E2 (coverage) experiments.

Matrix: target_provider 0..5 x seeds 1..5, 80 trials per run (2 gpt-4o-mini calls per trial).

The wrapped harness is experiments/run_dirbucket_quick_experiments.py (resolved next to this
file). The E1/E2 run cards name a DirBucket-only copy of it; the dirbucket_naive code path
of the two files is the same (replaying recorded trials through both gives byte-identical
trial logs).

Outputs (under --out_dir):
- dirbucket_naive/rag_multi_t{target}_seed{seed:02d}.json and .trials.jsonl
- paper_rag_multi_run_card.json (resolved commands)
- paper_rag_multi_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from summarize_rag_multi_dirbucket import compute_summary


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _parse_int_csv(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        p = part.strip()
        if not p:
            continue
        out.append(int(p))
    return out


def _q(cmd: List[str]) -> str:
    return " ".join(shlex.quote(x) for x in cmd)


def _build_cmd(
    python_bin: str,
    work_dir: Path,
    alignment_manifest: Path,
    out_json: Path,
    trial_log_jsonl: Path,
    seed: int,
    target_provider: int,
    trials: int,
    openai_model: str,
) -> List[str]:
    return [
        python_bin,
        str(_repo_root() / "run_dirbucket_quick_experiments.py"),
        "--work_dir",
        str(work_dir),
        "--wm_scheme",
        "dirbucket_naive",
        "--audit_mode",
        "aligned",
        "--retrieval_model",
        "all-MiniLM-L6-v2",
        "--openai_model",
        openai_model,
        "--answer_temp",
        "0.55",
        "--z_threshold",
        "4.0",
        "--min_pooled_T",
        "25",
        "--aligned_source_mode",
        "global_corpus",
        "--global_source_query_sim_threshold",
        "0.48",
        "--global_source_max_docs",
        "120",
        "--global_source_max_docs_per_provider",
        "0",
        "--global_source_selector",
        "query",
        "--competitive",
        "--direction_scheme",
        "naive",
        "--sim_threshold",
        "0.82",
        "--tie_margin",
        "0.05",
        "--alignment_manifest",
        str(alignment_manifest),
        "--exp",
        "rag_multi",
        "--trials",
        str(trials),
        "--target_provider",
        str(target_provider),
        "--answer_style",
        "ownwords",
        "--topk",
        "5",
        "--min_unique_providers",
        "3",
        "--do_wrongkey",
        "--log_traj",
        "--seed",
        str(seed),
        "--trial_log_jsonl",
        str(trial_log_jsonl),
        "--out_json",
        str(out_json),
        "--force_cross_provider",
        "--no-resume_trials",
    ]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run paper rag_multi DirBucket matrix (targets x seeds).")
    ap.add_argument("--work_dir", type=Path, required=True, help="DirBucket corpus work dir (e.g., work/farad_dirbucket_naive).")
    ap.add_argument("--alignment_manifest", type=Path, required=True, help="Shared alignment manifest JSONL.")
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("outputs/paper_rag_multi"),
        help="Output directory for rag_multi JSON + trial logs.",
    )
    ap.add_argument("--seeds", type=str, default="1,2,3,4,5", help="Comma-separated seeds.")
    ap.add_argument("--targets", type=str, default="0,1,2,3,4,5", help="Comma-separated target providers.")
    ap.add_argument("--trials", type=int, default=80, help="Queries per run.")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini", help="Answer-generation model.")
    ap.add_argument("--python_bin", type=str, default=sys.executable, help="Python executable for wrapped script.")
    ap.add_argument("--openai_api_key", type=str, default=None, help="Optional override for OPENAI_API_KEY.")
    ap.add_argument("--skip_existing", action="store_true", help="Skip runs when output JSON already exists.")
    ap.add_argument("--dry_run", action="store_true", help="Print commands without executing.")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    root = _repo_root()

    work_dir = args.work_dir.expanduser().resolve()
    alignment_manifest = args.alignment_manifest.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    run_dir = out_dir / "dirbucket_naive"
    run_dir.mkdir(parents=True, exist_ok=True)

    if not work_dir.exists():
        raise FileNotFoundError(f"work_dir not found: {work_dir}")
    if not alignment_manifest.exists():
        raise FileNotFoundError(f"alignment_manifest not found: {alignment_manifest}")

    seeds = _parse_int_csv(args.seeds)
    targets = _parse_int_csv(args.targets)
    if not seeds:
        raise SystemExit("No seeds parsed from --seeds.")
    if not targets:
        raise SystemExit("No targets parsed from --targets.")

    env = os.environ.copy()
    if args.openai_api_key:
        env["OPENAI_API_KEY"] = args.openai_api_key
    if not env.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is not set. Export it or pass --openai_api_key.")

    cwd_prefix = str(Path.cwd()) + os.sep

    def _portable(cmd: List[str]) -> List[str]:
        """Command as recorded in the run card: paths relative to the cwd, bare `python`."""
        return ["python"] + [x[len(cwd_prefix):] if x.startswith(cwd_prefix) else x for x in cmd[1:]]

    commands: List[Dict[str, object]] = []
    total = len(seeds) * len(targets)
    done = 0

    for seed in seeds:
        for target in targets:
            out_json = run_dir / f"rag_multi_t{target}_seed{seed:02d}.json"
            trial_log = run_dir / f"rag_multi_t{target}_seed{seed:02d}.trials.jsonl"
            cmd = _build_cmd(
                python_bin=args.python_bin,
                work_dir=work_dir,
                alignment_manifest=alignment_manifest,
                out_json=out_json,
                trial_log_jsonl=trial_log,
                seed=seed,
                target_provider=target,
                trials=args.trials,
                openai_model=args.openai_model,
            )
            card_cmd = _portable(cmd)
            commands.append({"seed": seed, "target_provider": target, "command": card_cmd, "command_shell": _q(card_cmd)})

            if args.skip_existing and out_json.exists():
                print(f"[SKIP] {out_json.name} exists")
                done += 1
                continue

            print(f"[RUN {done+1}/{total}] {out_json.name}")
            print(_q(cmd))
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=root, env=env)
            done += 1

    # Run-card for full matrix.
    run_card = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "entrypoint": "experiments/run_paper_rag_multi_dirbucket.py",
        "wrapped_script": "experiments/run_dirbucket_quick_experiments.py",
        "config_file": "configs/paper_rag_multi_config.json",
        "work_dir": _portable(["", str(work_dir)])[1],
        "alignment_manifest": _portable(["", str(alignment_manifest)])[1],
        "out_dir": _portable(["", str(out_dir)])[1],
        "seeds": seeds,
        "targets": targets,
        "trials": args.trials,
        "openai_model": args.openai_model,
        "commands": commands,
    }
    card_path = out_dir / "paper_rag_multi_run_card.json"
    card_path.write_text(json.dumps(run_card, indent=2), encoding="utf-8")
    print(f"[WRITE] {card_path}")

    if args.dry_run:
        print("[DRY-RUN] no summary generated.")
        return

    summary = compute_summary(run_dir)
    summary_path = out_dir / "paper_rag_multi_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[WRITE] {summary_path}")
    print(f"[DONE] Completed {len(seeds)*len(targets)} runs.")


if __name__ == "__main__":
    main()
