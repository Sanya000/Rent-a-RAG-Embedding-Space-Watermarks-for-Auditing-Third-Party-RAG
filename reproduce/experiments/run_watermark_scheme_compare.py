"""Driver for the scheme-comparison matrix (sanity, rag_multi, style sweep) of Sec. 5.2-5.4.

For every scheme and seed it calls run_dirbucket_quick_experiments.py once per target provider
and writes <out_dir>/<scheme>/{sanity_coop,sanity_unrel,rag_multi}_t{t}_seed{SS}.json and
style_sweep_seed{SS}.json. Existing outputs are skipped unless --overwrite/--overwrite_schemes.
The CLI defaults are not the paper settings; see reproduce/scripts/10_main_benchmark.sh.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


RUNNER = Path(__file__).resolve().parent / "run_dirbucket_quick_experiments.py"

SCHEMES = (
    "dirbucket_naive",
    "kgw_ward",
    "semstamp",
    "sim_attr",
    "overlap_attr",
    "entail_attr",
)


def _run(cmd: List[str], *, attempts: int = 3) -> None:
    print("[RUN ] " + " ".join(cmd))
    max_attempts = max(1, int(attempts))
    for i in range(1, max_attempts + 1):
        try:
            subprocess.run(cmd, check=True)
            return
        except subprocess.CalledProcessError:
            if i >= max_attempts:
                raise
            sleep_s = min(120.0, 15.0 * (2 ** (i - 1)))
            print(f"[WARN] command failed (attempt {i}/{max_attempts}); retrying in {sleep_s:.0f}s", flush=True)
            time.sleep(sleep_s)


def _maybe_run(out_path: Path, cmd: List[str], *, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if (not overwrite) and out_path.exists() and out_path.stat().st_size > 0:
        print(f"[SKIP] {out_path.as_posix()}")
        return
    _run(cmd)


def _validate_work_dirs(run_schemes: List[str], work_dirs: Dict[str, Path]) -> None:
    for scheme in run_schemes:
        wd = work_dirs[scheme]
        if not wd.exists():
            raise FileNotFoundError(
                f"Selected scheme '{scheme}' is missing work_dir: {wd}.\n"
                f"Either build that corpus first or remove it from --schemes."
            )


def _base_cmd(args: argparse.Namespace, *, scheme: str, work_dir: Path) -> List[str]:
    audit_mode = str(args.audit_mode)
    if scheme == "dirbucket_naive" and args.dirbucket_audit_mode is not None:
        audit_mode = str(args.dirbucket_audit_mode)
    if scheme == "kgw_ward" and args.kgw_audit_mode is not None:
        audit_mode = str(args.kgw_audit_mode)
    if scheme == "semstamp" and args.semstamp_audit_mode is not None:
        audit_mode = str(args.semstamp_audit_mode)
    if scheme == "sim_attr" and args.simattr_audit_mode is not None:
        audit_mode = str(args.simattr_audit_mode)
    if scheme == "overlap_attr" and args.overlap_audit_mode is not None:
        audit_mode = str(args.overlap_audit_mode)
    if scheme == "entail_attr" and args.entail_audit_mode is not None:
        audit_mode = str(args.entail_audit_mode)
    aligned_source_mode = str(args.dirbucket_aligned_source_mode)
    if scheme == "kgw_ward":
        aligned_source_mode = str(args.kgw_aligned_source_mode)
    if scheme == "semstamp":
        aligned_source_mode = str(args.semstamp_aligned_source_mode)
    if scheme == "sim_attr":
        aligned_source_mode = str(args.simattr_aligned_source_mode)
    if scheme == "overlap_attr":
        aligned_source_mode = str(args.overlap_aligned_source_mode)
    if scheme == "entail_attr":
        aligned_source_mode = str(args.entail_aligned_source_mode)

    cmd = [
        sys.executable,
        str(RUNNER),
        "--work_dir",
        str(work_dir),
        "--wm_scheme",
        scheme,
        "--audit_mode",
        audit_mode,
        "--retrieval_model",
        str(args.retrieval_model),
        "--openai_model",
        str(args.openai_model),
        "--answer_temp",
        str(args.answer_temp),
        "--z_threshold",
        str(args.z_threshold),
        "--min_pooled_T",
        str(args.min_pooled_T),
        "--aligned_source_mode",
        aligned_source_mode,
        "--global_source_query_sim_threshold",
        str(args.global_source_query_sim_threshold),
        "--global_source_max_docs",
        str(args.global_source_max_docs),
        "--global_source_max_docs_per_provider",
        str(args.global_source_max_docs_per_provider),
        "--global_source_selector",
        str(args.global_source_selector),
    ]

    if args.competitive:
        cmd.append("--competitive")
    else:
        cmd.append("--no-competitive")

    if scheme == "dirbucket_naive":
        cmd += [
            "--direction_scheme",
            str(args.dirbucket_direction_scheme),
            "--sim_threshold",
            str(args.sim_threshold),
            "--tie_margin",
            str(args.tie_margin),
        ]
        if args.dirbucket_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.dirbucket_alignment_manifest)]
    elif scheme == "kgw_ward":
        cmd += [
            "--kgw_seeding_scheme",
            str(args.kgw_seeding_scheme),
            "--kgw_gamma",
            str(args.kgw_gamma),
            "--kgw_delta",
            str(args.kgw_delta),
            "--kgw_n_min",
            str(args.kgw_n_min),
        ]
        if args.kgw_tokenizer_model:
            cmd += ["--kgw_tokenizer_model", str(args.kgw_tokenizer_model)]
        if args.kgw_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.kgw_alignment_manifest)]
        if args.kgw_ignore_repeated_ngrams:
            cmd.append("--kgw_ignore_repeated_ngrams")
        else:
            cmd.append("--no-kgw_ignore_repeated_ngrams")
    elif scheme == "semstamp":
        cmd += [
            "--sim_threshold",
            str(args.sim_threshold),
            "--tie_margin",
            str(args.tie_margin),
        ]
        if args.semstamp_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.semstamp_alignment_manifest)]
        if args.semstamp_centroids is not None:
            cmd += ["--semstamp_centroids", str(args.semstamp_centroids)]
        if args.semstamp_embed_model is not None:
            cmd += ["--semstamp_embed_model", str(args.semstamp_embed_model)]
        if args.semstamp_num_clusters is not None:
            cmd += ["--semstamp_num_clusters", str(args.semstamp_num_clusters)]
        if args.semstamp_gamma is not None:
            cmd += ["--semstamp_gamma", str(args.semstamp_gamma)]
        cmd += ["--semstamp_min_sent_words", str(args.semstamp_min_sent_words)]
    elif scheme == "sim_attr":
        if args.simattr_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.simattr_alignment_manifest)]
        cmd += [
            "--simattr_top_m",
            str(args.simattr_top_m),
            "--simattr_min_sent_words",
            str(args.simattr_min_sent_words),
        ]
    elif scheme == "overlap_attr":
        if args.overlap_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.overlap_alignment_manifest)]
        cmd += [
            "--overlap_top_m",
            str(args.overlap_top_m),
            "--overlap_ngram_n",
            str(args.overlap_ngram_n),
            "--overlap_min_sent_words",
            str(args.overlap_min_sent_words),
            "--overlap_idf_floor",
            str(args.overlap_idf_floor),
        ]
    elif scheme == "entail_attr":
        if args.entail_alignment_manifest is not None:
            cmd += ["--alignment_manifest", str(args.entail_alignment_manifest)]
        cmd += [
            "--entail_model",
            str(args.entail_model),
            "--entail_topk_passages_per_claim",
            str(args.entail_topk_passages_per_claim),
            "--entail_claim_min_words",
            str(args.entail_claim_min_words),
            "--entail_passage_min_words",
            str(args.entail_passage_min_words),
            "--entail_max_claims",
            str(args.entail_max_claims),
            "--entail_batch_size",
            str(args.entail_batch_size),
            "--entail_device",
            str(args.entail_device),
        ]

    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the paper experiment suite for DirBucket / KGW/WARD / SemStamp."
    )
    ap.add_argument("--work_dir_dirbucket", type=Path, default=Path("work/farad_dirbucket_naive"))
    ap.add_argument("--work_dir_kgw", type=Path, default=Path("work/farad_kgw_ward"))
    ap.add_argument("--work_dir_semstamp", type=Path, default=Path("work/farad_semstamp"))
    ap.add_argument("--work_dir_simattr", type=Path, default=None)
    ap.add_argument("--work_dir_overlap", type=Path, default=None)
    ap.add_argument("--work_dir_entail", type=Path, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("results/watermark_compare"))

    ap.add_argument("--schemes", nargs="+", choices=SCHEMES, default=["dirbucket_naive", "kgw_ward"])
    ap.add_argument("--overwrite_schemes", nargs="+", choices=SCHEMES, default=[])
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--providers", type=int, default=6)
    ap.add_argument("--target_provider", type=int, default=0)

    ap.add_argument("--trials_sanity", type=int, default=80)
    ap.add_argument("--trials_rag", type=int, default=80)
    ap.add_argument("--trials_style", type=int, default=30)
    ap.add_argument("--run_sanity", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run_rag_multi", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--run_rag_absent_target", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run_style", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--min_unique_providers", type=int, default=3)
    ap.add_argument("--force_cross_provider", action=argparse.BooleanOptionalAction, default=True)

    ap.add_argument("--retrieval_model", type=str, default="all-MiniLM-L6-v2")
    ap.add_argument("--openai_model", type=str, default="gpt-4o-mini")
    ap.add_argument("--answer_temp", type=float, default=0.55)

    ap.add_argument("--z_threshold", type=float, default=4.0)
    ap.add_argument("--min_pooled_T", type=int, default=10)
    ap.add_argument("--competitive", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--audit_mode", choices=["answer_only", "aligned"], default="answer_only")
    ap.add_argument("--dirbucket_audit_mode", choices=["answer_only", "aligned"], default=None)
    ap.add_argument("--kgw_audit_mode", choices=["answer_only", "aligned"], default=None)
    ap.add_argument("--semstamp_audit_mode", choices=["answer_only", "aligned"], default=None)
    ap.add_argument("--simattr_audit_mode", choices=["answer_only", "aligned"], default="aligned")
    ap.add_argument("--overlap_audit_mode", choices=["answer_only", "aligned"], default="aligned")
    ap.add_argument("--entail_audit_mode", choices=["answer_only", "aligned"], default="aligned")

    # DirBucket controls
    ap.add_argument("--dirbucket_direction_scheme", choices=["naive", "ortho"], default="naive")
    ap.add_argument("--sim_threshold", type=float, default=0.90)
    ap.add_argument("--tie_margin", type=float, default=0.05)
    ap.add_argument(
        "--dirbucket_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--dirbucket_alignment_manifest", type=Path, default=None)
    ap.add_argument(
        "--kgw_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--kgw_alignment_manifest", type=Path, default=None)
    ap.add_argument(
        "--semstamp_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--semstamp_alignment_manifest", type=Path, default=None)
    ap.add_argument(
        "--simattr_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--simattr_alignment_manifest", type=Path, default=None)
    ap.add_argument(
        "--overlap_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--overlap_alignment_manifest", type=Path, default=None)
    ap.add_argument(
        "--entail_aligned_source_mode",
        choices=["oracle", "global_corpus", "prompt_wm", "retrieved_src", "manifest_src"],
        default="oracle",
    )
    ap.add_argument("--entail_alignment_manifest", type=Path, default=None)
    ap.add_argument("--global_source_query_sim_threshold", type=float, default=0.55)
    ap.add_argument("--global_source_max_docs", type=int, default=120)
    ap.add_argument("--global_source_max_docs_per_provider", type=int, default=0)
    ap.add_argument("--global_source_selector", choices=["query", "answer", "max"], default="query")

    # KGW controls
    ap.add_argument("--kgw_tokenizer_model", type=str, default=None)
    ap.add_argument("--kgw_seeding_scheme", type=str, default="ff-position_prf-2-False-1548585")
    ap.add_argument("--kgw_gamma", type=float, default=0.25)
    ap.add_argument("--kgw_delta", type=float, default=3.5)
    ap.add_argument("--kgw_ignore_repeated_ngrams", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--kgw_n_min", type=int, default=3)
    ap.add_argument("--semstamp_centroids", type=Path, default=None)
    ap.add_argument("--semstamp_embed_model", type=str, default=None)
    ap.add_argument("--semstamp_num_clusters", type=int, default=None)
    ap.add_argument("--semstamp_gamma", type=float, default=None)
    ap.add_argument("--semstamp_min_sent_words", type=int, default=1)
    ap.add_argument("--simattr_top_m", type=int, default=3)
    ap.add_argument("--simattr_min_sent_words", type=int, default=4)
    ap.add_argument("--overlap_top_m", type=int, default=3)
    ap.add_argument("--overlap_ngram_n", type=int, default=3)
    ap.add_argument("--overlap_min_sent_words", type=int, default=4)
    ap.add_argument("--overlap_idf_floor", type=float, default=0.5)
    ap.add_argument("--entail_model", type=str, default="cross-encoder/nli-deberta-v3-base")
    ap.add_argument("--entail_topk_passages_per_claim", type=int, default=5)
    ap.add_argument("--entail_claim_min_words", type=int, default=4)
    ap.add_argument("--entail_passage_min_words", type=int, default=4)
    ap.add_argument("--entail_max_claims", type=int, default=8)
    ap.add_argument("--entail_batch_size", type=int, default=16)
    ap.add_argument("--entail_device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    sim_wd = args.work_dir_simattr if args.work_dir_simattr is not None else args.work_dir_dirbucket
    overlap_wd = args.work_dir_overlap if args.work_dir_overlap is not None else args.work_dir_dirbucket
    entail_wd = args.work_dir_entail if args.work_dir_entail is not None else args.work_dir_dirbucket
    work_dirs: Dict[str, Path] = {
        "dirbucket_naive": args.work_dir_dirbucket,
        "kgw_ward": args.work_dir_kgw,
        "semstamp": args.work_dir_semstamp,
        "sim_attr": sim_wd,
        "overlap_attr": overlap_wd,
        "entail_attr": entail_wd,
    }
    run_schemes = [s for s in SCHEMES if s in set(args.schemes)]
    overwrite_schemes = set(args.overwrite_schemes)
    _validate_work_dirs(run_schemes, work_dirs)

    for scheme in run_schemes:
        work_dir = work_dirs[scheme]
        scheme_out = args.out_dir / scheme
        scheme_out.mkdir(parents=True, exist_ok=True)
        scheme_overwrite = bool(args.overwrite or (scheme in overwrite_schemes))

        for seed in args.seeds:
            base = _base_cmd(args, scheme=scheme, work_dir=work_dir)
            if scheme_overwrite:
                # Force full recompute when overwriting, otherwise trial JSONL resume can
                # silently reuse old cumulative state.
                base.append("--no-resume_trials")

            # 1) Sanity (cooperative + unrelated), across all target providers.
            if bool(args.run_sanity) and int(args.trials_sanity) > 0:
                for t in range(args.providers):
                    out = scheme_out / f"sanity_coop_t{t}_seed{seed:02d}.json"
                    cmd = base + [
                        "--exp",
                        "sanity_single",
                        "--trials",
                        str(args.trials_sanity),
                        "--target_provider",
                        str(t),
                        "--answer_style",
                        "qa",
                        "--do_wrongkey",
                        "--seed",
                        str(seed),
                        "--out_json",
                        str(out),
                    ]
                    _maybe_run(out, cmd, overwrite=scheme_overwrite)

                    out = scheme_out / f"sanity_unrel_t{t}_seed{seed:02d}.json"
                    cmd = base + [
                        "--exp",
                        "sanity_single",
                        "--trials",
                        str(args.trials_sanity),
                        "--target_provider",
                        str(t),
                        "--answer_style",
                        "qa",
                        "--null_unrelated",
                        "--seed",
                        str(seed),
                        "--out_json",
                        str(out),
                    ]
                    _maybe_run(out, cmd, overwrite=scheme_overwrite)

            # 2) Multi-provider RAG, across all target providers (like sanity).
            # Filename includes target id so results can coexist and be aggregated later.
            if bool(args.run_rag_multi) and int(args.trials_rag) > 0:
                for t in range(args.providers):
                    out = scheme_out / f"rag_multi_t{t}_seed{seed:02d}.json"
                    cmd = base + [
                        "--exp",
                        "rag_multi",
                        "--trials",
                        str(args.trials_rag),
                        "--target_provider",
                        str(t),
                        "--answer_style",
                        "ownwords",
                        "--topk",
                        str(args.topk),
                        "--min_unique_providers",
                        str(args.min_unique_providers),
                        "--do_wrongkey",
                        "--log_traj",
                        "--seed",
                        str(seed),
                        "--out_json",
                        str(out),
                    ]
                    if args.force_cross_provider:
                        cmd.append("--force_cross_provider")
                    _maybe_run(out, cmd, overwrite=scheme_overwrite)

            # 3) Target-absent stress RAG (query from target doc, retrieval excludes target provider).
            if bool(args.run_rag_absent_target) and int(args.trials_rag) > 0:
                for t in range(args.providers):
                    out = scheme_out / f"rag_absent_t{t}_seed{seed:02d}.json"
                    cmd = base + [
                        "--exp",
                        "rag_absent_target",
                        "--trials",
                        str(args.trials_rag),
                        "--target_provider",
                        str(t),
                        "--answer_style",
                        "ownwords",
                        "--topk",
                        str(args.topk),
                        "--min_unique_providers",
                        str(args.min_unique_providers),
                        "--seed",
                        str(seed),
                        "--out_json",
                        str(out),
                    ]
                    if args.force_cross_provider:
                        cmd.append("--force_cross_provider")
                    _maybe_run(out, cmd, overwrite=scheme_overwrite)

            # 4) Style robustness sweep.
            if bool(args.run_style) and int(args.trials_style) > 0:
                out = scheme_out / f"style_sweep_seed{seed:02d}.json"
                cmd = base + [
                    "--exp",
                    "style_sweep",
                    "--trials",
                    str(args.trials_style),
                    "--target_provider",
                    str(args.target_provider),
                    "--topk",
                    str(args.topk),
                    "--min_unique_providers",
                    str(args.min_unique_providers),
                    "--do_wrongkey",
                    "--seed",
                    str(seed),
                    "--out_json",
                    str(out),
                ]
                if args.force_cross_provider:
                    cmd.append("--force_cross_provider")
                _maybe_run(out, cmd, overwrite=scheme_overwrite)

    print(f"[DONE] Results in: {args.out_dir}")


if __name__ == "__main__":
    main()
