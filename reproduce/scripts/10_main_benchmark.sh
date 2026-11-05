#!/usr/bin/env bash
# Primary benchmark matrix (Sec. 5.1-5.3): DirBucket vs KGW-adapted vs SemStamp-adapted.
#
# Paper items fed by this run (shipped as results/watermark_compare_global_tuned_mT20_kgwn3):
#   fig:wm-frontier-z-q2d            rag_multi_t{0-5}_seed{01-05}, all three schemes
#   fig:null-calibration(-single)    dirbucket_naive/sanity_{coop,unrel}_t{0-5}_seed{01-20}
#   tab:postselection-null           same 120 + 120 null audits
#   tab:baseline-diagnostics         {kgw_ward,semstamp}/sanity_coop_*, style_sweep_* (+ oracle row: 14_)
#   fig:benchmark_difficulty         dirbucket_naive rag_multi trial logs
#   fig:style-mixing-robustness-main baseline half (the stress half is 12_mixstress.sh)
#   tab:utility-fidelity             dirbucket_naive rag_multi trial logs
# Output: ${RUN_ROOT}/watermark_compare_global_tuned_mT20_kgwn3/{dirbucket_naive,kgw_ward,semstamp}/
#         *.json + *.trials.jsonl     (RUN_ROOT defaults to results/rerun)
#
# Cost (2 gpt-4o-mini calls per audited answer, roughly $1 per 1,000 answers):
#   STEPS=dirbucket   1,800 sanity + 2,400 rag_multi + 800 style-sweep = 5,000 answers
#   STEPS=null_extra  DirBucket null seeds 6-20: 5,400 answers
#   STEPS=kgw         5,000 answers (needs the Qwen2.5 tokenizer from HF)
#   STEPS=semstamp    5,000 answers
#   Total ~20k answers (~40k calls). Detection embeds every answer window with all-mpnet-base-v2;
#   use a GPU (CPU: ~10-30 s per answer).
#
# Per-run settings:
#   - query-similarity floor of the global_corpus alignment: 0.48 for DirBucket and KGW
#     (seeds 1-5), 0.55 for SemStamp and for the extra DirBucket null seeds 6-20.
#   - KGW sanity and style audits use T_min=20 (the "mT20" in the directory name); every
#     rag_multi audit and all other schemes use T_min=25. KGW uses n_min=3 ("kgwn3").
#   - SemStamp uses the K=8 corpus work/farad_semstamp_k8_relaxed.
# The non-watermark heuristics of fig:nonwm-critical-failure are run by 13_nonwm_baselines.sh.
#
# Knobs: STEPS (default "dirbucket null_extra kgw semstamp"), SEEDS (default "1 2 3 4 5"),
#        NULL_SEEDS (default 6..20), OUT_DIR, OVERWRITE=1, DRY_RUN=1.
# Runs can be split across machines by calling the script with different SEEDS / STEPS.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEPS="${STEPS:-dirbucket null_extra kgw semstamp}"
SEEDS="${SEEDS:-1 2 3 4 5}"
NULL_SEEDS="${NULL_SEEDS:-6 7 8 9 10 11 12 13 14 15 16 17 18 19 20}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/watermark_compare_global_tuned_mT20_kgwn3}"

require_openai_key
require_path "${MANIFEST}"
compare_common_args

for step in ${STEPS}; do
    case "${step}" in
        dirbucket)
            require_path "${WORK_DIRBUCKET}/corpus.jsonl"
            compare --out_dir "${OUT_DIR}" --schemes dirbucket_naive --seeds ${SEEDS} \
                --trials_sanity 30 --trials_rag 80 --trials_style 40 \
                --topk 5 --min_unique_providers 3 --min_pooled_T 25 \
                --dirbucket_aligned_source_mode global_corpus \
                --global_source_query_sim_threshold 0.48 --global_source_selector query \
                --run_sanity --run_rag_multi --no-run_rag_absent_target --run_style
            ;;
        null_extra)
            require_path "${WORK_DIRBUCKET}/corpus.jsonl"
            compare --out_dir "${OUT_DIR}" --schemes dirbucket_naive --seeds ${NULL_SEEDS} \
                --trials_sanity 30 --trials_rag 80 --trials_style 0 \
                --topk 5 --min_unique_providers 3 --min_pooled_T 25 \
                --dirbucket_aligned_source_mode global_corpus \
                --global_source_query_sim_threshold 0.55 --global_source_selector query \
                --run_sanity --no-run_rag_multi --no-run_rag_absent_target --no-run_style
            ;;
        kgw)
            require_path "${WORK_KGW}/corpus.jsonl"
            kgw_args=(--out_dir "${OUT_DIR}" --schemes kgw_ward --seeds ${SEEDS}
                --trials_sanity 30 --trials_rag 80 --trials_style 40
                --topk 5 --min_unique_providers 3 --kgw_n_min 3
                --kgw_aligned_source_mode global_corpus
                --global_source_query_sim_threshold 0.48 --global_source_selector query
                --no-run_rag_absent_target)
            compare "${kgw_args[@]}" --min_pooled_T 20 --run_sanity --no-run_rag_multi --run_style
            compare "${kgw_args[@]}" --min_pooled_T 25 --no-run_sanity --run_rag_multi --no-run_style
            ;;
        semstamp)
            require_path "${WORK_SEMSTAMP}/corpus.jsonl" "${WORK_SEMSTAMP}/semstamp_centroids.npz"
            compare --out_dir "${OUT_DIR}" --schemes semstamp --seeds ${SEEDS} \
                --trials_sanity 30 --trials_rag 80 --trials_style 40 \
                --topk 5 --min_unique_providers 3 --min_pooled_T 25 \
                --semstamp_aligned_source_mode global_corpus \
                --global_source_query_sim_threshold 0.55 --global_source_selector query \
                --run_sanity --run_rag_multi --no-run_rag_absent_target --run_style
            ;;
        *) die "unknown step '${step}' (dirbucket|null_extra|kgw|semstamp)" ;;
    esac
done

[[ "${DRY_RUN}" == "1" ]] || [[ ! -d "${OUT_DIR}/dirbucket_naive" ]] || \
    run "${PYTHON}" experiments/summarize_rag_multi_dirbucket.py \
        --results_dir "${OUT_DIR}/dirbucket_naive" --out_json "${OUT_DIR}/dirbucket_naive_rag_multi_summary.json"
