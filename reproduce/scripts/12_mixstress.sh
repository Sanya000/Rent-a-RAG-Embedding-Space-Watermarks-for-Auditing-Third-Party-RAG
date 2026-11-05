#!/usr/bin/env bash
# Provider-mixing stress run: DirBucket with 9 retrieved documents from >=5 providers, 160 answers.
#
# Paper: fig:style-mixing-robustness-main (stress half; the baseline half is the primary matrix
# of 10_main_benchmark.sh) and the App. marketplace-robustness text.
# Output: ${RUN_ROOT}/watermark_compare_global_mixstress_k9_u5/dirbucket_naive/rag_multi_t{t}_seed{SS}.json
# Cost: 5 seeds x 6 targets x 160 = 4,800 answers (~9.6k calls, ~$5).
#
# The shipped results/ directory holds target 0 with seeds 1-5 and targets 1-5 with seeds 2, 4, 5.
# Settings: topk 9, min_unique_providers 5, query floor 0.48, T_min 25.
# Knobs: SEEDS (default 1..5), OUT_DIR, OVERWRITE=1, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEEDS="${SEEDS:-1 2 3 4 5}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/watermark_compare_global_mixstress_k9_u5}"

require_openai_key
require_path "${MANIFEST}" "${WORK_DIRBUCKET}/corpus.jsonl"
compare_common_args
compare --out_dir "${OUT_DIR}" --schemes dirbucket_naive --seeds ${SEEDS} \
    --trials_sanity 0 --trials_rag 160 --trials_style 0 \
    --topk 9 --min_unique_providers 5 --min_pooled_T 25 \
    --dirbucket_aligned_source_mode global_corpus \
    --global_source_query_sim_threshold 0.48 --global_source_selector query \
    --no-run_sanity --run_rag_multi --no-run_rag_absent_target --no-run_style
