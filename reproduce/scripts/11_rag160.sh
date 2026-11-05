#!/usr/bin/env bash
# Long-horizon DirBucket audits: 10 seeds x 6 targets x 160 answers (60 audits).
#
# Paper: fig:evidence-accumulation-main (Sec. 5.4) and the headline numbers in Sec. 5.4 and the
# abstract ("median target Z crosses 4 after 10 answers; every audit detected within 23").
# Output: ${RUN_ROOT}/watermark_compare_global_tuned_mT20_kgwn3_rag160/dirbucket_naive/
#         rag_multi_t{0-5}_seed{01-10}.json + .trials.jsonl
# Cost: 9,600 answers (~19k gpt-4o-mini calls, roughly $10); GPU recommended for detection.
#
# Settings: query-similarity floor 0.55, T_min 25, topk 5, >=3 providers per bundle.
# Knobs: SEEDS (default 1..10), OUT_DIR, OVERWRITE=1, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEEDS="${SEEDS:-1 2 3 4 5 6 7 8 9 10}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/watermark_compare_global_tuned_mT20_kgwn3_rag160}"

require_openai_key
require_path "${MANIFEST}" "${WORK_DIRBUCKET}/corpus.jsonl"
compare_common_args
compare --out_dir "${OUT_DIR}" --schemes dirbucket_naive --seeds ${SEEDS} \
    --trials_sanity 30 --trials_rag 160 --trials_style 0 \
    --topk 5 --min_unique_providers 3 --min_pooled_T 25 \
    --dirbucket_aligned_source_mode global_corpus \
    --global_source_query_sim_threshold 0.55 --global_source_selector query \
    --no-run_sanity --run_rag_multi --no-run_rag_absent_target --no-run_style
