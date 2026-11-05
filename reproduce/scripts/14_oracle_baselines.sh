#!/usr/bin/env bash
# Oracle-aligned diagnostic: every scheme aligned against the documents actually in the prompt.
#
# Paper: tab:baseline-diagnostics (oracle row) and app:baseline-adaptations.
# Output: ${RUN_ROOT}/watermark_compare_oracle_aligned_baselines/<scheme>/
#         {sanity_coop,sanity_unrel,rag_multi}_t{0-5}_seed{01-05}.json, style_sweep_seed{01-05}.json
# Cost: 5,000 answers per scheme (~10k calls, ~$5 each). The three non-watermark heuristics can
#       be added via SCHEMES (entail_attr needs a GPU).
#
# Settings: aligned_source_mode oracle for every scheme, T_min 25, KGW n_min 5, SemStamp K=8.
# Knobs: SCHEMES (default "dirbucket_naive kgw_ward semstamp"), SEEDS, OVERWRITE=1, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCHEMES="${SCHEMES:-dirbucket_naive kgw_ward semstamp}"
SEEDS="${SEEDS:-1 2 3 4 5}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/watermark_compare_oracle_aligned_baselines}"

require_openai_key
require_path "${WORK_DIRBUCKET}/corpus.jsonl"
compare_common_args
compare --out_dir "${OUT_DIR}" --schemes ${SCHEMES} --seeds ${SEEDS} \
    --trials_sanity 30 --trials_rag 80 --trials_style 40 \
    --topk 5 --min_unique_providers 3 --min_pooled_T 25 --kgw_n_min 5 \
    --dirbucket_aligned_source_mode oracle --kgw_aligned_source_mode oracle \
    --semstamp_aligned_source_mode oracle --simattr_aligned_source_mode oracle \
    --overlap_aligned_source_mode oracle --entail_aligned_source_mode oracle \
    --global_source_query_sim_threshold 0.55 --global_source_selector query \
    --run_sanity --run_rag_multi --no-run_rag_absent_target --run_style
