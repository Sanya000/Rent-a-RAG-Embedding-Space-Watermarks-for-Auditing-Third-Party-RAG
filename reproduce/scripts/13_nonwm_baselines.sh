#!/usr/bin/env bash
# Non-watermark attribution heuristics: semantic similarity, lexical overlap, NLI entailment.
#
# Paper: fig:nonwm-critical-failure (Sec. 5.2, App. nonwm-calibration).
#   full  -> ${RUN_ROOT}/watermark_compare_nonwm_natural_full   (5 seeds; the figure)
#   pilot -> ${RUN_ROOT}/watermark_compare_nonwm_natural_pilot  (seed 1; in-text TPRs of
#            app:nonwm-calibration)
# Outputs: {sim_attr,overlap_attr,entail_attr}/{sanity_coop,sanity_unrel,rag_multi}_t{0-5}_seed{SS}.json
# Cost: per scheme and seed 360 sanity + 480 rag_multi answers -> full: 12,600 answers
#       (~25k calls, ~$13), pilot: 2,520 answers. entail_attr runs cross-encoder/nli-deberta-v3-base
#       on every claim/passage pair: use a GPU (several GPU-hours for the full run).
#
# Settings: global_source_selector=max, query floor 0.55, T_min 25.
# Knobs: VARIANTS (default "full pilot"), SCHEMES (default "sim_attr overlap_attr entail_attr"),
#        SEEDS (full; default 1..5), OVERWRITE=1, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

VARIANTS="${VARIANTS:-full pilot}"
SCHEMES="${SCHEMES:-sim_attr overlap_attr entail_attr}"
SEEDS="${SEEDS:-1 2 3 4 5}"

require_openai_key
require_path "${MANIFEST}" "${WORK_DIRBUCKET}/corpus.jsonl"
compare_common_args
for variant in ${VARIANTS}; do
    case "${variant}" in
        full) seeds="${SEEDS}" ;;
        pilot) seeds="1" ;;
        *) die "unknown variant '${variant}' (full|pilot)" ;;
    esac
    compare --out_dir "${RUN_ROOT}/watermark_compare_nonwm_natural_${variant}" \
        --schemes ${SCHEMES} --seeds ${seeds} \
        --trials_sanity 30 --trials_rag 80 --trials_style 0 \
        --topk 5 --min_unique_providers 3 --min_pooled_T 25 --kgw_n_min 5 \
        --simattr_aligned_source_mode global_corpus \
        --overlap_aligned_source_mode global_corpus \
        --entail_aligned_source_mode global_corpus \
        --global_source_query_sim_threshold 0.55 --global_source_selector max \
        --run_sanity --run_rag_multi --no-run_rag_absent_target --no-run_style
done
