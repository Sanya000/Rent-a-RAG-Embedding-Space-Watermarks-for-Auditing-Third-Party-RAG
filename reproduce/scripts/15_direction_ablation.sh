#!/usr/bin/env bash
# Direction ablation: keyed naive Gaussian directions vs per-provider orthogonalised directions.
#
# Paper: tab:ortho-ablation (App. ortho_eval).
# Needs: work/farad_dirbucket_naive and work/farad_dirbucket_ortho
#        (build the ortho corpus with SCHEMES=ortho scripts/02_build_dirbucket.sh).
# Output: ${RUN_ROOT}/direction_compare/{naive,ortho}/{sanity_coop,sanity_unrel}_t{0-5}_seed{SS}.json,
#         rag_multi_seed{SS}.json, style_sweep_seed{SS}.json; figures in ${RUN_ROOT}/figs/fig_compare_*
#         and the table numbers via experiments/export_paper_tikz_extra.py ortho-table.
# Cost: per scheme 5 seeds x (360 sanity + 80 rag_multi + 160 style) = 3,000 answers
#       (~6k calls, ~$3); GPU recommended.
#
# Settings: target provider 0, tau_sim 0.90, tie margin 0.05, T_min 10, competitive assignment,
# answers aligned to the prompted documents (--audit_mode aligned --aligned_source_mode oracle).
# Knobs: SEEDS, SCHEMES (default "naive ortho"), OVERWRITE=1, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SEEDS="${SEEDS:-1 2 3 4 5}"
SCHEMES="${SCHEMES:-naive ortho}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/direction_compare}"

require_openai_key
require_path "${WORK_DIRBUCKET}/corpus.jsonl" "${WORK_ORTHO}/corpus.jsonl"
ow="$(overwrite_flag)"
run "${PYTHON}" -u experiments/run_direction_scheme_compare.py \
    --work_dir_naive "${WORK_DIRBUCKET}" --work_dir_ortho "${WORK_ORTHO}" \
    --out_dir "${OUT_DIR}" \
    --trials_rag 80 --trials_style 40 --trials_sanity 30 \
    --seeds ${SEEDS} --schemes ${SCHEMES} \
    --sim_threshold 0.90 --competitive --tie_margin 0.05 --min_pooled_T 10 \
    --audit_mode aligned --aligned_source_mode oracle ${ow}
run "${PYTHON}" -u experiments/aggregate_direction_compare_and_plot.py \
    --results_root "${OUT_DIR}" --out_dir "${RUN_ROOT}/figs"
