#!/usr/bin/env bash
# Auditor hyperparameter sensitivity (tau_sim, tau_margin, T_min, W) on the stored answers.
#
# Paper: tab:sensitivity and app:sensitivity (324 combinations).
# Output: ${OUT_DIR}/{sensitivity_results.json (180 combos), sensitivity_extended.json (324),
#         SENSITIVITY_NOTE.md, tmin_sweep_dense.json, figures/*.pdf}
# No API calls. Phase 1 embeds every reference/answer window of seeds 1-3 once per W with
# all-mpnet-base-v2 (GPU: minutes; CPU: about 1 h per W); phase 2 is seconds per combination.
#
# The sweep re-applies the aligned detector with references merged per provider (as the
# primary harness does); at the operating point it gives the primary audit's median Z (11.85
# on seeds 1-3). The 30-audit T_min rows of tab:tmin-crossbenchmark come from scripts/25_tmin.sh.
# Knobs: SEEDS, TARGETS, OUT_DIR, STEPS (default "grid tmin plot"), DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
SEEDS="${SEEDS:-1,2,3}"
TARGETS="${TARGETS:-0,1,2,3,4,5}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/sensitivity_sweep}"
STEPS="${STEPS:-grid tmin plot}"
COMMON=(--work_dir "${WORK_DIRBUCKET}" --trial_logs_dir "${TRIAL_LOGS}"
        --seeds "${SEEDS}" --targets "${TARGETS}"
        --global_sim_threshold 0.48 --global_max_docs 120 --batch_size 512 --alpha 0.05)

require_path "${WORK_DIRBUCKET}/emb_text_wm.npz" "${TRIAL_LOGS}/rag_multi_t0_seed01.trials.jsonl"
for step in ${STEPS}; do
    case "${step}" in
        grid) run "${PYTHON}" -u experiments/detector_sensitivity_sweep.py --grid base,extended \
                  "${COMMON[@]}" --out_dir "${OUT_DIR}" ;;
        tmin) run "${PYTHON}" -u experiments/tmin_sweep_dense.py "${COMMON[@]}" \
                  --out "${OUT_DIR}/tmin_sweep_dense.json" ;;
        plot) run "${PYTHON}" experiments/plot_sensitivity_heatmap.py \
                  --extended "${OUT_DIR}/sensitivity_extended.json" \
                  --original "${OUT_DIR}/sensitivity_results.json" \
                  --tmin_dense "${OUT_DIR}/tmin_sweep_dense.json" --out_dir "${OUT_DIR}/figures" ;;
        *) die "unknown step ${step} (grid, tmin, plot)" ;;
    esac
done
run "${PYTHON}" experiments/paper_appendix_tables.py sensitivity --results_dir "$(dirname "${OUT_DIR}")"
