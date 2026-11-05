#!/usr/bin/env bash
# Regenerate the Sec. 5 figures and tables (TikZ, PDFs and table numbers) from result files.
#
# No API calls; CPU only (the utility step embeds queries with all-MiniLM-L6-v2, ~minutes).
# Inputs: the shipped summaries in results/ plus, for trial-log based items, the bundles
#         trial_logs_main / trial_logs_baselines and the corpora bundle (scripts/download_artifacts.sh).
#
# Paper item                       -> output (under ${FIGS_DIR}, default results/figs)
#   fig:wm-frontier-z-q2d           watermark_compare_global_tuned_mT20_kgwn3/fig_wm_frontier_z_q2d_tikz.tex
#   fig:null-calibration-single     watermark_compare_global_tuned_mT20_kgwn3/fig_null_calibration_panel_a_only_tikz.tex
#   fig:null-calibration            watermark_compare_global_tuned_mT20_kgwn3/fig_null_calibration_tikz.tex
#   fig:style-mixing-robustness-main  .../fig_style_mixing_robustness_main_tikz.tex      [trial logs]
#   fig:evidence-accumulation-main  watermark_compare_global_tuned_mT20_kgwn3_rag160/fig_evidence_accumulation_main_tikz.tex
#                                   (also prints first_q_median_target_z_ge_4 and q_all_detected)
#   fig:benchmark_difficulty        watermark_compare_global_tuned_mT20_kgwn3/fig_benchmark_task_difficulty.pdf [trial logs]
#   fig:nonwm-critical-failure      watermark_compare_nonwm_natural_full/fig_nonwm_critical_failure.pdf [trial logs]
#   tab:postselection-null          paper_tables/postselection_null.json   (resampling rows: needs trial logs)
#   tab:baseline-diagnostics        paper_tables/baseline_diagnostics.json
#   tab:ortho-ablation              paper_tables/ortho_ablation.json (+ fig_compare_*.png)
#   tab:utility-fidelity            watermark_compare_global_tuned_mT20_kgwn3/dirbucket_utility_extra_stats.json [trial logs + corpora]
#
# With the default RESULTS_ROOT=results the outputs replace the tracked files in results/figs;
# `git diff results/figs` compares them with the shipped versions.
#
# Knobs: RESULTS_ROOT (e.g. results/rerun after the 1x_*.sh runs), FIGS_DIR,
#        STEPS (default "tikz tables matplotlib utility"), DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RESULTS_ROOT="${RESULTS_ROOT:-results}"
FIGS_DIR="${FIGS_DIR:-${RESULTS_ROOT}/figs}"
STEPS="${STEPS:-tikz tables matplotlib utility}"

MAIN="${RESULTS_ROOT}/watermark_compare_global_tuned_mT20_kgwn3"
RAG160="${RESULTS_ROOT}/watermark_compare_global_tuned_mT20_kgwn3_rag160"
MIXSTRESS="${RESULTS_ROOT}/watermark_compare_global_mixstress_k9_u5"
NONWM_FULL="${RESULTS_ROOT}/watermark_compare_nonwm_natural_full"
ORACLE="${RESULTS_ROOT}/watermark_compare_oracle_aligned_baselines"
DIRECTION="${RESULTS_ROOT}/direction_compare"
MAIN_FIGS="${FIGS_DIR}/$(basename "${MAIN}")"
TABLES="${FIGS_DIR}/paper_tables"

has_trial_logs() {  # has_trial_logs <dir>: any *.trials.jsonl below <dir>
    [[ -d "$1" ]] && [[ -n "$(find "$1" -name '*.trials.jsonl' -print -quit)" ]]
}
skip() { echo "[skip] $*"; }

step_tikz() {
    run "${PYTHON}" experiments/export_paper_tikz_extra.py frontier \
        --results_root "${MAIN}" --out "${MAIN_FIGS}/fig_wm_frontier_z_q2d_tikz.tex"
    # The evidence figure applies BH over the T_min-eligible providers (--bh_family eligible,
    # the default); --bh_family all uses the runner's rule (ineligible providers enter with p=1).
    run "${PYTHON}" experiments/export_paper_tikz_extra.py evidence \
        --results_root "${RAG160}" --out "${FIGS_DIR}/$(basename "${RAG160}")/fig_evidence_accumulation_main_tikz.tex"
    run "${PYTHON}" experiments/export_null_calibration_tikz.py \
        --results-root "${MAIN}" \
        --out "${MAIN_FIGS}/fig_null_calibration_tikz.tex" \
        --out-panel-a-only "${MAIN_FIGS}/fig_null_calibration_panel_a_only_tikz.tex"
    if has_trial_logs "${MAIN}" && has_trial_logs "${MIXSTRESS}" || [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" experiments/export_style_mixing_robustness_tikz.py \
            --results_root "${MAIN}" --out_dir "${MAIN_FIGS}" \
            --out "${MAIN_FIGS}/fig_style_mixing_robustness_main_tikz.tex" \
            --mixing_stress_root "${MIXSTRESS}"
    else
        skip "style-mixing TikZ: needs trial logs (bundle trial_logs_main)"
    fi
}

step_tables() {
    run "${PYTHON}" experiments/export_paper_tikz_extra.py postselection-null \
        --results_root "${MAIN}" --out "${TABLES}/postselection_null.json"
    run "${PYTHON}" experiments/export_paper_tikz_extra.py baseline-diagnostics \
        --results_root "${MAIN}" --oracle_root "${ORACLE}" --out "${TABLES}/baseline_diagnostics.json"
    run "${PYTHON}" experiments/export_paper_tikz_extra.py ortho-table \
        --results_root "${DIRECTION}" --out "${TABLES}/ortho_ablation.json"
    run "${PYTHON}" experiments/summarize_rag_multi_dirbucket.py \
        --results_dir "${MAIN}/dirbucket_naive" --out_json "${TABLES}/dirbucket_rag_multi_summary.json"
}

step_matplotlib() {
    # The --work_dir_* corpora only feed the paraphrase-intensity and utility panels.
    run "${PYTHON}" experiments/aggregate_watermark_compare_and_plot.py \
        --results_root "${MAIN}" --out_dir "${MAIN_FIGS}" \
        --schemes dirbucket_naive kgw_ward semstamp sim_attr overlap_attr entail_attr \
        --work_dir_dirbucket "${WORK_DIRBUCKET}" --work_dir_kgw "${WORK_KGW}" \
        --work_dir_semstamp "${WORK_SEMSTAMP}" \
        --work_dir_simattr "${WORK_DIRBUCKET}" --work_dir_overlap "${WORK_DIRBUCKET}" \
        --work_dir_entail "${WORK_DIRBUCKET}" \
        --manifest_src "${MANIFEST}" \
        --mixing_stress_root "${MIXSTRESS}"
    run "${PYTHON}" experiments/aggregate_watermark_compare_and_plot.py \
        --results_root "${RAG160}" --out_dir "${FIGS_DIR}/$(basename "${RAG160}")" \
        --schemes dirbucket_naive --work_dir_dirbucket "${WORK_DIRBUCKET}" --manifest_src "${MANIFEST}"
    if has_trial_logs "${NONWM_FULL}" || [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" experiments/aggregate_watermark_compare_and_plot.py \
            --results_root "${NONWM_FULL}" --out_dir "${FIGS_DIR}/$(basename "${NONWM_FULL}")" \
            --schemes sim_attr overlap_attr entail_attr --work_dir_dirbucket "${WORK_DIRBUCKET}" \
            --manifest_src "${MANIFEST}"
    else
        skip "non-watermark calibration figure: needs trial logs (bundle trial_logs_baselines)"
    fi
    run "${PYTHON}" experiments/aggregate_direction_compare_and_plot.py \
        --results_root "${DIRECTION}" --out_dir "${FIGS_DIR}"
}

step_utility() {
    if has_trial_logs "${MAIN}/dirbucket_naive" || [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" experiments/compute_dirbucket_utility_diagnostics.py \
            --results_root "${MAIN}" --manifest_src "${MANIFEST}" \
            --work_dir_dirbucket "${WORK_DIRBUCKET}" \
            --output "${MAIN_FIGS}/dirbucket_utility_extra_stats.json"
    else
        skip "utility diagnostics: needs trial logs (bundle trial_logs_main) and corpora"
    fi
}

for step in ${STEPS}; do
    case "${step}" in
        tikz) step_tikz ;;
        tables) step_tables ;;
        matplotlib) step_matplotlib ;;
        utility) step_utility ;;
        *) die "unknown step '${step}' (tikz|tables|matplotlib|utility)" ;;
    esac
done
