#!/usr/bin/env bash
# Cross-benchmark support-gate (T_min) calibration (E6).
#
# Paper: tab:tmin-crossbenchmark (App. app:sensitivity). Pure post-processing of the stored
# per-run JSONs (shipped in results/): the gate + BH rule is re-applied over a T_min grid to the
# 30 primary FARAD audits and the 30 real-domain (E1) audits, including the wrong-key null.
# Output: ${OUT}. No API, no models; seconds.
# Knobs: PRIMARY_DIR, MULTIDOMAIN_DIR, OUT, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

PRIMARY_DIR="${PRIMARY_DIR:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
MULTIDOMAIN_DIR="${MULTIDOMAIN_DIR:-results/multidomain_full/dirbucket_naive}"
OUT="${OUT:-${RUN_ROOT}/e6_tmin/summary_cameraready.json}"

require_path "${PRIMARY_DIR}/rag_multi_t0_seed01.json" "${MULTIDOMAIN_DIR}/rag_multi_t0_seed01.json"
run "${PYTHON}" -u experiments/e6_tmin_crossbenchmark.py \
    --primary_dir "${PRIMARY_DIR}" --multidomain_dir "${MULTIDOMAIN_DIR}" --out "${OUT}"
