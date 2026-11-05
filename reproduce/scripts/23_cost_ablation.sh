#!/usr/bin/env bash
# Watermark cost-strength ablation over K in {6,12,24} x rho in {1,3,6}.
#
# Paper: tab:cost-strength (App. app:cost-strength); its candidates log also feeds
# scripts/22_llm_judge_utility.sh.
# Output: ${OUT_DIR}/{candidates_debug.jsonl, ablation_results.json, ABLATION_NOTE.md}
#
# Phase 1 (API) generates rho=6 rounds of K=24 gpt-4o-mini candidates for N_DOCS documents
# (default 54, ~16k calls, a few dollars); phases 2-3 replay every (K, rho) offline.
# SIMULATE_ONLY=1 (default when the candidates log exists, e.g. from bundle `appendix_aux`)
# re-runs only the deterministic offline phases.
# "Target Z" is gain(K,rho)/gain(24,6) x 11.85, the primary audit's median Z on seeds 1-3.
# Without the candidates log the script stops instead of silently calling the API; set
# ALLOW_API=1 (or SIMULATE_ONLY=0) to regenerate it with gpt-4o-mini.
# Knobs: OUT_DIR, SIMULATE_ONLY, ALLOW_API, N_DOCS, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SRC_LOG="${SRC_LOG:-results/watermark_cost_ablation/candidates_debug.jsonl}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/watermark_cost_ablation}"
N_DOCS="${N_DOCS:-54}"
if [[ -z "${SIMULATE_ONLY:-}" ]]; then
    if [[ -f "${SRC_LOG}" ]]; then
        SIMULATE_ONLY=1
    else
        echo "[note] ${SRC_LOG} not found (bundle appendix_aux); phase 1 would regenerate it with the API (~16k gpt-4o-mini calls)" >&2
        [[ "${ALLOW_API:-0}" == "1" || "${DRY_RUN}" == "1" ]] \
            || die "download bundle appendix_aux (scripts/download_artifacts.sh appendix_aux) for the API-free path, or set ALLOW_API=1 to call the API"
        SIMULATE_ONLY=0
    fi
fi

require_path "${MANIFEST}" "${WORK_DIRBUCKET}/provider_keys.json"
EXTRA=()
if [[ "${SIMULATE_ONLY}" == "1" ]]; then
    require_path "${SRC_LOG}"
    run mkdir -p "${OUT_DIR}"
    cmp -s "${SRC_LOG}" "${OUT_DIR}/candidates_debug.jsonl" \
        || run cp -f "${SRC_LOG}" "${OUT_DIR}/candidates_debug.jsonl"
    EXTRA=(--simulate_only)
else
    require_openai_key
fi
run "${PYTHON}" -u experiments/watermark_cost_ablation.py \
    --manifest "${MANIFEST}" --work_dir "${WORK_DIRBUCKET}" --out_dir "${OUT_DIR}" \
    --n_docs "${N_DOCS}" --seed 42 --model gpt-4o-mini --concurrency 20 --z_anchor 11.85 \
    "${EXTRA[@]}"
run "${PYTHON}" experiments/paper_appendix_tables.py cost-strength --results_dir "$(dirname "${OUT_DIR}")"
