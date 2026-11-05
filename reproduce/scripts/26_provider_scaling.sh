#!/usr/bin/env bash
# Marketplace-size scaling harness (controlled exposure, forced target inclusion).
#
# Paper: tab:marketplace-scaling (App. app:marketplace-robustness) and Sec. 5.6.
# Output: ${OUT_DIR}/m<m>/{corpus.jsonl, provider_keys.json, summary*.json}
# Cost per m: re-watermarks 20 docs/provider at rho=1, K=24 (2 gpt-4o-mini calls per sentence)
# and generates n_targets x 80 answers.
#
# RUNS lists "m:n_targets:T_min" triples (default "12:4:0 20:5:25", the runs in
# results/provider_scaling; T_min 0 = auto max(5, round(25*6/m)), i.e. 12 for m = 12).
# RUNS="6:6:25 12:6:25 20:6:25" AUDIT_SEEDS="42 43 44 45 46" runs 6 targets x 5 audit seeds
# per m (re-audits reuse the built corpus; each extra audit seed costs n_targets x 80 answers).
# Knobs: RUNS, AUDIT_SEEDS (default: none = --seed 42 only), OUT_DIR, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

RUNS="${RUNS:-12:4:0 20:5:25}"
AUDIT_SEEDS="${AUDIT_SEEDS:-}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/provider_scaling}"

require_openai_key
require_path "${MANIFEST}"
for spec in ${RUNS}; do
    IFS=':' read -r m n_targets tmin <<< "${spec}"
    common=(--manifest "${MANIFEST}" --out_dir "${OUT_DIR}" --m_providers "${m}"
            --docs_per_provider 20 --n_trials 80 --n_targets "${n_targets}" --T_min "${tmin}"
            --doc_max_chars 1500 --model gpt-4o-mini --concurrency 20 --seed 42)
    if [[ -z "${AUDIT_SEEDS}" ]]; then
        run "${PYTHON}" -u experiments/provider_scaling_experiment.py "${common[@]}"
    else
        for a in ${AUDIT_SEEDS}; do
            run "${PYTHON}" -u experiments/provider_scaling_experiment.py "${common[@]}" \
                --audit_seed "${a}" --summary_name "summary_tmin${tmin}_audit${a}.json"
        done
    fi
done
run "${PYTHON}" experiments/paper_appendix_tables.py marketplace-scaling --results_dir "$(dirname "${OUT_DIR}")"
