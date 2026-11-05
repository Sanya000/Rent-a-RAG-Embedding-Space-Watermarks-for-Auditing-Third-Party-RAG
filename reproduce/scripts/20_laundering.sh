#!/usr/bin/env bash
# Post-answer laundering attack: 6 conditions x 5 seeds x 6 targets (30 audits per condition).
#
# Paper: tab:answer-variation (Sec. 5.5); the "mpnet (primary)" row of tab:model-robustness
# is the `original` condition on seeds 1-3.
# Output: ${RUN_ROOT}/laundering_attack/laundering_results.json        (seeds 1-3)
#         ${RUN_ROOT}/laundering_attack_seeds45/laundering_results.json (seeds 4-5)
#         and the pooled 30-audit table printed by paper_appendix_tables.py.
#
# Default mode is an API-free re-score: transforms are read from the `laundering_cache` bundle
# (results/laundering_attack{,_seeds45}/answer_cache) and only the detector is re-run
# (CPU: roughly 1-3 s per answer, 1,440 answers per condition for seeds 1-3; GPU recommended).
# Without a cache the transforms are generated with ${LAUNDER_MODEL} (default gpt-4o-mini,
# temperature 0.3): about 4,800 calls per seed group.
#
# Settings:
#   - two invocations (seeds 1-3 and 4-5) with separate caches, pooled afterwards;
#   - CACHE_KEY=question (default): transforms are keyed on sha256(question)[:16], so trials that
#     repeat a question share one transform. CACHE_KEY=trial keys on question+answer (this
#     needs new transforms, i.e. API calls);
#   - neutral2 = neutral applied to the cached neutral answer.
# Knobs: TARGETS, CONDITIONS, SEEDS_A/SEEDS_B, CACHE_A/CACHE_B, OUT_A/OUT_B, TRIAL_LOGS,
#        LAUNDER_MODEL, CACHE_KEY, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
TARGETS="${TARGETS:-0,1,2,3,4,5}"
CONDITIONS="${CONDITIONS:-original,neutral,aggressive,compress50,compress25,neutral2}"
SEEDS_A="${SEEDS_A:-1,2,3}"
SEEDS_B="${SEEDS_B:-4,5}"
CACHE_A="${CACHE_A:-results/laundering_attack/answer_cache}"
CACHE_B="${CACHE_B:-results/laundering_attack_seeds45/answer_cache}"
OUT_A="${OUT_A:-${RUN_ROOT}/laundering_attack}"
OUT_B="${OUT_B:-${RUN_ROOT}/laundering_attack_seeds45}"
LAUNDER_MODEL="${LAUNDER_MODEL:-gpt-4o-mini}"
CACHE_KEY="${CACHE_KEY:-question}"

require_path "${WORK_DIRBUCKET}/corpus.jsonl" "${WORK_DIRBUCKET}/emb_text_wm.npz" \
    "${TRIAL_LOGS}/rag_multi_t0_seed01.trials.jsonl"

for spec in "${SEEDS_A}|${CACHE_A}|${OUT_A}" "${SEEDS_B}|${CACHE_B}|${OUT_B}"; do
    IFS='|' read -r seeds cache out <<< "${spec}"
    [[ -d "${cache}" ]] || require_openai_key
    run "${PYTHON}" -u experiments/post_answer_laundering_attack.py \
        --work_dir "${WORK_DIRBUCKET}" --trial_logs_dir "${TRIAL_LOGS}" \
        --seeds "${seeds}" --targets "${TARGETS}" --conditions "${CONDITIONS}" \
        --out_dir "${out}" --cache_dir "${cache}" --cache_key "${CACHE_KEY}" \
        --openai_model "${LAUNDER_MODEL}" --concurrency 20 \
        --sim_threshold 0.82 --tie_margin 0.05 --T_min 25 --alpha 0.05 \
        --global_sim_threshold 0.48 --global_max_docs 120 --win_words 28 --stride_words 14
done

run "${PYTHON}" experiments/paper_appendix_tables.py answer-variation \
    --results_dir "$(dirname "${OUT_A}")" \
    --launder_dirs "$(basename "${OUT_A}"),$(basename "${OUT_B}")"
