#!/usr/bin/env bash
# GPT-4o judge of laundered answers vs the original answer.
#
# Paper: tab:laundering-utility (App. app:laundering-utility) and the Sec. 5.5 sentence on
# completeness and preference at 25% compression.
# Input:  seeds 4-5 laundering cache (bundle `laundering_cache`) and the primary trial logs.
# Output: ${OUT_DIR}/utility_judge_results.json
# Cost: 4 conditions x 2 seeds x 6 targets x 10 = 480 GPT-4o calls (roughly $2).
#
# Settings: --n_per_pair 10 --seeds 4,5 --targets 0-5 --judge_model gpt-4o (120 per condition).
# Only the four laundering conditions are judged (the original answer is the reference).
# Knobs: CACHE_DIR, OUT_DIR, JUDGE_MODEL, N_PER_PAIR, SEEDS, TARGETS, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
CACHE_DIR="${CACHE_DIR:-results/laundering_attack_seeds45/answer_cache}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/laundering_attack}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
N_PER_PAIR="${N_PER_PAIR:-10}"
SEEDS="${SEEDS:-4,5}"
TARGETS="${TARGETS:-0,1,2,3,4,5}"

require_openai_key
require_path "${CACHE_DIR}/compress25" "${TRIAL_LOGS}/rag_multi_t0_seed04.trials.jsonl"
run "${PYTHON}" -u experiments/laundering_utility_judge.py \
    --cache_dir "${CACHE_DIR}" --trial_logs_dir "${TRIAL_LOGS}" \
    --conditions neutral,aggressive,compress50,compress25 \
    --seeds "${SEEDS}" --targets "${TARGETS}" --n_per_pair "${N_PER_PAIR}" \
    --judge_model "${JUDGE_MODEL}" --concurrency 20 --out_dir "${OUT_DIR}"
run "${PYTHON}" experiments/paper_appendix_tables.py laundering-utility \
    --results_dir "$(dirname "${OUT_DIR}")"
