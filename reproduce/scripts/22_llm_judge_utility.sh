#!/usr/bin/env bash
# Blind GPT-4o judge of watermarking utility at sentence and answer level.
#
# Paper: app:llm-judge (sentence/answer paragraphs) and the Sec. 5.8 judge sentence.
# Input:  results/watermark_cost_ablation/candidates_debug.jsonl (bundle `appendix_aux`, or
#         scripts/23_cost_ablation.sh), the primary trial logs, manifest and corpus.
# Output: ${OUT_DIR}/judge_results.json
# Cost: 200 sentence + 90 answer judgements and 90 source-answer generations with GPT-4o
#       (roughly $2).
#
# Settings (results/llm_judge_gpt4o): --n_sent 200 --n_per_target 15 --model gpt-4o --seed 42,
# i.e. 200 sentence pairs and 90 answer pairs. The sentence pairs are deterministic given the
# candidates log. The source answer is regenerated with the judge model (gpt-4o) and the
# watermarked answer is the logged gpt-4o-mini answer; SOURCE_ANSWER_MODEL=gpt-4o-mini makes
# the comparison symmetric.
# Knobs: OUT_DIR, JUDGE_MODEL, N_SENT, N_PER_TARGET, SOURCE_ANSWER_MODEL, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
DEBUG_LOG="${DEBUG_LOG:-results/watermark_cost_ablation/candidates_debug.jsonl}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/llm_judge_gpt4o}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o}"
N_SENT="${N_SENT:-200}"
N_PER_TARGET="${N_PER_TARGET:-15}"
EXTRA=()
[[ -z "${SOURCE_ANSWER_MODEL:-}" ]] || EXTRA=(--source_answer_model "${SOURCE_ANSWER_MODEL}")

require_openai_key
require_path "${DEBUG_LOG}" "${MANIFEST}" "${WORK_DIRBUCKET}/corpus.jsonl" \
    "${TRIAL_LOGS}/rag_multi_t0_seed01.trials.jsonl"
run "${PYTHON}" -u experiments/llm_judge_utility.py \
    --debug_log "${DEBUG_LOG}" --trial_logs_dir "${TRIAL_LOGS}" \
    --manifest "${MANIFEST}" --corpus "${WORK_DIRBUCKET}/corpus.jsonl" \
    --out_dir "${OUT_DIR}" --n_sent "${N_SENT}" --n_per_target "${N_PER_TARGET}" \
    --model "${JUDGE_MODEL}" --concurrency 15 --seed 42 "${EXTRA[@]}"
run "${PYTHON}" experiments/paper_appendix_tables.py llm-judge \
    --results_dir "$(dirname "${OUT_DIR}")" --judge_dir "$(basename "${OUT_DIR}")"
