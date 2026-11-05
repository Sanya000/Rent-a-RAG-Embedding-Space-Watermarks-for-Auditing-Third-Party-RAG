#!/usr/bin/env bash
# Model robustness: alternative embedding encoders and answer generators.
#
# Paper: tab:model-robustness (App. app:model-robustness).
#   STEPS=minilm  Part A, all-MiniLM-L6-v2: re-watermarks the 400-doc FARAD set at rho=1 in the
#                 encoder's space (~2 gpt-4o-mini calls per sentence) + 4 x 80 audited answers
#                 -> ${OUT_DIR}/part_a_minilm/summary_a.json
#   STEPS=gtr     Part A, sentence-transformers/gtr-t5-base, same protocol
#                 -> ${OUT_DIR_GTR}/part_a_minilm/summary_a.json (+ summary_gtr.json copy)
#   STEPS=gpt35   Part B, gpt-3.5-turbo answers on the primary corpus, 4 x 40 answers
#                 -> ${OUT_DIR}/part_b_gpt4o/summary_b.json
#   STEPS=qwen    Part C, local Qwen2.5-1.5B-Instruct answers (transformers, GPU advised,
#                 no API key), 4 x 40 answers -> ${OUT_DIR}/part_c_qwen/summary_c.json
# The "mpnet (primary)" row comes from scripts/20_laundering.sh (original, seeds 1-3).
# Knobs: STEPS (default "minilm gtr gpt35 qwen"), OUT_DIR, OUT_DIR_GTR, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
STEPS="${STEPS:-minilm gtr gpt35 qwen}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/model_robustness}"
OUT_DIR_GTR="${OUT_DIR_GTR:-${RUN_ROOT}/model_robustness_gtr}"
COMMON=(--manifest "${MANIFEST}" --wm_corpus "${WORK_DIRBUCKET}/corpus.jsonl"
        --wm_keys "${WORK_DIRBUCKET}/provider_keys.json" --trial_logs_dir "${TRIAL_LOGS}"
        --wm_model gpt-4o-mini --seed 42)

require_path "${MANIFEST}" "${WORK_DIRBUCKET}/corpus.jsonl" "${TRIAL_LOGS}/rag_multi_t0_seed01.trials.jsonl"
for step in ${STEPS}; do
    case "${step}" in
        minilm)
            require_openai_key
            run "${PYTHON}" -u experiments/model_robustness_check.py "${COMMON[@]}" \
                --out_dir "${OUT_DIR}" --parts A --concurrency 20 \
                --part_a_embed sentence-transformers/all-MiniLM-L6-v2 --part_a_rho 1 \
                --part_a_ndocs 400 --part_a_trials 80 --part_a_ntargets 4 ;;
        gtr)
            require_openai_key
            run "${PYTHON}" -u experiments/model_robustness_check.py "${COMMON[@]}" \
                --out_dir "${OUT_DIR_GTR}" --parts A --concurrency 20 \
                --part_a_embed sentence-transformers/gtr-t5-base --part_a_rho 1 \
                --part_a_ndocs 400 --part_a_trials 80 --part_a_ntargets 4
            run cp "${OUT_DIR_GTR}/part_a_minilm/summary_a.json" "${OUT_DIR_GTR}/summary_gtr.json" ;;
        gpt35)
            require_openai_key
            run "${PYTHON}" -u experiments/model_robustness_check.py "${COMMON[@]}" \
                --out_dir "${OUT_DIR}" --parts B --concurrency 15 --part_b_model gpt-3.5-turbo ;;
        qwen)
            run "${PYTHON}" -u experiments/model_robustness_check.py "${COMMON[@]}" \
                --out_dir "${OUT_DIR}" --parts C --part_c_hf_id Qwen/Qwen2.5-1.5B-Instruct ;;
        *) die "unknown step ${step} (minilm, gtr, gpt35, qwen)" ;;
    esac
done
run "${PYTHON}" experiments/paper_appendix_tables.py model-robustness --results_dir "$(dirname "${OUT_DIR}")"
