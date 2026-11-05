#!/usr/bin/env bash
# E2 selective watermarking: per-document keyed fraction c of watermarked sentences.
#
# Paper: tab:coverage (App. app:coverage); realized coverage 0.78 / 0.53 / 0.30.
# Steps (STEPS, default "build audit table"):
#   build  splice the deployed corpus with its source sentences (MiniLM alignment, no API)
#          -> work/coverage/c{25,50,75}/; byte-identical to the released corpora (bundle
#          `corpora_ext`).
#   audit  per coverage level 5 seeds x 6 targets x 60 answers with the paper's audit wrapper
#          (3 x 1,800 answers, ~10.8k gpt-4o-mini calls, roughly $5).
#   table  per-level summaries + the cross-coverage table -> ${OUT_DIR}/coverage_full_summary.json
# Knobs: STEPS, COVS (default "c25 c50 c75"), SEEDS, TARGETS, TRIALS, OUT_DIR, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEPS="${STEPS:-build audit table}"
COVS="${COVS:-c25 c50 c75}"
SEEDS="${SEEDS:-1,2,3,4,5}"
TARGETS="${TARGETS:-0,1,2,3,4,5}"
TRIALS="${TRIALS:-60}"
WORK_COV="${WORK_COV:-work/coverage}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/coverage}"

for step in ${STEPS}; do
    case "${step}" in
        build)
            require_path "${WORK_DIRBUCKET}/corpus.jsonl" "${MANIFEST}"
            run "${PYTHON}" -u experiments/build_coverage_corpus.py \
                --src_work_dir "${WORK_DIRBUCKET}" --manifest "${MANIFEST}" --out_root "${WORK_COV}" \
                --coverages 0.25,0.50,0.75 \
                --align_model sentence-transformers/all-MiniLM-L6-v2 --align_threshold 0.50 ;;
        audit)
            require_openai_key
            for cov in ${COVS}; do
                require_path "${WORK_COV}/${cov}/corpus.jsonl"
                run "${PYTHON}" -u experiments/run_paper_rag_multi_dirbucket.py \
                    --work_dir "${WORK_COV}/${cov}" --alignment_manifest "${MANIFEST}" \
                    --out_dir "${OUT_DIR}/${cov}" --seeds "${SEEDS}" --targets "${TARGETS}" \
                    --trials "${TRIALS}" --openai_model gpt-4o-mini --skip_existing
            done ;;
        table)
            for cov in ${COVS}; do
                require_path "${OUT_DIR}/${cov}/dirbucket_naive"
                run "${PYTHON}" -u experiments/summarize_rag_multi_dirbucket.py \
                    --results_dir "${OUT_DIR}/${cov}/dirbucket_naive" \
                    --out_json "${OUT_DIR}/${cov}/paper_rag_multi_summary_full.json"
            done
            run "${PYTHON}" experiments/paper_appendix_tables.py coverage --write \
                --results_dir "$(dirname "${OUT_DIR}")" ;;
        *) die "unknown step ${step} (build, audit, table)" ;;
    esac
done
