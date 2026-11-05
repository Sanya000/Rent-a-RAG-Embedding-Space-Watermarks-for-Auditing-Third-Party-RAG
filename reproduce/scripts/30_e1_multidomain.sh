#!/usr/bin/env bash
# E1 real-domain benchmark: clinical / CTI / legal, 2 providers per domain, 300 documents.
#
# Paper: tab:multidomain (App. app:multidomain), Sec. 5 real-domain paragraph, and the real-domain
# rows of tab:tmin-crossbenchmark.
# Steps (STEPS, default "manifest build audit analyze"):
#   manifest  fetch the three HF datasets through the datasets-server rows API and build
#             work/multidomain_manifest_p6_d300.jsonl; checked against the shipped text-free
#             record configs/e1_multidomain_manifest_record.jsonl (ids + sha256). Needs network.
#   build     watermark it at the primary setting (K=24, rho=6, gpt-4o-mini): ~300 x 12 x 6
#             calls (order of $5-10); LLM paraphrasing is not deterministic.
#   audit     5 seeds x 6 targets x 80 answers with the paper's audit wrapper (4,800 answers,
#             ~9.6k gpt-4o-mini calls, roughly $5; GPU recommended for detection).
#   analyze   summary + per-domain / sibling analysis + E6 T_min sweep (no API).
# Outputs: ${OUT_DIR}/{dirbucket_naive/, paper_rag_multi_summary.json, e1_domain_analysis.json},
#          ${RUN_ROOT}/e6_tmin/summary_cameraready.json
#
# Licences: qiaojin/PubMedQA is MIT; guychuk/cyber-threat-intelligence-reports and
# joelniklaus/legal_case_document_summarization list no licence on the HF Hub. The manifest,
# corpus and trial logs derived from them are not redistributed; only numeric summaries are.
# Knobs: STEPS, SEEDS, TARGETS, TRIALS, WORK_E1, MANIFEST_E1, OUT_DIR, WORKERS, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEPS="${STEPS:-manifest build audit analyze}"
SEEDS="${SEEDS:-1,2,3,4,5}"
TARGETS="${TARGETS:-0,1,2,3,4,5}"
TRIALS="${TRIALS:-80}"
WORKERS="${WORKERS:-24}"
MANIFEST_E1="${MANIFEST_E1:-work/multidomain_manifest_p6_d300.jsonl}"
WORK_E1="${WORK_E1:-work/multidomain_dirbucket_naive}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/multidomain_full}"
RECORD="configs/e1_multidomain_manifest_record.jsonl"

for step in ${STEPS}; do
    case "${step}" in
        manifest)
            run "${PYTHON}" -u experiments/build_manifest_multidomain.py \
                --out_manifest "${MANIFEST_E1}" --docs_per_provider 50 \
                --min_chars 1000 --max_chars 4000 --check_record "${RECORD}" ;;
        build)
            require_openai_key
            require_path "${MANIFEST_E1}"
            run "${PYTHON}" -u experiments/build_corpora_dirbucket.py \
                --manifest "${MANIFEST_E1}" --out_dir "${WORK_E1}" \
                --providers 6 --max_docs 300 --max_chars 4000 \
                --rounds 6 --k_batch 24 --workers "${WORKERS}" \
                --openai_model gpt-4o-mini --mmr_device cpu --direction_scheme naive --resume ;;
        audit)
            require_openai_key
            require_path "${MANIFEST_E1}" "${WORK_E1}/corpus.jsonl"
            run "${PYTHON}" -u experiments/run_paper_rag_multi_dirbucket.py \
                --work_dir "${WORK_E1}" --alignment_manifest "${MANIFEST_E1}" \
                --out_dir "${OUT_DIR}" --seeds "${SEEDS}" --targets "${TARGETS}" \
                --trials "${TRIALS}" --openai_model gpt-4o-mini --skip_existing ;;
        analyze)
            require_path "${OUT_DIR}/dirbucket_naive"
            run "${PYTHON}" -u experiments/summarize_rag_multi_dirbucket.py \
                --results_dir "${OUT_DIR}/dirbucket_naive" \
                --out_json "${OUT_DIR}/paper_rag_multi_summary.json"
            run "${PYTHON}" -u experiments/cr_e1_domain_analysis.py \
                --run_dir "${OUT_DIR}/dirbucket_naive" --out "${OUT_DIR}/e1_domain_analysis.json"
            run "${PYTHON}" -u experiments/e6_tmin_crossbenchmark.py \
                --multidomain_dir "${OUT_DIR}/dirbucket_naive" \
                --out "${RUN_ROOT}/e6_tmin/summary_cameraready.json"
            run "${PYTHON}" experiments/paper_appendix_tables.py multidomain \
                --results_dir "$(dirname "${OUT_DIR}")" ;;
        *) die "unknown step ${step} (manifest, build, audit, analyze)" ;;
    esac
done
