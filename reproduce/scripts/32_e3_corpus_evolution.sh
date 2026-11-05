#!/usr/bin/env bash
# E3 corpus evolution (staleness): audit stale answers against an evolved reference corpus.
#
# Paper: tab:corpus-evolution (App. app:corpus-evolution).
# Steps (STEPS, default "build replay merge"):
#   build   rewrite a nested fraction f in {10, 25, 50}% of the deployed docs without watermark
#           -> work/staleness/f{10,25,50}/. Rewrites are cached in work/staleness/evolved_rewrites.json
#           (bundle `corpora_ext`); with the cache this is API-free and byte-identical to the paper
#           corpora, without it 200 gpt-4o-mini calls (non-deterministic).
#   replay  replay the primary run's logged answers (5 seeds x 6 targets x 80) through the exact
#           primary detector with the evolved corpus as reference. No API; CPU-bound (roughly
#           1 h per condition and seed on CPU). One job per condition can run in parallel
#           (CONDITIONS="f25"); finished (condition, seed) replays are skipped on restart.
#   merge   combine the per-condition summaries -> ${OUT_DIR}/staleness_full_summary.json.
# f00 replays the primary audit exactly (median Z 12.30).
# Without the rewrite cache, build stops instead of silently calling the API; set ALLOW_API=1
# to regenerate the rewrites with gpt-4o-mini.
# Knobs: STEPS, CONDITIONS (default "f00 f10 f25 f50"), SEEDS, OUT_DIR, ALLOW_API, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

TRIAL_LOGS="${TRIAL_LOGS:-results/watermark_compare_global_tuned_mT20_kgwn3/dirbucket_naive}"
STEPS="${STEPS:-build replay merge}"
CONDITIONS="${CONDITIONS:-f00 f10 f25 f50}"
SEEDS="${SEEDS:-01,02,03,04,05}"
WORK_STALE="${WORK_STALE:-work/staleness}"
OUT_DIR="${OUT_DIR:-${RUN_ROOT}/staleness_full}"

for step in ${STEPS}; do
    case "${step}" in
        build)
            require_path "${WORK_DIRBUCKET}/corpus.jsonl"
            if [[ ! -f "${WORK_STALE}/evolved_rewrites.json" ]]; then
                echo "[note] ${WORK_STALE}/evolved_rewrites.json not found (bundle corpora_ext); rebuilding it needs the API (200 gpt-4o-mini calls)" >&2
                [[ "${ALLOW_API:-0}" == "1" || "${DRY_RUN}" == "1" ]] \
                    || die "download bundle corpora_ext (scripts/download_artifacts.sh corpora_ext) for the API-free path, or set ALLOW_API=1 to call the API"
                require_openai_key
            fi
            run "${PYTHON}" -u experiments/build_evolved_corpus.py \
                --src_work_dir "${WORK_DIRBUCKET}" --out_root "${WORK_STALE}" \
                --fractions 0.10,0.25,0.50 --model gpt-4o-mini --temperature 0.55 ;;
        replay)
            require_path "${TRIAL_LOGS}/rag_multi_t0_seed01.trials.jsonl"
            for tag in ${CONDITIONS}; do
                if [[ "${tag}" == "f00" ]]; then wd="${WORK_DIRBUCKET}"; else wd="${WORK_STALE}/${tag}"; fi
                require_path "${wd}/corpus.jsonl"
                run "${PYTHON}" -u experiments/staleness_reaudit.py \
                    --conditions "${tag}:${wd}" --trial_logs_dir "${TRIAL_LOGS}" \
                    --seeds "${SEEDS}" --max_trials 80 --out_dir "${OUT_DIR}/${tag}"
            done ;;
        merge)
            run "${PYTHON}" -u experiments/staleness_reaudit.py --merge --out_dir "${OUT_DIR}"
            run "${PYTHON}" experiments/paper_appendix_tables.py corpus-evolution \
                --results_dir "$(dirname "${OUT_DIR}")" ;;
        *) die "unknown step ${step} (build, replay, merge)" ;;
    esac
done
