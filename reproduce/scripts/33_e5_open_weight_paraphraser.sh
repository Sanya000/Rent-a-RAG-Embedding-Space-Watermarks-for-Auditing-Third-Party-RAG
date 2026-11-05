#!/usr/bin/env bash
# E5 open-weight paraphraser: Llama-3.1-8B-Instruct vs gpt-4o-mini on the same 50 documents.
#
# Paper: tab:paraphrase-model (App. app:model-robustness, "Paraphrase model").
# Steps (STEPS, default "manifest serve llama ref compare"):
#   manifest  work/e5_sub50_manifest.jsonl = first 50 FARAD manifest docs by sha256("e5:"+id)
#             (byte-identical to the released one).
#   serve     start an OpenAI-compatible server for ${MODEL} on 127.0.0.1:${PORT} unless one
#             already answers at ${E5_BASE_URL}. SERVER=shim (default; transformers, single
#             stream: ~5.4 h on one A100-40GB) or SERVER=vllm (`vllm serve`, much faster,
#             different sampler). The server is stopped on exit.
#   llama     build work/e5_llama8b at K=24, rho=3 through that endpoint (no OpenAI cost).
#   ref       build the gpt-4o-mini reference work/e5_ref_gpt4omini with identical settings
#             (~50 x 12 x 3 calls, under $1).
#   compare   gain / semantic similarity / changed fraction -> ${OUT}
# Paraphrasing is sampled; bundle `corpora_ext` holds the released corpora, and `STEPS=compare`
# on them regenerates compare.json.
# Knobs: STEPS, MODEL, PORT, SERVER, E5_BASE_URL, OUT, DRY_RUN=1.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

STEPS="${STEPS:-manifest serve llama ref compare}"
MODEL="${MODEL:-NousResearch/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-8123}"
SERVER="${SERVER:-shim}"
E5_BASE_URL="${E5_BASE_URL:-http://127.0.0.1:${PORT}/v1}"
E5_MANIFEST="${E5_MANIFEST:-work/e5_sub50_manifest.jsonl}"
OUT="${OUT:-${RUN_ROOT}/e5_paraphraser/compare.json}"
BUILD=(--manifest "${E5_MANIFEST}" --providers 6 --max_docs 50 --max_chars 4000
       --rounds 3 --k_batch 24 --mmr_device cpu --resume)
SERVER_PID=""

server_up() { curl -sf "${E5_BASE_URL%/v1}/health" >/dev/null 2>&1 || curl -sf "${E5_BASE_URL}/models" >/dev/null 2>&1; }
cleanup() { [[ -z "${SERVER_PID}" ]] || kill "${SERVER_PID}" 2>/dev/null || true; }
trap cleanup EXIT

for step in ${STEPS}; do
    case "${step}" in
        manifest)
            require_path "${MANIFEST}"
            run "${PYTHON}" experiments/make_e5_manifest.py --manifest "${MANIFEST}" \
                --out "${E5_MANIFEST}" --n_docs 50 ;;
        serve)
            [[ "${DRY_RUN}" == "1" ]] && { echo "+ (start ${SERVER} server for ${MODEL} on port ${PORT})"; continue; }
            if server_up; then echo "[serve] reusing endpoint ${E5_BASE_URL}"; continue; fi
            SERVER_LOG="$(dirname "${OUT}")/server.log"
            mkdir -p "$(dirname "${OUT}")"
            case "${SERVER}" in
                shim) "${PYTHON}" -u experiments/e5_shim_server.py --model "${MODEL}" --port "${PORT}" \
                          > "${SERVER_LOG}" 2>&1 & ;;
                vllm) vllm serve "${MODEL}" --port "${PORT}" --max-model-len 4096 \
                          > "${SERVER_LOG}" 2>&1 & ;;
                *) die "SERVER must be shim or vllm" ;;
            esac
            SERVER_PID=$!
            for _ in $(seq 1 180); do server_up && break; sleep 10; done
            server_up || die "server did not come up; see ${SERVER_LOG}"
            echo "[serve] ${MODEL} ready at ${E5_BASE_URL}" ;;
        llama)
            require_path "${E5_MANIFEST}"
            run env OPENAI_BASE_URL="${E5_BASE_URL}" OPENAI_API_KEY=local CUDA_VISIBLE_DEVICES= \
                "${PYTHON}" -u experiments/build_corpora_dirbucket.py "${BUILD[@]}" \
                --out_dir work/e5_llama8b --workers 6 --openai_model "${MODEL}" ;;
        ref)
            require_openai_key
            require_path "${E5_MANIFEST}"
            run env CUDA_VISIBLE_DEVICES= "${PYTHON}" -u experiments/build_corpora_dirbucket.py \
                "${BUILD[@]}" --out_dir work/e5_ref_gpt4omini --workers 8 --openai_model gpt-4o-mini ;;
        compare)
            require_path work/e5_llama8b/corpus.jsonl work/e5_ref_gpt4omini/corpus.jsonl
            run "${PYTHON}" -u experiments/e5_compare_paraphrasers.py \
                --corpora llama8b:work/e5_llama8b,gpt4omini:work/e5_ref_gpt4omini \
                --manifest "${E5_MANIFEST}" --out "${OUT}" ;;
        *) die "unknown step ${step} (manifest, serve, llama, ref, compare)" ;;
    esac
done
