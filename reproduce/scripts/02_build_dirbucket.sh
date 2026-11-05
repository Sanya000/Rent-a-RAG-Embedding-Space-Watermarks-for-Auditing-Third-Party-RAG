#!/usr/bin/env bash
# Build the DirBucket provider corpora (naive directions; optionally the orthogonalised ablation).
#
# Paper: Sec. 4 / App. A.1 (K=24 candidates, rho=6 rounds, tau_sem=0.70, delta_min=5e-4, h=96,
#        12 MMR-selected sentences per document); tab:ortho-ablation for SCHEMES="naive ortho".
# Outputs: work/farad_dirbucket_naive/{corpus.jsonl,corpus.shard0{0..3}.jsonl,provider_keys.json}
#          work/farad_dirbucket_ortho/... (only with SCHEMES containing ortho)
# Cost: one gpt-4o-mini call per sentence and round (24 paraphrases each), i.e. at most
#       400 docs x 12 sentences x 6 rounds ~ 29k calls per corpus (order of $10); a few GPU-hours
#       for the all-mpnet-base-v2 embeddings (CPU works but is much slower).
#
# Paraphrasing is sampled, so every build yields a new watermarked corpus. The released corpus is
# in bundle `corpora` (scripts/download_artifacts.sh corpora).
#
# Documents are selected from --farad_dir with 4 shards: the shard quota (100 docs per shard,
# shard = sha256("shard:13:<id>") mod 4) fixes the 400-document set (ids 0000-0445). Keep
# NUM_SHARDS=4 for the paper's document set; provider ids and keys (random.Random(13)) are
# deterministic.
#
# Knobs: SCHEMES ("naive" | "naive ortho"), SHARD_IDS (default "0 1 2 3": run a subset per
#        machine/GPU, then rerun with SHARD_IDS="" to merge), WORKERS (default 4).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

SCHEMES="${SCHEMES:-naive}"
NUM_SHARDS="${NUM_SHARDS:-4}"
SHARD_IDS="${SHARD_IDS-0 1 2 3}"
WORKERS="${WORKERS:-4}"

[[ "${NUM_SHARDS}" == "4" ]] || echo "[warn] NUM_SHARDS=${NUM_SHARDS}: the document set will differ from the paper's (4 shards)" >&2
[[ "${DRY_RUN}" == "1" || -d "${FARAD_DIR}" ]] \
    || die "missing ${FARAD_DIR}/: run scripts/00_get_farad.sh first (FARAD is not redistributed)"

for scheme in ${SCHEMES}; do
    case "${scheme}" in
        naive) out_dir="${WORK_DIRBUCKET}" ;;
        ortho) out_dir="${WORK_ORTHO}" ;;
        *) die "unknown scheme '${scheme}' (naive|ortho)" ;;
    esac
    if [[ -s "${out_dir}/corpus.jsonl" && "${OVERWRITE}" != "1" ]]; then
        echo "[skip] ${out_dir}/corpus.jsonl exists (OVERWRITE=1 to rebuild)"
        continue
    fi
    if [[ -n "${SHARD_IDS}" ]]; then
        require_openai_key
        mkdir -p "${out_dir}"
        for sid in ${SHARD_IDS}; do
            run "${PYTHON}" -u experiments/build_corpora_dirbucket.py \
                --out_dir "${out_dir}" \
                --direction_scheme "${scheme}" \
                --num_shards "${NUM_SHARDS}" --shard_id "${sid}" \
                --farad_dir "${FARAD_DIR}" \
                --providers 6 --max_docs 400 --seed 13 \
                --rewrite_mode summary --summary_max_sents 12 --summary_select mmr --mmr_device auto \
                --workers "${WORKERS}" \
                --gen_mode batch --rounds 6 --k_batch 24 --min_sem_sim 0.70 --min_delta_proj 5e-4 \
                --openai_model gpt-4o-mini \
                --timing_log --timing_every 1 \
                --resume
        done
    fi
    complete=1
    for ((sid = 0; sid < NUM_SHARDS; sid++)); do
        [[ -s "${out_dir}/corpus.shard$(printf '%02d' "${sid}").jsonl" ]] || complete=0
    done
    if [[ "${complete}" == "1" || "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" -u experiments/build_corpora_dirbucket.py \
            --farad_dir "${FARAD_DIR}" --out_dir "${out_dir}" \
            --num_shards "${NUM_SHARDS}" --max_docs 400 --merge_shards
    else
        echo "[info] ${out_dir}: not all ${NUM_SHARDS} shards are present yet; merge skipped"
    fi
done
