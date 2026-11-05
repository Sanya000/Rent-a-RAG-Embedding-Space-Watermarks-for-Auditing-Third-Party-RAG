#!/usr/bin/env bash
# Build the KGW-adapted and SemStamp-adapted corpus views used as watermark baselines.
#
# Paper: Sec. 5.1 ("three watermarked versions of the same corpus"), App. B.3 (baseline adaptations).
# Outputs:
#   work/farad_kgw_ward/{corpus.jsonl,provider_keys.json}            KGW, Qwen2.5-1.5B-Instruct,
#       gamma 0.25, delta 3.5, seeding ff-position_prf-2-False-1548585, 3 shards over the manifest
#   work/farad_semstamp_k8_relaxed/{corpus.jsonl,provider_keys.json,semstamp_centroids.npz}
#       SemStamp, K=8 k-means clusters on all-mpnet-base-v2, gamma 0.25, 4 shards over the manifest
# Cost:
#   KGW: local generation only (no API), ~400 docs x <=512 new tokens on one GPU (a few GPU-hours).
#   SemStamp: rejection-sampled gpt-4o-mini paraphrasing, ~100 h wall time over 4 shards
#       (~16 min per document) and tens of thousands of calls (order of $10-30).
#
# Generation is sampled; the released corpora are in bundle `corpora`
# (scripts/download_artifacts.sh corpora).
#
# SemStamp settings: K=8, rounds 6, k_batch 32, k_stage 24/12, min_sem_sim 0.75,
# cluster_margin 0.0, gen_min_sent_words 4. The released SemStamp corpus holds 386 documents.
#
# Knobs: BASELINES ("kgw semstamp"), KGW_SHARD_IDS ("0 1 2"), SEMSTAMP_SHARD_IDS ("0 1 2 3"),
#        DEVICE (auto|cuda|cpu). Run subsets of shards on different GPUs, then rerun with the
#        shard list empty to merge.
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

BASELINES="${BASELINES:-kgw semstamp}"
KGW_SHARD_IDS="${KGW_SHARD_IDS-0 1 2}"
SEMSTAMP_SHARD_IDS="${SEMSTAMP_SHARD_IDS-0 1 2 3}"
DEVICE="${DEVICE:-auto}"

require_path "${MANIFEST}"

shards_complete() {  # shards_complete <dir> <num_shards>
    local sid
    for ((sid = 0; sid < $2; sid++)); do
        [[ -s "$1/corpus.shard$(printf '%02d' "${sid}").jsonl" ]] || return 1
    done
}

build_kgw() {
    local out="${WORK_KGW}"
    if [[ -s "${out}/corpus.jsonl" && "${OVERWRITE}" != "1" ]]; then
        echo "[skip] ${out}/corpus.jsonl exists"; return
    fi
    mkdir -p "${out}"
    local sid
    for sid in ${KGW_SHARD_IDS}; do
        run "${PYTHON}" -u experiments/build_corpora_kgw.py \
            --num_shards 3 --shard_id "${sid}" \
            --manifest "${MANIFEST}" --out_dir "${out}" \
            --providers 6 --seed 13 --max_docs 400 \
            --wm_model Qwen/Qwen2.5-1.5B-Instruct --device "${DEVICE}" --dtype auto \
            --seeding_scheme ff-position_prf-2-False-1548585 --gamma 0.25 --delta 3.5 \
            --temperature 1.0 --top_p 1.0 --max_new_tokens 512 \
            --do_sample --ignore_repeated_ngrams --resume
    done
    if shards_complete "${out}" 3 || [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" -u experiments/build_corpora_kgw.py \
            --manifest "${MANIFEST}" --out_dir "${out}" --num_shards 3 --merge_shards
    else
        echo "[info] ${out}: not all 3 shards are present yet; merge skipped"
    fi
}

build_semstamp() {
    local out="${WORK_SEMSTAMP}"
    if [[ -s "${out}/corpus.jsonl" && "${OVERWRITE}" != "1" ]]; then
        echo "[skip] ${out}/corpus.jsonl exists"; return
    fi
    mkdir -p "${out}"
    local common=(
        --manifest "${MANIFEST}" --out_dir "${out}"
        --providers 6 --seed 13 --max_docs 400
        --embed_model sentence-transformers/all-mpnet-base-v2
        --num_clusters 8 --gamma 0.25 --device "${DEVICE}"
        --centroids_path "${out}/semstamp_centroids.npz" --centroid_min_words 4
        --gen_strategy batch --rounds 6 --k_batch 32 --k_stage1 24 --k_stage2 12
        --min_sem_sim 0.75 --cluster_margin 0.0 --gen_min_sent_words 4
        --keep_if_already_green --no-force_accept_last_round
        --openai_model gpt-4o-mini
    )
    run "${PYTHON}" -u experiments/build_corpora_semstamp.py "${common[@]}" --train_centroids_only
    local sid
    if [[ -n "${SEMSTAMP_SHARD_IDS}" ]]; then
        require_openai_key
        for sid in ${SEMSTAMP_SHARD_IDS}; do
            run "${PYTHON}" -u experiments/build_corpora_semstamp.py "${common[@]}" \
                --num_shards 4 --shard_id "${sid}" --resume
        done
    fi
    if shards_complete "${out}" 4 || [[ "${DRY_RUN}" == "1" ]]; then
        run "${PYTHON}" -u experiments/build_corpora_semstamp.py \
            --manifest "${MANIFEST}" --out_dir "${out}" --num_shards 4 --merge_shards
    else
        echo "[info] ${out}: not all 4 shards are present yet; merge skipped"
    fi
}

for b in ${BASELINES}; do
    case "${b}" in
        kgw) build_kgw ;;
        semstamp) build_semstamp ;;
        *) die "unknown baseline '${b}' (kgw|semstamp)" ;;
    esac
done
