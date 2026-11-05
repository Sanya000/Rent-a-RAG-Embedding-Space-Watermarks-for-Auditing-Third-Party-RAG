# Shared settings for reproduce/scripts/*.sh. Source it; do not execute it.
#
# Every script cds to reproduce/ so that the experiments see the original relative layout:
#   farad/    raw FARAD groups (scripts/00_get_farad.sh)
#   work/     manifest and watermarked corpora (download bundle `corpora` or 0*_build_*.sh)
#   results/  run outputs and the shipped paper summaries
#
# Knobs (environment variables):
#   PYTHON        interpreter (default: python)
#   DRY_RUN=1     print the commands without running them (no API key needed)
#   OVERWRITE=1   recompute outputs that already exist (also disables trial-log resume)
#   ALLOW_API=1   let 23_/32_ fall back to the paid API when their offline cache bundle is missing
#   RUN_ROOT      where the 1x_*.sh runs write (default: results/rerun, so the shipped paper
#                 summaries under results/ are never overwritten)
#   HF_HOME, HF_HUB_OFFLINE, CUDA_VISIBLE_DEVICES are honoured as usual.
set -euo pipefail

REPRO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPRO_DIR}"

PYTHON="${PYTHON:-python}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
RUN_ROOT="${RUN_ROOT:-results/rerun}"

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export MPLBACKEND="${MPLBACKEND:-Agg}"
# Fixed PDF CreationDate in matplotlib output, so regenerated PDFs are byte-reproducible.
export SOURCE_DATE_EPOCH="${SOURCE_DATE_EPOCH:-0}"

FARAD_DIR="${FARAD_DIR:-farad}"
MANIFEST="${MANIFEST:-work/farad_shared_manifest_p6_d400_seed13.jsonl}"
WORK_DIRBUCKET="${WORK_DIRBUCKET:-work/farad_dirbucket_naive}"
WORK_ORTHO="${WORK_ORTHO:-work/farad_dirbucket_ortho}"
WORK_KGW="${WORK_KGW:-work/farad_kgw_ward}"
WORK_SEMSTAMP="${WORK_SEMSTAMP:-work/farad_semstamp_k8_relaxed}"

die() { echo "error: $*" >&2; exit 1; }

# run CMD...: echo the command, then execute it unless DRY_RUN=1.
run() {
    printf '+'
    printf ' %q' "$@"
    printf '\n'
    [[ "${DRY_RUN}" == "1" ]] || "$@"
}

require_openai_key() {
    [[ "${DRY_RUN}" == "1" || -n "${OPENAI_API_KEY:-}" ]] \
        || die "OPENAI_API_KEY is not set; this step calls the OpenAI API (see the cost note in the script header)"
}

require_path() {
    local p
    for p in "$@"; do
        [[ "${DRY_RUN}" == "1" || -e "${p}" ]] \
            || die "missing ${p}: download the artifact bundles (scripts/download_artifacts.sh) or run the build scripts first"
    done
}

overwrite_flag() {
    [[ "${OVERWRITE}" == "1" ]] && echo "--overwrite" || true
}

# Arguments shared by every run_watermark_scheme_compare.py call of the paper (taken from the
# logged "[RUN ]" lines of the original runs). Scheme-specific flags only reach the runner for
# that scheme, so passing all of them is harmless. Per-run settings (seeds, trials, query floor,
# T_min, selector, phases) are added by each script.
compare_common_args() {
    COMPARE_COMMON=(
        --work_dir_dirbucket "${WORK_DIRBUCKET}"
        --work_dir_kgw "${WORK_KGW}"
        --work_dir_semstamp "${WORK_SEMSTAMP}"
        --work_dir_simattr "${WORK_DIRBUCKET}"
        --work_dir_overlap "${WORK_DIRBUCKET}"
        --work_dir_entail "${WORK_DIRBUCKET}"
        --providers 6
        --target_provider 0
        --retrieval_model all-MiniLM-L6-v2
        --openai_model gpt-4o-mini
        --answer_temp 0.55
        --dirbucket_audit_mode aligned --kgw_audit_mode aligned --semstamp_audit_mode aligned
        --simattr_audit_mode aligned --overlap_audit_mode aligned --entail_audit_mode aligned
        --z_threshold 4.0
        --dirbucket_direction_scheme naive
        --sim_threshold 0.82
        --tie_margin 0.05
        --global_source_max_docs 120
        --global_source_max_docs_per_provider 0
        --kgw_seeding_scheme ff-position_prf-2-False-1548585
        --kgw_gamma 0.25
        --kgw_delta 3.5
        --kgw_ignore_repeated_ngrams
        --semstamp_centroids "${WORK_SEMSTAMP}/semstamp_centroids.npz"
        --semstamp_embed_model sentence-transformers/all-mpnet-base-v2
        --semstamp_num_clusters 8
        --semstamp_gamma 0.25
        --semstamp_min_sent_words 4
        --simattr_top_m 3 --simattr_min_sent_words 4
        --overlap_top_m 3 --overlap_ngram_n 3 --overlap_min_sent_words 4 --overlap_idf_floor 0.5
        --entail_model cross-encoder/nli-deberta-v3-base
        --entail_topk_passages_per_claim 5 --entail_claim_min_words 4 --entail_passage_min_words 4
        --entail_max_claims 8 --entail_batch_size 16 --entail_device auto
        --dirbucket_alignment_manifest "${MANIFEST}"
        --kgw_alignment_manifest "${MANIFEST}"
        --semstamp_alignment_manifest "${MANIFEST}"
        --simattr_alignment_manifest "${MANIFEST}"
        --overlap_alignment_manifest "${MANIFEST}"
        --entail_alignment_manifest "${MANIFEST}"
        --force_cross_provider
        --competitive
    )
    local ow
    ow="$(overwrite_flag)"
    [[ -z "${ow}" ]] || COMPARE_COMMON+=("${ow}")
}

compare() {
    run "${PYTHON}" -u experiments/run_watermark_scheme_compare.py "${COMPARE_COMMON[@]}" "$@"
}
