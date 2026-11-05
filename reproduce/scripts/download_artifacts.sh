#!/usr/bin/env bash
# Download, verify and extract the Rent-a-RAG reproduction artifacts into reproduce/.
#
# Usage:
#   scripts/download_artifacts.sh                         # all bundles
#   scripts/download_artifacts.sh corpora trial_logs_main # selected bundles
#   scripts/download_artifacts.sh --list                  # show bundles
#
# Environment:
#   ARTIFACTS_BASE_URL  base URL that holds the rentarag_<bundle>.tar.gz files
#                       (http(s):// or file://). Required until the TODO default is filled in.
#   ARTIFACTS_CACHE     where tarballs are downloaded (default: reproduce/.artifacts_cache)
#   KEEP_TARBALLS=1     keep the downloaded tarballs after extraction
#
# Bundles extract to reproduce/work/... and reproduce/results/..., next to the
# result summaries already tracked in git. See reproduce/ARTIFACTS.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# TODO(authors): replace with the Zenodo / Hugging Face URL after uploading the bundles.
DEFAULT_BASE_URL="TODO_SET_ARTIFACTS_BASE_URL"
BASE_URL="${ARTIFACTS_BASE_URL:-${DEFAULT_BASE_URL}}"
CACHE_DIR="${ARTIFACTS_CACHE:-${REPRO_DIR}/.artifacts_cache}"
KEEP_TARBALLS="${KEEP_TARBALLS:-0}"

# bundle name | sha256 of rentarag_<bundle>.tar.gz | description
BUNDLES="
corpora|c55935314b21612b4fc451e34396d07809f57e70d5d7837ba811230c944efaec|FARAD manifest + DirBucket/KGW/SemStamp corpora (5.1 MB)
corpora_ext|a37dd786a154d0aac192d4d436b3eedd585db6e31e6852c123d73edff14c4923|coverage, staleness and E5 paraphraser corpora (5.8 MB)
trial_logs_main|8711e15cf8154861dbc939ab5569fc43b35a65cdf316701e92aae9bb15ae4910|primary, rag160 and mixstress trial logs (43.0 MB)
trial_logs_baselines|2f1262c27a5986fd945273316e17134de49b25745036e0ef9fb825b187a1f67d|non-watermark and oracle baseline trial logs (34.8 MB)
trial_logs_coverage|e3e79c3df4519dafbb7e3b55fb27fac40f3e1053cb0f155948718a5ed69eeb77|selective-coverage trial logs (5.0 MB)
laundering_cache|038fc1b66d284d3344f3c78a75bc987f952d5b742047f527913bb459a0bf4eec|cached laundered answers (4.0 MB)
appendix_aux|fa17d762f758cdb850eafd5132096ff2eecb6b51aaf5e2b3abbd3a33e88b8023|cost-ablation candidates + re-watermarked mini-corpora (13.8 MB)
"

die() { echo "error: $*" >&2; exit 1; }

bundle_field() {  # bundle_field <name> <column>
    printf '%s\n' "${BUNDLES}" | awk -F'|' -v n="$1" -v c="$2" '$1 == n { print $c }'
}

all_bundles() {
    printf '%s\n' "${BUNDLES}" | awk -F'|' 'NF == 3 { print $1 }'
}

sha256_of() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | awk '{ print $1 }'
    else
        shasum -a 256 "$1" | awk '{ print $1 }'
    fi
}

fetch() {  # fetch <url> <dest>
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 --progress-bar -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$2" "$1"
    else
        die "need curl or wget"
    fi
}

if [[ "${1:-}" == "--list" || "${1:-}" == "-l" ]]; then
    printf '%s\n' "${BUNDLES}" | awk -F'|' 'NF == 3 { printf "  %-22s %s\n", $1, $3 }'
    exit 0
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 0
fi

[[ "${BASE_URL}" == TODO* ]] && die "set ARTIFACTS_BASE_URL (artifact location not yet published; see reproduce/ARTIFACTS.md)"
BASE_URL="${BASE_URL%/}"

if [[ $# -gt 0 ]]; then
    SELECTED=("$@")
else
    read -r -a SELECTED <<< "$(all_bundles | tr '\n' ' ')"
fi
for b in "${SELECTED[@]}"; do
    [[ -n "$(bundle_field "$b" 2)" ]] || die "unknown bundle '$b' (try --list)"
done

mkdir -p "${CACHE_DIR}"
for b in "${SELECTED[@]}"; do
    file="rentarag_${b}.tar.gz"
    dest="${CACHE_DIR}/${file}"
    expected="$(bundle_field "$b" 2)"
    if [[ -f "${dest}" && "$(sha256_of "${dest}")" == "${expected}" ]]; then
        echo "[cached]   ${file}"
    else
        echo "[download] ${BASE_URL}/${file}"
        fetch "${BASE_URL}/${file}" "${dest}.part"
        mv "${dest}.part" "${dest}"
    fi
    actual="$(sha256_of "${dest}")"
    [[ "${actual}" == "${expected}" ]] || die "checksum mismatch for ${file}: got ${actual}, expected ${expected}"
    echo "[verified] ${file}"
    tar -xzf "${dest}" -C "${REPRO_DIR}" --no-same-owner
    echo "[extract]  ${file} -> ${REPRO_DIR}"
    [[ "${KEEP_TARBALLS}" == "1" ]] || rm -f "${dest}"
done
rmdir "${CACHE_DIR}" 2>/dev/null || true
echo "done: ${#SELECTED[@]} bundle(s) in ${REPRO_DIR}"
