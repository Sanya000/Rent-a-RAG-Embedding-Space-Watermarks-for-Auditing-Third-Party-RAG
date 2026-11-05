#!/usr/bin/env bash
# Build the shared FARAD manifest: 400 documents, 6 providers, seed 13, sources truncated to 4000 chars.
#
# Paper: App. B.1 (shared manifest fixing document ids and provider assignment across methods).
# Cost: none (CPU, seconds). Deterministic: the output is byte-identical to the paper's manifest.
# Input: farad/ (scripts/00_get_farad.sh). Output: work/farad_shared_manifest_p6_d400_seed13.jsonl
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

MANIFEST_SHA256="380ba2a014c693b43f7d05e567ef952c4e989a0f4cc44a6e3d8def52ecf6cc72"

[[ "${DRY_RUN}" == "1" || -d "${FARAD_DIR}" ]] \
    || die "missing ${FARAD_DIR}/: run scripts/00_get_farad.sh first (FARAD is not redistributed)"
mkdir -p "$(dirname "${MANIFEST}")"
run "${PYTHON}" -u experiments/build_corpora_shared_manifest.py \
    --farad_dir "${FARAD_DIR}" \
    --out_manifest "${MANIFEST}" \
    --providers 6 --seed 13 --max_docs 400 --resume

[[ "${DRY_RUN}" == "1" ]] && exit 0
got="$("${PYTHON}" -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "${MANIFEST}")"
if [[ "${got}" == "${MANIFEST_SHA256}" ]]; then
    echo "[ok] ${MANIFEST} matches the paper manifest"
else
    echo "[warn] ${MANIFEST} differs from the paper manifest (sha256 ${got})" >&2
fi
