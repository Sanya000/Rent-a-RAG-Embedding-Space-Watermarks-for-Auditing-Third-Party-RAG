#!/usr/bin/env bash
# Fetch the FARAD benchmark (released with WARD, Jovanovic et al., ICLR 2025) into reproduce/farad/.
#
# Paper: App. B.1 (benchmark construction). Cost: none (git clone, ~125 MB).
# Output: farad/0000.json ... farad/3590.json (3,591 groups).
#
# FARAD is not redistributed here; cite WARD and RepLiQA when you use it.
# Knobs: WARD_REPO (default https://github.com/eth-sri/ward), WARD_REF (branch/tag/commit).
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

WARD_REPO="${WARD_REPO:-https://github.com/eth-sri/ward}"
WARD_REF="${WARD_REF:-}"
# sha256 over `sha256sum` of every farad/*.json (sorted), for the copy used in the paper.
FARAD_TREE_SHA256="3955d8ca5c263990ec3b05d78a1eca0bb6db14147927077daffe75114fc81174"

farad_tree_sha256() {
    (cd "$1" && find . -name '*.json' | LC_ALL=C sort | xargs sha256sum | sha256sum | awk '{print $1}')
}

if [[ -d "${FARAD_DIR}" && -n "$(ls -A "${FARAD_DIR}" 2>/dev/null)" ]]; then
    echo "[skip] ${FARAD_DIR}/ already exists"
else
    command -v git >/dev/null || die "git is required"
    tmp="$(mktemp -d)"
    trap 'rm -rf "${tmp}"' EXIT
    if [[ -n "${WARD_REF}" ]]; then
        run git clone --quiet "${WARD_REPO}" "${tmp}/ward"
        run git -C "${tmp}/ward" checkout --quiet "${WARD_REF}"
    else
        run git clone --quiet --depth 1 "${WARD_REPO}" "${tmp}/ward"
    fi
    [[ "${DRY_RUN}" == "1" ]] || [[ -d "${tmp}/ward/farad" ]] || die "no farad/ directory in ${WARD_REPO}"
    run cp -r "${tmp}/ward/farad" "${FARAD_DIR}"
fi

[[ "${DRY_RUN}" == "1" ]] && exit 0
n="$(find "${FARAD_DIR}" -name '*.json' | wc -l | tr -d ' ')"
got="$(farad_tree_sha256 "${FARAD_DIR}")"
if [[ "${got}" == "${FARAD_TREE_SHA256}" ]]; then
    echo "[ok] ${FARAD_DIR}/: ${n} files, identical to the copy used in the paper"
else
    echo "[warn] ${FARAD_DIR}/: ${n} files (paper: 3591); content hash differs from the paper copy." >&2
    echo "       The manifest check in scripts/01_build_manifest.sh tells whether the 400 documents used match." >&2
fi
