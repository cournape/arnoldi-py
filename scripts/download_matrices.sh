#!/usr/bin/env bash
# AI note: this script was AI generated, except for the choices of matrices

# Download SuiteSparse .mat files used in this project.
# URL pattern: https://sparse.tamu.edu/mat/{group}/{name}.mat

set -euo pipefail

BASE_URL="https://sparse.tamu.edu/mat"
DEST_DIR="$(pwd)"

MATRICES=(
# Easy ones: small matrices, small amplitude
    "HB/1138_bus"
    "Bai/mhd1280b"
    "Bai/rdb1250"
# A bit harder: high amplitude, test relative convergence criteria
    "HB/bcsstk16"
    "HB/bcsstk18"
    "Nasa/nasasrb"
# Medium hard: ARPACK will take ~30 sec on my macbook M4 to find top 3 by real
# part
    "Bai/af23560"
    "Bai/olm5000"
)

for entry in "${MATRICES[@]}"; do
    group="${entry%/*}"
    name="${entry#*/}"
    url="${BASE_URL}/${group}/${name}.mat"
    dest="${DEST_DIR}/${name}.mat"

    if [[ -f "$dest" ]]; then
        echo "Already exists, skipping: ${name}.mat"
        continue
    fi

    echo "Downloading ${group}/${name} ..."
    wget --no-verbose --show-progress -O "$dest" "$url"
done

echo "Done."
