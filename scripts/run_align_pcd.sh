#!/usr/bin/env bash
# GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
# Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
# Upstream is included in this repo as a git submodule at: rkhs_ba/
# Upstream file: N/A (no 1:1 mapping claimed)
#
# This GCVO repository is NOT a verbatim copy of RKHS_BA. It includes substantial modifications,
# refactors, and additional original content (e.g., solver/optimization changes and new utilities).
#
# References:
# - RKHS_BA paper: R. Zhang et al., IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3593521
# - GCVO paper: R. Zhang et al., CVPR 2026 (see repo README for details)
#
# License: RKHS_BA is MIT-licensed (see rkhs_ba/ for the upstream LICENSE). This repo’s license is in
# the root LICENSE file. Contact (GCVO modifications): ray.zhang@tri.global

set -euo pipefail

# Example runner for gcvo_align_pcd
#
# Usage:
#   ./scripts/run_align_pcd.sh <build_dir> <params.yaml> <type:intensity|rgb|rgbg|fpfh> <source.pcd> <target.pcd>

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 <build_dir> <params.yaml> <type> <source.pcd> <target.pcd>" >&2
  exit 2
fi

BUILD_DIR="$1"
PARAMS="$2"
TYPE="$3"
SRC="$4"
TGT="$5"

"${BUILD_DIR}/gcvo_align_pcd" \
  --params "${PARAMS}" \
  --type "${TYPE}" \
  --source "${SRC}" \
  --target "${TGT}"
