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

# Example runner for gcvo_kitti_f2f
#
# Usage:
#   ./scripts/run_kitti_f2f.sh <build_dir> <params.yaml> <kitti_root> [sequence] [start] [count]
cd build_renamed
make -j
cd ..

BUILD_DIR="./build_renamed/"
PARAMS="gcvo_params/gcvo_driving_nonisotropic_gn.yaml"
KITTI_ROOT="/home/`whoami`/code/docker_home/cvo/data/kitti/dataset/"
SEQ="09"
START="0"
COUNT="100000"

for SEQ in 06 08
do
"${BUILD_DIR}/gcvo_kitti_f2f" \
  --params "${PARAMS}" \
  --kitti_root "${KITTI_ROOT}" \
  --sequence "${SEQ}" \
  --start "${START}" \
  --count "${COUNT}" \
  --voxel_mode fast \
  --random_downsample 4000 \
  --traj_file gcvo.${SEQ}.connection.kitti 
done
