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

export CC=gcc-10
export CXX=g++-10
export NVCC_PREPEND_FLAGS='-ccbin /usr/bin/g++-10'

cd build
make -j
cd ..

./build/gcvo_test_bunny_centroid --pivot origin --params gcvo_params/test.yaml --input_pcd_file demo_data/bunny.pcd --n 2000 --theta_deg 30 --t 0.5

