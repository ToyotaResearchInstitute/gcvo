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

cmake_minimum_required(VERSION 3.18)
project(gcvo_external_example LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(gcvo REQUIRED)

add_executable(register_clouds main.cu)
target_link_libraries(register_clouds PRIVATE gcvo::gcvo_s1)
target_compile_options(register_clouds PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)
