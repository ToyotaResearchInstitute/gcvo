/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: N/A (no 1:1 mapping claimed)
 *
 * This GCVO repository is NOT a verbatim copy of RKHS_BA. It includes substantial modifications,
 * refactors, and additional original content (e.g., solver/optimization changes and new utilities).
 *
 * References:
 * - RKHS_BA paper: R. Zhang et al., IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3593521
 * - GCVO paper: R. Zhang et al., CVPR 2026 (see repo README for details)
 *
 * License: RKHS_BA is MIT-licensed (see rkhs_ba/ for the upstream LICENSE). This repo’s license is in
 * the root LICENSE file. Contact (GCVO modifications): ray.zhang@tri.global
 */

#include "gcvo/utils/PointTypes19.hpp"
#include "gcvo/GCvoGPU.hpp"
#include "gcvo/impl/GCvoGPU_impl.cuh"
#include "gcvo/impl/GCvoPointCloud_covariance_impl.cuh"

template class gcvo::GCvoGPU<gcvo::PointS1>;
template void gcvo::GCvoPointCloudT<gcvo::PointS1>::compute_covariance(float, float, int, int, bool, bool);
