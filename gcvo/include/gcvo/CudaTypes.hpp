/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/CudaTypes.cuh, include/UnifiedCvo/cvo/CudaTypes.hpp (no 1:1 mapping claimed)
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

#pragma once

// Lightweight type aliases for the external CUDA point cloud + kd-tree library.
//
// NOTE:
//  - cuPointCloud is templated only on PointT
//  - cuKdTree is assumed to be templated on (PointT, KMax)
//    (forward-declare WITHOUT a default arg to avoid "redefinition of default argument".)

namespace perl_registration {
  template <typename T> class cuPointCloud;
  template <typename T, int KMax_> class cuKdTree;
}

namespace gcvo {

template <typename PointT>
using CuPointCloud = perl_registration::cuPointCloud<PointT>;

template <typename PointT, int KMax>
using CuKdTree = perl_registration::cuKdTree<PointT, KMax>;

}  // namespace gcvo
