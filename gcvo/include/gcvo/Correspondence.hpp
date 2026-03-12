/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/Association.hpp (no 1:1 mapping claimed)
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

/// @file Correspondence.hpp
/// @brief Sparse source-target correspondence output from GCvoGPU::align().

#pragma once

#include <Eigen/Sparse>
#include <vector>

namespace gcvo {

/// Sparse correspondence between source and target point clouds.
///
/// Populated by GCvoGPU::align() when return_correspondence is true. Contains
/// the kernel weights and the indices of points that contributed to at
/// least one non-zero correspondence entry.
struct Correspondence {
  /// Sparse weight matrix: rows = source indices, cols = target indices.
  /// Entry (i, j) is the kernel weight between source point i and target point j.
  using SparsePairs = Eigen::SparseMatrix<float, Eigen::RowMajor>;

  SparsePairs pairs;

  /// Source point indices that have at least one non-zero correspondence.
  std::vector<int> source_inliers;

  /// Target point indices that have at least one non-zero correspondence.
  std::vector<int> target_inliers;

  /// Reset all fields to empty.
  void clear() {
    pairs.resize(0, 0);
    pairs.setZero();
    source_inliers.clear();
    target_inliers.clear();
  }
};

} // namespace gcvo
