/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file:  include/UnifiedCvo/utils/data_type.hpp (no 1:1 mapping claimed)
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

#include <Eigen/Dense>
#include <vector>

namespace gcvo {

  template <typename Mat>
  using aligned_vector = std::vector<Mat, Eigen::aligned_allocator<Mat>>;

  using Mat33f      = Eigen::Matrix<float, 3, 3>;
  using Vec3f      = Eigen::Matrix<float, 3, 1>;
  using Vec6f      = Eigen::Matrix<float, 6, 1>;
  using Mat44f     = Eigen::Matrix<float, 4, 4>;

  using Mat33f_row  = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  using Mat34f_row  = Eigen::Matrix<float, 3, 4, Eigen::RowMajor>;
  using Mat44f_row  = Eigen::Matrix<float, 4, 4, Eigen::RowMajor>;

  using Mat33      = Eigen::Matrix<double, 3, 3>;
  using Vec3       = Eigen::Matrix<double, 3, 1>;
  using Vec6       = Eigen::Matrix<double, 6, 1>;
  using Mat44      = Eigen::Matrix<double, 4, 4>;

  using VecXf      = Eigen::Matrix<float, Eigen::Dynamic, 1>;
  using MatXXf     = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic>;

} // namespace gcvo
