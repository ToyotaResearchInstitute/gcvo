/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file:  include/UnifiedCvo/utils/CvoPoint.hpp (no 1:1 mapping claimed)
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

/// @file PointTypes19.hpp
/// @brief Pre-defined point type aliases and PCL registration for GCVO.
///
/// This header provides the four standard GCVO point types as convenient
/// aliases in the gcvo namespace, and registers each with PCL so that
/// standard PCL I/O (loadPCDFile, savePCDFile) works out of the box.
///
/// | Alias         | Feature dim | Typical use              |
/// |---------------|-------------|--------------------------|
/// | gcvo::PointS1 | 1           | LiDAR intensity          |
/// | gcvo::PointS3 | 3           | RGB camera               |
/// | gcvo::PointS5 | 5           | RGB + image gradients    |
/// | gcvo::PointS33| 33          | FPFH descriptor          |
///
/// @par Adding a new point type
/// Define a new alias and register it:
/// @code
///   namespace gcvo { using PointS8 = pcl::PointSemantic<8>; }
///   GCVO_REGISTER_POINTSEM(gcvo::PointS8, 8)
/// @endcode
/// Then create a corresponding instantiation .cu file (see README).
///
/// @note Requires PCL_NO_PRECOMPILE (set automatically by CMake).

#pragma once

#include "gcvo/utils/PointSemantic.hpp"

#include <pcl/register_point_struct.h>

namespace gcvo {

  using PointS1  = pcl::PointSemantic<1>;   ///< 1-dim features (LiDAR intensity).
  using PointS3  = pcl::PointSemantic<3>;   ///< 3-dim features (RGB).
  using PointS5  = pcl::PointSemantic<5>;   ///< 5-dim features (RGB + gradients).
  using PointS33 = pcl::PointSemantic<33>;  ///< 33-dim features (FPFH).

}  // namespace gcvo

/// Register a pcl::PointSemantic<N> alias with PCL's point struct system.
/// This enables PCL I/O, filters, and search on the custom point type.
#define GCVO_REGISTER_POINTSEM(TYPE, FEATURE_DIM)                      \
  POINT_CLOUD_REGISTER_POINT_STRUCT(                                     \
                                                              TYPE,      \
                                                              (float, x, x) \
                                                              (float, y, y) \
                                                              (float, z, z) \
                                                              (float, rgb, rgb) \
                                                              (float[FEATURE_DIM], features, features) \
                                                              (int, label, label) \
                                                              (float[3], normal, normal) \
                                                              (float[9], covariance, covariance) \
                                                              (float[3], cov_eigenvalues, cov_eigenvalues))

GCVO_REGISTER_POINTSEM(gcvo::PointS1, 1)
GCVO_REGISTER_POINTSEM(gcvo::PointS3, 3)
GCVO_REGISTER_POINTSEM(gcvo::PointS5, 5)
GCVO_REGISTER_POINTSEM(gcvo::PointS33, 33)
