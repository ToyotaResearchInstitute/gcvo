/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/CvoPoint.hpp (no 1:1 mapping claimed)
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

#include "gcvo/utils/PointSegmentedDistribution.hpp"

#include <pcl/point_types.h>

namespace gcvo {

// Canonical GCVO point type: PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>
// NOTE: Feature/label dimensions are compile-time template parameters.
template <unsigned int FEATURE_DIM, unsigned int NUM_CLASS>
using GCvoPointT = pcl::PointSegmentedDistribution<FEATURE_DIM, NUM_CLASS>;

// Map GCvoPointT specializations to common PCL types (best-effort).
template <typename PointT>
struct GCvoPointToPCL {
  using type = pcl::PointXYZ;
};

template <unsigned int NUM_CLASS>
struct GCvoPointToPCL<pcl::PointSegmentedDistribution<1, NUM_CLASS>> {
  using type = pcl::PointXYZI;
};

template <unsigned int NUM_CLASS>
struct GCvoPointToPCL<pcl::PointSegmentedDistribution<3, NUM_CLASS>> {
  using type = pcl::PointXYZRGB;
};

template <unsigned int NUM_CLASS>
struct GCvoPointToPCL<pcl::PointSegmentedDistribution<5, NUM_CLASS>> {
  using type = pcl::PointXYZRGB;
};

}  // namespace gcvo
