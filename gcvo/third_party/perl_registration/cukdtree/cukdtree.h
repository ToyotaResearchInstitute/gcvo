/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: thirdparty/cugicp/cukdtree/cukdtree.h  (no 1:1 mapping claimed)
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

#include <limits>
#include <memory>

#include <thrust/device_vector.h>

#include "cupointcloud/cupointcloud.h"

namespace perl_registration {

// A small, GPU-friendly KNN helper.
//
// NOTE:
//  - This is a simplified cuKdTree implementation intended for GCVO.
//  - It provides the same interface as the original perl_registration cuKdTree,
//    but uses a brute-force KNN kernel (no tree build) to keep the code small
//    and templated on KMax at compile time.

template <typename PointT, int KMax_>
class cuKdTree {
 public:
  using PointType = PointT;
  using SharedPtr = std::shared_ptr<cuKdTree<PointT, KMax_>>;
  using SharedConstPtr = std::shared_ptr<const cuKdTree<PointT, KMax_>>;
  using cuPointCloudSharedPtr = typename cuPointCloud<PointT>::SharedPtr;

  cuKdTree() = default;
  ~cuKdTree() = default;

  void SetInputCloud(cuPointCloudSharedPtr& d_cloud);

  // Device-vector output (kept for compatibility)
  int NearestKSearch(const cuPointCloudSharedPtr& d_query_points,
                     int k,
                     thrust::device_vector<int>& indices);

  // Raw-pointer output (preferred for GCVO to avoid realloc)
  int NearestKSearch(const cuPointCloudSharedPtr& d_query_points,
                     int k,
                     int* indices,
                     int indices_size);

  bool IsInputCloudSet() const { return input_cloud_set_; }

 private:
  cuPointCloudSharedPtr d_point_cloud_;
  bool input_cloud_set_ = false;
};

}  // namespace perl_registration

#include "cukdtree.cuh"
