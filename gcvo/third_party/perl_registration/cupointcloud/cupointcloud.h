/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: thirdparty/cugicp/cupointcloud/cupointcloud.h (no 1:1 mapping claimed)
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

#include <memory>

#include <Eigen/Core>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/swap.h>

namespace perl_registration {

template <typename PointT>
class cuPointCloud {
 public:
  using PointType = PointT;
  using DeviceVectorType = thrust::device_vector<PointT>;
  using HostVectorType = thrust::host_vector<PointT, Eigen::aligned_allocator<PointT>>;
  using SharedPtr = std::shared_ptr<cuPointCloud<PointT>>;
  using SharedConstPtr = std::shared_ptr<const cuPointCloud<PointT>>;

  // Stored on device
  DeviceVectorType points;

  // STL-ish aliases
  using value_type = PointT;
  using reference = PointT&;
  using const_reference = const PointT&;
  using difference_type = typename DeviceVectorType::difference_type;
  using size_type = typename DeviceVectorType::size_type;

  using iterator = typename DeviceVectorType::iterator;
  using const_iterator = typename DeviceVectorType::const_iterator;
  inline iterator begin() { return points.begin(); }
  inline iterator end() { return points.end(); }
  inline const_iterator cbegin() const { return points.cbegin(); }
  inline const_iterator cend() const { return points.cend(); }

  inline size_type size() const { return points.size(); }
  inline bool empty() const { return points.empty(); }
  inline void reserve(size_type n) { points.reserve(n); }

  cuPointCloud() = default;
  explicit cuPointCloud(size_t n, const PointType& value = PointType()) : points(n, value) {}

  cuPointCloud(const cuPointCloud& rhs) = default;
  cuPointCloud& operator=(const cuPointCloud& rhs) = default;

  ~cuPointCloud() = default;

  inline void swap(cuPointCloud& rhs) { thrust::swap(points, rhs.points); }
};

}  // namespace perl_registration
