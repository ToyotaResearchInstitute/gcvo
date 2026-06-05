/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/CvoPointCloud.hpp (no 1:1 mapping claimed)
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

/// @file GCvoPointCloud.hpp
/// @brief Lightweight, header-only point cloud container for GCVO.
///
/// GCvoPointCloudT<PointT> wraps a std::vector<PointT> and provides:
///   - Construction from any PCL point cloud (auto-maps intensity/rgb to features[]).
///   - Rigid-body transform that also rotates normals and covariance.
///   - GPU-accelerated per-point covariance/normal computation via compute_covariance().
///
/// This header is CUDA-free. The compute_covariance() body lives in
/// gcvo/include/gcvo/impl/GCvoPointCloud_covariance_impl.cuh and is explicitly
/// instantiated in the per-type .cu files under gcvo/src/instantiations/.
///
/// @par Example
/// @code
///   pcl::PointCloud<pcl::PointXYZI> pcl_cloud;
///   pcl::io::loadPCDFile("scan.pcd", pcl_cloud);
///
///   gcvo::GCvoPointCloudT<gcvo::PointS1> cloud(pcl_cloud);
///   // For DENSE/RESCALED kernels, compute per-point covariance first:
///   cloud.compute_covariance();
/// @endcode

#pragma once

#include <Eigen/Dense>

#include <pcl/point_cloud.h>

#include <type_traits>
#include <vector>

namespace gcvo {

/// @cond INTERNAL
namespace detail {
  // SFINAE traits to detect optional fields on PCL point types.
  template <typename T, typename = void>
  struct has_normal : std::false_type {};
  template <typename T>
  struct has_normal<T, std::void_t<decltype(std::declval<T>().normal[0])>> : std::true_type {};

  template <typename T, typename = void>
  struct has_covariance : std::false_type {};
  template <typename T>
  struct has_covariance<T, std::void_t<decltype(std::declval<T>().covariance[0])>> : std::true_type {};

  template <typename T, typename = void>
  struct has_intensity : std::false_type {};
  template <typename T>
  struct has_intensity<T, std::void_t<decltype(std::declval<T>().intensity)>> : std::true_type {};

  template <typename T, typename = void>
  struct has_rgb : std::false_type {};
  template <typename T>
  struct has_rgb<T, std::void_t<decltype(std::declval<T>().rgb),
                                decltype(std::declval<T>().r),
                                decltype(std::declval<T>().g),
                                decltype(std::declval<T>().b)>> : std::true_type {};
}
/// @endcond

/// Lightweight point cloud container for GCVO.
///
/// Stores points in a contiguous std::vector<PointT>. Provides constructors
/// that convert from standard PCL clouds (PointXYZI, PointXYZRGB, etc.) with
/// automatic feature mapping.
///
/// @tparam PointT  Point type, typically pcl::PointSemantic<FEATURE_DIM>.
///   Must provide x, y, z and features[FEATURE_DIMENSION].
template <typename PointT>
class GCvoPointCloudT {
public:
  using PointType = PointT;

  /// Default constructor (empty cloud).
  GCvoPointCloudT() = default;

  /// Construct from a PCL cloud of the same point type (zero-copy style).
  explicit GCvoPointCloudT(const pcl::PointCloud<PointT>& pc) {
    points_.assign(pc.begin(), pc.end());
  }

  /// Construct from a PCL cloud of a different point type with best-effort conversion.
  ///
  /// Always copies xyz. Additionally:
  /// - If PclPointT has intensity and PointT::FEATURE_DIMENSION >= 1:
  ///   sets features[0] = intensity / 255.
  /// - If PclPointT has rgb and PointT::FEATURE_DIMENSION >= 3:
  ///   sets features[0..2] = r,g,b / 255 and copies the packed rgb field.
  ///
  /// @tparam PclPointT  Any PCL point type with x, y, z fields.
  template <typename PclPointT>
  explicit GCvoPointCloudT(const pcl::PointCloud<PclPointT>& pc) {
    points_.resize(pc.size());
    for (size_t i = 0; i < pc.size(); ++i) {
      PointT p;
      p.x = pc[i].x;
      p.y = pc[i].y;
      p.z = pc[i].z;

      if constexpr (detail::has_intensity<PclPointT>::value) {
        if constexpr (PointT::FEATURE_DIMENSION >= 1) {
          p.features[0] = static_cast<float>(pc[i].intensity) / 255.0f;
        }
      }

      if constexpr (detail::has_rgb<PclPointT>::value) {
        p.rgb = pc[i].rgb;
        if constexpr (PointT::FEATURE_DIMENSION >= 3) {
          p.features[0] = static_cast<float>(pc[i].r) / 255.0f;
          p.features[1] = static_cast<float>(pc[i].g) / 255.0f;
          p.features[2] = static_cast<float>(pc[i].b) / 255.0f;
        }
      }

      points_[i] = p;
    }
  }

  /// @name Size and access
  /// @{
  int num_points() const { return static_cast<int>(points_.size()); }
  int size() const { return num_points(); }

  const std::vector<PointT>& points() const { return points_; }
  std::vector<PointT>& points() { return points_; }

  /// Return a copy of the point vector.
  std::vector<PointT> get_points() const { return points_; }

  const PointT& point_at(size_t idx) const { return points_.at(idx); }
  PointT& point_at(size_t idx) { return points_.at(idx); }

  const PointT& operator[](size_t idx) const { return points_[idx]; }
  PointT& operator[](size_t idx) { return points_[idx]; }
  /// @}

  /// @name Mutation
  /// @{
  void reserve(size_t n) { points_.reserve(n); }
  void clear() { points_.clear(); }
  void push_back(const PointT& p) { points_.push_back(p); }
  /// @}

  /// Compute per-point covariance, normals, and eigenvalues via GPU KNN.
  ///
  /// Required before calling GCvoGPU::align() with DENSE or RESCALED kernels.
  /// Populates the normal[3], covariance[9], and cov_eigenvalues[3] fields
  /// of each point.
  ///
  /// Implementation lives in gcvo/include/gcvo/impl/GCvoPointCloud_covariance_impl.cuh
  /// and is explicitly instantiated in the per-type .cu files.
  ///
  /// @param min_range         Ignore points closer than this to the origin.
  /// @param max_range         Ignore points farther than this from the origin.
  /// @param num_neighbors     Number of nearest neighbors for covariance estimation.
  /// @param num_threads       CPU thread count (unused in current GPU impl).
  /// @param is_rescaled_kernel  If true, apply the RESCALED kernel normalization
  ///   to the covariance (divides by the largest eigenvalue).
  /// @param use_kdtree        If true, use GPU KD-tree for KNN; otherwise brute-force.
  void compute_covariance(float min_range = 0.0f,
                          float max_range = 1000.0f,
                          int num_neighbors = 12,
                          int num_threads = 8,
                          bool is_rescaled_kernel = false,
                          bool use_kdtree = true);

  /// Clamp per-point covariance eigenvalues to [min_eig, max_eig] and reconstruct.
  ///
  /// Call after compute_covariance() when using DENSE/RESCALED kernels.
  /// Ensures all eigenvalues are in [min_eig, max_eig] (max_eig <= 0 means no upper clamp).
  /// When paired with use_ell2_in_kernel=0, the clamped covariance fully controls kernel shape.
  void clamp_covariance_eigenvalues(float min_eig, float max_eig) {
    if constexpr (!detail::has_covariance<PointT>::value) return;
    for (auto& p : points_) {
      Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> cov(p.covariance);
      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> es(cov);
      Eigen::Vector3f eigs = es.eigenvalues().cwiseMax(min_eig);
      if (max_eig > 0.0f) eigs = eigs.cwiseMin(max_eig);
      cov = (es.eigenvectors() * eigs.asDiagonal() * es.eigenvectors().transpose()).eval();
    }
  }

  /// Apply a rigid transform to @p input and write the result to @p output.
  ///
  /// Transforms xyz by T. If PointT has normal[], rotates normals (no
  /// translation). If PointT has covariance[], rotates covariance as
  /// R * Cov * R^T.
  ///
  /// @param T                      4x4 rigid transform matrix.
  /// @param input                  Source cloud.
  /// @param output                 Destination cloud (resized to match input).
  /// @param update_normal_and_cov  If false, skip normal/covariance rotation
  ///   (faster when covariance will be recomputed anyway).
  static void transform(const Eigen::Matrix4f& T,
                        const GCvoPointCloudT& input,
                        GCvoPointCloudT& output,
                        bool update_normal_and_cov = true) {
    output.points_.resize(input.points_.size());
    Eigen::Matrix3f R = T.block<3, 3>(0, 0);
    Eigen::Vector3f t = T.block<3, 1>(0, 3);

    for (size_t i = 0; i < input.points_.size(); ++i) {
      PointT p = input.points_[i];
      Eigen::Vector3f x(p.x, p.y, p.z);
      Eigen::Vector3f xt = R * x + t;
      p.x = xt.x();
      p.y = xt.y();
      p.z = xt.z();

      if constexpr (detail::has_normal<PointT>::value) {
        if (update_normal_and_cov) {
          Eigen::Vector3f n(p.normal[0], p.normal[1], p.normal[2]);
          Eigen::Vector3f nt = R * n;
          p.normal[0] = nt.x();
          p.normal[1] = nt.y();
          p.normal[2] = nt.z();
        }
      }

      if constexpr (detail::has_covariance<PointT>::value) {
        if (update_normal_and_cov) {
          Eigen::Map<Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> cov(p.covariance);
          cov = (R * cov * R.transpose()).eval();
        }
      }

      output.points_[i] = p;
    }
  }

private:
  std::vector<PointT> points_;
};

} // namespace gcvo
