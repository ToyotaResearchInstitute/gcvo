/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/CvoGPU.hpp (no 1:1 mapping claimed)
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

/// @file GCvoGPU.hpp
/// @brief GPU-accelerated point cloud registration solver for GCVO.
///
/// This is the main public header for the GCVO solver. It is CUDA-free: no
/// device code or CUDA headers are required to include it. All CUDA kernels
/// are compiled via explicit template instantiation in per-point-type .cu
/// files (see gcvo/src/instantiations/).
///
/// @par Transform convention
/// All public interfaces use:
///   p_source = T_s2t * p_target
/// i.e., T_s2t maps target-frame coordinates into the source frame.
///
/// @par Typical usage
/// @code
///   #include "gcvo/GCvoGPU.hpp"
///   #include "gcvo/utils/PointTypes19.hpp"
///
///   gcvo::GCvoGPU<gcvo::PointS1> solver("params.yaml");
///   gcvo::GCvoPointCloudT<gcvo::PointS1> src(pcl_cloud), tgt(pcl_cloud);
///   auto result = solver.align(src, tgt, Eigen::Matrix4f::Identity());
///   // result.T_s2t is the estimated rigid transform.
/// @endcode
///
/// @par Linking
/// Link your application against the per-point-type static library:
///   target_link_libraries(my_app PRIVATE gcvo_s1)   # for PointS1
/// Or link all types at once:
///   target_link_libraries(my_app PRIVATE gcvo)       # all point types

#pragma once

#include "gcvo/Correspondence.hpp"
#include "gcvo/GCvoParams.hpp"
#include "gcvo/utils/GCvoPointCloud.hpp"

#include <Eigen/Dense>
#include <string>

namespace gcvo {

/// Result of a single align() call.
struct GCvoResultInfo {
  /// Return code from the optimizer (0 = converged normally).
  int return_code = 0;

  /// Estimated rigid transform: p_source = T_s2t * p_target.
  Eigen::Matrix4f T_s2t = Eigen::Matrix4f::Identity();

  /// Sparse source-target correspondence weights (populated only if requested).
  Correspondence correspondence;

  /// Wall-clock time spent inside align() (seconds).
  double registration_seconds = 0.0;

  /// Number of Gauss-Newton iterations executed.
  int num_iters = 0;
};

/// GCVO point cloud registration solver.
///
/// Maximizes a kernelized inner product between two point clouds using
/// Gauss-Newton iterations on SE(3). The kernel combines a geometric term
/// (Euclidean or Mahalanobis distance) and an appearance term (per-point
/// feature vector).
///
/// @tparam PointT  Point type, typically pcl::PointSemantic<FEATURE_DIM>.
///   Must provide: x,y,z (position), features[] with FEATURE_DIMENSION,
///   normal[3], covariance[9], cov_eigenvalues[3].
///
/// @par Internal convention
/// Internally the solver optimizes T_t2s = inverse(T_s2t), because the
/// legacy CUDA code applies the transform to source points (source->target)
/// during correspondence building. The public API inverts this before returning.
template <typename PointT>
class GCvoGPU {
public:
  using PointType = PointT;
  using PointCloud = GCvoPointCloudT<PointT>;

  /// Construct solver from a YAML parameter file.
  /// @param yaml_param_file  Path to a GCvoParams YAML file.
  ///   Unspecified keys use compiled-in defaults (see GCvoParams).
  explicit GCvoGPU(const std::string& yaml_param_file);

  /// Construct solver from an in-memory GCvoParams struct.
  explicit GCvoGPU(const GCvoParams& params);

  ~GCvoGPU();

  GCvoGPU(const GCvoGPU&) = delete;
  GCvoGPU& operator=(const GCvoGPU&) = delete;

  /// @name Parameter access
  /// @{
  GCvoParams& params() { return params_; }
  const GCvoParams& params() const { return params_; }

  /// Device-side parameter pointer (for internal CUDA kernels).
  const GCvoParams* params_gpu() const { return params_gpu_; }

  /// Overwrite both host and device parameters.
  /// Useful for tweaking settings between align() calls (e.g. max_iterations).
  void write_params(const GCvoParams& p_cpu);
  /// @}

  /// @name Registration
  /// @{

  /// Align source to target and return a result struct.
  ///
  /// @param source            Source point cloud.
  /// @param target            Target point cloud.
  /// @param T_s2t_init        Initial guess: p_source = T_s2t_init * p_target.
  /// @param return_correspondence If true, populate result.correspondence with the
  ///   sparse kernel weights after the final iteration.
  /// @return GCvoResultInfo containing T_s2t, iteration count, and timing.
  GCvoResultInfo align(const PointCloud& source,
                      const PointCloud& target,
                      const Eigen::Matrix4f& T_s2t_init,
                      bool return_correspondence = false) const;

  /// Legacy align() overload that writes outputs into references.
  ///
  /// @param[in]  source       Source point cloud.
  /// @param[in]  target       Target point cloud.
  /// @param[in]  T_s2t_init   Initial guess.
  /// @param[out] T_s2t_out    Estimated transform.
  /// @param[out] correspondence  If non-null, filled with sparse correspondence.
  /// @param[out] num_iters    If non-null, set to the iteration count.
  /// @param[out] registration_seconds  If non-null, set to wall-clock time.
  /// @return Optimizer return code (0 = converged).
  int align(const PointCloud& source,
            const PointCloud& target,
            const Eigen::Matrix4f& T_s2t_init,
            Eigen::Matrix4f& T_s2t_out,
            Correspondence* correspondence = nullptr,
            int* num_iters = nullptr,
            double* registration_seconds = nullptr) const;

  /// @}

  /// Compute the function angle (overlap proxy) between two clouds.
  ///
  /// Returns cos(theta) = <f(X), f(Y)> / (||f(X)|| * ||f(Y)||), where
  /// f maps a cloud into the RKHS. Values close to 1.0 indicate high overlap.
  ///
  /// @param source           Source point cloud.
  /// @param target           Target point cloud.
  /// @param T_s2t            Pose at which to evaluate.
  /// @param l              Kernel length-scale.
  /// @param is_approximate   If true, use the self-inner-product approximation
  ///   (faster, sufficient for most uses).
  /// @return Cosine of the function-space angle in [0, 1].
  float cos(const PointCloud& source,
                       const PointCloud& target,
                       const Eigen::Matrix4f& T_s2t,
                       float l,
                       bool is_approximate = true) const;

private:
  /// Evaluate the GPU inner product <f(X), f(Y)> at the given pose/l.
  /// Optionally writes the sparse correspondence matrix to @p corresp_out.
  float inner_product_(const PointCloud& source,
                           const PointCloud& target,
                           const Eigen::Matrix4f& T_s2t,
                           float l,
                           Correspondence* corresp_out = nullptr) const;

  GCvoParams params_;       ///< Host-side parameters.
  GCvoParams* params_gpu_ = nullptr;  ///< Device-side parameters (cudaMalloc'd).
};

}  // namespace gcvo
