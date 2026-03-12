/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/PointSegmentedDistribution.hpp (no 1:1 mapping claimed)
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

/// @file PointSemantic.hpp
/// @brief Custom PCL point type used by GCVO.
///
/// pcl::PointSemantic<FEATURE_DIM> is the primary point struct for all GCVO
/// operations. It extends the standard PCL point with:
///   - A compile-time-sized feature vector (intensity, RGB, FPFH, etc.)
///   - Per-point normal, covariance, and eigenvalues (populated by compute_covariance())
///   - A semantic label field
///
/// The struct is 16-byte aligned for GPU compatibility and works transparently
/// with both host (C++) and device (CUDA) code.
///
/// @par Pre-defined aliases (in gcvo namespace, see PointTypes19.hpp)
///   - PointS1  = PointSemantic<1>   (LiDAR intensity)
///   - PointS3  = PointSemantic<3>   (RGB camera)
///   - PointS5  = PointSemantic<5>   (RGB + image gradients)
///   - PointS33 = PointSemantic<33>  (FPFH descriptor)
///
/// @par PCL registration
/// Each alias is registered with PCL via GCVO_REGISTER_POINTSEM (see
/// PointTypes19.hpp), enabling standard PCL I/O (loadPCDFile, savePCDFile).
/// Requires PCL_NO_PRECOMPILE to be defined (set by CMake).

#pragma once

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/impl/point_types.hpp>

#include <cstring>
#include <iostream>

namespace pcl {

/// PCL-compatible point type with a compile-time feature vector.
///
/// @tparam FEATURE_DIM  Number of floats in the per-point feature vector.
///   Controls the size of the appearance kernel in the solver.
///
/// @par Memory layout (all floats unless noted)
/// | Field              | Count          | Description                                    |
/// |--------------------|----------------|------------------------------------------------|
/// | x, y, z            | 3 (+1 padding) | 3D position (via PCL_ADD_POINT4D)              |
/// | rgb                | 1 (packed)     | Packed RGB color (via PCL_ADD_RGB)              |
/// | features[]         | FEATURE_DIM    | Per-point appearance descriptor                |
/// | label              | 1 (int)        | Semantic class label (-1 = unlabeled)          |
/// | normal[]           | 3              | Unit surface normal (set by compute_covariance)|
/// | covariance[]       | 9              | 3x3 row-major covariance matrix                |
/// | cov_eigenvalues[]  | 3              | Eigenvalues of the covariance                  |
template <unsigned int FEATURE_DIM>
struct
#ifdef __CUDACC__
__align__(16)
#else
alignas(16)
#endif
PointSemantic {
  /// Compile-time constants used by CUDA kernels to size feature loops.
  static const unsigned int FEATURE_DIMENSION = FEATURE_DIM;
  static const unsigned int NORMAL_DIMENSION = 3;
  static const unsigned int COVARIANCE_DIMENSION = 9;
  static const unsigned int COV_EIGENVALUES_DIMENSION = 3;

  PCL_ADD_POINT4D;                   ///< x, y, z, data[3] (padding)
  PCL_ADD_RGB;                       ///< Packed RGB + individual r, g, b accessors
  float features[FEATURE_DIM];      ///< Per-point feature vector (normalized to [0,1])
  int label;                         ///< Semantic class label (-1 = unlabeled)
  float normal[3];                   ///< Unit surface normal
  float covariance[9];               ///< 3x3 covariance, row-major
  float cov_eigenvalues[3];          ///< Eigenvalues of the covariance

  /// Runtime query for the feature vector length.
  unsigned int feature_dimension() const { return FEATURE_DIM; }

#ifdef __CUDACC__
  inline __host__ __device__ ~PointSemantic() {}
#else
  inline ~PointSemantic() {}
#endif

  /// Default constructor: zero-initializes all fields, label = -1.
#ifdef __CUDACC__
  inline __host__ __device__
#endif
  PointSemantic() {
    this->x = this->y = this->z = 0.0f;
    label = -1;
    this->r = this->g = this->b = 0;
    std::memset(features, 0, sizeof(float) * FEATURE_DIM);
    std::memset(normal, 0, sizeof(float) * 3);
    std::memset(covariance, 0, sizeof(float) * 9);
    std::memset(cov_eigenvalues, 0, sizeof(float) * 3);
  }

  /// Construct with explicit xyz position; all other fields zero-initialized.
#ifdef __CUDACC__
  inline __host__ __device__
#endif
  explicit PointSemantic(float a, float b, float c) {
    this->x = a;
    this->y = b;
    this->z = c;
    label = -1;
    this->r = this->g = this->b = 0;
    std::memset(features, 0, sizeof(float) * FEATURE_DIM);
    std::memset(normal, 0, sizeof(float) * 3);
    std::memset(covariance, 0, sizeof(float) * 9);
    std::memset(cov_eigenvalues, 0, sizeof(float) * 3);
  }

#ifdef __CUDACC__
  inline __host__ __device__
#endif
  PointSemantic(const PointSemantic& other) {
    *this = other;
  }

#ifdef __CUDACC__
  inline __host__ __device__
#endif
  PointSemantic& operator=(const PointSemantic& other) {
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
      this->r = other.r;
      this->g = other.g;
      this->b = other.b;
      label = other.label;
      std::memcpy(features, other.features, sizeof(float) * FEATURE_DIM);
      std::memcpy(normal, other.normal, sizeof(float) * 3);
      std::memcpy(covariance, other.covariance, sizeof(float) * 9);
      std::memcpy(cov_eigenvalues, other.cov_eigenvalues, sizeof(float) * 3);
    }
    return *this;
  }
};

/// Convert a PointSemantic cloud to a standard PCL PointXYZRGB cloud.
///
/// Copies xyz and rgb; discards features, normals, covariance.
/// Useful for visualization with standard PCL viewers.
template <unsigned int FEATURE_DIM, typename PointWithXYZRGB>
void PointSemantic_to_PointXYZRGB(const pcl::PointCloud<pcl::PointSemantic<FEATURE_DIM>>& pc_sem,
                                   pcl::PointCloud<PointWithXYZRGB>& pc_rgb) {
  pc_rgb.resize(pc_sem.size());
  for (size_t i = 0; i < pc_rgb.size(); i++) {
    auto& p_rgb = pc_rgb[i];
    auto& p_sem = pc_sem[i];
    p_rgb.x = p_sem.x;
    p_rgb.y = p_sem.y;
    p_rgb.z = p_sem.z;
    p_rgb.r = p_sem.r;
    p_rgb.g = p_sem.g;
    p_rgb.b = p_sem.b;
  }
  pc_rgb.header = pc_sem.header;
}

} // namespace pcl
