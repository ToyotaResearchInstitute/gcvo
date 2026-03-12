/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/CvoState.cuh (no 1:1 mapping claimed)
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

#include "gcvo/CudaTypes.hpp"
#include "gcvo/GCvoParams.hpp"
#include "gcvo/Correlation.hpp"

#include "cupointcloud/cupointcloud.h"

#include <Eigen/Dense>
#include <memory>
#include <thrust/device_vector.h>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace gcvo {

// Per-alignment GPU state (templated on point type).
template <typename PointT>
struct GCvoStateT {
  using PointType = PointT;
  using CuPointCloudT = CuPointCloud<PointT>;

  using Mat66 = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>;
  using Vec6  = Eigen::Matrix<float, 6, 1>;

  // NOTE: This state follows the **legacy CVO roles**:
  //   - fixed:  points in the "source" frame (cloud_x in old code)
  //   - moving: points in the "target" frame, which are transformed into the
  //             fixed frame each iteration (cloud_y in old code)
  GCvoStateT(std::shared_ptr<CuPointCloudT> fixed_points,   // fixed (source) in source frame
            std::shared_ptr<CuPointCloudT> moving_points,  // moving (target) in target frame
            const GCvoParams& params)
      : num_fixed(static_cast<int>(fixed_points ? fixed_points->size() : 0)),
        num_moving(static_cast<int>(moving_points ? moving_points->size() : 0)),
        l(params.l_init),
        cloud_fixed(std::move(fixed_points)),
        cloud_moving_init(std::move(moving_points)),
        cloud_moving_tf(std::make_shared<CuPointCloudT>(num_moving)) {

    // Allocate corr with the *maximum* neighbor capacity we will use.
    const int corr_rows = num_fixed;
    const int corr_cols = std::max(1, params.max_neighbors);
    corr = init_Correlation_gpu(corr_rows, corr_cols, corr_host);

    // Transform (inverse(T_s2t)) on GPU, used to map moving->fixed.
    cudaMalloc((void**)&R_gpu, sizeof(Eigen::Matrix3f));
    cudaMalloc((void**)&t_gpu, sizeof(Eigen::Vector3f));

    // Reusable GN buffers (avoid per-iter malloc/free)
    H_rows.resize(static_cast<size_t>(num_fixed));
    g_rows.resize(static_cast<size_t>(num_fixed));

    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
      fprintf(stderr, "GCVO GCvoStateT init failed: %s\n", cudaGetErrorString(err));
      std::exit(EXIT_FAILURE);
    }
  }

  ~GCvoStateT() {
    cudaFree(R_gpu);
    cudaFree(t_gpu);
    delete_Correlation_gpu(corr, &corr_host);
  }

  void reset_state_at_new_iter(int num_neighbors) {
    // corr was allocated with corr_host.max_entries_per_row; clear only the used prefix.
    clear_Correlation(&corr_host, num_neighbors);
  }

  int num_fixed = 0;
  int num_moving = 0;

  float l = 0.0f;

  // Sparse kernel matrix corr: num_sources=num_fixed, max_entries_per_row=corr_host.max_entries_per_row (max neighbors)
  Correlation* corr = nullptr;
  Correlation corr_host;

  // Transform (moving->fixed) on GPU
  Eigen::Matrix3f* R_gpu = nullptr;
  Eigen::Vector3f* t_gpu = nullptr;

  // Clouds (legacy naming: x=fixed, y=moving)
  std::shared_ptr<CuPointCloudT> cloud_fixed;       // fixed in fixed frame
  std::shared_ptr<CuPointCloudT> cloud_moving_init; // moving in its own frame
  std::shared_ptr<CuPointCloudT> cloud_moving_tf;   // moving transformed into fixed frame

  // GN work buffers (reused per-iteration)
  thrust::device_vector<Mat66> H_rows;
  thrust::device_vector<Vec6>  g_rows;
};

}  // namespace gcvo
