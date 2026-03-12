/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: src/utils/CvoPointCovariance.cu (no 1:1 mapping claimed)
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

#include "gcvo/utils/GCvoPointCloud.hpp"
#include "gcvo/utils/PointTypes19.hpp"

#include "cupointcloud/cupointcloud.h"
#include "cukdtree/cukdtree.h"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <cstdio>

#ifndef GCVO_CUDA_THREADS
#define GCVO_CUDA_THREADS 256
#endif

namespace gcvo {
namespace {

inline void gcvo_cov_gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GCVO CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) std::exit(static_cast<int>(code));
  }
}
#define GCVO_COV_CUDA_CHECK(ans) ::gcvo::gcvo_cov_gpuAssert((ans), __FILE__, __LINE__, true)

constexpr int GCVO_COV_KMAX = 64;

template <typename PointT>
__global__ void compute_covariance_knn_kernel(PointT* points,
                                              int n,
                                              const int* nn_inds,
                                              int k,
                                              float min_range,
                                              float max_range) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;

  PointT& pi = points[i];

  const float r2 = pi.x * pi.x + pi.y * pi.y + pi.z * pi.z;
  const float min2 = min_range * min_range;
  const float max2 = max_range * max_range;

  auto set_default = [&]() {
    #pragma unroll
    for (int j = 0; j < 9; ++j) pi.covariance[j] = 0.0f;
    pi.covariance[0] = 1e-3f;
    pi.covariance[4] = 1e-3f;
    pi.covariance[8] = 1e-3f;
    if constexpr (gcvo::detail::has_normal<PointT>::value) {
      pi.normal[0] = 0.0f;
      pi.normal[1] = 0.0f;
      pi.normal[2] = 0.0f;
    }
  };

  if (r2 < min2 || r2 > max2) {
    set_default();
    return;
  }

  float mx = 0.0f, my = 0.0f, mz = 0.0f;
  int cnt = 0;

  for (int j = 0; j < k; ++j) {
    const int idx = nn_inds[i * k + j];
    if (idx < 0 || idx >= n) break;
    const PointT& pj = points[idx];
    mx += pj.x;
    my += pj.y;
    mz += pj.z;
    ++cnt;
  }

  if (cnt < 3) {
    set_default();
    return;
  }

  const float inv = 1.0f / static_cast<float>(cnt);
  mx *= inv;
  my *= inv;
  mz *= inv;

  float c00 = 0.0f, c01 = 0.0f, c02 = 0.0f;
  float c11 = 0.0f, c12 = 0.0f;
  float c22 = 0.0f;

  for (int j = 0; j < cnt; ++j) {
    const int idx = nn_inds[i * k + j];
    if (idx < 0 || idx >= n) break;
    const PointT& pj = points[idx];
    const float dx = pj.x - mx;
    const float dy = pj.y - my;
    const float dz = pj.z - mz;
    c00 += dx * dx;
    c01 += dx * dy;
    c02 += dx * dz;
    c11 += dy * dy;
    c12 += dy * dz;
    c22 += dz * dz;
  }

  const float denom = 1.0f / static_cast<float>(cnt - 1);
  c00 *= denom;
  c01 *= denom;
  c02 *= denom;
  c11 *= denom;
  c12 *= denom;
  c22 *= denom;

  const float reg = 1e-6f;
  c00 += reg;
  c11 += reg;
  c22 += reg;

  pi.covariance[0] = c00;
  pi.covariance[1] = c01;
  pi.covariance[2] = c02;
  pi.covariance[3] = c01;
  pi.covariance[4] = c11;
  pi.covariance[5] = c12;
  pi.covariance[6] = c02;
  pi.covariance[7] = c12;
  pi.covariance[8] = c22;

  if constexpr (gcvo::detail::has_normal<PointT>::value) {
    pi.normal[0] = 0.0f;
    pi.normal[1] = 0.0f;
    pi.normal[2] = 0.0f;
  }
}

}  // namespace

template <typename PointT>
void GCvoPointCloudT<PointT>::compute_covariance(float min_range,
                                                float max_range,
                                                int num_neighbors,
                                                int /*num_threads*/,
                                                bool /*is_rescaled_kernel*/,
                                                bool use_kdtree) {
  static_assert(gcvo::detail::has_covariance<PointT>::value,
                "PointT must have covariance[9] field for compute_covariance().");

  if (points_.empty()) return;

  const int n = static_cast<int>(points_.size());
  const int k = std::max(1, std::min(num_neighbors, GCVO_COV_KMAX));

  using CuCloud = perl_registration::cuPointCloud<PointT>;
  using HostVec = thrust::host_vector<PointT, Eigen::aligned_allocator<PointT>>;

  HostVec h(points_.begin(), points_.end());
  auto d_cloud = std::make_shared<CuCloud>();
  d_cloud->points = h;

  thrust::device_vector<int> d_inds;
  d_inds.resize(static_cast<size_t>(n) * static_cast<size_t>(k));
  cudaMemset(thrust::raw_pointer_cast(d_inds.data()), -1, sizeof(int) * d_inds.size());

  if (use_kdtree) {
    using KdTree = perl_registration::cuKdTree<PointT, GCVO_COV_KMAX>;
    KdTree tree;
    tree.SetInputCloud(d_cloud);

    tree.NearestKSearch(d_cloud,
                        k,
                        thrust::raw_pointer_cast(d_inds.data()),
                        static_cast<int>(d_inds.size()));
    GCVO_COV_CUDA_CHECK(cudaDeviceSynchronize());
    GCVO_COV_CUDA_CHECK(cudaGetLastError());
  } else {
    // Fallback (no KD-tree): leave indices as -1 (will default covariances).
  }

  const int threads = GCVO_CUDA_THREADS;
  const int blocks = (n + threads - 1) / threads;

  compute_covariance_knn_kernel<PointT><<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_cloud->points.data()),
      n,
      thrust::raw_pointer_cast(d_inds.data()),
      k,
      min_range,
      max_range);
  GCVO_COV_CUDA_CHECK(cudaDeviceSynchronize());
  GCVO_COV_CUDA_CHECK(cudaGetLastError());

  HostVec h_out = d_cloud->points;
  points_.assign(h_out.begin(), h_out.end());
}

}  // namespace gcvo
