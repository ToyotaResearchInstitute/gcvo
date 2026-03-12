/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: thirdparty/cugicp/cukdtree/cukdtree.cuh  (no 1:1 mapping claimed)
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

#include <cuda_runtime.h>

#include <thrust/device_vector.h>

#include "common/gpuutil.h"
#include "cukdtree/cukdtree.h"

namespace perl_registration {

namespace detail {

template <typename PointT>
__device__ __forceinline__ float pDist(const PointT& a, const PointT& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

template <int KMax>
__device__ __forceinline__ void insert_best(float d, int idx, float* best_d, int* best_i, int k) {
  // Keep best_d sorted ascending (insertion sort into fixed array).
  int pos = k;
  for (int i = 0; i < k; ++i) {
    if (d < best_d[i]) {
      pos = i;
      break;
    }
  }
  if (pos >= k) return;
  for (int j = k - 1; j > pos; --j) {
    best_d[j] = best_d[j - 1];
    best_i[j] = best_i[j - 1];
  }
  best_d[pos] = d;
  best_i[pos] = idx;
}

template <typename PointT, int KMax>
__global__ void knn_bruteforce_kernel(const PointT* ref,
                                     int n_ref,
                                     const PointT* query,
                                     int n_query,
                                     int k,
                                     int* out_indices,
                                     int out_stride) {
  const int qid = blockIdx.x * blockDim.x + threadIdx.x;
  if (qid >= n_query) return;

  float best_d[KMax];
  int best_i[KMax];
  #pragma unroll
  for (int i = 0; i < KMax; ++i) {
    best_d[i] = 1e30f;
    best_i[i] = -1;
  }

  const PointT q = query[qid];
  for (int rid = 0; rid < n_ref; ++rid) {
    const float d = pDist<PointT>(ref[rid], q);
    insert_best<KMax>(d, rid, best_d, best_i, k);
  }

  int* row = out_indices + qid * out_stride;
  for (int i = 0; i < k; ++i) row[i] = best_i[i];
}

}  // namespace detail

template <typename PointT, int KMax_>
void cuKdTree<PointT, KMax_>::SetInputCloud(cuPointCloudSharedPtr& d_cloud) {
  d_point_cloud_ = d_cloud;
  input_cloud_set_ = static_cast<bool>(d_point_cloud_);
}

template <typename PointT, int KMax_>
int cuKdTree<PointT, KMax_>::NearestKSearch(const cuPointCloudSharedPtr& d_query_points,
                                            int k,
                                            thrust::device_vector<int>& indices) {
  const int kk = (k > KMax_) ? KMax_ : k;
  const size_t n = d_query_points->size();
  indices.resize(n * static_cast<size_t>(kk));
  return NearestKSearch(d_query_points, kk,
                        thrust::raw_pointer_cast(indices.data()),
                        static_cast<int>(indices.size()));
}

template <typename PointT, int KMax_>
int cuKdTree<PointT, KMax_>::NearestKSearch(const cuPointCloudSharedPtr& d_query_points,
                                            int k,
                                            int* indices,
                                            int indices_size) {
  if (!input_cloud_set_ || !d_point_cloud_) return 0;
  const int kk = (k > KMax_) ? KMax_ : k;
  const int n_query = static_cast<int>(d_query_points->size());
  const int need = n_query * kk;
  if (indices_size < need) return 0;

  const int threads = 128;
  const int blocks = (n_query + threads - 1) / threads;
  detail::knn_bruteforce_kernel<PointT, KMax_><<<blocks, threads>>>(
      thrust::raw_pointer_cast(d_point_cloud_->points.data()),
      static_cast<int>(d_point_cloud_->size()),
      thrust::raw_pointer_cast(d_query_points->points.data()),
      n_query,
      kk,
      indices,
      kk);
  cudaSafe(cudaDeviceSynchronize());
  cudaSafe(cudaGetLastError());
  return kk;
}

}  // namespace perl_registration
