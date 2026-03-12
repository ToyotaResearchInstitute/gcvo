/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: src/cvo/SparseKernelMat.cu (no 1:1 mapping claimed)
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

#include "gcvo/Correlation.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

#ifndef GCVO_GPU_CHECK
static inline void gcvo_sparse_gpuErrorCheck(cudaError_t code, const char* file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    std::exit(code);
  }
}
#define GCVO_GPU_CHECK(ans) gcvo_sparse_gpuErrorCheck((ans), __FILE__, __LINE__)
#endif

namespace gcvo {

inline unsigned int count_entries(Correlation* A_host) {
  return A_host ? A_host->total_entries : 0u;
}

inline void compute_total_entries(Correlation* A_host) {
  if (!A_host) return;
  A_host->total_entries = 0;
  std::vector<unsigned int> v(A_host->num_sources);
  cudaMemcpy(v.data(), A_host->num_entries_per_row, sizeof(unsigned int) * A_host->num_sources, cudaMemcpyDeviceToHost);
  A_host->total_entries = std::accumulate(v.begin(), v.end(), 0u);
}

inline unsigned int max_entries_in_row(Correlation* A_host) {
  if (!A_host) return 0u;
  std::vector<unsigned int> v(A_host->num_sources);
  cudaMemcpy(v.data(), A_host->num_entries_per_row, sizeof(unsigned int) * A_host->num_sources, cudaMemcpyDeviceToHost);
  return *std::max_element(v.begin(), v.end());
}

inline float correlation_sum(Correlation* A_host) {
  if (!A_host) return 0.0f;
  thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->weights);
  thrust::device_vector<float> v(A_ptr, A_ptr + A_host->num_sources * A_host->max_entries_per_row);
  return thrust::reduce(v.begin(), v.end(), 0.0f);
}

inline float correlation_sum(Correlation* A_host, int num_neighbors) {
  if (!A_host) return 0.0f;
  thrust::device_ptr<float> A_ptr = thrust::device_pointer_cast(A_host->weights);
  thrust::device_vector<float> v(A_ptr, A_ptr + A_host->num_sources * num_neighbors);
  return thrust::reduce(v.begin(), v.end(), 0.0f);
}

inline void clear_Correlation(Correlation* A_host) {
  if (!A_host) return;
  A_host->total_entries = 0;
  cudaMemset(A_host->weights, 0, A_host->num_sources * A_host->max_entries_per_row * sizeof(float));
  cudaMemset(A_host->target_indices, -1, A_host->num_sources * A_host->max_entries_per_row * sizeof(int));
  cudaMemset(A_host->num_entries_per_row, 0, A_host->num_sources * sizeof(unsigned int));
}

inline void clear_Correlation(Correlation* A_host, int num_neighbors) {
  if (!A_host) return;
  A_host->total_entries = 0;
  cudaMemset(A_host->weights, 0, A_host->num_sources * num_neighbors * sizeof(float));
  cudaMemset(A_host->target_indices, -1, A_host->num_sources * num_neighbors * sizeof(int));
  cudaMemset(A_host->num_entries_per_row, 0, A_host->num_sources * sizeof(unsigned int));
}

inline Correlation* init_Correlation_gpu(int row, int col, Correlation& A_host) {
  Correlation* A_dev = nullptr;
  GCVO_GPU_CHECK(cudaMalloc((void**)&A_dev, sizeof(Correlation)));

  A_host.num_sources = row;
  A_host.max_entries_per_row = col;
  A_host.total_entries = 0;

  GCVO_GPU_CHECK(cudaMalloc((void**)&A_host.weights, sizeof(float) * row * col));
  GCVO_GPU_CHECK(cudaMalloc((void**)&A_host.target_indices, sizeof(int) * row * col));
  GCVO_GPU_CHECK(cudaMalloc((void**)&A_host.num_entries_per_row, sizeof(unsigned int) * row));

  clear_Correlation(&A_host);

  GCVO_GPU_CHECK(cudaMemcpy((void*)A_dev, &A_host, sizeof(Correlation), cudaMemcpyHostToDevice));
  return A_dev;
}

inline void delete_Correlation_gpu(Correlation* A_gpu, Correlation* A_host) {
  if (!A_host) return;
  cudaFree(A_host->weights);
  cudaFree(A_host->target_indices);
  cudaFree(A_host->num_entries_per_row);
  if (A_gpu) cudaFree(A_gpu);
}

} // namespace gcvo
