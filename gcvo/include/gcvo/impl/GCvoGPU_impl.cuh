/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: src/cvo/CvoGPU.cu, src/cvo/CvoGPU.cpp, include/UnifiedCvo/cvo/CvoGPU_impl.hpp, 
 *                include/UnifiedCvo/cvo/CvoGPU_impl.cuh (no 1:1 mapping claimed)
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

// Template implementations for gcvo::GCvoGPU<PointT>
//
// This file must be compiled with NVCC (it contains CUDA kernels).
//
// NOTE (March 2026): GCVO currently implements **only** the legacy second-order
// (Gauss-Newton) update path. Legacy first-order knobs in GCvoParams such as
// `dl`, `dl_step`, and `step` are intentionally ignored here.

#include "gcvo/GCvoGPU.hpp"

#include "gcvo/GCvoState.cuh"
#include "gcvo/impl/Correlation_impl.cuh"
#include "gcvo/LieGroup.hpp"

#include "cupointcloud/cupointcloud.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

#include <chrono>
#include <cmath>
#include <queue>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <utility>

#ifndef GCVO_CUDA_THREADS
#define GCVO_CUDA_THREADS 256
#endif

namespace gcvo {

// -----------------------------
// CUDA error helper
// -----------------------------
inline void gcvo_gpu_assert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GCVO GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) std::exit(static_cast<int>(code));
  }
}

#ifndef GCVO_GPU_CHECK
#define GCVO_GPU_CHECK(ans) ::gcvo::gcvo_gpu_assert((ans), __FILE__, __LINE__)
#endif

// -----------------------------
// Small math helpers
// -----------------------------
__host__ __device__ inline Eigen::Matrix3f skew3(const Eigen::Vector3f& v) {
  Eigen::Matrix3f m;
  m << 0.0f, -v.z(), v.y(),
       v.z(), 0.0f, -v.x(),
      -v.y(), v.x(), 0.0f;
  return m;
}

inline Eigen::Matrix4f exp_se3(const Eigen::Matrix<float, 6, 1>& xi) {
  const Eigen::Vector3f w = xi.head<3>();
  const Eigen::Vector3f v = xi.tail<3>();

  const float theta = w.norm();
  const Eigen::Matrix3f W = skew3(w);
  const Eigen::Matrix3f W2 = W * W;

  Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
  Eigen::Matrix3f V = Eigen::Matrix3f::Identity();

  if (theta < 1e-6f) {
    // Series expansions
    R = Eigen::Matrix3f::Identity() + W + 0.5f * W2;
    V = Eigen::Matrix3f::Identity() + 0.5f * W + (1.0f / 6.0f) * W2;
  } else {
    const float s = std::sin(theta);
    const float c = std::cos(theta);
    const float theta2 = theta * theta;
    const float theta3 = theta2 * theta;

    R = Eigen::Matrix3f::Identity() + (s / theta) * W + ((1.0f - c) / theta2) * W2;
    V = Eigen::Matrix3f::Identity() + ((1.0f - c) / theta2) * W + ((theta - s) / theta3) * W2;
  }

  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3, 3>(0, 0) = R;
  T.block<3, 1>(0, 3) = V * v;
  return T;
}

template <int N>
__host__ __device__ inline float squared_dist_arr(const float* a, const float* b) {
  float s = 0.0f;
  #pragma unroll
  for (int i = 0; i < N; ++i) {
    const float d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

template <typename PointT>
__host__ __device__ inline float squared_dist_xyz(const PointT& a, const PointT& b) {
  const float dx = a.x - b.x;
  const float dy = a.y - b.y;
  const float dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ Eigen::Matrix3f cov_sum_inv_plus_l2I(const float* cov_a,
                                                                  const float* cov_b,
                                                                  float l2) {
  // Covariance arrays are stored in row-major order [r*3+c].
  using Mat33RM = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
  Eigen::Matrix3f cov = Eigen::Map<const Mat33RM>(cov_a) + Eigen::Map<const Mat33RM>(cov_b);
  cov(0, 0) += l2;
  cov(1, 1) += l2;
  cov(2, 2) += l2;
  return cov.inverse();
}

// -----------------------------
// Point transform (thrust)
// -----------------------------
namespace cuda_detail {

template <typename PointT>
struct TransformPointRT {
  const Eigen::Matrix3f* R;
  const Eigen::Vector3f* t;
  const bool update_normal_and_cov;

  __host__ __device__ TransformPointRT(const Eigen::Matrix3f* R_, const Eigen::Vector3f* t_,
                                      bool update_) : R(R_), t(t_), update_normal_and_cov(update_) {}

  __host__ __device__ PointT operator()(const PointT& p_in) const {
    PointT p_out(p_in);

    Eigen::Vector3f x(p_in.x, p_in.y, p_in.z);
    Eigen::Vector3f x_t = (*R) * x + (*t);
    p_out.x = x_t.x();
    p_out.y = x_t.y();
    p_out.z = x_t.z();

    if (update_normal_and_cov) {
      if constexpr (gcvo::detail::has_normal<PointT>::value) {
        Eigen::Vector3f n(p_in.normal[0], p_in.normal[1], p_in.normal[2]);
        Eigen::Vector3f n_t = (*R) * n;  // rotation only
        p_out.normal[0] = n_t.x();
        p_out.normal[1] = n_t.y();
        p_out.normal[2] = n_t.z();
      }
      if constexpr (gcvo::detail::has_covariance<PointT>::value) {
        Eigen::Matrix3f C;
        C << p_in.covariance[0], p_in.covariance[1], p_in.covariance[2],
             p_in.covariance[3], p_in.covariance[4], p_in.covariance[5],
             p_in.covariance[6], p_in.covariance[7], p_in.covariance[8];
        Eigen::Matrix3f C_t = (*R) * C * R->transpose();
        #pragma unroll
        for (int r = 0; r < 3; ++r)
          for (int c = 0; c < 3; ++c)
            p_out.covariance[r * 3 + c] = C_t(r, c);
      }
    }
    return p_out;
  }
};

template <typename PointT>
inline void transform_pointcloud_thrust(const std::shared_ptr<CuPointCloud<PointT>>& in,
                                        const std::shared_ptr<CuPointCloud<PointT>>& out,
                                        const Eigen::Matrix3f* R_gpu,
                                        const Eigen::Vector3f* t_gpu,
                                        bool update_normal_and_cov) {
  thrust::transform(in->points.begin(), in->points.end(), out->points.begin(),
                   TransformPointRT<PointT>(R_gpu, t_gpu, update_normal_and_cov));
}

}  // namespace cuda_detail

// -----------------------------
// Correlation kernel
// -----------------------------

template <typename PointT>
__global__ void compute_correlation_kernel(const GCvoParams* params,
                                           const PointT* src_tf,
                                           int num_src,
                                           const PointT* tgt,
                                           int num_tgt,
                                           int k,
                                           float l,
                                           // output
                                           Correlation* corr) {
  constexpr int FD = static_cast<int>(PointT::FEATURE_DIMENSION);

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_src) return;

  const float amplitude2   = params->amplitude * params->amplitude;
  const float c_l2   = params->c_l * params->c_l;
  const float c_amplitude2 = params->c_amplitude * params->c_amplitude;
  const float sparsity_cutoff = params->sparsity_cutoff;

  // Early-gating thresholds (mirrors legacy CUDA code)
  float d2_c_thres = 1.0f;
  if (params->use_features) {
    d2_c_thres = -2.0f * c_l2 * logf(sparsity_cutoff / c_amplitude2);
  }
  float d2_thres = 1.0f;
  if (params->use_geometry) {
    d2_thres = -2.0f * logf(sparsity_cutoff / amplitude2);
  }

  unsigned int num_matches = 0;
  const PointT& p = src_tf[i];
  const float l2 = l * l;

  for (int idx = 0; idx < num_tgt && num_matches < static_cast<unsigned int>(k); ++idx) {
    const PointT& q = tgt[idx];

    float w_feat = 1.0f;
    if (params->use_features) {
      const float d2 = squared_dist_arr<FD>(p.features, q.features);
      if (d2 > d2_c_thres) continue;
      w_feat = c_amplitude2 * expf(-d2 / (2.0f * c_l2));
    }

    float w_geo = 1.0f;
    if (params->use_geometry) {
      if (params->kernel_type == GCvoKernelType::SCALAR) {
        const float d2 = squared_dist_xyz(p, q);
        if (d2 > params->kernel_eval_max_dist
            || d2 / l2  > d2_thres) continue;
        w_geo = amplitude2 * expf(-d2 / (2.0f * l2));
      } else {
        const Eigen::Matrix3f cov_inv = cov_sum_inv_plus_l2I(p.covariance, q.covariance, l2);
        Eigen::Vector3f dp(p.x - q.x, p.y - q.y, p.z - q.z);
        const float d2 = dp.dot(cov_inv * dp);
        if (!isfinite(d2) || d2 > params->kernel_eval_max_dist
            || d2 > d2_thres) continue;
        w_geo = amplitude2 * expf(-0.5f * d2);
      }
    }

    const float w_ij = w_geo * w_feat;
    if (w_ij > sparsity_cutoff) {
      corr->weights[i * k + num_matches] = w_ij;
      corr->target_indices[i * k + num_matches] = idx;
      num_matches++;
    }
  }

  corr->num_entries_per_row[i] = num_matches;
}

// -----------------------------
// GN Hessian kernel (LEGACY GCvoGPU.cu math)
//   - fixed/moving roles match legacy implementation
//   - residual r = x - y (x=fixed, y=moving transformed into fixed frame)
//   - Jacobian J = [ -skew(y)  I ]
// -----------------------------

template <typename PointT>
__global__ void compute_hessian_gn_kernel(const GCvoParams* params,
                                          const PointT* fixed,      // fixed points (cloud_x)
                                          const PointT* moving_tf,  // moving points in fixed frame (cloud_y transformed)
                                          const Correlation* corr,
                                          int num_neighbors,
                                          float l,
                                          // outputs
                                          Eigen::Matrix<float, 6, 6, Eigen::RowMajor>* H_i,
                                          Eigen::Matrix<float, 6, 1>* g_i) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= corr->num_sources) return;

  Eigen::Matrix<float, 6, 6, Eigen::RowMajor> Hi = Eigen::Matrix<float, 6, 6, Eigen::RowMajor>::Zero();
  Eigen::Matrix<float, 6, 1> gi = Eigen::Matrix<float, 6, 1>::Zero();

  const PointT& px = fixed[i];
  const Eigen::Vector3f x(px.x, px.y, px.z);

  const float l2 = l * l;
  const bool use_covariance = (params->kernel_type != GCvoKernelType::SCALAR);
  const float l2_inv = 1.0f / l2;

  // Legacy GCvoGPU.cu uses `num_neighbors` as the stride for corr (even though corr->max_entries_per_row == num_neighbors).
  for (int j = 0; j < num_neighbors; ++j) {
    const int idx = corr->target_indices[i * num_neighbors + j];
    if (idx < 0) break;
    const float w_ij = corr->weights[i * num_neighbors + j];
    const PointT& py = moving_tf[idx];
    const Eigen::Vector3f y(py.x, py.y, py.z);

    // Residual (legacy): r = x - y
    const Eigen::Vector3f r = x - y;

    // Jacobian (legacy): J = [ -skew(y), I ]
    const Eigen::Matrix3f skew_y = skew3(y);
    Eigen::Matrix<float, 3, 6> J;
    J.block<3, 3>(0, 0) = -skew_y;
    J.block<3, 3>(0, 3) = Eigen::Matrix3f::Identity();

    if (!use_covariance) {
      const float w = l2_inv;
      Hi.noalias() += w_ij * (J.transpose() * w * J);
      gi.noalias() += w_ij * (J.transpose() * w * r);
    } else {
      const Eigen::Matrix3f cov_inv = cov_sum_inv_plus_l2I(px.covariance, py.covariance, l2);
      Hi.noalias() += w_ij * (J.transpose() * cov_inv * J);
      gi.noalias() += w_ij * (J.transpose() * cov_inv * r);
    }
  }

  H_i[i] = Hi;
  g_i[i] = gi;
}

// -----------------------------
// Correspondence extraction
// -----------------------------
inline void gpu_correspondence_to_cpu(const Correlation& corr_dev,
                                   Correspondence& corresp,
                                   int num_rows,
                                   int num_cols,
                                   int num_neighbors) {
  const int rows = num_rows;
  const int cols = num_neighbors;
  if (rows <= 0 || cols <= 0) return;
  if (corr_dev.total_entries == 0) return;

  const int stride = num_neighbors;

  thrust::device_ptr<float> weights_ptr = thrust::device_pointer_cast(corr_dev.weights);
  thrust::device_ptr<int> indices_ptr = thrust::device_pointer_cast(corr_dev.target_indices);
  thrust::device_ptr<unsigned int> entries_ptr = thrust::device_pointer_cast(corr_dev.num_entries_per_row);

  // Copy full-stride buffers; we'll only read the first `cols` entries per row.
  thrust::host_vector<float> weights_host(weights_ptr, weights_ptr + rows * stride);
  thrust::host_vector<int> indices_host(indices_ptr, indices_ptr + rows * stride);
  thrust::host_vector<unsigned int> entries_host(entries_ptr, entries_ptr + rows);

  corresp.pairs.resize(rows, num_cols);
  corresp.source_inliers.clear();
  corresp.target_inliers.clear();

  for (int i = 0; i < rows; ++i) {
    if (entries_host[i] == 0) continue;
    corresp.source_inliers.push_back(i);
    for (int j = 0; j < cols; ++j) {
      const int idx = indices_host[i * stride + j];
      if (idx < 0) break;
      corresp.target_inliers.push_back(idx);
      corresp.pairs.insert(i, idx) = weights_host[i * stride + j];
    }
  }
  corresp.pairs.makeCompressed();
}

// -----------------------------
// Fast reductions on device buffers (avoid per-iter host vectors / device_vector copies)
// -----------------------------
inline unsigned int total_entries_device(Correlation& corr_host) {
  thrust::device_ptr<unsigned int> entries_ptr = thrust::device_pointer_cast(corr_host.num_entries_per_row);
  const unsigned int sum = thrust::reduce(thrust::device, entries_ptr, entries_ptr + corr_host.num_sources, 0u);
  corr_host.total_entries = sum;
  return sum;
}

inline unsigned int max_entries_device(const Correlation& corr_host) {
  thrust::device_ptr<unsigned int> entries_ptr = thrust::device_pointer_cast(corr_host.num_entries_per_row);
  return *thrust::max_element(thrust::device, entries_ptr, entries_ptr + corr_host.num_sources);
}

inline float correlation_sum_device(const Correlation& corr_host, int num_neighbors) {
  thrust::device_ptr<float> weights_ptr = thrust::device_pointer_cast(corr_host.weights);
  return thrust::reduce(thrust::device, weights_ptr, weights_ptr + corr_host.num_sources * num_neighbors, 0.0f);
}

// -----------------------------
// L decay indicator — queue-based (original)
// -----------------------------
static inline bool should_decay_l(std::queue<float>& decay_q_start,
                                    std::queue<float>& decay_q_end,
                                    float& decay_sum_start,
                                    float& decay_sum_end,
                                    float indicator,
                                    const GCvoParams& params) {
  const int win = params.indicator_window;

  if (static_cast<int>(decay_q_start.size()) < win) {
    decay_q_start.push(indicator);
    decay_sum_start += indicator;
    return false;
  }
  if (static_cast<int>(decay_q_end.size()) < win) {
    decay_q_end.push(indicator);
    decay_sum_end += indicator;
  }

  if (static_cast<int>(decay_q_end.size()) >= win) {
    const float ratio = (decay_sum_start > 1e-12f) ? (decay_sum_end / decay_sum_start) : 1.0f;
    const float lo = 1.0f - params.indicator_threshold;
    const float hi = 1.0f + params.indicator_threshold;
    if (ratio > lo && ratio < hi) {
      std::queue<float> empty1, empty2;
      std::swap(decay_q_start, empty1);
      std::swap(decay_q_end, empty2);
      decay_sum_start = 0.0f;
      decay_sum_end   = 0.0f;
      return true;
    }
    // slide window
    decay_sum_end   -= decay_q_end.front();
    decay_sum_start += decay_q_end.front();
    decay_q_start.push(decay_q_end.front());
    decay_q_end.pop();
    decay_sum_start -= decay_q_start.front();
    decay_q_start.pop();
    decay_q_end.push(indicator);
    decay_sum_end += indicator;
  }
  return false;
}

// -----------------------------
// L decay indicator — EMA-based
// -----------------------------
// Maintains two EMA accumulators (slow="start", fast="end") to detect when
// the inner-product indicator has stabilized. Triggers l decay when
// end_ema / start_ema falls within [1 ± indicator_threshold].
//
// All mutable state (ema, start_ema, end_ema, flags, cooldown) is passed by
// reference so each align() call has its own independent state.
//
// alpha: EMA smoothing factor for the raw indicator (0 < alpha <= 1).
// cooldown_iters: after a decay, suppress further decays for this many iters;
//                 initialize cooldown_iters_left = l_decay_start to skip
//                 the warm-up phase.
static inline bool should_decay_l_ema(
    float& ema,               // state: smoothed indicator EMA
    bool&  ema_initialized,   // state: init flag for ema
    int&   cooldown_iters_left, // state: remaining cooldown iters (init to l_decay_start)
    float& start_ema,         // state: slow EMA ("start" window)
    float& end_ema,           // state: fast EMA ("end" window)
    bool&  ema_windows_inited,// state: init flag for start/end EMAs
    int&   ema_warmup_left,   // state: warmup counter (init to indicator_window)
    float  indicator,         // input: raw indicator value (e.g. correlation_sum)
    const GCvoParams& params,
    float  alpha            = 0.2f,   // smoothing for raw indicator
    float  min_indicator    = 1e-6f,  // guard against near-zero denom
    int    cooldown_iters   = 0       // 0 disables post-decay cooldown
) {
  // Step 0: decrement cooldown counter each iteration.
  if (cooldown_iters_left > 0) {
    cooldown_iters_left--;
  }

  // Step 1: update smoothed EMA of the raw indicator.
  if (!ema_initialized) {
    ema = indicator;
    ema_initialized = true;
  } else {
    ema = (1.0f - alpha) * ema + alpha * indicator;
  }

  // Step 2: update slow ("start") and fast ("end") EMA windows.
  // EMA rates derived from indicator_window so that effective memory
  // is ~win iters (fast) and ~2*win iters (slow).
  const int   win    = std::max(1, params.indicator_window);
  const float a_fast = std::min(1.0f, 2.0f / float(win     + 1));
  const float a_slow = std::min(1.0f, 2.0f / float(2*win   + 1));

  if (!ema_windows_inited) {
    start_ema = ema;
    end_ema   = ema;
    ema_windows_inited = true;
    return false;
  }

  start_ema = (1.0f - a_slow) * start_ema + a_slow * ema;
  end_ema   = (1.0f - a_fast) * end_ema   + a_fast * ema;

  // Step 3: fire decay if indicator has stabilized, cooldown has elapsed, and
  // the windows have had enough iterations to diverge meaningfully.
  if (cooldown_iters_left > 0) return false;

  if (ema_warmup_left > 0) {
    ema_warmup_left--;
    return false;
  }

  const float denom = std::max(std::fabs(start_ema), min_indicator);
  const float ratio = end_ema / denom;
  const float lo    = 1.0f - params.indicator_threshold;
  const float hi    = 1.0f + params.indicator_threshold;

  if (ratio > lo && ratio < hi) {
    if (cooldown_iters > 0) cooldown_iters_left = cooldown_iters;
    // Reset windows so next decay cycle starts fresh.
    ema_windows_inited = false;
    ema_warmup_left    = win;
    return true;
  }
  return false;
}

// -----------------------------
// PointCloud -> GPU helper
// -----------------------------
template <typename PointT>
static std::shared_ptr<CuPointCloud<PointT>> pointcloud_to_gpu(const GCvoPointCloudT<PointT>& cloud) {
  using GpuCloud = CuPointCloud<PointT>;
  typename GpuCloud::HostVectorType host(cloud.points().begin(), cloud.points().end());
  auto gpu_cloud = std::make_shared<GpuCloud>();
  gpu_cloud->points = host;
  return gpu_cloud;
}

// -----------------------------
// GCvoGPU methods
// -----------------------------

template <typename PointT>
GCvoGPU<PointT>::GCvoGPU(const std::string& yaml_param_file) {
  read_GCvoParams_yaml(yaml_param_file.c_str(), &params_);
  GCVO_GPU_CHECK(cudaMalloc((void**)&params_gpu_, sizeof(GCvoParams)));
  GCVO_GPU_CHECK(cudaMemcpy(params_gpu_, &params_, sizeof(GCvoParams), cudaMemcpyHostToDevice));
}

template <typename PointT>
GCvoGPU<PointT>::GCvoGPU(const GCvoParams& params) : params_(params) {
  GCVO_GPU_CHECK(cudaMalloc((void**)&params_gpu_, sizeof(GCvoParams)));
  GCVO_GPU_CHECK(cudaMemcpy(params_gpu_, &params_, sizeof(GCvoParams), cudaMemcpyHostToDevice));
}

template <typename PointT>
GCvoGPU<PointT>::~GCvoGPU() {
  if (params_gpu_) cudaFree(params_gpu_);
  params_gpu_ = nullptr;
}

template <typename PointT>
void GCvoGPU<PointT>::write_params(const GCvoParams& p_cpu) {
  params_ = p_cpu;
  GCVO_GPU_CHECK(cudaMemcpy(params_gpu_, &params_, sizeof(GCvoParams), cudaMemcpyHostToDevice));
}

template <typename PointT>
static Eigen::Matrix<float, 6, 1> compute_gn_update(const GCvoParams& params_cpu,
                                                    const GCvoParams* params_gpu,
                                                    GCvoStateT<PointT>& state,
                                                    int num_neighbors,
                                                    // optional debug outputs
                                                    typename GCvoStateT<PointT>::Mat66* H_out = nullptr,
                                                    typename GCvoStateT<PointT>::Vec6* g_out = nullptr) {
  using Mat66 = typename GCvoStateT<PointT>::Mat66;
  using Vec6  = typename GCvoStateT<PointT>::Vec6;

  const int threads = GCVO_CUDA_THREADS;
  const int blocks = (state.num_fixed + threads - 1) / threads;

  compute_hessian_gn_kernel<PointT><<<blocks, threads>>>(
      params_gpu,
      thrust::raw_pointer_cast(state.cloud_fixed->points.data()),
      thrust::raw_pointer_cast(state.cloud_moving_tf->points.data()),
      state.corr,
      num_neighbors,
      state.l,
      thrust::raw_pointer_cast(state.H_rows.data()),
      thrust::raw_pointer_cast(state.g_rows.data()));
  GCVO_GPU_CHECK(cudaPeekAtLastError());

  const Mat66 H0 = Mat66::Zero();
  const Vec6 g0 = Vec6::Zero();
  const Mat66 H = thrust::reduce(thrust::device, state.H_rows.begin(), state.H_rows.end(), H0);
  const Vec6 g = thrust::reduce(thrust::device, state.g_rows.begin(), state.g_rows.end(), g0);

  if (H_out) *H_out = H;
  if (g_out) *g_out = g;

  // Optional connection term: adds an antisymmetric correction to the Hessian.
  Mat66 Hmod = H;
  const Eigen::Vector3f gw = g.template head<3>();
  const Eigen::Vector3f gv = g.template tail<3>();
  const Eigen::Matrix3f gw_skew = skew3(gw);
  const Eigen::Matrix3f gv_skew = skew3(gv);

  Mat66 gamma = Mat66::Zero();
  gamma.template block<3,3>(0,0) = -0.5f * gw_skew;
  gamma.template block<3,3>(3,0) = -gv_skew;

  Hmod = (Hmod + gamma.transpose()).eval();

  // GN solve: dx = -H^{-1} Jr
  const Mat66 Hsym = 0.5f * (Hmod + Hmod.transpose());
  Eigen::LDLT<Mat66> ldlt(Hsym);
  Vec6 dx = -ldlt.solve(g);
  if (!dx.array().isFinite().all()) {
    dx.setZero();
  }
  return dx;
}

template <typename PointT>
int GCvoGPU<PointT>::align(const PointCloud& source,
                                     const PointCloud& target,
                                     const Eigen::Matrix4f& T_s2t_init,
                                     Eigen::Matrix4f& T_s2t_out,
                                     Correspondence* correspondence,
                                     int* num_iters,
                                     double* registration_seconds) const {
  if (source.size() == 0 || target.size() == 0) {
    if (num_iters) *num_iters = 0;
    if (registration_seconds) *registration_seconds = 0.0;
    T_s2t_out = T_s2t_init;
    return -1;
  }

  auto fixed_gpu = pointcloud_to_gpu<PointT>(source);
  auto moving_gpu = pointcloud_to_gpu<PointT>(target);

  GCvoParams params_align = params_;
  GCvoStateT<PointT> state(fixed_gpu, moving_gpu, params_align);

  int k = std::max(1, params_align.max_neighbors);
  int k_last_used = k;

  const Eigen::Matrix4f T_t2s_init = T_s2t_init.inverse();
  Eigen::Matrix3f R = T_t2s_init.block<3, 3>(0, 0);
  Eigen::Vector3f t = T_t2s_init.block<3, 1>(0, 3);

  // Queue-based l-decay state (original)
  std::queue<float> decay_q_start, decay_q_end;
  float decay_sum_start = 0.0f;
  float decay_sum_end   = 0.0f;
  // EMA-based l-decay state (all per-call, no static storage)
  float ema_val       = 0.0f;
  bool  ema_init      = false;
  int   ema_cooldown  = params_align.l_decay_start;
  float ema_start     = 0.0f;
  float ema_end       = 0.0f;
  bool  ema_wins_init = false;
  int   ema_warmup    = params_align.indicator_window;

  cudaEvent_t ev_start, ev_stop;
  cudaEventCreate(&ev_start);
  cudaEventCreate(&ev_stop);
  cudaEventRecord(ev_start, 0);

  int it = 0;
  for (; it < params_align.max_iterations; ++it) {
    if (params_align.verbose) std::cout << "===========================\n";

    k_last_used = k;
    state.reset_state_at_new_iter(k);

    // Inverse transform for mapping moving->fixed.
    const Eigen::Matrix3f R_inv = R.transpose();
    const Eigen::Vector3f t_inv = -R_inv * t;
    GCVO_GPU_CHECK(cudaMemcpyAsync(state.R_gpu, &R_inv, sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
    GCVO_GPU_CHECK(cudaMemcpyAsync(state.t_gpu, &t_inv, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));

    // Transform moving cloud into fixed frame.
    const bool update_normal_cov = (params_align.kernel_type != GCvoKernelType::SCALAR);
    cuda_detail::transform_pointcloud_thrust<PointT>(state.cloud_moving_init, state.cloud_moving_tf,
                                                     state.R_gpu, state.t_gpu, update_normal_cov);

    // Fill correlation (bruteforce)
    const int threads = GCVO_CUDA_THREADS;
    const int blocks = (state.num_fixed + threads - 1) / threads;
    compute_correlation_kernel<PointT><<<blocks, threads>>>(
        params_gpu_,
        thrust::raw_pointer_cast(state.cloud_fixed->points.data()), state.num_fixed,
        thrust::raw_pointer_cast(state.cloud_moving_tf->points.data()), state.num_moving,
        k,
        state.l,
        state.corr);
    GCVO_GPU_CHECK(cudaPeekAtLastError());

    const unsigned int nnz_sum = total_entries_device(state.corr_host);
    if (nnz_sum < 100) {
      if (params_align.verbose) {
        std::cout << "[gcvo][debug] it=" << it << " total_entries=" << nnz_sum << " -> break (too sparse)\n";
      }
      break;
    }

    typename GCvoStateT<PointT>::Mat66 H_dbg;
    typename GCvoStateT<PointT>::Vec6  g_dbg;
    const bool dbg = (params_align.verbose != 0);

    const Eigen::Matrix<float, 6, 1> dx = compute_gn_update<PointT>(
        params_align, params_gpu_, state, k,
        dbg ? &H_dbg : nullptr,
        dbg ? &g_dbg : nullptr);

    const float dx_norm = dx.norm();
    if (dx_norm < params_align.tol_2) {
      if (params_align.kernel_type == GCvoKernelType::SCALAR) {
        if ( state.l <= params_align.l_min) {
          if (dbg)  {
            std::cout << "[gcvo][debug] it=" << it << " dx_norm=" << dx_norm << " < tol -> break\n";
          }
          break;
        }
      } else
        break;
    }

    const Eigen::Matrix<double, 6, 1> dx_d = dx.cast<double>();
    const Eigen::Matrix<double, 3, 4, Eigen::RowMajor> dRt = gcvo::liegroup::Exp_SE3<double, Eigen::RowMajor>(dx_d, true);
    const Eigen::Matrix3d dR = dRt.block<3, 3>(0, 0);
    const Eigen::Vector3d dt = dRt.block<3, 1>(0, 3);

    // RIGHT composition on T_t2s
    t = (R.cast<double>() * dt + t.cast<double>()).cast<float>();
    R = (R.cast<double>() * dR).cast<float>();

    if (dbg) {
      const float sum_corr = correlation_sum_device(state.corr_host, k);
      std::cout << "[gcvo][debug] it=" << it
                << " k=" << k
                << " l=" << state.l
                << " total_entries=" << nnz_sum
                << " dx_norm=" << dx_norm << "\n";
      std::cout << "[gcvo][debug] correlation_sum = " << sum_corr << "\n";
      std::cout << "[gcvo][debug] dx^T = " << dx.transpose() << "\n";
      std::cout << "[gcvo][debug] dT =\n" << dRt << "\n";
      std::cout << "[gcvo][debug] g^T = " << g_dbg.transpose() << "\n";
      std::cout << "[gcvo][debug] H =\n" << H_dbg << "\n";
      std::cout << "[gcvo][debug] R =\n" << R << "\n";
      std::cout << "[gcvo][debug] t^T = " << t.transpose() << "\n";
      std::cout << "[gcvo][debug] dist = " << dx_norm << "\n";
    }

    // Optionally decay l for SCALAR kernel
    if (params_align.kernel_type == GCvoKernelType::SCALAR) {
      const float corr_indicator = correlation_sum_device(state.corr_host, k);
      bool decay = false;
      if (params_align.use_ema_indicator) {
        decay = should_decay_l_ema(
            ema_val, ema_init, ema_cooldown,
            ema_start, ema_end, ema_wins_init, ema_warmup,
            corr_indicator, params_align);
      } else {
        decay = (it > params_align.l_decay_start) &&
                should_decay_l(decay_q_start, decay_q_end, decay_sum_start, decay_sum_end,
                                 corr_indicator, params_align);
      }
      if (decay && state.l > params_align.l_min) {
        state.l = std::max(params_align.l_min, state.l * params_align.l_decay_rate);
      }
    }

    // Adaptive k: resize based on current max neighbors used.
    if (params_align.neighbor_decay) {
      const unsigned int max_nn = max_entries_device(state.corr_host);
      if (params_align.verbose) {
        std::cout << "[gcvo][debug] max number of neighbors is " << max_nn << "\n";
      }
      k = std::min(params_align.max_neighbors,
                   std::max(1, static_cast<int>(max_nn * 1.2f)));
    }
  }

  cudaEventRecord(ev_stop, 0);
  cudaEventSynchronize(ev_stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, ev_start, ev_stop);
  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_stop);

  if (registration_seconds) *registration_seconds = static_cast<double>(ms) / 1000.0;
  if (num_iters) *num_iters = it;

  // Return to public convention: T_s2t = inverse(T_t2s)
  Eigen::Matrix4f T_t2s = Eigen::Matrix4f::Identity();
  T_t2s.block<3, 3>(0, 0) = R;
  T_t2s.block<3, 1>(0, 3) = t;
  T_s2t_out = T_t2s.inverse();

  if (correspondence) {
    gpu_correspondence_to_cpu(state.corr_host, *correspondence, state.num_fixed, state.num_moving, k_last_used);
  }
  return 0;
}

template <typename PointT>
GCvoResultInfo GCvoGPU<PointT>::align(const PointCloud& source,
                                    const PointCloud& target,
                                    const Eigen::Matrix4f& T_s2t_init,
                                    bool return_correspondence) const {
  GCvoResultInfo out;
  out.T_s2t = T_s2t_init;
  out.return_code = align(source, target, T_s2t_init, out.T_s2t,
                          return_correspondence ? &out.correspondence : nullptr,
                          &out.num_iters, &out.registration_seconds);
  return out;
}

template <typename PointT>
float GCvoGPU<PointT>::inner_product_(const PointCloud& source,
                                         const PointCloud& target,
                                         const Eigen::Matrix4f& T_s2t,
                                         float l,
                                         Correspondence* corresp_out) const {
  if (source.size() == 0 || target.size() == 0) return 0.0f;

  auto src_gpu = pointcloud_to_gpu<PointT>(source);
  auto tgt_gpu = pointcloud_to_gpu<PointT>(target);
  GCvoStateT<PointT> state(src_gpu, tgt_gpu, params_);
  state.l = l;

  const int k = std::max(1, params_.max_neighbors);

  // Apply inverse(T_s2t) to map moving->fixed.
  const Eigen::Matrix4f T_inv = T_s2t.inverse();
  const Eigen::Matrix3f R_inv = T_s2t.block<3, 3>(0, 0);
  const Eigen::Vector3f t_inv = T_s2t.block<3, 1>(0, 3);
  GCVO_GPU_CHECK(cudaMemcpy(state.R_gpu, &R_inv, sizeof(Eigen::Matrix3f), cudaMemcpyHostToDevice));
  GCVO_GPU_CHECK(cudaMemcpy(state.t_gpu, &t_inv, sizeof(Eigen::Vector3f), cudaMemcpyHostToDevice));

  const bool update_normal_cov = (params_.kernel_type != GCvoKernelType::SCALAR);
  cuda_detail::transform_pointcloud_thrust<PointT>(state.cloud_moving_init, state.cloud_moving_tf,
                                                   state.R_gpu, state.t_gpu, update_normal_cov);
  GCVO_GPU_CHECK(cudaPeekAtLastError());

  state.reset_state_at_new_iter(k);

  const int threads = GCVO_CUDA_THREADS;
  const int blocks = (state.num_fixed + threads - 1) / threads;

  compute_correlation_kernel<PointT><<<blocks, threads>>>(
      params_gpu_,
      thrust::raw_pointer_cast(state.cloud_fixed->points.data()), state.num_fixed,
      thrust::raw_pointer_cast(state.cloud_moving_tf->points.data()), state.num_moving,
      k,
      state.l,
      state.corr);
  GCVO_GPU_CHECK(cudaPeekAtLastError());
  total_entries_device(state.corr_host);

  const float ip = correlation_sum_device(state.corr_host, k);

  if (corresp_out) {
    gpu_correspondence_to_cpu(state.corr_host, *corresp_out, state.num_fixed, state.num_moving, k);
  }
  return ip;
}

template <typename PointT>
float GCvoGPU<PointT>::cos(const PointCloud& source,
                                     const PointCloud& target,
                                     const Eigen::Matrix4f& T_s2t,
                                     float l,
                                     bool is_approximate) const {
  if (source.size() == 0 || target.size() == 0) return 0.0f;
  const float fxy = inner_product_(source, target, T_s2t, l, nullptr);

  float fx_norm = 0.0f;
  float fy_norm = 0.0f;
  if (is_approximate) {
    fx_norm = std::sqrt(static_cast<float>(source.size()));
    fy_norm = std::sqrt(static_cast<float>(target.size()));
  } else {
    const Eigen::Matrix4f I = Eigen::Matrix4f::Identity();
    fx_norm = std::sqrt(std::max(0.0f, inner_product_(source, source, I, l, nullptr)));
    fy_norm = std::sqrt(std::max(0.0f, inner_product_(target, target, I, l, nullptr)));
  }

  const float denom = fx_norm * fy_norm;
  if (denom < 1e-12f) return 0.0f;
  return fxy / denom;
}

}  // namespace gcvo
