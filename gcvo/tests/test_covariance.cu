/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: N/A (no 1:1 mapping claimed)
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

// gcvo/tests/test_covariance.cu
//
// Covariance / KD-tree sanity test for GCVO:
//  1. Basic checks: covariance is symmetric, positive diagonal, finite.
//  2. Geometric check: CPU brute-force KNN recomputes the sample covariance
//     and verifies it matches the GPU result.  Also checks that neighbors
//     fall within the covariance lipsoid (bounded Mahalanobis distance).

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gcvo/utils/PointTypes19.hpp"
#include "gcvo/utils/GCvoPointCloud.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

namespace {

struct Args {
  std::string input_pcd_file;
  int n = 2000;
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --input_pcd_file <bunny.pcd> [--n 2000]\n\n"
      << "Tests that compute_covariance() produces valid and geometrically\n"
      << "consistent covariance matrices.\n";
}

static bool parse_args(int argc, char** argv, Args& a) {
  for (int i = 1; i < argc; ++i) {
    const std::string k(argv[i]);
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "Missing value for " << name << "\n";
        std::exit(2);
      }
      return std::string(argv[++i]);
    };

    if (k == "--input_pcd_file") a.input_pcd_file = need("--input_pcd_file");
    else if (k == "--n") a.n = std::stoi(need("--n"));
    else if (k == "-h" || k == "--help") return false;
    else {
      std::cerr << "Unknown arg: " << k << "\n";
      return false;
    }
  }
  return !a.input_pcd_file.empty() && a.n > 10;
}

using PointT = gcvo::PointS1;
using CloudT = gcvo::GCvoPointCloudT<PointT>;

static CloudT load_xyz_as_pointsem1(const std::string& fname) {
  pcl::PointCloud<pcl::PointXYZ> pc;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(fname, pc) != 0) {
    throw std::runtime_error("Failed to load PCD: " + fname);
  }

  pcl::PointCloud<PointT> out;
  out.resize(pc.size());
  for (size_t i = 0; i < pc.size(); ++i) {
    PointT p;
    p.x = pc[i].x;
    p.y = pc[i].y;
    p.z = pc[i].z;
    p.features[0] = 0.0f;
    p.label = -1;
    out[i] = p;
  }
  return CloudT(out);
}

static CloudT subsample(const CloudT& in, int n, std::mt19937& rng) {
  const int N = in.size();
  if (n <= 0 || n >= N) return in;

  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::shuffle(idx.begin(), idx.end(), rng);
  idx.resize(n);

  pcl::PointCloud<PointT> pc;
  pc.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    pc[static_cast<size_t>(i)] = in.points()[static_cast<size_t>(idx[static_cast<size_t>(i)])];
  }
  return CloudT(pc);
}

// Brute-force KNN on CPU.  Returns indices of k nearest neighbors for point i
// (sorted by distance, includes self).
static std::vector<int> cpu_knn(const CloudT& cloud, int i, int k) {
  const int n = cloud.size();
  const auto& pi = cloud[i];
  const Eigen::Vector3f xi(pi.x, pi.y, pi.z);

  // Compute distances to all points
  std::vector<std::pair<float, int>> dists(n);
  for (int j = 0; j < n; ++j) {
    const auto& pj = cloud[j];
    const Eigen::Vector3f xj(pj.x, pj.y, pj.z);
    dists[j] = {(xi - xj).squaredNorm(), j};
  }

  // Partial sort to get k nearest
  std::partial_sort(dists.begin(), dists.begin() + k, dists.end());
  std::vector<int> inds(k);
  for (int j = 0; j < k; ++j) inds[j] = dists[j].second;
  return inds;
}

// Compute sample covariance from a set of neighbor indices (same formula as GPU kernel).
static Eigen::Matrix3f cpu_sample_covariance(const CloudT& cloud,
                                              const std::vector<int>& inds,
                                              Eigen::Vector3f& mean_out) {
  const int cnt = static_cast<int>(inds.size());

  // Compute mean
  Eigen::Vector3f mean = Eigen::Vector3f::Zero();
  for (int idx : inds) {
    const auto& p = cloud[idx];
    mean += Eigen::Vector3f(p.x, p.y, p.z);
  }
  mean /= static_cast<float>(cnt);
  mean_out = mean;

  // Compute covariance with Bessel correction (1/(cnt-1))
  Eigen::Matrix3f cov = Eigen::Matrix3f::Zero();
  for (int idx : inds) {
    const auto& p = cloud[idx];
    const Eigen::Vector3f d = Eigen::Vector3f(p.x, p.y, p.z) - mean;
    cov += d * d.transpose();
  }
  cov /= static_cast<float>(cnt - 1);

  // Add the same regularization as the GPU kernel
  cov(0, 0) += 1e-6f;
  cov(1, 1) += 1e-6f;
  cov(2, 2) += 1e-6f;

  return cov;
}

static int failures = 0;

static void check(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "FAIL: " << msg << "\n";
    ++failures;
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    usage(argv[0]);
    return 2;
  }

  std::mt19937 rng(42);

  CloudT cloud = load_xyz_as_pointsem1(a.input_pcd_file);
  check(cloud.size() > 10, "input cloud too small");
  cloud = subsample(cloud, a.n, rng);

  const int n = cloud.size();
  const int K = 12;  // same as the num_neighbors we pass to compute_covariance
  std::cout << "Loaded " << n << " points from " << a.input_pcd_file << "\n";

  // ---- Test 1: basic validity checks ----
  std::cout << "Running compute_covariance (K=" << K << ", use_kdtree=true)...\n";
  cloud.compute_covariance(0.0f, 1000.0f, K, 8, false, true);

  int cov_ok = 0;

  for (int i = 0; i < n; ++i) {
    const auto& p = cloud[i];

    // Covariance: symmetric, positive diagonal, finite
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> cov(p.covariance);
    bool cov_valid = true;
    for (int r = 0; r < 3; ++r) {
      for (int c = 0; c < 3; ++c)
        if (!std::isfinite(cov(r, c))) cov_valid = false;
      if (cov(r, r) < -1e-6f) cov_valid = false;
    }
    for (int r = 0; r < 3; ++r)
      for (int c = r + 1; c < 3; ++c)
        if (std::abs(cov(r, c) - cov(c, r)) > 1e-5f) cov_valid = false;
    if (cov_valid) ++cov_ok;
  }

  const float cov_pct = 100.0f * cov_ok / n;
  std::cout << "  covariance valid:   " << cov_ok << "/" << n << " (" << cov_pct << "%)\n";
  check(cov_pct > 95.0f, "too many invalid covariance matrices");

  // ---- Test 2: geometric consistency (CPU KNN vs GPU covariance) ----
  //
  // For a random subset of points:
  //   a) Brute-force KNN on CPU to find the same K neighbors.
  //   b) Recompute the sample covariance on CPU — should match GPU result.
  //   c) For each neighbor, compute squared Mahalanobis distance from the
  //      local mean.  Since the covariance was estimated FROM these neighbors,
  //      they should lie well within the lipsoid.

  const int num_probe = std::min(100, n);
  std::vector<int> probe_ids(n);
  std::iota(probe_ids.begin(), probe_ids.end(), 0);
  std::shuffle(probe_ids.begin(), probe_ids.end(), rng);
  probe_ids.resize(num_probe);

  int cov_match_ok = 0;
  int mahal_ok = 0;
  float max_frob_rel = 0.0f;
  float max_mahal = 0.0f;

  // Chi-squared 3 DOF, 99.5th percentile — generous bound for K=12 sample covariance.
  // The neighbors DEFINED the covariance, so they should fit well within it.
  const float mahal_sq_thresh = 12.84f;

  for (int pi : probe_ids) {
    const auto& pt = cloud[pi];
    Eigen::Map<const Eigen::Matrix<float, 3, 3, Eigen::RowMajor>> cov_gpu(pt.covariance);

    // Skip default/degenerate covariances
    if (cov_gpu.trace() < 1e-4f) continue;

    // a) CPU KNN
    std::vector<int> nn = cpu_knn(cloud, pi, K);

    // b) CPU sample covariance
    Eigen::Vector3f mean_cpu;
    Eigen::Matrix3f cov_cpu = cpu_sample_covariance(cloud, nn, mean_cpu);

    // Compare: relative Frobenius norm of the difference
    const float frob_diff = (cov_cpu - cov_gpu.template cast<float>()).norm();
    const float frob_ref = cov_cpu.norm();
    const float frob_rel = (frob_ref > 1e-8f) ? (frob_diff / frob_ref) : frob_diff;

    if (frob_rel > max_frob_rel) max_frob_rel = frob_rel;

    // Allow up to 10% relative error (floating-point order-of-ops differences
    // between CPU and GPU can cause small discrepancies).
    if (frob_rel < 0.10f) ++cov_match_ok;

    // c) Mahalanobis check: each neighbor should lie within the lipsoid.
    Eigen::Matrix3f cov_inv = cov_gpu.inverse();
    bool all_in = true;
    for (int idx : nn) {
      const auto& pj = cloud[idx];
      const Eigen::Vector3f d = Eigen::Vector3f(pj.x, pj.y, pj.z) - mean_cpu;
      const float mahal_sq = d.transpose() * cov_inv * d;
      if (mahal_sq > max_mahal) max_mahal = mahal_sq;
      if (!std::isfinite(mahal_sq) || mahal_sq > mahal_sq_thresh) {
        all_in = false;
      }
    }
    if (all_in) ++mahal_ok;
  }

  const float match_pct = 100.0f * cov_match_ok / num_probe;
  const float mahal_pct = 100.0f * mahal_ok / num_probe;

  std::cout << "\n  --- Geometric consistency (probed " << num_probe << " points) ---\n";
  std::cout << "  CPU vs GPU cov match: " << cov_match_ok << "/" << num_probe
            << " (" << match_pct << "%, max rel Frobenius = " << max_frob_rel << ")\n";
  std::cout << "  Mahalanobis in-bound: " << mahal_ok << "/" << num_probe
            << " (" << mahal_pct << "%, max d^2 = " << max_mahal
            << ", thresh = " << mahal_sq_thresh << ")\n";

  check(match_pct > 90.0f, "CPU vs GPU covariance mismatch on too many points");
  check(mahal_pct > 90.0f, "too many points with neighbors outside covariance lipsoid");

  if (failures == 0) {
    std::cout << "\nPASS\n";
    return 0;
  } else {
    std::cerr << "\n" << failures << " check(s) failed.\n";
    return 1;
  }
}
