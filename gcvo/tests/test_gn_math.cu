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

// gcvo/tests/test_gn_math.cpp
//
// Minimal GN math sanity test for GCVO:
//  - builds a synthetic point cloud
//  - applies a RANDOM ROTATION (and optional translation) to make a target cloud
//  - checks that ONE GN iteration increases the INNER PRODUCT (maximization objective)
//
// Notes:
//  - We print inner product (NOT MSE).
//  - We use a deterministic RNG seed.
//  - This test intentionally "reaches into" GCvoGPU internals to call inner_product_
//    (private) using the common test trick `#define private public`.
//
// Build suggestion (example):
//   add_executable(test_gn_math gcvo/tests/test_gn_math.cpp)
//   target_link_libraries(test_gn_math PRIVATE gcvo)
// Run:
//   ./test_gn_math --params params.yaml [--n 2000] [--theta_deg 15] [--t 0.0]

#include <Eigen/Dense>

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "gcvo/utils/PointTypes19.hpp"
#include "gcvo/utils/GCvoPointCloud.hpp"
#include "gcvo/GCvoParams.hpp"

#include "pcl/io/pcd_io.h"
#include "pcl/point_types.h"

// --- test hack: expose private for inner_product_ ---
#define private public
#include "gcvo/GCvoGPU.hpp"
#include "gcvo/impl/GCvoGPU_impl.cuh"
#include "gcvo/impl/GCvoPointCloud_covariance_impl.cuh"
#undef private

namespace {

struct Args {
  std::string params = "gcvo_params/test.yaml";
  int n = 2000;
  float theta_deg = 15.0f;   // max rotation magnitude
  float trans_mag = 0.0f;    // optional translation magnitude
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " [--params <params.yaml>] [--input_pcd_file bunny.pcd] [--n 2000] [--theta_deg 15] [--t 0.0]\n\n"
      << "This test prints INNER PRODUCT before/after one GN iteration under a random rotation.\n";
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

    if (k == "--params") a.params = need("--params");
    else if (k == "--n") a.n = std::stoi(need("--n"));
    else if (k == "--theta_deg") a.theta_deg = std::stof(need("--theta_deg"));
    else if (k == "--t") a.trans_mag = std::stof(need("--t"));
    else if (k == "--input_pcd_file") need("--input_pcd_file");  // accepted, unused
    else if (k == "-h" || k == "--help") return false;
    else {
      std::cerr << "Unknown arg: " << k << "\n";
      return false;
    }
  }
  return !a.params.empty() && a.n > 10;
}

static Eigen::Matrix3f random_rotation(std::mt19937& rng, float theta_max_deg) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Eigen::Vector3f axis(nd(rng), nd(rng), nd(rng));
  const float n = axis.norm();
  if (n < 1e-8f) axis = Eigen::Vector3f(1, 0, 0);
  else axis /= n;

  std::uniform_real_distribution<float> ud(-theta_max_deg, theta_max_deg);
  const float theta = ud(rng) * static_cast<float>(M_PI) / 180.0f;

  return Eigen::AngleAxisf(theta, axis).toRotationMatrix();
}

static Eigen::Vector3f random_translation(std::mt19937& rng, float mag) {
  if (mag <= 0.0f) return Eigen::Vector3f::Zero();
  std::uniform_real_distribution<float> ud(-mag, mag);
  return Eigen::Vector3f(ud(rng), ud(rng), ud(rng));
}

static Eigen::Matrix4f make_T(const Eigen::Matrix3f& R, const Eigen::Vector3f& t) {
  Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
  T.block<3,3>(0,0) = R;
  T.block<3,1>(0,3) = t;
  return T;
}

// Apply transform to a point (xyz only)
template <typename PointT>
static PointT transform_point_xyz(const PointT& p, const Eigen::Matrix4f& T) {
  PointT out = p;
  Eigen::Vector3f x(p.x, p.y, p.z);
  Eigen::Vector3f y = T.block<3,3>(0,0) * x + T.block<3,1>(0,3);
  out.x = y.x();
  out.y = y.y();
  out.z = y.z();
  return out;
}


  
static gcvo::GCvoPointCloudT<gcvo::PointS1> read_cloud_xyz(std::string & fname) {
  using P = gcvo::PointS1;

  pcl::PointCloud<P> pc;
  pcl::io::loadPCDFile<P>(fname, pc);
  return gcvo::GCvoPointCloudT<P>(pc);
}

  
static gcvo::GCvoPointCloudT<gcvo::PointS1> make_random_cloud_xyzI(std::mt19937& rng, int n) {
  using P = gcvo::PointS1;

  std::uniform_real_distribution<float> up(-20.0f, 20.0f);
  std::uniform_real_distribution<float> ui(0.0f, 1.0f);

  pcl::PointCloud<P> pc;
  pc.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    P p;
    p.x = up(rng);
    p.y = up(rng);
    p.z = up(rng);

    // intensity feature in [0,1] stored in features[0]
    p.features[0] = ui(rng);

    // minimal semantics (all zeros ok)
    p.label = -1;
    pc[static_cast<size_t>(i)] = p;
  }

  return gcvo::GCvoPointCloudT<P>(pc);
}

static void require(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "TEST FAILED: " << msg << "\n";
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    usage(argv[0]);
    return 2;
  }

  using PointT = gcvo::PointS1;
  using CloudT = gcvo::GCvoPointCloudT<PointT>;

  // Deterministic RNG
  std::mt19937 rng(0);

  // Make synthetic source cloud
  //CloudT src = make_random_cloud_xyzI(rng, a.n);
  CloudT src = make_random_cloud_xyzI(rng, a.n);
  

  // Apply a random rotation (and optional translation) to create target.
  const Eigen::Matrix3f R_gt = random_rotation(rng, a.theta_deg);
  const Eigen::Vector3f t_gt = random_translation(rng, a.trans_mag);

  const Eigen::Matrix4f T_gt = make_T(R_gt, t_gt);

  pcl::PointCloud<PointT> tgt_pcl;
  tgt_pcl.resize(static_cast<size_t>(a.n));
  for (int i = 0; i < a.n; ++i) {
    tgt_pcl[static_cast<size_t>(i)] = transform_point_xyz(src.points()[static_cast<size_t>(i)], T_gt);
    // keep same features for this test
    tgt_pcl[static_cast<size_t>(i)].features[0] = src.points()[static_cast<size_t>(i)].features[0];
  }
  CloudT tgt(tgt_pcl);

  // Build solver
  gcvo::GCvoGPU<PointT> solver(a.params);

  // Enforce a "single GN step" behavior via params.
  // (We keep the YAML as base, but clamp max_iterations here for the test.)
  gcvo::GCvoParams p = solver.params();
  // For a pure GN smoke test, brute-force is often simpler/stabler on small synthetic clouds:
  // (leave as-is if your YAML sets kdtree)
  p.max_iterations = 1;
  // p.use_kdtree = 0;
  // Make sure GN is on
  // Enable printing inside align() if you wired verbose to logs
  // p.verbose = 1;
  solver.write_params(p);

  const float l = solver.params().l_init;

  // Evaluate inner product at identity init
  const Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
  const float ip_before = solver.inner_product_(src, tgt, T_init, l, nullptr);

  // Run ONE iteration alignment
  Eigen::Matrix4f T_out = Eigen::Matrix4f::Identity();
  int iters = 0;
  double secs = 0.0;
  const int rc = solver.align(src, tgt, T_init, T_out, nullptr, &iters, &secs);

  const float ip_after = solver.inner_product_(src, tgt, T_out, l, nullptr);

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "==== test_gn_math (inner product) ====\n";
  std::cout << "n=" << a.n
            << " theta_deg(max)=" << a.theta_deg
            << " trans_mag=" << a.trans_mag << "\n";
  std::cout << "return_code=" << rc
            << " iters=" << iters
            << " sec=" << secs << "\n";
  std::cout << "l=" << l << "\n";
  std::cout << "IP_before=" << ip_before << "\n";
  std::cout << "IP_after =" << ip_after << "\n";
  std::cout << "Delta_IP =" << (ip_after - ip_before) << "\n";

  // We are maximizing: after one GN step we expect IP to not decrease (tiny tolerance).
  require(std::isfinite(ip_before) && std::isfinite(ip_after), "inner product is not finite");
  require(ip_after + 1e-4f >= ip_before, "inner product did not increase (or decreased too much)");

  // std::cout<<"T_gt is "<<T_gt<<", T_pred is "<<T_out<<"\n";

  std::cout << "PASS\n";
  return 0;
}
