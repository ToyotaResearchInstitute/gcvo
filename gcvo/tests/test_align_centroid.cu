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

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "gcvo/utils/PointTypes19.hpp"
#include "gcvo/utils/GCvoPointCloud.hpp"
#include "gcvo/GCvoParams.hpp"
#include "gcvo/LieGroup.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

// --- test hack: expose private for inner_product_ ---
#define private public
#include "gcvo/GCvoGPU.hpp"
#include "gcvo/impl/GCvoGPU_impl.cuh"
#include "gcvo/impl/GCvoPointCloud_covariance_impl.cuh"
#undef private

namespace {

struct Args {
  std::string params;
  std::string input_pcd_file;

  int n = 2000;              // number of points to use (subsample if pcd larger)
  float theta_deg = 15.0f;   // max rotation magnitude
  float trans_mag = 0.0f;    // optional extra translation magnitude (added after pivot logic)
  std::string pivot = "origin"; // origin|centroid
  bool save_pcd = false;     // write before/after PCD files (requires GCVO_TEST_SAVE_PCD compile def)
};

static void usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " --params <params.yaml> --input_pcd_file <bunny.pcd>\n"
      << "    [--n 2000] [--theta_deg 15] [--t 0.0] [--pivot origin|centroid] [--save_pcd]\n\n"
      << "Notes:\n"
      << "  - If --pivot centroid, the GT transform will generally have non-zero translation even if --t=0.\n"
      << "  - This test prints INNER PRODUCT before/after align(), plus centroid diagnostics and SE(3) log error.\n";
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
    else if (k == "--input_pcd_file") a.input_pcd_file = need("--input_pcd_file");
    else if (k == "--n") a.n = std::stoi(need("--n"));
    else if (k == "--theta_deg") a.theta_deg = std::stof(need("--theta_deg"));
    else if (k == "--t") a.trans_mag = std::stof(need("--t"));
    else if (k == "--pivot") a.pivot = need("--pivot");
    else if (k == "--save_pcd") a.save_pcd = true;
    else if (k == "-h" || k == "--help") return false;
    else {
      std::cerr << "Unknown arg: " << k << "\n";
      return false;
    }
  }

  if (a.params.empty() || a.input_pcd_file.empty()) return false;
  if (a.n < 10) return false;
  if (!(a.pivot == "origin" || a.pivot == "centroid")) {
    std::cerr << "Invalid --pivot: " << a.pivot << " (use origin|centroid)\n";
    return false;
  }
  return true;
}

static Eigen::Matrix3f random_rotation(std::mt19937& rng, float theta_max_deg) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  Eigen::Vector3f axis(nd(rng), nd(rng), nd(rng));
  float n = axis.norm();
  if (n < 1e-8f) axis = Eigen::Vector3f(1, 0, 0);
  else axis /= n;

  std::uniform_real_distribution<float> ud(-theta_max_deg, theta_max_deg);
  float theta = ud(rng) * static_cast<float>(M_PI) / 180.0f;

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

static void require(bool cond, const std::string& msg) {
  if (!cond) {
    std::cerr << "TEST FAILED: " << msg << "\n";
    std::exit(1);
  }
}

#ifdef GCVO_TEST_SAVE_PCD
static void save_stacked_pcd(const gcvo::GCvoPointCloudT<gcvo::PointS1>& src,
                              const gcvo::GCvoPointCloudT<gcvo::PointS1>& tgt,
                              const Eigen::Matrix4f& T_s2t,
                              const std::string& filename) {
  pcl::PointCloud<pcl::PointXYZRGB> pc;
  // Add src points in red
  for (const auto& p : src.points()) {
    pcl::PointXYZRGB pt;
    pt.x = p.x; pt.y = p.y; pt.z = p.z;
    pt.r = 255; pt.g = 0; pt.b = 0;
    pc.push_back(pt);
  }
  // Add tgt points transformed by T_s2t, in blue
  for (const auto& p : tgt.points()) {
    Eigen::Vector3f pw = T_s2t.block<3,3>(0,0) * Eigen::Vector3f(p.x, p.y, p.z) + T_s2t.block<3,1>(0,3);
    pcl::PointXYZRGB pt;
    pt.x = pw.x(); pt.y = pw.y(); pt.z = pw.z();
    pt.r = 0; pt.g = 0; pt.b = 255;
    pc.push_back(pt);
  }
  pcl::io::savePCDFileASCII(filename, pc);
  std::cout << "Saved " << pc.size() << " points to " << filename << "\n";
}
#endif

using PointT = gcvo::PointS1;
using CloudT = gcvo::GCvoPointCloudT<PointT>;

static CloudT load_xyz_as_pointsem1(const std::string& fname) {
  pcl::PointCloud<pcl::PointXYZ> pc;
  if (pcl::io::loadPCDFile<pcl::PointXYZ>(fname, pc) != 0) {
    throw std::runtime_error("Failed to load PCD as pcl::PointXYZ: " + fname);
  }

  pcl::PointCloud<PointT> out;
  out.resize(pc.size());
  for (size_t i = 0; i < pc.size(); ++i) {
    PointT p;
    p.x = pc[i].x;
    p.y = pc[i].y;
    p.z = pc[i].z;

    // intensity feature in [0,1]
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

static Eigen::Vector3f centroid(const CloudT& c) {
  Eigen::Vector3f sum = Eigen::Vector3f::Zero();
  for (const auto& p : c.points()) sum += Eigen::Vector3f(p.x, p.y, p.z);
  if (c.size() == 0) return Eigen::Vector3f::Zero();
  return sum / static_cast<float>(c.size());
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    usage(argv[0]);
    return 2;
  }

  // Deterministic RNG
  std::mt19937 rng(0);

  // Load bunny as XYZ and convert to PointSem1, then subsample to --n
  CloudT src_full = load_xyz_as_pointsem1(a.input_pcd_file);
  require(src_full.size() > 10, "input cloud too small");

  CloudT src = subsample(src_full, a.n, rng);
  const int n = src.size();

  // Compute pivot (origin or centroid)
  const Eigen::Vector3f c = centroid(src);
  const Eigen::Vector3f pivot = (a.pivot == "centroid") ? c : Eigen::Vector3f::Zero();

  // Sample GT rotation and optional extra translation
  const Eigen::Matrix3f R_gt = random_rotation(rng, a.theta_deg);
  const Eigen::Vector3f t_user = random_translation(rng, a.trans_mag);

  // IMPORTANT:
  // If we rotate about pivot 'p', then:
  //   x' = R (x - p) + p + t_user = R x + (p - R p + t_user)
  // So the equivalent rigid transform in the original frame is:
  const Eigen::Vector3f t_gt = pivot - R_gt * pivot + t_user;
  const Eigen::Matrix4f T_gt_src2tgt = make_T(R_gt, t_gt);

  // Build target by applying T_gt_src2tgt to src (xyz only, features copied)
  pcl::PointCloud<PointT> tgt_pcl;
  tgt_pcl.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    const auto& ps = src.points()[static_cast<size_t>(i)];
    auto pt = transform_point_xyz(ps, T_gt_src2tgt);
    pt.features[0] = ps.features[0];
    tgt_pcl[static_cast<size_t>(i)] = pt;
  }
  CloudT tgt(tgt_pcl);

  // --- Diagnostics: centroid consistency check ---
  const Eigen::Vector3f c_s = centroid(src);
  const Eigen::Vector3f c_t = centroid(tgt);
  const Eigen::Vector3f c_pred = R_gt * c_s + t_gt;
  const float centroid_err = (c_t - c_pred).norm();

  std::cout << std::fixed << std::setprecision(6);
  std::cout << "==== test_gn_math (inner product) ====\n";
  std::cout << "n=" << n
            << " theta_deg(max)=" << a.theta_deg
            << " t_user_mag=" << a.trans_mag
            << " pivot=" << a.pivot << "\n";
  std::cout << "centroid(src)^T = " << c_s.transpose() << "\n";
  std::cout << "centroid(tgt)^T = " << c_t.transpose() << "\n";
  std::cout << "centroid_pred(R_gt*c_s+t_gt)^T = " << c_pred.transpose()
            << "  err=" << centroid_err << "\n";
  std::cout << "t_gt^T (includes pivot effect) = " << t_gt.transpose() << "\n";

  // Build solver
  gcvo::GCvoGPU<PointT> solver(a.params);

  // Force GN-only + iteration count for this test
  gcvo::GCvoParams p = solver.params();
  // If you want debug prints from align():
  // p.verbose = 1;
  solver.write_params(p);

  const float l = solver.params().l_init;

  if (solver.params().kernel_type != gcvo::GCvoKernelType::SCALAR) {
    std::cout<<"Compute covariance for both src and tgt...";
    src.compute_covariance();
    tgt.compute_covariance();
    std::cout<<" cov computed.\n";
  }

  // Evaluate inner product at identity init
  const Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
  std::cout<<"eval init inner product....\n";
  const float ip_before = solver.inner_product_(src, tgt, T_init, l, nullptr);
  std::cout<<"eval init inner product done\n";  

#ifdef GCVO_TEST_SAVE_PCD
  if (a.save_pcd) {
    save_stacked_pcd(src, tgt, Eigen::Matrix4f::Identity(), "before_align.pcd");
  }
#endif

  // Run alignment
  Eigen::Matrix4f T_out = Eigen::Matrix4f::Identity();
  int iters = 0;
  double secs = 0.0;
  const int rc = solver.align(src, tgt, T_init, T_out, nullptr, &iters, &secs);

  const float ip_after = solver.inner_product_(src, tgt, T_out, l, nullptr);

#ifdef GCVO_TEST_SAVE_PCD
  if (a.save_pcd) {
    save_stacked_pcd(src, tgt, T_out, "after_align.pcd");
  }
#endif

  std::cout << "return_code=" << rc
            << " iters=" << iters
            << " sec=" << secs << "\n";
  std::cout << "l=" << l << "\n";
  std::cout << "IP_before=" << ip_before << "\n";
  std::cout << "IP_after =" << ip_after << "\n";
  std::cout << "Delta_IP =" << (ip_after - ip_before) << "\n";

  //require(std::isfinite(ip_before) && std::isfinite(ip_after), "inner product is not finite");
  //require(ip_after + 1e-4f >= ip_before, "inner product did not increase (or decreased too much)");

  // --- Convention check ---
  // We GENERATED:  p_t = T_gt_src2tgt * p_s   (src->tgt)
  // Your GCVO API RETURNS: p_s = T_s2t * p_t  (tgt->src)
  // Therefore, the GT for returned transform is:
  const Eigen::Matrix4f T_gt_s2t = T_gt_src2tgt.inverse();

  std::cout << "T_gt_s2t (expected, tgt->src) =\n" << T_gt_s2t << "\n";
  std::cout << "T_pred_s2t (returned)         =\n" << T_out << "\n";

  // Error: T_err = T_gt^{-1} * T_pred
  const Eigen::Matrix4f T_err = T_gt_s2t.inverse() * T_out;
  const auto xi_err = gcvo::liegroup::Log_SE3<float, Eigen::ColMajor>(T_err);
  const float err_norm = xi_err.norm();
  const float init_err_norm =  gcvo::liegroup::Log_SE3<float, Eigen::ColMajor>(T_gt_s2t.inverse()).norm();
  
  std::cout << "SE3 log error norm of init value = "<< init_err_norm<<"\n";
  std::cout << "SE3 log error norm = " << err_norm << "\n";
  std::cout << "xi_err^T = " << xi_err.transpose() << "\n";
  require(init_err_norm > err_norm, "SE3 log error did not decrease");
  require(err_norm < 1e-2, "SE3 log error too large");

  // IMPORTANT: Do NOT require tiny error unless you run enough iterations and the case is in basin.
  // For a smoke test, just print it.
  std::cout << "PASS\n";
  return 0;
}
