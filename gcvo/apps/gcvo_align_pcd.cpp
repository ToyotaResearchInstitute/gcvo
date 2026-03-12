/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: src/experiments/main_cvo_gpu_align_pcd.cpp (no 1:1 mapping claimed)
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

/// @file gcvo_align_pcd.cpp
/// @brief Generic PCD alignment runner supporting all GCVO point types.
///
/// Aligns PCD files in two modes:
///   - Pair mode:     --source <a.pcd> --target <b.pcd>
///   - Sequence mode: --pcd_list <list.txt>  (one path per line, aligns consecutive pairs)
///
/// The --type flag selects the point type: intensity (PointS1), rgb (PointS3),
/// rgbg (PointS5), or fpfh (PointS33). Each type dispatches to the
/// corresponding GCvoGPU<PointT> instantiation.
///
/// For DENSE/RESCALED kernels, per-point covariance is computed automatically.
///
/// Links against: gcvo (all point types).
///
/// @par Example
/// @code
///   ./gcvo_align_pcd --params params.yaml --type intensity \
///     --source a.pcd --target b.pcd
/// @endcode

#include "gcvo/GCvoGPU.hpp"
#include "gcvo/utils/PointTypes19.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <type_traits>
#include <vector>

namespace {

struct Args {
  std::string params;
  std::string type = "intensity";  // intensity|rgb|rgbg|fpfh

  // Single pair mode
  std::string source_pcd;
  std::string target_pcd;

  // Sequence mode
  std::string pcd_list;

  bool return_corresp = false;
};

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0 << " --params <params.yaml> --type <intensity|rgb|rgbg|fpfh> --source <a.pcd> --target <b.pcd> [--return_corresp]\n"
      << "  " << argv0 << " --params <params.yaml> --type <intensity|rgb|rgbg|fpfh> --pcd_list <list.txt> [--return_corresp]\n\n"
      << "list.txt: one PCD path per line. Consecutive pairs (i,i+1) are aligned.\n";
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
    else if (k == "--type") a.type = need("--type");
    else if (k == "--source") a.source_pcd = need("--source");
    else if (k == "--target") a.target_pcd = need("--target");
    else if (k == "--pcd_list") a.pcd_list = need("--pcd_list");
    else if (k == "--return_corresp") a.return_corresp = true;
    else if (k == "-h" || k == "--help") return false;
    else {
      std::cerr << "Unknown arg: " << k << "\n";
      return false;
    }
  }

  if (a.params.empty()) return false;
  const bool pair_mode = !a.source_pcd.empty() && !a.target_pcd.empty();
  const bool list_mode = !a.pcd_list.empty();
  if (!(pair_mode ^ list_mode)) return false;
  return true;
}

static std::vector<std::string> read_lines(const std::string& path) {
  std::ifstream ifs(path);
  if (!ifs) throw std::runtime_error("Failed to open " + path);
  std::vector<std::string> out;
  std::string line;
  while (std::getline(ifs, line)) {
    if (line.empty()) continue;
    out.push_back(line);
  }
  return out;
}

template <typename PointT>
gcvo::GCvoPointCloudT<PointT> load_as_gcvo_cloud(const std::string& pcd_path) {
  if constexpr (std::is_same_v<PointT, gcvo::PointS1>) {
    pcl::PointCloud<pcl::PointXYZI> pc;
    if (pcl::io::loadPCDFile(pcd_path, pc) != 0) {
      throw std::runtime_error("Failed to load " + pcd_path);
    }
    return gcvo::GCvoPointCloudT<PointT>(pc);
  } else if constexpr (std::is_same_v<PointT, gcvo::PointS3> || std::is_same_v<PointT, gcvo::PointS5>) {
    pcl::PointCloud<pcl::PointXYZRGB> pc;
    if (pcl::io::loadPCDFile(pcd_path, pc) != 0) {
      throw std::runtime_error("Failed to load " + pcd_path);
    }
    return gcvo::GCvoPointCloudT<PointT>(pc);
  } else {
    // FPFH/PointS33: expects the PCD was saved with fields matching PointT (features[33], etc).
    pcl::PointCloud<PointT> pc;
    if (pcl::io::loadPCDFile(pcd_path, pc) != 0) {
      throw std::runtime_error("Failed to load " + pcd_path);
    }
    return gcvo::GCvoPointCloudT<PointT>(pc);
  }
}

static void print_mat4(const Eigen::Matrix4f& T) {
  std::cout << std::fixed << std::setprecision(6);
  for (int r = 0; r < 4; ++r) {
    for (int c = 0; c < 4; ++c) {
      std::cout << T(r, c);
      if (!(r == 3 && c == 3)) std::cout << ' ';
    }
  }
  std::cout << "\n";
}

template <typename PointT>
static void maybe_compute_covariance(const gcvo::GCvoGPU<PointT>& solver,
                                     gcvo::GCvoPointCloudT<PointT>& src,
                                     gcvo::GCvoPointCloudT<PointT>& tgt) {
  const auto& p = solver.params();
  if (static_cast<int>(p.kernel_type) == 0) return;

  // Follow legacy main_cvo_gpu_lidar_raw_intensity.cpp behavior.
  const bool is_rescaled = (p.kernel_type == gcvo::GCvoKernelType::RESCALED);
  src.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
  tgt.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
}

template <typename PointT>
int run_pair(const Args& a, const std::string& src_path, const std::string& tgt_path,
             const Eigen::Matrix4f& init_T_s2t) {
  gcvo::GCvoGPU<PointT> solver(a.params);

  auto src = load_as_gcvo_cloud<PointT>(src_path);
  auto tgt = load_as_gcvo_cloud<PointT>(tgt_path);

  maybe_compute_covariance<PointT>(solver, src, tgt);

  gcvo::GCvoResultInfo r = solver.align(src, tgt, init_T_s2t, a.return_corresp);
  std::cout << "# src=" << src_path << " tgt=" << tgt_path
            << " iters=" << r.num_iters
            << " sec=" << r.registration_seconds
            << " code=" << r.return_code << "\n";
  // r.T_s2t uses legacy convention: p_source = T_s2t * p_target
  print_mat4(r.T_s2t);
  return r.return_code;
}

template <typename PointT>
int run_sequence_list(const Args& a, const std::vector<std::string>& paths) {
  if (paths.size() < 2) {
    std::cerr << "Need >=2 PCDs in list\n";
    return 2;
  }
  gcvo::GCvoGPU<PointT> solver(a.params);

  // Accumulate poses in the first frame (frame_0) coordinate.
  // Convention:
  //   align(source=i, target=i+1) returns T_i_ip1 such that p_i = T_i_ip1 * p_{i+1}
  // Therefore:
  //   T_0_{i+1} = T_0_i * T_i_{i+1}
  Eigen::Matrix4f T_0_i = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f init = Eigen::Matrix4f::Identity();

  for (size_t i = 0; i + 1 < paths.size(); ++i) {
    auto src = load_as_gcvo_cloud<PointT>(paths[i]);
    auto tgt = load_as_gcvo_cloud<PointT>(paths[i + 1]);

    maybe_compute_covariance<PointT>(solver, src, tgt);

    gcvo::GCvoResultInfo r = solver.align(src, tgt, init, false);

    const Eigen::Matrix4f T_i_ip1 = r.T_s2t;
    T_0_i = T_0_i * T_i_ip1;

    std::cout << "# pair=" << i << " iters=" << r.num_iters
              << " sec=" << r.registration_seconds
              << " code=" << r.return_code << "\n";
    std::cout << "# T_i_from_i+1\n";
    print_mat4(T_i_ip1);
    std::cout << "# T_0_from_i+1\n";
    print_mat4(T_0_i);

    init = T_i_ip1;  // warm start
  }
  return 0;
}

template <typename PointT>
int dispatch_run(const Args& a) {
  const Eigen::Matrix4f init = Eigen::Matrix4f::Identity();
  if (!a.pcd_list.empty()) {
    return run_sequence_list<PointT>(a, read_lines(a.pcd_list));
  }
  return run_pair<PointT>(a, a.source_pcd, a.target_pcd, init);
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    if (a.type == "intensity") {
      return dispatch_run<gcvo::PointS1>(a);
    } else if (a.type == "rgb") {
      return dispatch_run<gcvo::PointS3>(a);
    } else if (a.type == "rgbg") {
      return dispatch_run<gcvo::PointS5>(a);
    } else if (a.type == "fpfh") {
      return dispatch_run<gcvo::PointS33>(a);
    }

    std::cerr << "Unknown --type: " << a.type << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
