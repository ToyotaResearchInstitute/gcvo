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

/// @file gcvo_run_pcd.cu
/// @brief General-purpose PCD runner (pair or sequence mode).
///
/// Supports two modes:
///   - Pair mode:     --source <a.pcd> --target <b.pcd>
///   - Sequence mode: --pcd_dir <dir> [--count N]
///     Collects all *.pcd files sorted lexicographically, registers
///     consecutive pairs, and accumulates a trajectory.
///
/// The --type flag selects: intensity (PointS1), rgb (PointS3), or fpfh (PointS33).
/// If --count exceeds available files, uses all available files.
///
/// Outputs:
///   - Per-pair relative transforms and timing to stdout.
///   - Accumulated trajectory in KITTI format to --traj_file (optional).
///   - Live PCL viewer with --visualize (requires -DGCVO_BUILD_VIZ=ON).
///
/// Links against: gcvo_s1 gcvo_s3 gcvo_s33.
///
/// @par Example
/// @code
///   ./gcvo_run_pcd --params params.yaml --pcd_dir /data/pcds/ \
///     --type intensity --count 50 --traj_file traj.txt
/// @endcode

#include "gcvo/GCvoGPU.hpp"
#include "gcvo/utils/PointTypes19.hpp"

#ifdef GCVO_USE_VIZ
#  include "gcvo/utils/MapViewer.hpp"
#endif

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Dense>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string params;
  // pair mode
  std::string source_pcd;
  std::string target_pcd;
  // sequence mode
  std::string pcd_dir;
  // output
  std::string traj_file;
  std::string type = "intensity";  // intensity|rgb|fpfh
  int count = 0;  // 0 = use all files; >0 = max number of frames
  bool visualize = false;
};

static void print_usage(const char* argv0) {
  std::cerr
    << "Usage (pair mode):\n"
    << "  " << argv0
    << " --params <p.yaml> --source <a.pcd> --target <b.pcd>"
       " [--type intensity|rgb|fpfh] [--traj_file <out.txt>]\n"
    << "Usage (sequence mode):\n"
    << "  " << argv0
    << " --params <p.yaml> --pcd_dir <dir>"
       " [--type intensity|rgb|fpfh] [--count N] [--traj_file <out.txt>] [--visualize]\n"
    << "\n"
    << "In sequence mode PCD files are sorted lexicographically by path (timestamps).\n"
    << "--visualize requires building with -DGCVO_BUILD_VIZ=ON.\n";
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
    if      (k == "--params")    a.params     = need("--params");
    else if (k == "--source")    a.source_pcd = need("--source");
    else if (k == "--target")    a.target_pcd = need("--target");
    else if (k == "--pcd_dir")   a.pcd_dir    = need("--pcd_dir");
    else if (k == "--traj_file") a.traj_file  = need("--traj_file");
    else if (k == "--type")      a.type       = need("--type");
    else if (k == "--count")     a.count      = std::stoi(need("--count"));
    else if (k == "--visualize") a.visualize  = true;
    else if (k == "-h" || k == "--help") return false;
    else { std::cerr << "Unknown arg: " << k << "\n"; return false; }
  }
  if (a.params.empty()) return false;
  const bool pair_mode = !a.source_pcd.empty() && !a.target_pcd.empty();
  const bool seq_mode  = !a.pcd_dir.empty();
  return pair_mode || seq_mode;
}

// ---- helpers ---------------------------------------------------------------

template <typename PointT>
gcvo::GCvoPointCloudT<PointT> load_cloud(const std::string& path) {
  if constexpr (std::is_same_v<PointT, gcvo::PointS1>) {
    pcl::PointCloud<pcl::PointXYZI> pc;
    if (pcl::io::loadPCDFile(path, pc) != 0)
      throw std::runtime_error("Failed to load " + path);
    return gcvo::GCvoPointCloudT<PointT>(pc);
  } else if constexpr (std::is_same_v<PointT, gcvo::PointS3>) {
    pcl::PointCloud<pcl::PointXYZRGB> pc;
    if (pcl::io::loadPCDFile(path, pc) != 0)
      throw std::runtime_error("Failed to load " + path);
    return gcvo::GCvoPointCloudT<PointT>(pc);
  } else {
    pcl::PointCloud<PointT> pc;
    if (pcl::io::loadPCDFile(path, pc) != 0)
      throw std::runtime_error("Failed to load " + path);
    return gcvo::GCvoPointCloudT<PointT>(pc);
  }
}

// Write one KITTI pose line: 12 floats (3×4 row-major, no last row).
static void write_kitti_pose(std::ofstream& ofs, const Eigen::Matrix4f& T) {
  ofs << std::fixed << std::setprecision(6);
  for (int r = 0; r < 3; ++r)
    for (int c = 0; c < 4; ++c) {
      ofs << T(r, c);
      if (!(r == 2 && c == 3)) ofs << ' ';
    }
  ofs << '\n';
}

static void print_mat4(const Eigen::Matrix4f& T) {
  std::cout << std::fixed << std::setprecision(6);
  for (int r = 0; r < 4; ++r)
    for (int c = 0; c < 4; ++c) {
      std::cout << T(r, c);
      if (!(r == 3 && c == 3)) std::cout << ' ';
    }
  std::cout << "\n";
}

// Return all *.pcd paths under dir, sorted lexicographically.
static std::vector<std::string> collect_pcds(const std::string& dir) {
  std::vector<std::string> files;
  for (const auto& entry : fs::directory_iterator(dir))
    if (entry.is_regular_file() && entry.path().extension() == ".pcd")
      files.push_back(entry.path().string());
  std::sort(files.begin(), files.end());
  return files;
}

// ---- pair mode -------------------------------------------------------------

template <typename PointT>
int run_pair(const Args& a) {
  gcvo::GCvoGPU<PointT> solver(a.params);

  auto src = load_cloud<PointT>(a.source_pcd);
  auto tgt = load_cloud<PointT>(a.target_pcd);

  const bool need_cov    = (solver.params().kernel_type != gcvo::GCvoKernelType::SCALAR);
  const bool is_rescaled = (solver.params().kernel_type == gcvo::GCvoKernelType::RESCALED);
  if (need_cov) {
    src.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
    tgt.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
  }

  const Eigen::Matrix4f T_init = Eigen::Matrix4f::Identity();
  gcvo::GCvoResultInfo r = solver.align(src, tgt, T_init, false);

  std::cout << "# src=" << a.source_pcd << " tgt=" << a.target_pcd
            << " iters=" << r.num_iters
            << " sec="   << r.registration_seconds
            << " code="  << r.return_code << "\n";
  print_mat4(r.T_s2t);

  if (!a.traj_file.empty()) {
    std::ofstream ofs(a.traj_file);
    if (!ofs) throw std::runtime_error("Cannot open traj file: " + a.traj_file);
    write_kitti_pose(ofs, Eigen::Matrix4f::Identity());
    write_kitti_pose(ofs, r.T_s2t);
  }

  return r.return_code;
}

// ---- sequence mode ---------------------------------------------------------

template <typename PointT>
int run_sequence(const Args& a) {
  auto files = collect_pcds(a.pcd_dir);
  if (files.size() < 2) {
    std::cerr << "Need at least 2 PCD files in " << a.pcd_dir << "\n";
    return 2;
  }
  // Cap to --count frames if specified (0 = use all).
  if (a.count > 0 && static_cast<size_t>(a.count) < files.size()) {
    files.resize(a.count);
  }
  std::cout << "Using " << files.size() << " PCD files from " << a.pcd_dir;
  if (a.count > 0 && static_cast<size_t>(a.count) > files.size())
    std::cout << " (requested " << a.count << ", only " << files.size() << " available)";
  std::cout << "\n";

  gcvo::GCvoGPU<PointT> solver(a.params);
  const bool need_cov    = (solver.params().kernel_type != gcvo::GCvoKernelType::SCALAR);
  const bool is_rescaled = (solver.params().kernel_type == gcvo::GCvoKernelType::RESCALED);

  std::ofstream traj_ofs;
  if (!a.traj_file.empty()) {
    traj_ofs.open(a.traj_file);
    if (!traj_ofs) throw std::runtime_error("Cannot open traj file: " + a.traj_file);
  }

#ifdef GCVO_USE_VIZ
  std::unique_ptr<gcvo::MapViewer> viewer;
  if (a.visualize) {
    viewer = std::make_unique<gcvo::MapViewer>("GCVO PCD sequence");
  }
#else
  if (a.visualize)
    std::cerr << "Warning: --visualize ignored (rebuild with -DGCVO_BUILD_VIZ=ON)\n";
#endif

  Eigen::Matrix4f T_world_i = Eigen::Matrix4f::Identity();
  Eigen::Matrix4f T_init    = Eigen::Matrix4f::Identity();

  // Frame 0 has identity pose.
  if (traj_ofs.is_open()) write_kitti_pose(traj_ofs, T_world_i);

#ifdef GCVO_USE_VIZ
  if (viewer) viewer->add_pose(T_world_i);
#endif

  for (size_t f = 0; f + 1 < files.size(); ++f) {
    auto src = load_cloud<PointT>(files[f]);
    auto tgt = load_cloud<PointT>(files[f + 1]);

    if (need_cov) {
      src.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
      tgt.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
    }

    gcvo::GCvoResultInfo r = solver.align(src, tgt, T_init, false);

    const Eigen::Matrix4f T_world_f = T_world_i;           // pose of frame f
    T_world_i                        = T_world_f * r.T_s2t; // pose of frame f+1

    std::cout << "# frame=" << f << "->" << (f + 1)
              << " iters=" << r.num_iters
              << " sec="   << r.registration_seconds
              << " code="  << r.return_code << "\n";

    if (traj_ofs.is_open()) write_kitti_pose(traj_ofs, T_world_i);
    T_init = r.T_s2t;

#ifdef GCVO_USE_VIZ
    if (viewer) {
      viewer->add_cloud(src.points(), T_world_f);
      viewer->add_pose(T_world_i);
      // On the last pair also add the final target frame.
      if (f + 2 == files.size())
        viewer->add_cloud(tgt.points(), T_world_i);
      viewer->spin_once();
    }
#endif
  }

#ifdef GCVO_USE_VIZ
  if (viewer) {
    std::cout << "Processing complete. Close viewer window to exit.\n";
    viewer->spin();
  }
#endif

  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    print_usage(argv[0]);
    return 2;
  }

  const bool seq_mode = !a.pcd_dir.empty();

  try {
    if (a.type == "intensity")
      return seq_mode ? run_sequence<gcvo::PointS1>(a)  : run_pair<gcvo::PointS1>(a);
    if (a.type == "rgb")
      return seq_mode ? run_sequence<gcvo::PointS3>(a)  : run_pair<gcvo::PointS3>(a);
    if (a.type == "fpfh")
      return seq_mode ? run_sequence<gcvo::PointS33>(a) : run_pair<gcvo::PointS33>(a);
    std::cerr << "Unknown --type: " << a.type << "\n";
    return 2;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
