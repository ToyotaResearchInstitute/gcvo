/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file:  src/experiments/main_cvo_gpu_lidar_raw_intensity.cpp (no 1:1 mapping claimed)
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

/// @file gcvo_kitti_f2f.cpp
/// @brief KITTI frame-to-frame odometry runner.
///
/// Reads KITTI velodyne .bin files and runs frame-to-frame registration
/// using PointS1 (XYZ + intensity). Outputs:
///   - Per-pair relative transforms and timing to stdout.
///   - Accumulated trajectory in KITTI format (12 floats per line, 3x4
///     row-major) to --traj_file.
///
/// Stops gracefully if --count exceeds the number of available frames.
///
/// Links against: gcvo_s1.
///
/// @par Example
/// @code
///   ./gcvo_kitti_f2f --params params.yaml \
///     --kitti_root /data/kitti/odometry \
///     --sequence 05 --start 0 --count 100 \
///     --traj_file kitti_05.txt --visualize
/// @endcode

#include "gcvo/GCvoGPU.hpp"
#include "gcvo/utils/PointTypes19.hpp"
#include "gcvo/utils/VoxelMapFirstPoint.hpp"
#include "gcvo/utils/VoxelMapFirstPoint_impl.hpp"

#ifdef GCVO_USE_VIZ
#  include "gcvo/utils/MapViewer.hpp"
#endif

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <Eigen/Dense>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct Args {
  std::string params;
  std::string kitti_root;
  std::string sequence = "00";
  std::string traj_file;
  int start = 0;
  int count = 10;
  bool visualize = false;
  std::string voxel_mode = "pcl"; // "pcl", "fast", or "none"
  int random_downsample = 0;      // 0 = disabled; >0 = target number of points
  int max_iter = 10000;           // overrides yaml max_iterations
};

static void print_usage(const char* argv0) {
  std::cerr
      << "Usage:\n"
      << "  " << argv0
      << " --params <params.yaml> --kitti_root <KITTI_ODOM_ROOT>"
         " [--sequence 00] [--start 0] [--count 10]"
         " [--traj_file <out.txt>] [--visualize] [--voxel_mode pcl|fast|none]"
         " [--random_downsample <N>] [--max_iter <N>]\n\n"
      << "Reads velodyne/*.bin and runs frame-to-frame registration using PointS1 (XYZI).\n"
      << "--voxel_mode: 'pcl' (default), 'fast', or 'none' to skip voxel downsampling.\n"
      << "--random_downsample N: randomly subsample to N points (applied after voxel, or alone if voxel_mode=none).\n"
      << "--max_iter N: override max iterations from YAML (default 10000).\n"
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
    if (k == "--params") a.params = need("--params");
    else if (k == "--kitti_root") a.kitti_root = need("--kitti_root");
    else if (k == "--sequence") a.sequence = need("--sequence");
    else if (k == "--start") a.start = std::stoi(need("--start"));
    else if (k == "--count") a.count = std::stoi(need("--count"));
    else if (k == "--traj_file") a.traj_file = need("--traj_file");
    else if (k == "--visualize") a.visualize = true;
    else if (k == "--voxel_mode") a.voxel_mode = need("--voxel_mode");
    else if (k == "--random_downsample") a.random_downsample = std::stoi(need("--random_downsample"));
    else if (k == "--max_iter") a.max_iter = std::stoi(need("--max_iter"));
    else if (k == "-h" || k == "--help") return false;
    else {
      std::cerr << "Unknown arg: " << k << "\n";
      return false;
    }
  }
  return !(a.params.empty() || a.kitti_root.empty() || a.count < 2);
}

static std::string kitti_bin_path(const Args& a, int frame) {
  std::ostringstream ss;
  ss << a.kitti_root << "/sequences/" << a.sequence << "/velodyne/";
  ss << std::setw(6) << std::setfill('0') << frame << ".bin";
  return ss.str();
}

static pcl::PointCloud<pcl::PointXYZI>::Ptr load_kitti_bin_raw(const std::string& path) {
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs) {
    throw std::runtime_error("Failed to open " + path);
  }
  ifs.seekg(0, std::ios::end);
  const std::streamsize nbytes = ifs.tellg();
  ifs.seekg(0, std::ios::beg);
  if (nbytes % (4 * sizeof(float)) != 0) {
    throw std::runtime_error("Unexpected KITTI bin size: " + path);
  }
  const size_t npts = static_cast<size_t>(nbytes / (4 * sizeof(float)));

  auto ptr = std::make_shared<pcl::PointCloud<pcl::PointXYZI>>();
  ptr->resize(npts);

  for (size_t i = 0; i < npts; ++i) {
    float x, y, z, r;
    ifs.read(reinterpret_cast<char*>(&x), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&y), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&z), sizeof(float));
    ifs.read(reinterpret_cast<char*>(&r), sizeof(float));
    (*ptr)[i].x = x;
    (*ptr)[i].y = y;
    (*ptr)[i].z = z;
    (*ptr)[i].intensity = r * 255.0f;  // store in [0,255] so GCvoPointCloudT maps to features[0] in [0,1]
    if (i == 1) {
      std::cout<<" raw intensity = "<<r<<", p.intensity is "<<(*ptr)[i].intensity<<"\n";
    }
  }
  return ptr;
}

static pcl::PointCloud<pcl::PointXYZI> downsample(
    const pcl::PointCloud<pcl::PointXYZI>::Ptr& ptr, float voxel_size = 0.25f) {
  pcl::PointCloud<pcl::PointXYZI> pc;
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(ptr);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(pc);
  return pc;
}

static pcl::PointCloud<pcl::PointXYZI> downsample_fast(
    pcl::PointCloud<pcl::PointXYZI>& raw, float voxel_size = 0.25f) {
  cvo::VoxelMapFirstPoint<pcl::PointXYZI> vmap(voxel_size);
  for (auto& p : raw) vmap.insert_point(&p);
  auto sampled = vmap.sample_points();
  pcl::PointCloud<pcl::PointXYZI> out;
  out.reserve(sampled.size());
  for (auto* p : sampled) out.push_back(*p);
  return out;
}

static pcl::PointCloud<pcl::PointXYZI> downsample_random(
    const pcl::PointCloud<pcl::PointXYZI>& cloud, int target_n,
    std::mt19937& rng) {
  pcl::PointCloud<pcl::PointXYZI> out;
  const int n = static_cast<int>(cloud.size());
  if (target_n >= n) {
    out = cloud;
    return out;
  }
  // Fisher-Yates partial shuffle to pick target_n indices without replacement.
  std::vector<int> idx(n);
  std::iota(idx.begin(), idx.end(), 0);
  for (int i = 0; i < target_n; ++i) {
    std::uniform_int_distribution<int> dist(i, n - 1);
    std::swap(idx[i], idx[dist(rng)]);
  }
  out.resize(target_n);
  for (int i = 0; i < target_n; ++i) out[i] = cloud[idx[i]];
  return out;
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

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, a)) {
    print_usage(argv[0]);
    return 2;
  }

  try {
    gcvo::GCvoGPU<gcvo::PointS1> solver(a.params);
    solver.params().max_iterations = a.max_iter;

    std::ofstream traj_ofs;
    if (!a.traj_file.empty()) {
      traj_ofs.open(a.traj_file);
      if (!traj_ofs) throw std::runtime_error("Cannot open traj file: " + a.traj_file);
    }

#ifdef GCVO_USE_VIZ
    std::unique_ptr<gcvo::MapViewer> viewer;
    if (a.visualize)
      viewer = std::make_unique<gcvo::MapViewer>("GCVO KITTI f2f");
#else
    if (a.visualize)
      std::cerr << "Warning: --visualize ignored (rebuild with -DGCVO_BUILD_VIZ=ON)\n";
#endif

    // Accumulate poses in the first frame (frame_0) coordinate.
    // Convention:
    //   align(source=i, target=i+1) returns T_i_ip1 such that p_i = T_i_ip1 * p_{i+1}
    // Therefore:
    //   T_0_{i+1} = T_0_i * T_i_{i+1}
    Eigen::Matrix4f T_0_i = Eigen::Matrix4f::Identity();
    Eigen::Matrix4f init = Eigen::Matrix4f::Identity();

    const bool need_cov = (static_cast<int>(solver.params().kernel_type) != 0);
    const bool is_rescaled = (solver.params().kernel_type == gcvo::GCvoKernelType::RESCALED);

    // Write identity pose for the first frame.
    if (traj_ofs.is_open()) write_kitti_pose(traj_ofs, T_0_i);
#ifdef GCVO_USE_VIZ
    if (viewer) viewer->add_pose(T_0_i);
#endif

    std::mt19937 rng(42);  // fixed seed for reproducibility

    int frames_processed = 0;
    for (int f = a.start; f < a.start + a.count - 1; ++f) {
      const std::string p0 = kitti_bin_path(a, f);
      const std::string p1 = kitti_bin_path(a, f + 1);

      // Stop gracefully if we run out of frames.
      if (!fs::exists(p0) || !fs::exists(p1)) {
        std::cout << "Stopping at frame " << f << ": input file(s) not found.\n";
        break;
      }

      // IO: read raw point clouds from disk (not timed).
      auto raw0 = load_kitti_bin_raw(p0);
      auto raw1 = load_kitti_bin_raw(p1);

      // --- Start timing: downsample + covariance + align ---
      const auto t_start = std::chrono::steady_clock::now();

      // Downsample: voxel first (unless "none"), then random subsample if requested.
      pcl::PointCloud<pcl::PointXYZI> pc0, pc1;
      if (a.voxel_mode == "none") {
        pc0 = *raw0;
        pc1 = *raw1;
      } else if (a.voxel_mode == "fast") {
        pc0 = downsample_fast(*raw0);
        pc1 = downsample_fast(*raw1);
      } else {
        pc0 = downsample(raw0);
        pc1 = downsample(raw1);
      }
      if (a.random_downsample > 0) {
        pc0 = downsample_random(pc0, a.random_downsample, rng);
        pc1 = downsample_random(pc1, a.random_downsample, rng);
      }
      gcvo::GCvoPointCloudT<gcvo::PointS1> src(pc0);
      gcvo::GCvoPointCloudT<gcvo::PointS1> tgt(pc1);

      if (need_cov) {
        src.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
        tgt.compute_covariance(0.1f, 100.0f, 12, 8, is_rescaled, true);
      }

      gcvo::GCvoResultInfo r = solver.align(src, tgt, init, false);

      const auto t_end = std::chrono::steady_clock::now();
      const double total_sec = std::chrono::duration<double>(t_end - t_start).count();
      // --- End timing ---

      const Eigen::Matrix4f T_0_f  = T_0_i;       // pose of frame f
      const Eigen::Matrix4f T_i_ip1 = r.T_s2t;
      T_0_i = T_0_f * T_i_ip1;

      std::cout << "# frame=" << f << "->" << (f + 1)
                << " iters=" << r.num_iters
                << " align_sec=" << r.registration_seconds
                << " total_sec=" << total_sec
                << " pts_src=" << src.size()
                << " pts_tgt=" << tgt.size()
                << " code=" << r.return_code << "\n";
      std::cout << "# T_i_from_i+1\n";
      print_mat4(T_i_ip1);
      std::cout << "# T_0_from_i+1\n";
      print_mat4(T_0_i);

      if (traj_ofs.is_open()) write_kitti_pose(traj_ofs, T_0_i);
      init = T_i_ip1;
      ++frames_processed;

#ifdef GCVO_USE_VIZ
      if (viewer) {
        viewer->add_cloud(src.points(), T_0_f);
        viewer->add_pose(T_0_i);
        viewer->spin_once();
      }
#endif
    }

    std::cout << "Processed " << frames_processed << " pairs ("
              << (frames_processed + 1) << " frames).\n";

#ifdef GCVO_USE_VIZ
    if (viewer) {
      std::cout << "Processing complete. Close viewer window to exit.\n";
      viewer->spin();
    }
#endif

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
