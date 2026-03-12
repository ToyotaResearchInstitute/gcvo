/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/viewer.hpp  (no 1:1 mapping claimed)
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

// Independent PCL-based map viewer for GCVO.
//
// Supports incrementally adding point clouds (of any type with .x .y .z)
// transformed to a common world frame, and visualising the estimated
// trajectory as a polyline.
//
// The entire class is compiled away when GCVO_USE_VIZ is not defined,
// so callers should guard #include and usage with #ifdef GCVO_USE_VIZ.

#pragma once

#ifdef GCVO_USE_VIZ

#include <Eigen/Dense>

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace gcvo {

class MapViewer {
public:
  explicit MapViewer(const std::string& window_name = "GCVO Map Viewer");
  ~MapViewer();

  MapViewer(const MapViewer&) = delete;
  MapViewer& operator=(const MapViewer&) = delete;

  /// Add pre-transformed world-frame points (xyz interleaved, 3 floats per pt).
  void add_points(const float* xyz, std::size_t num_points);

  /// Convenience: transform points to world frame and add them.
  /// Works with any container whose elements expose .x .y .z (pcl points,
  /// gcvo point types, std::vector<PointT>, etc.).
  template <typename Container>
  void add_cloud(const Container& pts, const Eigen::Matrix4f& T_world) {
    const Eigen::Matrix3f R = T_world.block<3, 3>(0, 0);
    const Eigen::Vector3f t = T_world.block<3, 1>(0, 3);
    std::vector<float> buf(pts.size() * 3);
    for (std::size_t i = 0; i < pts.size(); ++i) {
      Eigen::Vector3f pw = R * Eigen::Vector3f(pts[i].x, pts[i].y, pts[i].z) + t;
      buf[3 * i]     = pw.x();
      buf[3 * i + 1] = pw.y();
      buf[3 * i + 2] = pw.z();
    }
    add_points(buf.data(), pts.size());
  }

  /// Append a pose to the trajectory being visualised.
  void add_pose(const Eigen::Matrix4f& T_world);

  /// Non-blocking update — call after each frame.
  void spin_once(int time_ms = 100);

  /// Blocking — blocks until the viewer window is closed.
  void spin();

  bool was_stopped() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace gcvo

#endif  // GCVO_USE_VIZ
