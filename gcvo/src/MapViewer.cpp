/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: src/utils/viewer.cpp (no 1:1 mapping claimed)
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

#include "gcvo/utils/MapViewer.hpp"

#ifdef GCVO_USE_VIZ

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <string>

namespace gcvo {

struct MapViewer::Impl {
  pcl::visualization::PCLVisualizer::Ptr viewer;
  pcl::PointCloud<pcl::PointXYZ>::Ptr    map_cloud;
  std::vector<Eigen::Vector3f>           traj_origins;
  std::size_t drawn_segments = 0;
  bool        cloud_dirty    = false;

  explicit Impl(const std::string& name)
    : viewer(new pcl::visualization::PCLVisualizer(name)),
      map_cloud(new pcl::PointCloud<pcl::PointXYZ>) {
    viewer->setBackgroundColor(0.0, 0.0, 0.0);
    viewer->addCoordinateSystem(1.0);
    viewer->initCameraParameters();
  }

  void refresh() {
    // Re-upload the accumulated point cloud when it changed.
    if (cloud_dirty) {
      viewer->removePointCloud("map");
      pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
          white(map_cloud, 255, 255, 255);
      viewer->addPointCloud<pcl::PointXYZ>(map_cloud, white, "map");
      viewer->setPointCloudRenderingProperties(
          pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "map");
      cloud_dirty = false;
    }

    // Draw any new trajectory line segments.
    while (drawn_segments + 1 < traj_origins.size()) {
      const auto& a = traj_origins[drawn_segments];
      const auto& b = traj_origins[drawn_segments + 1];
      pcl::PointXYZ pa, pb;
      pa.x = a.x(); pa.y = a.y(); pa.z = a.z();
      pb.x = b.x(); pb.y = b.y(); pb.z = b.z();
      const std::string id = "traj_" + std::to_string(drawn_segments);
      viewer->addLine(pa, pb, 1.0, 0.0, 0.0, id);  // red polyline
      ++drawn_segments;
    }
  }
};

// ---- public API ------------------------------------------------------------

MapViewer::MapViewer(const std::string& window_name)
    : impl_(std::make_unique<Impl>(window_name)) {}

MapViewer::~MapViewer() = default;

void MapViewer::add_points(const float* xyz, std::size_t n) {
  for (std::size_t i = 0; i < n; ++i) {
    pcl::PointXYZ pt;
    pt.x = xyz[3 * i];
    pt.y = xyz[3 * i + 1];
    pt.z = xyz[3 * i + 2];
    impl_->map_cloud->push_back(pt);
  }
  impl_->cloud_dirty = true;
}

void MapViewer::add_pose(const Eigen::Matrix4f& T_world) {
  impl_->traj_origins.push_back(T_world.block<3, 1>(0, 3));
}

void MapViewer::spin_once(int time_ms) {
  impl_->refresh();
  impl_->viewer->spinOnce(time_ms);
}

void MapViewer::spin() {
  impl_->refresh();
  impl_->viewer->spin();
}

bool MapViewer::was_stopped() const {
  return impl_->viewer->wasStopped();
}

}  // namespace gcvo

#endif  // GCVO_USE_VIZ
