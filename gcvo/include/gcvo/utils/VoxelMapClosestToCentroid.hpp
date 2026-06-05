/*
 * VoxelMapClosestToCentroid.hpp
 *
 * Voxel downsampling that selects the point closest to the running centroid of
 * its cell. Incremental: O(1) per insertion. Useful when the "first point"
 * heuristic produces systematic spatial bias (e.g. for LiDAR scans where
 * insertion order correlates with sensor scan angle).
 *
 * Differs from VoxelMapFirstPoint in that each cell tracks:
 *   - running centroid c = (1/n) Σ p_i
 *   - current representative pt_rep, defined as the inserted point with the
 *     smallest |pt - c|² at the time of its insertion (incremental approximation
 *     of "closest to final centroid")
 *
 * When a new point arrives:
 *   1. Update centroid: c_new = (c_old · n + p_new) / (n+1)
 *   2. If |p_new - c_new|² < |pt_rep - c_new|², replace pt_rep with p_new
 *
 * This is not exactly equivalent to "find argmin |p - centroid_final|" but is a
 * strong incremental heuristic that doesn't require storing all per-voxel points.
 */

#pragma once

#include "gcvo/utils/VoxelMapFirstPoint.hpp"  // reuse VoxelIndex + hash

#include <unordered_map>
#include <memory_resource>
#include <optional>
#include <vector>
#include <cstdint>

namespace cvo {

  template <typename PointType>
  struct VoxelCentroidCell {
    float cx = 0.0f, cy = 0.0f, cz = 0.0f;   // running centroid
    int   count = 0;
    PointType* pt_rep = nullptr;             // current representative (closest to centroid so far)
  };

  template <typename PointType>
  class VoxelMapClosestToCentroid {
   public:
    explicit VoxelMapClosestToCentroid(float voxelSize,
                                       size_t reserveVoxels = 0,
                                       float maxLoadFactor = 0.7f,
                                       size_t initialBufferBytes = 0,
                                       std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
        : voxelSize_(voxelSize),
          invVoxelSize_(1.0f / voxelSize),
          maxLoadFactor_(maxLoadFactor),
          mem_(initialBufferBytes, upstream) {
      vmap_.emplace(&mem_);
      vmap_->max_load_factor(maxLoadFactor_);
      if (reserveVoxels > 0) vmap_->reserve(reserveVoxels);
    }

    // Insert a point. Updates the cell's running centroid and may swap the
    // representative if the new point is closer to the new centroid.
    bool insert_point(PointType* pt) {
      VoxelIndex idx;
      idx.ix = static_cast<int32_t>(std::lrintf(pt->x * invVoxelSize_));
      idx.iy = static_cast<int32_t>(std::lrintf(pt->y * invVoxelSize_));
      idx.iz = static_cast<int32_t>(std::lrintf(pt->z * invVoxelSize_));

      auto& map = *vmap_;
      auto [it, inserted] = map.try_emplace(idx);
      VoxelCentroidCell<PointType>& cell = it->second;

      // Update centroid online: c_new = (c_old * n + p) / (n+1)
      const float n_old = static_cast<float>(cell.count);
      const float n_new = n_old + 1.0f;
      cell.cx = (cell.cx * n_old + pt->x) / n_new;
      cell.cy = (cell.cy * n_old + pt->y) / n_new;
      cell.cz = (cell.cz * n_old + pt->z) / n_new;
      cell.count++;

      // Compare distance to NEW centroid
      const float dx_new = pt->x - cell.cx;
      const float dy_new = pt->y - cell.cy;
      const float dz_new = pt->z - cell.cz;
      const float d2_new = dx_new*dx_new + dy_new*dy_new + dz_new*dz_new;

      if (cell.pt_rep == nullptr) {
        cell.pt_rep = pt;
        return true;
      }
      const float dx_rep = cell.pt_rep->x - cell.cx;
      const float dy_rep = cell.pt_rep->y - cell.cy;
      const float dz_rep = cell.pt_rep->z - cell.cz;
      const float d2_rep = dx_rep*dx_rep + dy_rep*dy_rep + dz_rep*dz_rep;

      if (d2_new < d2_rep) cell.pt_rep = pt;
      return inserted;
    }

    size_t size() const { return vmap_->size(); }

    std::vector<PointType*> sample_points() const {
      std::vector<PointType*> res;
      res.reserve(vmap_->size());
      for (const auto& kv : *vmap_) {
        if (kv.second.pt_rep) res.push_back(kv.second.pt_rep);
      }
      return res;
    }

   private:
    float voxelSize_ = 0.1f;
    float invVoxelSize_ = 10.0f;
    float maxLoadFactor_ = 0.7f;
    std::pmr::monotonic_buffer_resource mem_;
    using MapType = std::pmr::unordered_map<VoxelIndex, VoxelCentroidCell<PointType>>;
    std::optional<MapType> vmap_;
  };

}  // namespace cvo
