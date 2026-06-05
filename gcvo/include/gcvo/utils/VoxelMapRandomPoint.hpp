/*
 * VoxelMapRandomPoint.hpp
 *
 * Voxel downsampling that selects a uniformly random point from each cell using
 * incremental reservoir sampling (reservoir size 1):
 *
 *   On the k-th insertion to a cell (k = 1, 2, ...):
 *     - with probability 1/k, the new point becomes the cell's representative
 *
 * Result: any point in the cell is equally likely to be chosen, O(1) per
 * insertion, no storage of intermediate points. Useful to test whether the
 * spatial bias of "first-point wins" voxel selection affects downstream
 * alignment quality.
 */

#pragma once

#include "gcvo/utils/VoxelMapFirstPoint.hpp"  // reuse VoxelIndex + hash

#include <unordered_map>
#include <memory_resource>
#include <optional>
#include <vector>
#include <cstdint>
#include <random>

namespace cvo {

  template <typename PointType>
  struct VoxelRandomCell {
    int count = 0;
    PointType* pt_rep = nullptr;
  };

  template <typename PointType>
  class VoxelMapRandomPoint {
   public:
    explicit VoxelMapRandomPoint(float voxelSize,
                                 uint64_t seed = 3141592ull,
                                 size_t reserveVoxels = 0,
                                 float maxLoadFactor = 0.7f,
                                 size_t initialBufferBytes = 0,
                                 std::pmr::memory_resource* upstream = std::pmr::get_default_resource())
        : voxelSize_(voxelSize),
          invVoxelSize_(1.0f / voxelSize),
          maxLoadFactor_(maxLoadFactor),
          mem_(initialBufferBytes, upstream),
          rng_(seed) {
      vmap_.emplace(&mem_);
      vmap_->max_load_factor(maxLoadFactor_);
      if (reserveVoxels > 0) vmap_->reserve(reserveVoxels);
    }

    // Reservoir sampling (size 1): on the k-th insertion to a cell, replace the
    // representative with probability 1/k. After N inserts, any of the N is
    // equally likely (probability 1/N) to be the representative.
    bool insert_point(PointType* pt) {
      VoxelIndex idx;
      idx.ix = static_cast<int32_t>(std::lrintf(pt->x * invVoxelSize_));
      idx.iy = static_cast<int32_t>(std::lrintf(pt->y * invVoxelSize_));
      idx.iz = static_cast<int32_t>(std::lrintf(pt->z * invVoxelSize_));

      auto& map = *vmap_;
      auto [it, inserted] = map.try_emplace(idx);
      VoxelRandomCell<PointType>& cell = it->second;

      cell.count++;
      if (cell.pt_rep == nullptr) {
        cell.pt_rep = pt;
      } else {
        // P(replace) = 1 / count
        std::uniform_int_distribution<int> dist(1, cell.count);
        if (dist(rng_) == 1) cell.pt_rep = pt;
      }
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
    std::mt19937_64 rng_;
    using MapType = std::pmr::unordered_map<VoxelIndex, VoxelRandomCell<PointType>>;
    std::optional<MapType> vmap_;
  };

}  // namespace cvo
