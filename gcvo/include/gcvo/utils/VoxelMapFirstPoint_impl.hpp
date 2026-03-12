/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/VoxelMap_impl.hpp (no 1:1 mapping claimed)
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

#pragma once

#include "gcvo/utils/VoxelMapFirstPoint.hpp"
#include <random>

namespace cvo {

  template <typename PointType>
  VoxelMapFirstPoint<PointType>::VoxelMapFirstPoint(float voxelSize,
                                                    size_t reserveVoxels,
                                                    float maxLoadFactor,
                                                    size_t initialBufferBytes,
                                                    std::pmr::memory_resource* upstream)
      : voxelSize_(voxelSize),
        invVoxelSize_(1.0f / voxelSize),
        reserveVoxels_(reserveVoxels),
        maxLoadFactor_(maxLoadFactor),
        mem_(initialBufferBytes, upstream) {

    // std::cout << "Assigned voxelSize_" << voxelSize << std::endl;
    std::srand(3141592);	// want to be deterministic.

    vmap_.emplace(&mem_);
    vmap_->max_load_factor(maxLoadFactor_);
    if (reserveVoxels_ > 0) {
      vmap_->reserve(reserveVoxels_);
    }
  }

  template <typename PointType>
  VoxelMapFirstPoint<PointType>::~VoxelMapFirstPoint() {
    //std::cout<<"Voxel map destructed\n";
  }

  template <typename PointType>
  void VoxelMapFirstPoint<PointType>::reserve(size_t reserveVoxels) {
    reserveVoxels_ = reserveVoxels;
    vmap_->reserve(reserveVoxels_);
  }

  template <typename PointType>
  void VoxelMapFirstPoint<PointType>::set_max_load_factor(float maxLoadFactor) {
    maxLoadFactor_ = maxLoadFactor;
    vmap_->max_load_factor(maxLoadFactor_);
  }

  template <typename PointType>
  void VoxelMapFirstPoint<PointType>::reset(size_t reserveVoxels) {
    // monotonic_buffer_resource does NOT reuse freed memory; release + reconstruct is the intended pattern.
    vmap_.reset();      // destroy map first (it references mem_)
    mem_.release();     // free all blocks obtained from upstream
    vmap_.emplace(&mem_);

    vmap_->max_load_factor(maxLoadFactor_);

    if (reserveVoxels > 0) {
      reserveVoxels_ = reserveVoxels;
    }
    if (reserveVoxels_ > 0) {
      vmap_->reserve(reserveVoxels_);
    }
  }

  template <typename PointType>
  bool VoxelMapFirstPoint<PointType>::insert_point(PointType* pt) {
    // 1. find coresponding voxel coordinates
    const VoxelIndex idx = point_to_voxel_index(pt);

    // 2. insert point to map (first-point wins)
    //std::cout<<"insert the point to "<<idx.ix<<", "<<idx.iy<<", "<<idx.iz<<"\n";
    auto& map = *vmap_;

    // Only constructs the value if insertion happens (fast path).
    const float xc = static_cast<float>(idx.ix) * voxelSize_;
    const float yc = static_cast<float>(idx.iy) * voxelSize_;
    const float zc = static_cast<float>(idx.iz) * voxelSize_;

    auto [it, inserted] = map.try_emplace(idx, xc, yc, zc, pt);
    return inserted;
  }

  template <typename PointType>
  bool VoxelMapFirstPoint<PointType>::delete_point(PointType* pt) {
    // 1. convert to integer coord to look up its voxel
    const VoxelIndex idx = point_to_voxel_index(pt);
    auto& map = *vmap_;
    auto it = map.find(idx);
    if (it == map.end())
      return false;

    // 2. remove this point from the voxel (only one point stored)
    if (it->second.pt != pt)
      return false;

    map.erase(it);
    return true;
  }

  template <typename PointType>
  bool VoxelMapFirstPoint<PointType>::delete_point_BA(PointType* pt, const VoxelFirstPoint<PointType>* voxel) {
    // After BA, an ActivePoint's ref frame changes pose, need to provide its containing voxel for removal
    VoxelIndex idx;
    idx.ix = static_cast<int32_t>(std::lrintf(voxel->xc * invVoxelSize_));
    idx.iy = static_cast<int32_t>(std::lrintf(voxel->yc * invVoxelSize_));
    idx.iz = static_cast<int32_t>(std::lrintf(voxel->zc * invVoxelSize_));

    auto& map = *vmap_;
    auto it = map.find(idx);
    if (it == map.end())
      return false;

    if (it->second.pt != pt)
      return false;

    map.erase(it);
    return true;
  }

  template <typename PointType>
  const VoxelFirstPoint<PointType>* VoxelMapFirstPoint<PointType>::query_point(const PointType* pt) const {
    // 1. convert to integer coord to look up its voxel
    const VoxelIndex idx = point_to_voxel_index(pt);
    const auto& map = *vmap_;
    auto it = map.find(idx);
    if (it == map.end())
      return nullptr;
    return &it->second;
  }

  template <typename PointType>
  const VoxelFirstPoint<PointType>* VoxelMapFirstPoint<PointType>::query_point(float globalX, float globalY, float globalZ) const {
    const VoxelIndex idx = point_to_voxel_index(globalX, globalY, globalZ);
    const auto& map = *vmap_;
    auto it = map.find(idx);
    if (it == map.end())
      return nullptr;
    return &it->second;
  }

  template <typename PointType>
  size_t VoxelMapFirstPoint<PointType>::size() {
    return vmap_->size();
  }

  template <typename PointType>
  const std::vector<PointType*> VoxelMapFirstPoint<PointType>::sample_points() const {
    std::vector<PointType*> res;
    res.reserve(vmap_->size());

    for (const auto& kv : *vmap_) {
      if (kv.second.pt)
        res.push_back(kv.second.pt);
    }
    return res;
  }

  template <typename PointType>
  VoxelIndex VoxelMapFirstPoint<PointType>::point_to_voxel_index(const PointType* pt) const {
    //std::cout<<"point world pos is "<<p_wld.transpose();
    // 2. find its corresponding voxel
    // (No heap allocations here; keep it tight.)
    VoxelIndex idx;
    idx.ix = static_cast<int32_t>(std::lrintf(pt->x * invVoxelSize_));
    idx.iy = static_cast<int32_t>(std::lrintf(pt->y * invVoxelSize_));
    idx.iz = static_cast<int32_t>(std::lrintf(pt->z * invVoxelSize_));
    return idx;
  }

  template <typename PointType>
  VoxelIndex VoxelMapFirstPoint<PointType>::point_to_voxel_index(float globalX, float globalY, float globalZ) const {
    // 1. get pt coord in world frame
    Eigen::Vector4f p_wld;
    p_wld << globalX, globalY, globalZ, 1.0;
    //std::cout<<"point world pos is "<<p_wld.transpose();
    // 2. find its corresponding voxel
    VoxelIndex idx;
    idx.ix = static_cast<int32_t>(std::lrintf(p_wld(0) * invVoxelSize_));
    idx.iy = static_cast<int32_t>(std::lrintf(p_wld(1) * invVoxelSize_));
    idx.iz = static_cast<int32_t>(std::lrintf(p_wld(2) * invVoxelSize_));
    return idx;
  }

}
