/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/utils/VoxelMap.hpp  (no 1:1 mapping claimed)
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

#include <unordered_map>
#include <memory_resource>
#include <optional>
#include <iostream>
#include <functional>
#include <ostream>
#include <vector>
#include <math.h>
#include <cstdint>
#include <string>

#include <Eigen/Core>

namespace cvo
{
    // used as hash key in voxelMap
    struct VoxelIndex {
        int32_t ix;
        int32_t iy;
        int32_t iz;

      bool selected = false;
        bool operator==(const VoxelIndex& other) const {
          return (ix == other.ix && iy == other.iy && iz == other.iz);
        }

        friend std::ostream& operator<<(std::ostream& os, const VoxelIndex& vi) {
            os << "ix: " << vi.ix << ", iy: " << vi.iy << ", iz: " << vi.iz;
            return os;
        }
    };

    // A minimal voxel record for first-point wins downsampling.
    // Stores the voxel center (for debug/pcd) and the one kept point.
    template <typename PointType>
    struct VoxelFirstPoint {
        float xc;
        float yc;
        float zc;
        PointType* pt;

        VoxelFirstPoint() : xc(0.0f), yc(0.0f), zc(0.0f), pt(nullptr) {}
        VoxelFirstPoint(float x, float y, float z, PointType* p) : xc(x), yc(y), zc(z), pt(p) {}
    };

    template <typename PointType>
    class VoxelMapFirstPoint {
    public:
        // reserveVoxels and maxLoadFactor are optional perf knobs.
        // initialBufferBytes controls the initial block size taken from the upstream resource
        // (still monotonic; call reset() to release between frames).
        VoxelMapFirstPoint(float voxelSize,
                           size_t reserveVoxels = 0,
                           float maxLoadFactor = 0.7f,
                           size_t initialBufferBytes = 0,
                           std::pmr::memory_resource* upstream = std::pmr::get_default_resource());
        ~VoxelMapFirstPoint();

        // Optional perf controls (can also be set via ctor)
        void reserve(size_t reserveVoxels);
        void set_max_load_factor(float maxLoadFactor);

        // For monotonic_buffer_resource: prefer reset() (releases memory) when rebuilding often.
        void reset(size_t reserveVoxels = 0);

        /**
         * @brief insert a 3D point into the voxel map
         * @param pt: a 3D point
         * @return true if insertion is successful; False if point already exists
         */
        bool insert_point(PointType* pt);

        /**
         * @brief remove a 3D point from the voxel map
         * @param pt: a 3D point
         * @return true if deletion is successful; False if point doesn't exist in map
         */
        bool delete_point(PointType* pt);

        /**
         * @brief After BA, an ActivePoint's ref frame changes pose, need to provide its containing voxel for removal
         * @param pt: a 3D point
         * @param voxel: the voxel that stores the given pt
         * @return true if deletion is successful; False if point doesn't exist in map
         */
        bool delete_point_BA(PointType* pt, const VoxelFirstPoint<PointType>* voxel);

        /**
         * @brief query a 3D point to find the voxel containing it in the voxel map
         * @param pt: a 3D point
         * @return nullptr if no voxel exists at the given pt, or the voxel
         */
        const VoxelFirstPoint<PointType>* query_point(const PointType* pt) const;
        const VoxelFirstPoint<PointType>* query_point(float globalX, float globalY, float globalZ) const;

        const VoxelFirstPoint<PointType>* query_point_raycasting(const PointType * pt, float minDist=0.5, float maxDist=55.0);

        /**
         * @brief obtain the frameIds that have seen this voxel
         * @param pt: a 3D point
         * @return a set of frameIds, empty if point isn't inside a voxel
         */
      //        std::unordered_set<int> voxel_seen_frames(PointType* pt) const;

        /**
         * @brief returns the number of voxels
         * @return number of voxels in the voxelmap
         */
        size_t size();

        /**
         * @brief takes one point from every exisitng voxel
         * @return a vector of points
         */
        const std::vector<PointType*> sample_points() const;

        // updateCovis Debug use
        // public:
        void save_voxels_pcd(std::string filename) const;

        void save_points_pcd(std::string filename) const;

    private:
        /**
         * @brief finds the voxel index that contains a given point
         * @param pt: a 3D point
         * @return the integer voxel index (ix,iy,iz)
         */
        VoxelIndex point_to_voxel_index(const PointType* pt) const;
        VoxelIndex point_to_voxel_index(float globalX, float globalY, float globalZ) const;

    private:
        float voxelSize_ = 0.1f;                   // Edge length of a single voxel cubic
        float invVoxelSize_ = 10.0f;               // cached 1 / voxelSize_

        size_t reserveVoxels_ = 0;
        float maxLoadFactor_ = 0.7f;

        std::pmr::monotonic_buffer_resource mem_;

        using MapType = std::pmr::unordered_map<VoxelIndex, VoxelFirstPoint<PointType>>;
        std::optional<MapType> vmap_;
    };
}

// Hash for VoxelIndex (integer key)
namespace std {
  template<>
  struct hash<cvo::VoxelIndex> {
    size_t operator()(cvo::VoxelIndex const& vi) const noexcept {
      // Simple prime-mix hashing (fast, good enough for voxel indices)
      const uint64_t x = static_cast<uint64_t>(static_cast<uint32_t>(vi.ix));
      const uint64_t y = static_cast<uint64_t>(static_cast<uint32_t>(vi.iy));
      const uint64_t z = static_cast<uint64_t>(static_cast<uint32_t>(vi.iz));
      return static_cast<size_t>((x * 73856093ull) ^ (y * 19349663ull) ^ (z * 83492791ull));
    }
  };
}
