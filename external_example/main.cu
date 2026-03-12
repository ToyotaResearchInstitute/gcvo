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

#include <iostream>
#include <Eigen/Dense>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "gcvo/GCvoGPU.hpp"
#include "gcvo/utils/GCvoPointCloud.hpp"
#include "gcvo/utils/PointTypes19.hpp"

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: register_clouds <params.yaml> <source.pcd> <target.pcd>\n";
        return 1;
    }

    const std::string params_file = argv[1];
    const std::string source_file = argv[2];
    const std::string target_file = argv[3];

    // Load standard PCL clouds
    pcl::PointCloud<pcl::PointXYZI> pcl_src, pcl_tgt;
    if (pcl::io::loadPCDFile(source_file, pcl_src) < 0) {
        std::cerr << "Failed to load " << source_file << "\n";
        return 1;
    }
    if (pcl::io::loadPCDFile(target_file, pcl_tgt) < 0) {
        std::cerr << "Failed to load " << target_file << "\n";
        return 1;
    }

    std::cout << "Source: " << pcl_src.size() << " points\n";
    std::cout << "Target: " << pcl_tgt.size() << " points\n";

    // Wrap into GCVO point clouds (auto-maps intensity -> features[0])
    gcvo::GCvoPointCloudT<gcvo::PointS1> src(pcl_src);
    gcvo::GCvoPointCloudT<gcvo::PointS1> tgt(pcl_tgt);

    // Create solver from YAML parameter file
    gcvo::GCvoGPU<gcvo::PointS1> solver(params_file);

    // Align with identity initial guess
    Eigen::Matrix4f init = Eigen::Matrix4f::Identity();
    auto result = solver.align(src, tgt, init);

    std::cout << "Converged in " << result.num_iters << " iterations ("
              << result.registration_seconds << " s)\n";
    std::cout << "Transform (T_s2t):\n" << result.T_s2t << "\n";

    return 0;
}
