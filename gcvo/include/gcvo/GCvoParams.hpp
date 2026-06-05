/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/CvoParams.hpp (no 1:1 mapping claimed)
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

/// @file GCvoParams.hpp
/// @brief Runtime parameters for the GCVO solver, loaded from YAML.
///
/// All parameters have compiled-in defaults. A YAML file only needs to
/// specify the values you want to override. See the README "GCvoParams
/// reference" section for a complete description of each parameter.
///
/// @par Example YAML
/// @code{.yaml}
///   l_init: 1.0
///   amplitude: 0.1
///   max_neighbors: 64
///   kernel_type: 0
/// @endcode

#pragma once
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <yaml-cpp/yaml.h>

namespace gcvo {

  /// Kernel type selector for the geometric distance term.
  enum struct GCvoKernelType {
    SCALAR = 0,   ///< Isotropic: single scalar length-scale everywhere.
    DENSE,        ///< Isotropic with per-point covariance in Mahalanobis distance.
    RESCALED      ///< Anisotropic: per-point covariance rescales the kernel.
  };

  /// Runtime parameters for GCvoGPU.
  ///
  /// Loaded from YAML via read_GCvoParams_yaml(). Missing keys use the
  /// default values shown below.
  struct GCvoParams {

    // -- Kernel shape --
    float l_init;           ///< Initial kernel length-scale (default 0.5).
    float l_min;            ///< Minimum length-scale floor (default 0.05).
    float amplitude;              ///< Gaussian kernel amplitude (default 0.1).
    float sparsity_cutoff;           ///< Sparsity threshold -- entries below this are dropped (default 0.0006).
    float c_l;              ///< Coefficient scaling the geometric length-scale (default 0.15).
    float c_amplitude;            ///< Coefficient scaling the feature length-scale (default 0.6).

    // -- Convergence --
    int max_iterations;             ///< Maximum Gauss-Newton iterations (default 10000).
    float tol;                ///< Primary convergence threshold on step norm (default 5e-5).
    float tol_2;              ///< Secondary (tighter) convergence threshold (default 1.2e-5).

    // -- Neighbour search --
    int max_neighbors; ///< Max neighbours per point in the correspondence matrix (default 512).

    // -- Length-scale schedule --
    float l_decay_rate;     ///< Multiplicative decay factor for l (default 0.9).
    int l_decay_start;      ///< Iterations before first decay (default 30).
    int indicator_window;         ///< Window size for the l-decay stability indicator (default 15).
    float indicator_threshold;  ///< Threshold below which IP change is "stable" (default 0.2).

    // -- Feature / geometry toggles --
    int use_geometry;    ///< Include geometric (XYZ) kernel term (default 1).
    int use_features;   ///< Include feature (appearance) kernel term (default 0).
    int use_kdtree;      ///< Kept for YAML backward compat; ignored by the solver (default 0).
    int export_correspondence; ///< Return sparse correspondence matrix in result (default 0).
    GCvoKernelType kernel_type; ///< Kernel type: SCALAR(0), DENSE(1), RESCALED(2) (default SCALAR).
    float kernel_eval_max_dist; ///< Hard distance cutoff for kernel evaluation (default inf).
    int verbose;          ///< 1 = print per-iteration diagnostics (default 0).

    /// Adaptive k: shrink nearest_neighbors per iteration based on max actually used.
    /// 0 = disabled (fixed k), 1 = enabled.
    int neighbor_decay;

    /// Add the SE(3) connection-term correction (`-gamma^T`) to the GN curvature
    /// matrix. 0 = off, 1 = on (default).
    int use_connection_term;

    /// Clamp per-point covariance eigenvalues to >= this value (0 = disabled, default 0).
    float cov_eig_min;

    /// Clamp per-point covariance eigenvalues to <= this value (0 = disabled, default 0).
    float cov_eig_max;

    /// Include ℓ²·I regularization in cov_sum_inv_plus_l2I during kernel evaluation (default 1).
    /// Set to 0 when eigenvalue clamping already ensures conditioning.
    int use_ell2_in_kernel;

    /// Euclidean squared-distance pre-filter for DENSE/RESCALED kernel evaluation (default inf).
    /// Pairs with squared Euclidean distance > this are skipped before computing cov_inv.
    /// E.g. set to 1.0 to ignore pairs more than 1 m apart.
    float kernel_euclidean_max_dist;

    GCvoParams()
      : l_init(0.5),
        l_min(0.05),
        amplitude(0.1),
        sparsity_cutoff(0.0006),
        c_l(0.15),
        c_amplitude(0.6),
        max_iterations(10000),
        tol(0.00005f),
        tol_2(0.000012f),
        max_neighbors(512),
        l_decay_rate(0.9f),
        l_decay_start(30),
        indicator_window(15),
        indicator_threshold(0.2f),
        use_geometry(1),
        use_features(0),
        use_kdtree(0),
        export_correspondence(0),
        kernel_type(GCvoKernelType::SCALAR),
        kernel_eval_max_dist(std::numeric_limits<float>::max()),
        verbose(0),
        neighbor_decay(0),
        use_connection_term(1),
        cov_eig_min(0.0f),
        cov_eig_max(0.0f),
        use_ell2_in_kernel(1),
        kernel_euclidean_max_dist(std::numeric_limits<float>::max()) {}
  };

  /// Load GCvoParams from a YAML file, overwriting only the keys present.
  ///
  /// @param filename  Path to the YAML parameter file.
  /// @param params    Output params struct (fields not in the file keep their current value).
  inline void read_GCvoParams_yaml(const char *filename, GCvoParams * params) {
    YAML::Node fs = YAML::LoadFile(filename);

    if (fs["l_init"]) params->l_init = fs["l_init"].as<float>();
    if (fs["l_min"]) params->l_min = fs["l_min"].as<float>();
    if (fs["amplitude"]) params->amplitude = fs["amplitude"].as<float>();
    if (fs["sparsity_cutoff"]) params->sparsity_cutoff = fs["sparsity_cutoff"].as<float>();
    if (fs["c_l"]) params->c_l = fs["c_l"].as<float>();
    if (fs["c_amplitude"]) params->c_amplitude = fs["c_amplitude"].as<float>();
    if (fs["max_iterations"]) params->max_iterations = fs["max_iterations"].as<int>();
    if (fs["tol"]) params->tol = fs["tol"].as<float>();
    if (fs["tol_2"]) params->tol_2 = fs["tol_2"].as<float>();

    if (fs["l_decay_rate"]) params->l_decay_rate = fs["l_decay_rate"].as<float>();
    if (fs["l_decay_start"]) params->l_decay_start = fs["l_decay_start"].as<int>();

    if (fs["indicator_window"]) params->indicator_window = fs["indicator_window"].as<int>();
    if (fs["indicator_threshold"]) params->indicator_threshold = fs["indicator_threshold"].as<float>();

    if (fs["use_geometry"]) params->use_geometry = fs["use_geometry"].as<int>();
    if (fs["use_features"]) params->use_features = fs["use_features"].as<int>();
    if (fs["use_kdtree"]) params->use_kdtree = fs["use_kdtree"].as<int>();
    if (fs["export_correspondence"]) params->export_correspondence = fs["export_correspondence"].as<int>();

    if (fs["kernel_type"]) params->kernel_type = static_cast<gcvo::GCvoKernelType>(fs["kernel_type"].as<int>());
    if (fs["kernel_eval_max_dist"]) params->kernel_eval_max_dist = fs["kernel_eval_max_dist"].as<float>();
    if (fs["verbose"]) params->verbose = fs["verbose"].as<int>();

    if (fs["max_neighbors"]) params->max_neighbors = fs["max_neighbors"].as<int>();

    if (fs["neighbor_decay"]) params->neighbor_decay = fs["neighbor_decay"].as<int>();

    if (fs["use_connection_term"]) params->use_connection_term = fs["use_connection_term"].as<int>();

    if (fs["cov_eig_min"]) params->cov_eig_min = fs["cov_eig_min"].as<float>();
    if (fs["cov_eig_max"]) params->cov_eig_max = fs["cov_eig_max"].as<float>();
    if (fs["use_ell2_in_kernel"]) params->use_ell2_in_kernel = fs["use_ell2_in_kernel"].as<int>();
    if (fs["kernel_euclidean_max_dist"]) params->kernel_euclidean_max_dist = fs["kernel_euclidean_max_dist"].as<float>();
  }

} // namespace gcvo
