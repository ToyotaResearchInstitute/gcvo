/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: include/UnifiedCvo/cvo/SparseKernelMat.hpp (no 1:1 mapping claimed)
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

namespace gcvo {

// This is allocated on GPU.
struct Correlation {
  int num_sources;
  int max_entries_per_row;
  unsigned int total_entries;
  float* weights;
  int* target_indices;
  unsigned int* num_entries_per_row;
};

unsigned int count_entries(Correlation* A_host);
void compute_total_entries(Correlation* A_host);
unsigned int max_entries_in_row(Correlation* A_host);

float correlation_sum(Correlation* A_host);
float correlation_sum(Correlation* A_host, int num_neighbors);

void clear_Correlation(Correlation* A_host);
void clear_Correlation(Correlation* A_host, int num_neighbors);

Correlation* init_Correlation_gpu(int row, int col, Correlation& A_host);
void delete_Correlation_gpu(Correlation* A_gpu, Correlation* A_host);

} // namespace gcvo
