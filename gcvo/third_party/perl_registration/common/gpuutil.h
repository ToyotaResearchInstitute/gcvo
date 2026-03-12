/*
 * GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
 * Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
 * Upstream is included in this repo as a git submodule at: rkhs_ba/
 * Upstream file: thirdparty/cugicp/common/gpuutil.h (no 1:1 mapping claimed)
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

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

// Minimal CUDA error helper (mirrors the upstream perl_registration utility).

namespace perl_registration {

inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
  if (code != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line);
    if (abort) std::exit(static_cast<int>(code));
  }
}

}  // namespace perl_registration

#define cudaSafe(ans) ::perl_registration::gpuAssert((ans), __FILE__, __LINE__, true)
