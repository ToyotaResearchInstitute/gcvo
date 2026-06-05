# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build Commands

```bash
# Configure and build (Release)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j

# With visualization support
cmake .. -DCMAKE_BUILD_TYPE=Release -DGCVO_BUILD_VIZ=ON

# Run all tests
cd build && ctest --output-on-failure

# Run a single test
cd build && ctest -R gcvo_test_gn --output-on-failure

# Install for downstream find_package(gcvo)
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install -DCMAKE_BUILD_TYPE=Release
make -j && make install
```

**CMake options:** `GCVO_BUILD_APPS` (ON), `GCVO_BUILD_TESTS` (ON), `GCVO_BUILD_VIZ` (OFF), `GCVO_TEST_SAVE_PCD` (OFF), `GCVO_CUDA_THREADS` (256).

## Running the Apps

```bash
# Stanford Bunny demo (intensity)
./build/gcvo/gcvo_align_pcd --source demo_data/bunny.pcd --target demo_data/bunny.pcd \
  --params gcvo_params/test.yaml --type intensity

# KITTI frame-to-frame odometry
./build/gcvo_kitti_f2f --params gcvo_params/gcvo_driving_nonisotropic_gn.yaml \
  --kitti_root /data/kitti/dataset --sequence 05 \
  --start 0 --count 999999 --voxel_mode fast \
  --first_frame_l_init 1.0 --traj_file traj.kitti

# Evaluate trajectory against LiDAR-frame ground truth
~/code/docker_home/cvo/odometry_eval/KITTI/cpp/evaluate_odometry \
  ~/code/docker_home/cvo/RKHS_BA/ground_truth/kitti/lidar/05.txt traj.kitti

# Sequence registration (multiple PCD files)
./build/gcvo/gcvo_run_pcd --pcd_list file_list.txt \
  --params gcvo_params/test.yaml --type intensity
```

## Architecture

GCVO is a GPU point-cloud registration library using kernelized inner-product maximization on SE(3). It runs Gauss-Newton optimization with a coarse-to-fine length-scale schedule.

### Header-Only Core + Explicit Instantiation

All CUDA kernels live in `gcvo/include/gcvo/impl/*.cuh` header files. These are compiled once per point type in tiny instantiation files under `gcvo/src/instantiations/`. This means:

- **Application code is CUDA-free** — `.cpp` files can include `GCvoGPU.hpp` and link a precompiled static library without needing NVCC.
- **Per-point-type static libraries** (`gcvo_s1`, `gcvo_s3`, `gcvo_s5`, `gcvo_s33`) — link only what you need. `gcvo` links all four.
- **`gcvo_headers`** — INTERFACE target for custom point types (user provides their own instantiation `.cu` file).

### Key Classes

- **`GCvoGPU<PointT>`** (`GCvoGPU.hpp`): Main solver. `align()` returns `GCvoResultInfo` with `T_s2t` (transform), iteration count, timing. Transform convention: `p_source = T_s2t * p_target`. `cos()` returns an overlap proxy (function-space angle between 0–1). `write_params()` updates parameters between calls without reconstructing the solver.
- **`GCvoPointCloudT<PointT>`** (`utils/GCvoPointCloud.hpp`): Point cloud container with PCL conversion constructors and GPU covariance computation. Call `compute_covariance()` before using DENSE or RESCALED kernel types.
- **`GCvoParams`** (`GCvoParams.hpp`): Runtime parameters loaded from YAML. Kernel shape, convergence, neighbor search, length-scale schedule.
- **`GCvoStateT<PointT>`** (`GCvoState.cuh`): Per-alignment GPU state (device arrays, sparse kernel matrix, GN work buffers).
- **`pcl::PointSemantic<FEATURE_DIM>`** (`utils/PointSemantic.hpp`): Custom PCL point type with compile-time feature vector size. Fields include `xyz`, `rgb`, `features[]`, `normal[]`, `covariance[]`, `label`.

### Point Type Aliases (PointTypes19.hpp)

| Alias | Feature Dim | Use Case |
|-------|------------|----------|
| `PointS1` | 1 | LiDAR intensity |
| `PointS3` | 3 | RGB |
| `PointS5` | 5 | RGB + gradients |
| `PointS33` | 33 | FPFH descriptor |

### Adding a New Point Type

1. Define an alias in a header (e.g., `using PointS10 = pcl::PointSemantic<10>;`)
2. Create `gcvo/src/instantiations/GCvoGPU_pointsem10.cu` that includes the impl headers and explicit-instantiates `GCvoGPU<PointS10>` and `GCvoPointCloudT<PointS10>::compute_covariance`
3. Add `gcvo_add_point_type(gcvo_s10 gcvo/src/instantiations/GCvoGPU_pointsem10.cu)` in CMakeLists.txt

### Kernel Types

Three kernel types (set via `kernel_type` in YAML):
- **SCALAR** (0): Isotropic, single length-scale
- **DENSE** (1): Isotropic + per-point covariance in Mahalanobis distance (requires `compute_covariance()`)
- **RESCALED** (2): Anisotropic, per-point covariance rescales kernel

### Key YAML Parameters

The most impactful parameters in YAML configs under `gcvo_params/`:
- `l_init` / `l_min`: Starting and minimum kernel length-scale for the coarse-to-fine schedule
- `l_decay_rate` / `l_decay_start`: Length-scale decay factor and iteration offset
- `max_neighbors`: KD-tree neighbors per source point (higher = denser kernel, slower)
- `sparsity_cutoff`: Kernel value threshold below which pairs are pruned
- `kernel_type`: 0=SCALAR, 1=DENSE, 2=RESCALED
- `use_geometry` / `use_features`: Toggle XYZ and feature-vector contributions to the kernel
- `tol` / `tol_2`: Convergence thresholds (inner-product change and gradient norm)
- `indicator_window` / `use_ema_indicator`: Stability detection before triggering length-scale decay
- `use_connection_term`: Add SE(3) Christoffel-symbol correction to curvature matrix (default 1)
- `use_symmetrization`: Symmetrize `B_gn` before LDLT (default 1; setting 0 is often better)
- `use_h1_term` / `use_h2_term` / `use_h3_term`: Include H1/H2/H3 exact-Hessian correction terms (default 0; H4 = GN term always active)
- `cov_eig_min` / `cov_eig_max`: Clamp per-point covariance eigenvalues (0 = disabled)
- `use_ell2_in_kernel`: Add ℓ²·I regularization in `cov_sum_inv_plus_l2I` (default 1; set 0 when eigenvalue clamping is active)
- `kernel_euclidean_max_dist`: Euclidean squared-distance pre-filter for DENSE/RESCALED kernel (default inf; e.g. 1.0 = 1 m cutoff)

### KITTI Evaluation Notes

- **Ground truth**: Use LiDAR-frame GT from `~/code/docker_home/cvo/RKHS_BA/ground_truth/kitti/lidar/` (not the camera-frame poses in the KITTI dataset itself).
- **Broken sequences**: Seqs 01, 04, 10 have GT frame issues — exclude from comparisons.
- **First-frame initialization**: With `l_init=0.1`, the first frame (identity init, ~0.5 m displacement) gets stuck in a local minimum. Use `--first_frame_l_init 1.0` to set a coarser scale for the first pair only; subsequent frames use warm-start and work fine with `l_init=0.1`.
- **Best baseline so far**: `use_symmetrization: 0` (no_sym) gives ~1–1.6% t_rel on working sequences.

### Third-Party Vendored Code

`gcvo/third_party/perl_registration/` contains GPU KD-tree (`cukdtree/`) and GPU point cloud (`cupointcloud/`) from the RKHS-BA project. Forward-declared in `CudaTypes.hpp`.

### Tests

Tests use `#define private public` to access internal methods, so they link `gcvo_headers` (not `gcvo_s1`) to avoid duplicate symbols. Test parameters are configurable via CMake cache variables (`GCVO_TEST_*`).

| Test | What it checks |
|---|---|
| `gcvo_test_gn` | One GN step increases inner product |
| `gcvo_test_bunny_centroid` | Full alignment on Stanford Bunny; SE(3) error < 0.01 |
| `gcvo_test_covariance` | GPU KNN covariance fields are finite and nonzero |
| `gcvo_test_hessian_terms` | H1/H2 GPU kernel matches analytical CPU formula; vanishes at r=0 |
| `gcvo_test_two_frame_kitti` | H4/H4+H2/H4+H1+H2/H4+H1+H2+H3 all return code=0 on KITTI seq05 (gated by `-DGCVO_TEST_KITTI_ROOT=...`) |

## Experimentation Workflow

Before making any experimental or exploratory code change:

1. **Checkout a new branch** — never experiment on `main`:
   ```bash
   git checkout -b experiment/<short-description>
   ```
2. **Implement** the change.
3. **Build:**
   ```bash
   cd build && make -j
   ```
4. **Run the test suite** — all tests must still pass:
   ```bash
   cd build && ctest --output-on-failure
   ```
5. **Run evaluation scripts** and record metrics (SE(3) error, iteration count, timing):
   ```bash
   bash scripts/run_bunny.sh

   # KITTI full-sequence eval on working sequences (skip 01, 04, 10):
   BIN=build/gcvo_kitti_f2f
   GT=~/code/docker_home/cvo/RKHS_BA/ground_truth/kitti/lidar
   EVAL=~/code/docker_home/cvo/odometry_eval/KITTI/cpp/evaluate_odometry
   for SEQ in 00 02 03 05 06 07 08 09; do
     $BIN --params gcvo_params/variant_<name>.yaml \
          --kitti_root /path/to/kitti/dataset --sequence $SEQ \
          --start 0 --count 999999 --voxel_mode fast \
          --first_frame_l_init 1.0 \
          --traj_file eval_<name>_${SEQ}.kitti
     $EVAL ${GT}/${SEQ}.txt eval_<name>_${SEQ}.kitti
   done
   ```
6. **Compare against baseline** (`no_sym` variant). If t_rel or r_rel improve across most sequences, keep the branch; otherwise discard it.
7. **Save results** to a dated experiment directory. Every experiment should produce
   a self-contained folder so the result can be reproduced later without digging through
   command history:
   ```bash
   DATE=$(date +%Y-%m-%d)
   DIR=results/${DATE}/<exp_name>
   mkdir -p ${DIR}/params

   # 1. Copy effective YAML(s) into params/
   cp gcvo_params/variant_<name>.yaml ${DIR}/params/best_config.yaml

   # 2. Write launch.sh with the full reproduction commands
   #    (binary path, KITTI root, all CLI flags, eval pipeline)
   cat > ${DIR}/launch.sh << 'EOF'
   ...
   EOF

   # 3. Snapshot the eval script
   cp scripts/eval_table.py ${DIR}/eval_table.py

   # 4. Save the evaluation table output (mean±std across sequences)
   python3 scripts/eval_table.py > ${DIR}/results_table.txt

   # 5. Write test.md with: per-sequence t_rel/r_rel, mean±std, config description,
   #    key findings, and file listing.
   ```
   See `results/2026-05-31/kitti_centroid_voxel_final/` for a template (test.md +
   launch.sh + eval_table.py + params/best_config.yaml + results_table.txt).

---

### Downstream Usage

`external_example/` shows how to consume the installed library via `find_package(gcvo)`. Application `.cpp` files only need to include `GCvoGPU.hpp` and link against `gcvo::gcvo_s1` (or whichever point-type library is needed) — no NVCC required in the consumer's build.
