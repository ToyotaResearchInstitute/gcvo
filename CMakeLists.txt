# GCVO PROVENANCE NOTICE (derived from RKHS_BA, MIT license)
# Portions of this file are derived from RKHS_BA (MIT): https://github.com/UMich-CURLY/RKHS_BA
# Upstream is included in this repo as a git submodule at: rkhs_ba/
# Upstream file: rkhs_ba/CMakeLists.txt (no 1:1 mapping claimed)
#
# This GCVO repository is NOT a verbatim copy of RKHS_BA. It includes substantial modifications,
# refactors, and additional original content (e.g., solver/optimization changes and new utilities).
#
# References:
# - RKHS_BA paper: R. Zhang et al., IEEE TPAMI 2025, doi: 10.1109/TPAMI.2025.3593521
# - GCVO paper: R. Zhang et al., CVPR 2026 (see repo README for details)
#
# License: RKHS_BA is MIT-licensed (see rkhs_ba/ for the upstream LICENSE). This repo’s license is in
# the root LICENSE file. Contact (GCVO modifications): ray.zhang@tri.global

cmake_minimum_required(VERSION 3.18)

project(gcvo LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

option(GCVO_BUILD_APPS "Build example runners" ON)
option(GCVO_BUILD_TESTS "Build unit tests" ON)
option(GCVO_BUILD_VIZ  "Enable PCL visualizer support (requires PCL visualization)" OFF)
option(GCVO_TEST_SAVE_PCD "Enable --save_pcd in tests to write alignment PCD files" OFF)

# Threads-per-block used by GCVO CUDA kernels (compile-time constant)
set(GCVO_CUDA_THREADS 256 CACHE STRING "Threads per CUDA block for GCVO kernels")

# ---- Dependencies ----
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io filters)
find_package(CUDAToolkit REQUIRED)

if (GCVO_BUILD_VIZ)
  find_package(PCL REQUIRED COMPONENTS visualization)
endif()

# CUDA point cloud / KD-tree headers are vendored under gcvo/third_party/perl_registration
set(GCVO_PERL_REG_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/gcvo/third_party/perl_registration)

# ---- Header-only interface (includes + compile defs + transitive deps) ----
add_library(gcvo_headers INTERFACE)
target_include_directories(gcvo_headers INTERFACE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/gcvo/include>
  $<BUILD_INTERFACE:${GCVO_PERL_REG_INCLUDE_DIR}>
  $<INSTALL_INTERFACE:include>
  $<INSTALL_INTERFACE:include/gcvo/third_party>
  ${PCL_INCLUDE_DIRS}
)
target_link_libraries(gcvo_headers INTERFACE
  Eigen3::Eigen
  yaml-cpp
  CUDA::cudart
  ${PCL_LIBRARIES}
)
target_compile_definitions(gcvo_headers INTERFACE
  PCL_NO_PRECOMPILE
  ${PCL_DEFINITIONS}
  GCVO_CUDA_THREADS=${GCVO_CUDA_THREADS}
)

# ---- Per-point-type static libraries ----
# Each library compiles exactly one instantiation .cu file.
# Usage: link your app against the type(s) it needs (e.g. gcvo_s1 for PointS1).

function(gcvo_add_point_type TARGET_NAME SOURCE_FILE)
  add_library(${TARGET_NAME} STATIC ${SOURCE_FILE})
  target_link_libraries(${TARGET_NAME} PUBLIC gcvo_headers)
  target_compile_options(${TARGET_NAME} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
endfunction()

gcvo_add_point_type(gcvo_s1  gcvo/src/instantiations/GCvoGPU_pointsem1.cu)
gcvo_add_point_type(gcvo_s3  gcvo/src/instantiations/GCvoGPU_pointsem3.cu)
gcvo_add_point_type(gcvo_s5  gcvo/src/instantiations/GCvoGPU_pointsem5.cu)
gcvo_add_point_type(gcvo_s33 gcvo/src/instantiations/GCvoGPU_pointsem33.cu)

# Convenience target: links all point types.
add_library(gcvo INTERFACE)
target_link_libraries(gcvo INTERFACE gcvo_s1 gcvo_s3 gcvo_s5 gcvo_s33)

# ---- Optional visualizer library (PIMPL, compiled as plain C++) ----
if (GCVO_BUILD_VIZ)
  add_library(gcvo_viz STATIC gcvo/src/MapViewer.cpp)
  target_include_directories(gcvo_viz PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/gcvo/include
    ${PCL_INCLUDE_DIRS}
  )
  target_link_libraries(gcvo_viz PUBLIC Eigen3::Eigen ${PCL_LIBRARIES})
  target_compile_definitions(gcvo_viz PUBLIC GCVO_USE_VIZ)
endif()

# ---- Install rules ----
install(DIRECTORY gcvo/include/gcvo/ DESTINATION include/gcvo
  PATTERN "*~" EXCLUDE
  PATTERN "*.backup" EXCLUDE
)
install(DIRECTORY gcvo/third_party/perl_registration/ DESTINATION include/gcvo/third_party)

install(TARGETS gcvo_headers gcvo_s1 gcvo_s3 gcvo_s5 gcvo_s33 gcvo
  EXPORT gcvo-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
)

install(EXPORT gcvo-targets
  FILE gcvo-targets.cmake
  NAMESPACE gcvo::
  DESTINATION lib/cmake/gcvo
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  cmake/gcvo-config.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/gcvo-config.cmake"
  INSTALL_DESTINATION lib/cmake/gcvo
)

write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/gcvo-config-version.cmake"
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/gcvo-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/gcvo-config-version.cmake"
  DESTINATION lib/cmake/gcvo
)

# ---- Apps ----
if (GCVO_BUILD_APPS)
  # gcvo_align_pcd: dispatches to all 4 types at runtime via --type flag
  add_executable(gcvo_align_pcd gcvo/apps/gcvo_align_pcd.cpp)
  target_link_libraries(gcvo_align_pcd PRIVATE gcvo)

  # gcvo_kitti_f2f: PointS1 only (LiDAR intensity)
  add_executable(gcvo_kitti_f2f gcvo/apps/gcvo_kitti_f2f.cpp)
  target_link_libraries(gcvo_kitti_f2f PRIVATE gcvo_s1)

  # gcvo_run_pcd: dispatches to PointS1/S3/S33 via --type flag
  add_executable(gcvo_run_pcd gcvo/apps/gcvo_run_pcd.cu)
  target_link_libraries(gcvo_run_pcd PRIVATE gcvo_s1 gcvo_s3 gcvo_s33)

  if (GCVO_BUILD_VIZ)
    target_link_libraries(gcvo_kitti_f2f PRIVATE gcvo_viz)
    target_link_libraries(gcvo_run_pcd   PRIVATE gcvo_viz)
  endif()
endif()

# ---- Tests ----
if (GCVO_BUILD_TESTS)
  enable_testing()

  # Default arguments (override with -DGCVO_TEST_* on the cmake command line)
  set(GCVO_TEST_PARAMS           "${CMAKE_CURRENT_SOURCE_DIR}/gcvo_params/test.yaml" CACHE FILEPATH "Params YAML used by all tests")
  set(GCVO_TEST_PCD              "${CMAKE_CURRENT_SOURCE_DIR}/demo_data/bunny.pcd"  CACHE FILEPATH "Input PCD used by all tests")
  set(GCVO_TEST_N                2000   CACHE STRING "Number of points for all tests")
  # test_gn: single-step IP sanity check — small rotation, no translation
  set(GCVO_TEST_GN_THETA         30     CACHE STRING "Max rotation (deg) for test_gn")
  set(GCVO_TEST_GN_T             0.0    CACHE STRING "Translation magnitude for test_gn")
  # test_bunny: full alignment + strict IP check — keep rotation moderate
  set(GCVO_TEST_BUNNY_THETA      30     CACHE STRING "Max rotation (deg) for test_bunny")
  set(GCVO_TEST_BUNNY_T          0.5    CACHE STRING "Translation magnitude for test_bunny")
  # test_bunny_centroid: full alignment + SE3 error check — tolerates larger motion
  set(GCVO_TEST_CENTROID_THETA   30     CACHE STRING "Max rotation (deg) for test_bunny_centroid")
  set(GCVO_TEST_CENTROID_T       0.5    CACHE STRING "Translation magnitude for test_bunny_centroid")
  set(GCVO_TEST_PIVOT            origin CACHE STRING "Pivot mode for test_bunny_centroid (origin|centroid)")

  # NOTE: test_gn and test_bunny_centroid include GCvoGPU_impl.cuh directly
  # (the #define private public hack), so they compile their own GCvoGPU<PointS1>
  # instantiation.  They link gcvo_headers (not gcvo_s1) to avoid duplicate symbols.

  add_executable(gcvo_test_gn gcvo/tests/test_gn_math.cu)
  target_link_libraries(gcvo_test_gn PRIVATE gcvo_headers)
  target_compile_options(gcvo_test_gn PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  add_test(
    NAME gcvo_test_gn
    COMMAND gcvo_test_gn
      --params         ${GCVO_TEST_PARAMS}
      --input_pcd_file ${GCVO_TEST_PCD}
      --n              ${GCVO_TEST_N}
      --theta_deg      ${GCVO_TEST_GN_THETA}
      --t              ${GCVO_TEST_GN_T}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  #add_executable(gcvo_test_bunny gcvo/tests/test_align_bunny.cu)
  #target_link_libraries(gcvo_test_bunny PRIVATE gcvo_headers)
  #target_compile_options(gcvo_test_bunny PRIVATE
  #  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  #)
  #add_test(
  #  NAME gcvo_test_bunny
  #  COMMAND gcvo_test_bunny
  #    --params         ${GCVO_TEST_PARAMS}
  #    --input_pcd_file ${GCVO_TEST_PCD}
  #    --n              ${GCVO_TEST_N}
  #    --theta_deg      ${GCVO_TEST_BUNNY_THETA}
  #    --t              ${GCVO_TEST_BUNNY_T}
  #  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  #)

  add_executable(gcvo_test_bunny_centroid gcvo/tests/test_align_centroid.cu)
  target_link_libraries(gcvo_test_bunny_centroid PRIVATE gcvo_headers)
  target_compile_options(gcvo_test_bunny_centroid PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  if (GCVO_TEST_SAVE_PCD)
    target_compile_definitions(gcvo_test_bunny_centroid PRIVATE GCVO_TEST_SAVE_PCD)
  endif()
  add_test(
    NAME gcvo_test_bunny_centroid
    COMMAND gcvo_test_bunny_centroid
      --pivot          ${GCVO_TEST_PIVOT}
      --params         ${GCVO_TEST_PARAMS}
      --input_pcd_file ${GCVO_TEST_PCD}
      --n              ${GCVO_TEST_N}
      --theta_deg      ${GCVO_TEST_CENTROID_THETA}
      --t              ${GCVO_TEST_CENTROID_T}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # Covariance / KD-tree test: loads bunny.pcd, computes covariance, checks output
  add_executable(gcvo_test_covariance gcvo/tests/test_covariance.cu)
  target_link_libraries(gcvo_test_covariance PRIVATE gcvo_s1)
  add_test(
    NAME gcvo_test_covariance
    COMMAND gcvo_test_covariance
      --input_pcd_file ${GCVO_TEST_PCD}
      --n              ${GCVO_TEST_N}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # Hessian term unit tests (H1, H2, H3 analytical vs GPU kernel).
  # Links gcvo_headers to avoid duplicate symbols (same pattern as test_gn).
  add_executable(gcvo_test_hessian_terms gcvo/tests/test_hessian_terms.cu)
  target_link_libraries(gcvo_test_hessian_terms PRIVATE gcvo_headers)
  target_compile_options(gcvo_test_hessian_terms PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
  )
  add_test(
    NAME gcvo_test_hessian_terms
    COMMAND gcvo_test_hessian_terms --params ${GCVO_TEST_PARAMS}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  )

  # 2-frame KITTI test — only registered when GCVO_TEST_KITTI_ROOT is set.
  set(GCVO_TEST_KITTI_ROOT "" CACHE PATH "KITTI odometry dataset root (enables 2-frame test)")
  if (GCVO_TEST_KITTI_ROOT)
    add_executable(gcvo_test_two_frame_kitti gcvo/tests/test_two_frame_kitti.cu)
    target_link_libraries(gcvo_test_two_frame_kitti PRIVATE gcvo_s1)
    target_compile_definitions(gcvo_test_two_frame_kitti PRIVATE
      GCVO_TEST_KITTI_ROOT="${GCVO_TEST_KITTI_ROOT}")
    add_test(
      NAME gcvo_test_two_frame_kitti
      COMMAND gcvo_test_two_frame_kitti
      WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    add_executable(gcvo_test_pitch_diagnostic gcvo/tests/test_pitch_diagnostic.cu)
    target_link_libraries(gcvo_test_pitch_diagnostic PRIVATE gcvo_s1)
    target_compile_definitions(gcvo_test_pitch_diagnostic PRIVATE
      GCVO_TEST_KITTI_ROOT="${GCVO_TEST_KITTI_ROOT}")
    # No add_test — this is a manual diagnostic.

    add_executable(gcvo_test_symmetry_real gcvo/tests/test_symmetry_real.cu)
    target_link_libraries(gcvo_test_symmetry_real PRIVATE gcvo_s1)
    target_compile_definitions(gcvo_test_symmetry_real PRIVATE
      GCVO_TEST_KITTI_ROOT="${GCVO_TEST_KITTI_ROOT}")
  endif()

endif()
