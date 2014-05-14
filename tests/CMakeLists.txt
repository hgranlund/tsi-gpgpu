cmake_minimum_required(VERSION 2.8)

project(tests)

include_directories(
 ${CMAKE_SOURCE_DIR}/src/common
 ${CMAKE_SOURCE_DIR}/include
 ${CMAKE_SOURCE_DIR}/src/kNN
 ${CMAKE_SOURCE_DIR}/src/kNN/kernels
 ${CMAKE_SOURCE_DIR}/src/kNN/kd-tree
 ${CMAKE_SOURCE_DIR}/src/kNN/brute-force
 ${CMAKE_SOURCE_DIR}/tests/common
 ${CMAKE_SOURCE_DIR}/tests/common/inc
)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib/
)
set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)

set(GENCODE -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
# set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX

option(CUDPP_GENCODE_SM10
       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
       OFF)

option(CUDPP_GENCODE_SM13
       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
       OFF)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

if (CUDPP_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(CUDPP_GENCODE_SM10)

if (CUDPP_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(CUDPP_GENCODE_SM13)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)

################################
# Testing
################################
if (BUILD_TESTS)

  add_subdirectory(lib/gtest-1.7.0)
  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  set(common_files
    common/test-common.cuh
    common/test-common.cu
    common/test-common.h
  )

  set(test_files
    kNN/kernels/test_redution.cu
    kNN/kernels/test-multiple-radix-select.cu
    kNN/kernels/test-radix-select.cu
    kNN/kernels/test-quick-select.cu

    kNN/brute_force/test-knn-brute-force-reduce.cu
    kNN/brute_force/test-knn-brute-force-bitonic.cu

    kNN/kd-tree/test-kd-search.cu
    kNN/kd-tree/test-kd-search-openmp.cu
    kNN/kd-tree/test-kd-tree-build.cu
    kNN/kd-tree/test-search-rosetta.cu
  )

  cuda_add_executable(runUnitTests
    ${common_files}
    ${test_files}
  )

  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests knn_gpgpu)

  add_test(run_test runUnitTests)
  add_custom_target(test runUnitTests)

  cuda_add_executable(time_kd_tree_build
    kNN/kd-tree/time-kd-tree-build.cu
    OPTIONS ${GENCODE}  ${VERBOSE_PTXAS}
  )

  target_link_libraries(time_kd_tree_build knn_gpgpu)
  add_custom_target(kd_tree_build time_kd_tree_build)

  cuda_add_executable(time_kd_search
    kNN/kd-tree/time-kd-search.cu
    OPTIONS ${GENCODE}  ${VERBOSE_PTXAS}
  )

  target_link_libraries(time_kd_search knn_gpgpu)
  add_custom_target(kd-search time_kd_search)

  cuda_add_executable(time_kd_search_openmp
    kNN/kd-tree/time-kd-search-openmp.cu
    OPTIONS ${GENCODE}  ${VERBOSE_PTXAS}
  )

  target_link_libraries(time_kd_search_openmp knn_gpgpu)
  add_custom_target(kd_search_openmp time_kd_search_openmp)

  endif()
