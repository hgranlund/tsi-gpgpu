cmake_minimum_required(VERSION 2.8)

project(tests)

include_directories(
 ${CMAKE_SOURCE_DIR}/src/common
 ${CMAKE_SOURCE_DIR}/tests/common/inc
 ${CMAKE_SOURCE_DIR}/include
 ${CMAKE_SOURCE_DIR}/src/kNN
 ${CMAKE_SOURCE_DIR}/src/kNN/kernels
 ${CMAKE_SOURCE_DIR}/src/kNN/kd-tree
 ${CMAKE_SOURCE_DIR}/src/kNN/brute-force
 # ${CMAKE_CURRENT_SOURCE_DIR}/common
)

link_directories(
  ${CMAKE_SOURCE_DIR}/lib/
)

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)


option(CUDPP_GENCODE_SM10
       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
       ON)

option(CUDPP_GENCODE_SM13
       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
       ON)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       ON)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       ON)

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

  set(test_files
    #kNN/test-kNN-brute-force.cpp
    kNN/kernels/test_redution.cu
    kNN/kernels/test-multiple-radix-select.cu
    kNN/kernels/test-radix-select.cu
    kNN/kernels/test-quick-select.cu
    kNN/brute_force/test-knn-brute-force-reduce.cu
    kNN/brute_force/test-knn-brute-force-bitonic.cu
    kNN/kd-tree/test-kd-tree-naive.cu
  )

  cuda_add_executable(runUnitTests
    ${test_files}
  )

  target_link_libraries(runUnitTests gtest gtest_main )
  # target_link_libraries(runUnitTests common_debug )

  target_link_libraries(runUnitTests knn_gpgpu)

  add_test(run_test runUnitTests)
  add_custom_target(test runUnitTests)


  cuda_add_executable(performance kNN/performance_tests/knn-brute-force-performance-test.cu)
  target_link_libraries(performance knn_gpgpu)
  add_custom_target(time performance)

  set(profile_files
    #kNN/test-kNN-brute-force.cpp
    # kNN/kernels/test_redution.cu
    # kNN/kernels/test-multiple-radix-select.cu
    # kNN/brute_force/test-knn-brute-force-reduce.cu
    # kNN/brute_force/test-knn-brute-force-bitonic.cu
    kNN/kd-tree/test-kd-tree-naive.cu
    # kNN/kernels/test-quick-select.cu

  )



  cuda_add_executable(nvproc_test
    ${profile_files}
    OPTIONS ${GENCODE}  ${VERBOSE_PTXAS}
  )

  target_link_libraries(nvproc_test gtest gtest_main )
  # target_link_libraries(runUnitTests common_debug )

  target_link_libraries(nvproc_test knn_gpgpu)

  add_custom_target(profile_test nvproc_test)

  endif()
