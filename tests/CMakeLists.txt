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
    kNN/kernels/test-radix-select.cu
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
    # kNN/kernels/test-radix-select.cu
    # kNN/brute_force/test-knn-brute-force-reduce.cu
    # kNN/brute_force/test-knn-brute-force-bitonic.cu
    kNN/kd-tree/test-kd-tree-naive.cu
  )

  cuda_add_executable(nvproc_test
    ${profile_files}
  )

  target_link_libraries(nvproc_test gtest gtest_main )
  # target_link_libraries(runUnitTests common_debug )

  target_link_libraries(nvproc_test knn_gpgpu)

  add_custom_target(profile_test nvproc_test)

  endif()
