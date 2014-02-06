cmake_minimum_required(VERSION 2.8)

project(tests)

include_directories(
 ${CMAKE_SOURCE_DIR}/src/kNN
 ${CMAKE_SOURCE_DIR}/src/kNN/brute-force
)

################################
# Testing
################################
if (BUILD_TESTS)
  add_subdirectory(lib/gtest-1.6.0)
  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  cuda_add_executable(runUnitTests kNN/test-kNN-brute-force.cpp kNN/brute_force/test_knn_brute_force.cu)
  # add_executable(runUnitTests kNN/test.cpp)

  target_link_libraries(runUnitTests gtest gtest_main)

  target_link_libraries(runUnitTests knn_gpgpu -lcuda)

  add_test(run_test runUnitTests)
  add_custom_target(test runUnitTests)


  cuda_add_executable(performance kNN/performance_tests/knn-brute-force-performance-test.cu)
  target_link_libraries(performance knn_gpgpu -lcuda)
  add_custom_target(time performance)

endif()
