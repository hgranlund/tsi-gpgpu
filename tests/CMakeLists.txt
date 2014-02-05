cmake_minimum_required(VERSION 2.8)

project(tests)

include_directories(
 ${CMAKE_SOURCE_DIR}/src/kNN
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
endif()
