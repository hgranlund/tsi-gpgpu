cmake_minimum_required(VERSION 2.8)

project(tests)




################################
# Testing
################################
if (BUILD_TESTS)
  add_subdirectory(lib/gtest-1.6.0)
  enable_testing()

  # Include the gtest library. gtest_SOURCE_DIR is available due to
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

  # add_executable(runUnitTests kNN/test-kNN-brute-force.cpp)
  add_executable(runUnitTests kNN/test.cpp)

  # Standard linking to gtest stuff.
  target_link_libraries(runUnitTests gtest gtest_main)

  # Extra linking for the project.
  # target_link_libraries(runUnitTests knn_gpgpu)

  # add_test(run_test runUnitTests)
  # add_custom_target(test runUnitTests)
endif()
