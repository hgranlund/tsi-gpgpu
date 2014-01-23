cmake_minimum_required(VERSION 2.8)

project(tsi-gpgpu)
# SET(CMAKE_C_COMPILER gcc)
# SET(CMAKE_CXX_COMPILER g++)
# set(CUDA_HOST_COMPILER gcc)
# set(CMAKE_MODULE_PATH "/usr/share/cmake/Modules" ${CMAKE_MODULE_PATH})
find_package(CUDA REQUIRED)

message(STATUS "CUDA_HOST_COMPILER ${CUDA_HOST_COMPILER}.")
message(STATUS "CMAKE_C_COMPILER ${CMAKE_C_COMPILER}.")
message(STATUS "CMAKE_CXX_COMPILER ${CMAKE_CXX_COMPILER}.")

SET( CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}" )
if (CUDA_FOUND)
        add_subdirectory(src/kNN)
else()
        message("CUDA not found, doing something alternatively")
endif()

## Set the directory where the binaries will be stored
set(EXECUTABLE_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/bin
  CACHE PATH
  "Directory where all executables will be stored"
  )

# Set the directory where the libraries will be stored
set(LIBRARY_OUTPUT_PATH
  ${PROJECT_BINARY_DIR}/lib
  CACHE PATH
  "Directory where all the libraries will be stored"
  )

set(CMAKE_MODULE_PATH
  ${CMAKE_SOURCE_DIR}/cmake
  )

set(tsi-gpgpu_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  )

include_directories(${gpgpu_INCLUDE_DIRS})
