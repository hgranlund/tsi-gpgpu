cmake_minimum_required(VERSION 2.8)

project(tsi_gpgpu)

find_package(CUDA REQUIRED)

option(BUILD_SHARED_LIBS
  "BUILD_SHARED_LIBS On to build shared libraries, off for static libraries."
  OFF
  )

option(CUDA_VERBOSE_PTXAS
  "On to enable verbose output from the PTXAS assembler."
  OFF
  )

option(BUILD_TESTS
  "Build all tests."
  ON
  )

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

set(tsi_gpgpu_INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  )

include_directories(${tsi_gpgpu_INCLUDE_DIRS})

if(BUILD_SHARED_LIBS)
  set(LIB_TYPE SHARED)
else()
  set(LIB_TYPE STATIC)
  set(STATIC_LIB 1)
endif(BUILD_SHARED_LIBS)

file(COPY tests/data DESTINATION tests)

if (CUDA_FOUND)
        add_subdirectory(src/kNN)
        add_subdirectory(tests)
else()
        message("CUDA not found, doing something else")
endif()

include(CPack)
