cmake_minimum_required(VERSION 2.8)

project(KNN)
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
        message("CUDA found, using device squaring!")
        add_subdirectory(garcia)

else()
        message("CUDA not found, doing something alternatively")
endif()


