

# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/garcia
  )

CUDA_ADD_EXECUTABLE(knn_cuda_with_indexes_mod  garcia/knn_cuda_with_indexes_mod.cu)
TARGET_LINK_LIBRARIES(knn_cuda_with_indexes_mod -lcuda)

# ADD_APP(knn_cuda_with_indexes)
# ADD_APP(knn_cublas_with_indexes)
# ADD_APP(knn_cublas_with_indexes_mod)
