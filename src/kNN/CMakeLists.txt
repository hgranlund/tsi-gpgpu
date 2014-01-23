

# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/garcia
  )

MACRO(ADD_APP APP FILE)
  CUDA_ADD_EXECUTABLE(${APP}  ${FILE})
  TARGET_LINK_LIBRARIES(${APP} -lcuda)
  # SET(all-bin "${all-bin} ${APP}")
ENDMACRO(ADD_APP)

ADD_APP(knn_cuda_with_indexes_mod  garcia/knn_cuda_with_indexes_mod.cu)
ADD_APP(knn_cuda_with_indexes  garcia/knn_cuda_with_indexes.cu)
ADD_APP(knn_cublas_with_indexes  garcia/knn_cublas_with_indexes.cu)
ADD_APP(knn_cublas_with_indexes_mod  garcia/knn_cublas_with_indexes_mod.cu)
CUDA_ADD_CUBLAS_TO_TARGET( knn_cublas_with_indexes )
CUDA_ADD_CUBLAS_TO_TARGET( knn_cublas_with_indexes_mod )

