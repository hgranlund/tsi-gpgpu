

# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} )

MACRO(ADD_APP APP)
  CUDA_ADD_EXECUTABLE(${APP} ${APP}.cu)
  TARGET_LINK_LIBRARIES(${APP} -lcuda)
  SET(all-bin "${all-bin} ${APP}")
ENDMACRO(ADD_APP)

ADD_APP(knn_cuda_with_indexes_mod)
# ADD_APP(knn_cuda_with_indexes)
# ADD_APP(knn_cublas_with_indexes)
# ADD_APP(knn_cublas_with_indexes_mod)


# Some untested, alternative statements of how to add libraries to an CUDA executable

# CUDA_ADD_LIBRARY(externals externalStuff.cu)
#
# target_link_libraries(gpuSquareDemo externals)
