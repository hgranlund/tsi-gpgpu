

# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcuda -lpthread)


set(CUDASRCS
        knn_cuda_with_indexes_imporved.cu
)

CUDA_ADD_EXECUTABLE(knn_cuda_with_indexes_imporved ${CUDASRCS})
target_link_libraries(knn_cuda_with_indexes_imporved -lcuda)

# Some untested, alternative statements of how to add libraries to an CUDA executable

# CUDA_ADD_LIBRARY(externals externalStuff.cu)
#
# target_link_libraries(gpuSquareDemo externals)
