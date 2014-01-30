# Append current NVCC flags by something, eg comput capability
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -lcuda)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/garcia
  )

set(CUHFILES
kNN-brute-force.cuh
bitonic_sort.cuh
)

set(CUFILES
kNN-brute-force.cu
bitonic_sort.cu
)

set(CFILES

)


set(HFILES_PUBLIC
  ../../include/knn_gpgpu.h
  )

source_group("CUDA Source Files" FILES ${CUFILES})
source_group("CUDA Header Files" FILES ${CUHFILES})

set(GENCODE_SM10 -gencode=arch=compute_10,code=sm_10 -gencode=arch=compute_10,code=compute_10)
set(GENCODE_SM13 -gencode=arch=compute_13,code=sm_13 -gencode=arch=compute_13,code=compute_13)
set(GENCODE_SM20 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_20,code=compute_20)
set(GENCODE_SM30 -gencode=arch=compute_30,code=sm_30 -gencode=arch=compute_30,code=compute_30)
set(GENCODE_SM35 -gencode=arch=compute_35,code=sm_35 -gencode=arch=compute_35,code=compute_35)

set(GENCODE -gencode=arch=compute_10,code=compute_10) # at least generate PTX

option(CUDPP_GENCODE_SM10
       "ON to generate code for Compute Capability 1.0 devices (e.g. Tesla C870)"
       OFF)

option(CUDPP_GENCODE_SM13
       "ON to generate code for Compute Capability 1.3 devices (e.g. Tesla C1060)"
       OFF)

option(CUDPP_GENCODE_SM20
       "ON to generate code for Compute Capability 2.0 devices (e.g. Tesla C2050)"
       ON)

option(CUDPP_GENCODE_SM30
       "ON to generate code for Compute Capability 3.0 devices (e.g. Tesla K10)"
       OFF)

option(CUDPP_GENCODE_SM35
       "ON to generate code for Compute Capability 3.5 devices (e.g. Tesla K20)"
       OFF)

if (CUDPP_GENCODE_SM10)
  set(GENCODE ${GENCODE} ${GENCODE_SM10})
endif(CUDPP_GENCODE_SM10)

if (CUDPP_GENCODE_SM13)
  set(GENCODE ${GENCODE} ${GENCODE_SM13})
endif(CUDPP_GENCODE_SM13)

if (CUDPP_GENCODE_SM20)
  set(GENCODE ${GENCODE} ${GENCODE_SM20})
endif(CUDPP_GENCODE_SM20)

if (CUDPP_GENCODE_SM30)
  set(GENCODE ${GENCODE} ${GENCODE_SM30})
endif(CUDPP_GENCODE_SM30)

if (CUDPP_GENCODE_SM35)
  set(GENCODE ${GENCODE} ${GENCODE_SM35})
endif(CUDPP_GENCODE_SM35)

if (CUDA_VERBOSE_PTXAS)
  set(VERBOSE_PTXAS --ptxas-options=-v)
endif (CUDA_VERBOSE_PTXAS)



cuda_add_library(knn_gpgpu ${LIB_TYPE}
  ${CUHFILES}
  ${HFILES_PUBLIC}
  ${CUFILES}
  ${CFILES}
  OPTIONS ${GENCODE} ${CUDA_NVCC_FLAGS} ${VERBOSE_PTXAS}
  )


install(FILES ${HFILES_PUBLIC}
  DESTINATION include
  )

install(TARGETS knn_gpgpu
  DESTINATION lib
  EXPORT knn_gpgpu-targets
  )

install(EXPORT knn-targets
  DESTINATION lib
  )

