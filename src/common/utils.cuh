#ifndef _CU_UTILS_
#define _CU_UTILS_

void cuSetDevice(int device);
int cuGetDevice();
int cuGetDeviceCount();
size_t getFreeBytesOnGpu();

#endif
