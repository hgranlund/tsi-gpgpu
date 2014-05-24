#ifndef _CU_UTILS_
#define _CU_UTILS_

void cuSetDevice(int devive);
int cuGetDevice();
int cuGetDeviceCount();
size_t getFreeBytesOnGpu();

#endif
