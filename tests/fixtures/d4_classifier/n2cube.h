#pragma once
#include <cstdint>
struct DPUKernel {};
struct DPUTask {};
const int N2CUBE_EXCEPTION_MODE_RET_ERR_CODE = 1;
const int T_MODE_NORMAL = 0;
void dpuSetExceptionMode(int);
int dpuOpen();
int dpuClose();
DPUKernel* dpuLoadKernel(const char*);
int dpuDestroyKernel(DPUKernel*);
DPUTask* dpuCreateTask(DPUKernel*, int);
int dpuDestroyTask(DPUTask*);
int dpuGetInputTensorSize(DPUTask*, const char*, int = 0);
int8_t* dpuGetInputTensorAddress(DPUTask*, const char*, int = 0);
float dpuGetInputTensorScale(DPUTask*, const char*, int = 0);
float dpuGetOutputTensorScale(DPUTask*, const char*, int = 0);
int dpuRunTask(DPUTask*);
int dpuGetOutputTensorSize(DPUTask*, const char*, int = 0);
int8_t* dpuGetOutputTensorAddress(DPUTask*, const char*, int = 0);
