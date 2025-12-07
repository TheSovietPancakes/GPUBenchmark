

#include "../opencl_backend.hpp"
#include <dlfcn.h>
#include <iostream>

bool CLBackend::init() {
  const char* clLibNames[] = {"OpenCL", "libOpenCL.dylib"};
  for (const char* libName : clLibNames) {
    clHandle = dlopen(libName, RTLD_NOW);
    if (clHandle)
      break;
  }
  if (!clHandle) {
    std::cerr << "Failed to load HIP runtime: " << dlerror() << "\n";
    shutdown();
    return false;
  }

#define LOAD_CL_SYMBOL(sym)                                                                                                                          \
  sym = (sym##_t)dlsym(clHandle, #sym);                                                                                                              \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol for OpenCL so " #sym ": " << dlerror() << "\n";                                                              \
    dlclose(clHandle);                                                                                                                               \
    clHandle = nullptr;                                                                                                                              \
    return false;                                                                                                                                    \
  }

  // Load all required OpenCL Driver API functions
  LOAD_CL_SYMBOL(clGetDeviceInfo);
  LOAD_CL_SYMBOL(clGetPlatformIDs);
  LOAD_CL_SYMBOL(clGetDeviceIDs);
  LOAD_CL_SYMBOL(clCreateContext);
  LOAD_CL_SYMBOL(clCreateCommandQueueWithProperties);
  LOAD_CL_SYMBOL(clCreateProgramWithSource);
  LOAD_CL_SYMBOL(clBuildProgram);
  LOAD_CL_SYMBOL(clCreateKernel);
  LOAD_CL_SYMBOL(clEnqueueNDRangeKernel);
  LOAD_CL_SYMBOL(clWaitForEvents);
  LOAD_CL_SYMBOL(clGetEventProfilingInfo);
  LOAD_CL_SYMBOL(clReleaseEvent);
  LOAD_CL_SYMBOL(clReleaseCommandQueue);
  LOAD_CL_SYMBOL(clReleaseContext);
  LOAD_CL_SYMBOL(clReleaseProgram);
  LOAD_CL_SYMBOL(clReleaseKernel);
  LOAD_CL_SYMBOL(clCreateBuffer);
  LOAD_CL_SYMBOL(clGetPlatformInfo);
  LOAD_CL_SYMBOL(clEnqueueReadBuffer);
  LOAD_CL_SYMBOL(clReleaseMemObject);
  LOAD_CL_SYMBOL(clSetKernelArg);

#undef LOAD_CL_SYMBOL

  return true;
}
