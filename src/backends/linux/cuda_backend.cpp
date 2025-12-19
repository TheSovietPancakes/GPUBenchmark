#include "../cuda_backend.hpp"
#include <dlfcn.h>
#include <iostream>

bool CudaBackend::init() {
  cudaHandle = dlopen("libcuda.so", RTLD_NOW);
  if (!cudaHandle) {
    std::cerr << "Failed to load libcuda.so: " << dlerror() << "\n";
    shutdown();
    return false;
  }

  nvmlHandle = dlopen("libnvidia-ml.so", RTLD_NOW);
  if (!nvmlHandle) {
    std::cerr << "Failed to load NVML: " << dlerror() << "\n";
    shutdown();
    return false;
  }

#define LOAD_CUDA_SYMBOL(sym)                                                                                                                        \
  sym = (sym##_t)dlsym(cudaHandle, #sym);                                                                                                            \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol for CUDA so " #sym ": " << dlerror() << "\n";                                                                \
    dlclose(cudaHandle);                                                                                                                             \
    cudaHandle = nullptr;                                                                                                                            \
    return false;                                                                                                                                    \
  }

#define LOAD_NVML_SYMBOL(sym)                                                                                                                        \
  sym = (sym##_t)dlsym(nvmlHandle, #sym);                                                                                                            \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol for NVML so " #sym ": " << dlerror() << "\n";                                                                \
    dlclose(nvmlHandle);                                                                                                                             \
    nvmlHandle = nullptr;                                                                                                                            \
    return false;                                                                                                                                    \
  }

  // Load all required CUDA Driver API functions
  LOAD_CUDA_SYMBOL(cuInit);
  LOAD_CUDA_SYMBOL(cuMemAlloc);
  LOAD_CUDA_SYMBOL(cuMemFree);
  LOAD_CUDA_SYMBOL(cuMemAllocHost);
  LOAD_CUDA_SYMBOL(cuMemFreeHost);
  LOAD_CUDA_SYMBOL(cuDeviceGetCount);
  LOAD_CUDA_SYMBOL(cuDeviceGet);
  LOAD_CUDA_SYMBOL(cuDeviceGetName);
  LOAD_CUDA_SYMBOL(cuCtxCreate);
  LOAD_CUDA_SYMBOL(cuCtxDestroy);
  LOAD_CUDA_SYMBOL(cuCtxSynchronize);
  LOAD_CUDA_SYMBOL(cuModuleLoadData);
  LOAD_CUDA_SYMBOL(cuModuleUnload);
  LOAD_CUDA_SYMBOL(cuModuleGetFunction);
  LOAD_CUDA_SYMBOL(cuLaunchKernel);
  LOAD_CUDA_SYMBOL(cuMemcpyHtoD);
  LOAD_CUDA_SYMBOL(cuMemcpyDtoH);
  LOAD_CUDA_SYMBOL(cuMemsetD8);
  LOAD_CUDA_SYMBOL(cuEventCreate);
  LOAD_CUDA_SYMBOL(cuEventRecord);
  LOAD_CUDA_SYMBOL(cuEventSynchronize);
  LOAD_CUDA_SYMBOL(cuEventElapsedTime);
  LOAD_CUDA_SYMBOL(cuEventDestroy);
  LOAD_CUDA_SYMBOL(cuStreamCreate);
  LOAD_CUDA_SYMBOL(cuStreamDestroy);
  LOAD_CUDA_SYMBOL(cuDeviceTotalMem);
  LOAD_CUDA_SYMBOL(cuDeviceComputeCapability);
  LOAD_CUDA_SYMBOL(cuDeviceGetAttribute);

  LOAD_CUDA_SYMBOL(cuGetErrorString);

  // Load NVML functions
  LOAD_NVML_SYMBOL(nvmlInit);
  LOAD_NVML_SYMBOL(nvmlShutdown);
  LOAD_NVML_SYMBOL(nvmlDeviceGetHandleByIndex);
  LOAD_NVML_SYMBOL(nvmlDeviceGetUtilizationRates);
  LOAD_NVML_SYMBOL(nvmlDeviceGetTemperature);
  LOAD_NVML_SYMBOL(nvmlDeviceGetMemoryInfo);

#undef LOAD_CUDA_SYMBOL
#undef LOAD_NVML_SYMBOL

  if (cuInit(0) != 0) {
    std::cerr << "Failed to initialize CUDA Driver API.\n";
    shutdown();
    return false;
  }
  if (nvmlInit() != 0) {
    std::cerr << "Failed to initialize NVML.\n";
    shutdown();
    return false;
  }
  return true;
}
