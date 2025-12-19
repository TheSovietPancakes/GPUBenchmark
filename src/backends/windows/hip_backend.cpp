#include "../hip_backend.hpp"
#include <iostream>
#include <windows.h>

bool HIPBackend::init() {
  const char* hipLibNames[] = {"hipamd64.dll", "hiprtc.dll", "hiprtc64.dll"};
  for (const char* libName : hipLibNames) {
    hipHandle = LoadLibraryA(libName);
    if (hipHandle)
      break;
  }
  if (!hipHandle) {
    std::cerr << "Failed to load HIP DLL.\n";
    shutdown();
    return false;
  }

  const char* rsmiLibNames[] = {"rocm_smi64.dll", "rocm_smi.dll"};
  for (const char* libName : rsmiLibNames) {
    rsmiHandle = LoadLibraryA(libName);
    if (rsmiHandle)
      break;
  }
  if (!rsmiHandle) {
    std::cerr << "Failed to load RSMI DLL.\n";
    shutdown();
    return false;
  }

#define LOAD_HIP_SYMBOL(sym)                                                                                                                         \
  sym = (sym##_t)GetProcAddress(static_cast<HMODULE>(hipHandle), #sym);                                                                              \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol " #sym " for HIP.dll.\n";                                                                                    \
    shutdown();                                                                                                                                      \
    return false;                                                                                                                                    \
  }

#define LOAD_RSMI_SYMBOL(sym)                                                                                                                        \
  sym = (sym##_t)GetProcAddress(static_cast<HMODULE>(rsmiHandle), #sym);                                                                             \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol " #sym " for RSMI.dll.\n";                                                                                   \
    shutdown();                                                                                                                                      \
    return false;                                                                                                                                    \
  }

  LOAD_HIP_SYMBOL(hipInit)
  LOAD_HIP_SYMBOL(hipDeviceReset)
  LOAD_HIP_SYMBOL(hipSetDevice)
  LOAD_HIP_SYMBOL(hipGetDeviceCount)
  LOAD_HIP_SYMBOL(hipGetDevice)
  LOAD_HIP_SYMBOL(hipGetDeviceProperties)
  LOAD_HIP_SYMBOL(hipMalloc)
  LOAD_HIP_SYMBOL(hipHostMalloc)
  LOAD_HIP_SYMBOL(hipHostFree)
  LOAD_HIP_SYMBOL(hipFree)
  LOAD_HIP_SYMBOL(hipMemcpy)
  LOAD_HIP_SYMBOL(hipMemset)
  LOAD_HIP_SYMBOL(hipEventCreate)
  LOAD_HIP_SYMBOL(hipEventDestroy)
  LOAD_HIP_SYMBOL(hipEventRecord)
  LOAD_HIP_SYMBOL(hipEventSynchronize)
  LOAD_HIP_SYMBOL(hipEventElapsedTime)
  LOAD_HIP_SYMBOL(hipStreamCreate)
  LOAD_HIP_SYMBOL(hipStreamDestroy)
  LOAD_HIP_SYMBOL(hipModuleLoadData)
  LOAD_HIP_SYMBOL(hipModuleUnload)
  LOAD_HIP_SYMBOL(hipModuleGetFunction)
  LOAD_HIP_SYMBOL(hipModuleLaunchKernel)
  LOAD_HIP_SYMBOL(hipGetErrorString)

  LOAD_RSMI_SYMBOL(rsmi_init)
  LOAD_RSMI_SYMBOL(rsmi_shut_down)
  LOAD_RSMI_SYMBOL(rsmi_dev_temp_metric_get)
  LOAD_RSMI_SYMBOL(rsmi_dev_memory_total_get)
  LOAD_RSMI_SYMBOL(rsmi_dev_memory_usage_get)
  LOAD_RSMI_SYMBOL(rsmi_dev_name_get)
  LOAD_RSMI_SYMBOL(rsmi_dev_busy_percent_get)

  if (hipInit(0) != hipSuccess) {
    std::cerr << "Failed to initialize HIP runtime.\n";
    shutdown();
    return false;
  }
  if (rsmi_init(0) != 0) {
    std::cerr << "Failed to initialize RSMI.\n";
    shutdown();
    return false;
  }
  return true;
}
