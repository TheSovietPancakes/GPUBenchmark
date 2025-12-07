#include "../hip_backend.hpp"
#include <dlfcn.h>
#include <iostream>

bool HIPBackend::init() {
  hipHandle = dlopen("libamdhip64.so", RTLD_NOW);
  if (!hipHandle) {
    std::cerr << "Failed to load HIP runtime: " << dlerror() << "\n";
    shutdown();
    return false;
  }
  const char* rsmiLibNames[] = {"/opt/rocm/lib/librocm_smi64.so", "librocm_smi64.so",
                                "librocm_smi64.so.5",   // fedora
                                "librocm_smi64.so.1.0", // debian
                                "librocm_smi64.so.6"};
  for (const char* libName : rsmiLibNames) {
    rsmiHandle = dlopen(libName, RTLD_NOW);
    if (rsmiHandle)
      break;
  }
  if (!rsmiHandle) {
    std::cerr << "Failed to load RSMI: " << dlerror() << "\n";
    shutdown();
    return false;
  }

#define LOAD_HIP_SYMBOL(sym)                                                                                                                         \
  sym = (sym##_t)dlsym(hipHandle, #sym);                                                                                                             \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol for HIP so " #sym ": " << dlerror() << "\n";                                                                 \
    shutdown();                                                                                                                                      \
    return false;                                                                                                                                    \
  }

#define LOAD_RSMI_SYMBOL(sym)                                                                                                                        \
  sym = (sym##_t)dlsym(rsmiHandle, #sym);                                                                                                            \
  if (!sym) {                                                                                                                                        \
    std::cerr << "Failed to load symbol for RSMI so " #sym ": " << dlerror() << "\n";                                                                \
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
