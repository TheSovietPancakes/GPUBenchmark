#include "cuda_backend.hpp"
#include "../shared/shared.hpp"
#include "modules/cuda_kernels.hpp"
#include <algorithm>
#include <array>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
// #include "/opt/cuda/include/cuda_runtime.h"

CudaBackend::cuInit_t CudaBackend::cuInit = nullptr;
CudaBackend::cuMemAlloc_t CudaBackend::cuMemAlloc = nullptr;
CudaBackend::cuMemFree_t CudaBackend::cuMemFree = nullptr;
CudaBackend::cuDeviceGetCount_t CudaBackend::cuDeviceGetCount = nullptr;
CudaBackend::cuDeviceGet_t CudaBackend::cuDeviceGet = nullptr;
CudaBackend::cuDeviceGetName_t CudaBackend::cuDeviceGetName = nullptr;
CudaBackend::cuCtxCreate_t CudaBackend::cuCtxCreate = nullptr;
CudaBackend::cuCtxDestroy_t CudaBackend::cuCtxDestroy = nullptr;
CudaBackend::cuCtxSynchronize_t CudaBackend::cuCtxSynchronize = nullptr;
CudaBackend::cuModuleLoadData_t CudaBackend::cuModuleLoadData = nullptr;
CudaBackend::cuModuleUnload_t CudaBackend::cuModuleUnload = nullptr;
CudaBackend::cuModuleGetFunction_t CudaBackend::cuModuleGetFunction = nullptr;
CudaBackend::cuLaunchKernel_t CudaBackend::cuLaunchKernel = nullptr;
CudaBackend::cuMemcpyHtoD_t CudaBackend::cuMemcpyHtoD = nullptr;
CudaBackend::cuMemcpyDtoH_t CudaBackend::cuMemcpyDtoH = nullptr;
CudaBackend::cuMemsetD8_t CudaBackend::cuMemsetD8 = nullptr;
CudaBackend::cuEventCreate_t CudaBackend::cuEventCreate = nullptr;
CudaBackend::cuEventRecord_t CudaBackend::cuEventRecord = nullptr;
CudaBackend::cuEventSynchronize_t CudaBackend::cuEventSynchronize = nullptr;
CudaBackend::cuEventElapsedTime_t CudaBackend::cuEventElapsedTime = nullptr;
CudaBackend::cuEventDestroy_t CudaBackend::cuEventDestroy = nullptr;
CudaBackend::cuStreamCreate_t CudaBackend::cuStreamCreate = nullptr;
CudaBackend::cuStreamDestroy_t CudaBackend::cuStreamDestroy = nullptr;
CudaBackend::cuDeviceTotalMem_t CudaBackend::cuDeviceTotalMem = nullptr;
CudaBackend::cuDeviceComputeCapability_t CudaBackend::cuDeviceComputeCapability = nullptr;
CudaBackend::cuDeviceGetAttribute_t CudaBackend::cuDeviceGetAttribute = nullptr;

CudaBackend::cuGetErrorString_t CudaBackend::cuGetErrorString = nullptr;

// NVML
CudaBackend::nvmlInit_t CudaBackend::nvmlInit = nullptr;
CudaBackend::nvmlShutdown_t CudaBackend::nvmlShutdown = nullptr;
CudaBackend::nvmlDeviceGetHandleByIndex_t CudaBackend::nvmlDeviceGetHandleByIndex = nullptr;
CudaBackend::nvmlDeviceGetUtilizationRates_t CudaBackend::nvmlDeviceGetUtilizationRates = nullptr;
CudaBackend::nvmlDeviceGetTemperature_t CudaBackend::nvmlDeviceGetTemperature = nullptr;
CudaBackend::nvmlDeviceGetMemoryInfo_t CudaBackend::nvmlDeviceGetMemoryInfo = nullptr;

// ------------------------
// Error checking macro
// ------------------------
#define CUDA_ERR(call)                                                                                                                               \
  do {                                                                                                                                               \
    CUresult err = call;                                                                                                                             \
    if (err != 0) {                                                                                                                                  \
      const char* msg;                                                                                                                               \
      cuGetErrorString(err, &msg);                                                                                                                   \
      std::cerr << "CUDA Driver API error: " << msg << " (" << err << ")\n";                                                                         \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

#define NVML_ERR(call)                                                                                                                               \
  do {                                                                                                                                               \
    CudaBackend::nvmlReturn_t err = call;                                                                                                            \
    if (err != 0) {                                                                                                                                  \
      std::cerr << "NVML error: " << err << "\n";                                                                                                    \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

bool getDeviceProperties(CudaBackend::CUdevice dev, CudaBackend::cudaDeviceProp* prop) {
  char name[256];
  int major = 0, minor = 0;
  size_t totalMem = 0;

  if (CudaBackend::cuDeviceGetName(name, 256, dev) != 0)
    return false;
  if (CudaBackend::cuDeviceComputeCapability(&major, &minor, dev) != 0)
    return false;
  if (CudaBackend::cuDeviceTotalMem(&totalMem, dev) != 0)
    return false;

  strcpy(prop->name, name);
  prop->major = major;
  prop->minor = minor;
  prop->totalGlobalMem = totalMem;

  int maxThreadsPerBlock = 0;
  if (CudaBackend::cuDeviceGetAttribute(&maxThreadsPerBlock, 1, dev) != 0)
    return false;
  prop->maxThreadsPerBlock = maxThreadsPerBlock;

  // You can query more attributes similarly:
  int multiProcessorCount = 0;
  CudaBackend::cuDeviceGetAttribute(&multiProcessorCount, 16, dev);
  prop->multiProcessorCount = multiProcessorCount;

  return true;
}

bool CudaBackend::gpuUtilizationSafe(void* nvmlDevice) {
  nvmlUtilization_t utilization;
  NVML_ERR(nvmlDeviceGetUtilizationRates(nvmlDevice, &utilization));
  if (utilization.gpu > 7) {
    std::cout << CUDA << "Skipping benchmark on this device due to high utilization (" << RED << utilization.gpu << "%" << RESET << ")\n";
    return false;
  }
  return true;
}

bool CudaBackend::memUtilizationSafe(void* nvmlDevice) {
  nvmlMemory_t memoryInfo;
  NVML_ERR(nvmlDeviceGetMemoryInfo(nvmlDevice, &memoryInfo));
  // This bench will use rougly 2 GB of memory. Check if the available VRAM is sufficient.
  const unsigned long long requiredMem = 2ull * 1024 * 1024 * 1024;
  if (memoryInfo.free < requiredMem) {
    std::cout << CUDA << "Skipping benchmark on this device due to insufficient free memory (" << RED << (memoryInfo.free / (1024 * 1024))
              << "mb/2048mb required free" << RESET << ")\n";
    return false;
  }
  double usagePercent = (double)memoryInfo.used / (double)memoryInfo.total * 100.0;
  std::string_view memColor;
  if (usagePercent < 50.0) {
    memColor = GREEN;
  } else if (usagePercent < 75.0) {
    memColor = "\033[33m"; // Yellow
  } else {
    memColor = RED;
  }
  std::cout << CUDA << "GPU Memory: " << memColor << (memoryInfo.used / (1024 * 1024)) << " MB / " << (memoryInfo.total / (1024 * 1024)) << " MB ("
            << std::fixed << std::setprecision(2) << usagePercent << "%" << RESET << ")\n";
  return true;
}

unsigned int CudaBackend::getAndPrintTemperature(void* nvmlDevice) {
  unsigned int temp = 0;
  NVML_ERR(nvmlDeviceGetTemperature(nvmlDevice, 0, &temp));
  std::string_view tempColor;
  if (temp < 50) {
    tempColor = GREEN;
  } else if (temp < 75) {
    tempColor = "\033[33m"; // Yellow
  } else {
    tempColor = RED;
  }
  std::cout << CUDA << "GPU Temperature before benchmark: " << tempColor << temp << RESET << " C\n";
  return temp;
}

bool CudaBackend::slowBenchmarks(float linearSetTime, float linearMultiplyTime) {
  constexpr const float slowMSThreshold = 50.0f;
  if (linearSetTime > slowMSThreshold || linearMultiplyTime > slowMSThreshold) {
    // These words are randomly selected to make sure that the user is paying attention! Seriously, these tests
    // may actually take forever, so the user better know what they're in for.
    const char* confirmWords[] = {"YES", "CUDA", "CONTINUE", "YEAH", "SURE", "GOAHEAD", "FINE", "WHYNOT", "AFFIRMATIVE", "LETSGO", "OKAY"};
    const int randIdx = static_cast<int>(time(nullptr)) % (std::size(confirmWords));
    using namespace std::string_literals;
    const std::string message = "The previous test benchmarks either took a very long time or did not complete at all. "
                                "This may indicate a hardware, driver, or other issue. Continuing to the full test suite may "
                                "take an excessively long time or fail. To proceed, please type '"s +
                                confirmWords[randIdx] + "'. All other responses will be treated as a 'no'.: ";
    wrapped_print(message, std::string(RED) + "[CUDA] ");
    std::string userInput;
    std::cin >> userInput;
    if (!stringsRoughlyMatch(userInput, confirmWords[randIdx])) {
      std::cout << CUDA << "Aborting further benchmarks on this device.\n";
      return false;
    }
    return true;
  }
  return false;
}

void CudaBackend::prepareDeviceForBenchmarking(int dev) {
  // Take the ptx file compiled at build time and load it.
  CUcontext context;
  CUDA_ERR(cuCtxCreate(&context, 0, dev));
  CUmodule module;
  // Read from file
  CUDA_ERR(cuModuleLoadData(&module, cudaKernels_ptx));

  cudaDeviceProp prop;
  getDeviceProperties(dev, &prop);
  std::cout << CUDA << "Running benches on '" << prop.name << "'\n";
  // Try getting GPU usage of this device, to see if running a benchmark is applicable
  // or if the device is in too much use that it might skew results.
  void* nvmlDevice = nullptr;
  NVML_ERR(nvmlDeviceGetHandleByIndex(dev, &nvmlDevice));
  if (!nvmlDevice) {
    std::cout << CUDA << "Skipping benchmark on this device due to inability to get NVML handle.\n";
    return;
  }

  if (!gpuUtilizationSafe(nvmlDevice))
    return;

  if (!memUtilizationSafe(nvmlDevice))
    return;

  // Temperature before (Fun metric, why not)
  getAndPrintTemperature(nvmlDevice);

  // Get the kernel functions
  CUfunction linearSetKernel;
  CUfunction linearMultiplyKernel;
  CUDA_ERR(cuModuleGetFunction(&linearSetKernel, module, "linearSetKernel"));
  CUDA_ERR(cuModuleGetFunction(&linearMultiplyKernel, module, "linearMultiplyKernel"));
  unsigned int threadsPerBlock = std::clamp(prop.maxThreadsPerBlock, 128, 1024);
  std::cout << CUDA << "Running simple tests...\n";
  float linearSetTime = runLinearSetBenchmark(threadsPerBlock, linearSetKernel);
  float linearMultiplyTime = runLinearMultiplyBenchmark(threadsPerBlock, linearMultiplyKernel);

  // Check to see if those first two rather simple tests took a while.
  // They are only 16 million elements, so if the GPU is running these tests slowly,
  // prompt the user if they would really like to continune with the more intensive tests.
  // //! FOR REFERENCE: my 4060 TI completes both tests in ONE FIFTH of a ms. So if it seriously takes this long, something is up.
  if (slowBenchmarks(linearSetTime, linearMultiplyTime)) {
    CUDA_ERR(cuModuleUnload(module));
    CUDA_ERR(cuCtxDestroy(context));
    return;
  }
  std::cout << CUDA << "All set. Starting full test suite...\n";
  CUfunction fmaKernel, intThroughputKernel, sharedMemoryKernel;
  CUDA_ERR(cuModuleGetFunction(&fmaKernel, module, "fmaKernel"));
  CUDA_ERR(cuModuleGetFunction(&intThroughputKernel, module, "integerThroughputKernel"));
  CUDA_ERR(cuModuleGetFunction(&sharedMemoryKernel, module, "sharedMemoryKernel"));
  runFmaBenchmark(threadsPerBlock, fmaKernel);
  runIntegerThroughputBenchmark(threadsPerBlock, intThroughputKernel);
  runSharedMemoryBenchmark(threadsPerBlock, sharedMemoryKernel);
  // Unload context, module, functions, free data, get ready for next device
  CUDA_ERR(cuModuleUnload(module));
  CUDA_ERR(cuCtxDestroy(context));
}

void CudaBackend::runBenchmark() {
  if (!cuMemAlloc) // Simple check to see if CUDA has been loaded. It SHOULD be, but you never know.
    return;

  // List the name of each available CUDA-compliant device.
  // Allocate and subsequently free 1 MB on the each.
  int deviceCount = 0;
  CUDA_ERR(cuDeviceGetCount(&deviceCount));

  for (int dev = 0; dev < deviceCount; ++dev) {
    prepareDeviceForBenchmarking(dev);
  }
}

#define CUDA_BENCHMARK_KERNEL(kernelFunc, blocks, threadsPerBlock, args, milliseconds)                                                               \
  CUstream stream;                                                                                                                                   \
  CUDA_ERR(cuStreamCreate(&stream, 0));                                                                                                              \
  CUevent startEvent, stopEvent;                                                                                                                     \
  CUDA_ERR(cuEventCreate(&startEvent, 0));                                                                                                           \
  CUDA_ERR(cuEventCreate(&stopEvent, 0));                                                                                                            \
  CUDA_ERR(cuEventRecord(startEvent, stream));                                                                                                       \
  CUDA_ERR(cuLaunchKernel(kernelFunc, blocks, 1, 1, threadsPerBlock, 1, 1, 0, stream, args, nullptr));                                               \
  CUDA_ERR(cuEventRecord(stopEvent, stream));                                                                                                        \
  CUDA_ERR(cuEventSynchronize(stopEvent));                                                                                                           \
  CUDA_ERR(cuEventElapsedTime(&milliseconds, startEvent, stopEvent));                                                                                \
  CUDA_ERR(cuEventDestroy(startEvent));                                                                                                              \
  CUDA_ERR(cuEventDestroy(stopEvent));                                                                                                               \
  CUDA_ERR(cuStreamDestroy(stream));

float CudaBackend::runLinearSetBenchmark(unsigned int threadsPerBlock, CudaBackend::CUfunction linearSetFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  std::cout << CUDA << "1) Linear Set (~" << N / 1000000 << "M elements)..." << std::flush;

  float* h_data = new float[N];
  CUdeviceptr d_data = 0;
  CUDA_ERR(cuMemAlloc(&d_data, N * sizeof(float)));

  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_data};
  float milliseconds = 0;
  std::cout << "\r" << CUDA << "1) Linear Set (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  CUDA_BENCHMARK_KERNEL(linearSetFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << CUDA << "1) Linear Set (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  CUDA_ERR(cuMemcpyDtoH(h_data, d_data, N * sizeof(float)));

  bool valid = true;
  for (unsigned long long i = 0ull; i < N; ++i) {
    if (h_data[i] != static_cast<float>(i)) {
      valid = false;
      std::cerr << "Data verification failed at index " << i << ": expected " << i << ", got " << h_data[i] << "\n";
      break;
    }
  }
  std::cout << "\r" << CUDA << "1) Linear Set (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  CUDA_ERR(cuMemFree(d_data));
  delete[] h_data;
  return valid ? milliseconds : 0.0f;
}

float CudaBackend::runLinearMultiplyBenchmark(unsigned int threadsPerBlock, CudaBackend::CUfunction linearMultiplyFunc) {
  constexpr const unsigned long long N = (1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats (since two inputs)
  std::cout << CUDA << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Preparing..." << std::flush;
  float* h_in1 = new float[N];
  float* h_in2 = new float[N];
  float* h_out = new float[N];
  for (unsigned long long i = 0ull; i < N; ++i) {
    h_in1[i] = static_cast<float>(i);
    h_in2[i] = static_cast<float>(i) / 2;
  }
  CUdeviceptr d_in1 = 0, d_in2 = 0, d_out = 0;
  CUDA_ERR(cuMemAlloc(&d_in1, N * sizeof(float)));
  CUDA_ERR(cuMemAlloc(&d_in2, N * sizeof(float)));
  CUDA_ERR(cuMemAlloc(&d_out, N * sizeof(float)));
  CUDA_ERR(cuMemsetD8(d_out, 0, N * sizeof(float)));
  CUDA_ERR(cuMemcpyHtoD(d_in1, h_in1, N * sizeof(float)));
  CUDA_ERR(cuMemcpyHtoD(d_in2, h_in2, N * sizeof(float)));

  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_in1, &d_in2, &d_out};
  float milliseconds = 0;
  std::cout << "\r" << CUDA << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  CUDA_BENCHMARK_KERNEL(linearMultiplyFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << CUDA << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  CUDA_ERR(cuMemcpyDtoH(h_out, d_out, N * sizeof(float)));

  bool valid = true;
  for (unsigned long long i = 0ull; i < N; ++i) {
    float expected = static_cast<float>(i) * static_cast<float>(i) / 2;
    constexpr static const float epsilon = 1e-5f;
    if (std::abs(h_out[i] - expected) > epsilon) {
      valid = false;
      std::cerr << " Data verification failed at index " << i << ": expected " << expected << ", got " << h_out[i] << "\n";
      break;
    }
  }
  std::cout << "\r" << CUDA << "2) Linear Multiply (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  CUDA_ERR(cuMemFree(d_in1));
  CUDA_ERR(cuMemFree(d_in2));
  CUDA_ERR(cuMemFree(d_out));
  delete[] h_in1;
  delete[] h_in2;
  delete[] h_out;
  return valid ? milliseconds : 0.0f;
}

float CudaBackend::runFmaBenchmark(unsigned int threadsPerBlock, CudaBackend::CUfunction fmaFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  constexpr const unsigned int totalIterations = 3000;
  std::cout << CUDA << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Preparing..." << std::flush;
  float* h_out = new float[N];
  CUdeviceptr d_out = 0;
  CUDA_ERR(cuMemAlloc(&d_out, N * sizeof(float)));
  CUDA_ERR(cuMemsetD8(d_out, 0, N * sizeof(float)));

  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_out, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << CUDA << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..." << std::flush;
  CUDA_BENCHMARK_KERNEL(fmaFunc, blocks, threadsPerBlock, args, milliseconds);

  std::cout << "\r" << CUDA << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET;
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CUDA_ERR(cuMemFree(d_out));
  delete[] h_out;
  return milliseconds;
}

float CudaBackend::runIntegerThroughputBenchmark(unsigned int threadsPerBlock, CudaBackend::CUfunction intThroughputFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(unsigned int); // 2GB worth of uints
  constexpr const unsigned int totalIterations = 5000;
  std::cout << CUDA << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << std::flush;
  // No host side preparation needed.
  CUdeviceptr d_out = 0;
  CUDA_ERR(cuMemAlloc(&d_out, N * sizeof(unsigned int)));
  CUDA_ERR(cuMemsetD8(d_out, 0, N * sizeof(unsigned int)));
  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_out, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << CUDA << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  CUDA_BENCHMARK_KERNEL(intThroughputFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << CUDA << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Verifying..."
            << std::flush;
  std::cout << "\r" << CUDA << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << GREEN
            << " PASSED" << RESET;
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CUDA_ERR(cuMemFree(d_out));
  return milliseconds;
}

float CudaBackend::runSharedMemoryBenchmark(unsigned int threadsPerBlock, CudaBackend::CUfunction sharedMemoryFunc) {
  constexpr const unsigned long long N = (1024 * 1024 * 1024) / sizeof(float); // 1GB worth of floats
  constexpr const unsigned int totalIterations = 2000;
  std::cout << CUDA << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << std::flush;
  float* h_out = new float[N];
  CUdeviceptr d_out = 0;
  CUDA_ERR(cuMemAlloc(&d_out, N * sizeof(float)));
  CUDA_ERR(cuMemsetD8(d_out, 0, N * sizeof(float)));

  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_out, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << CUDA << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  CUDA_BENCHMARK_KERNEL(sharedMemoryFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << CUDA << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET;
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CUDA_ERR(cuMemFree(d_out));
  delete[] h_out;
  return milliseconds;
}

void CudaBackend::shutdown() {
  if (cudaHandle) {
    cuMemAlloc = nullptr;
    cuMemFree = nullptr;
    cuDeviceGetCount = nullptr;
    cuDeviceGet = nullptr;
    cuDeviceGetName = nullptr;
    cuCtxCreate = nullptr;
    cuCtxDestroy = nullptr;
    cuCtxSynchronize = nullptr;
    cuModuleLoadData = nullptr;
    cuModuleUnload = nullptr;
    cuModuleGetFunction = nullptr;
    cuLaunchKernel = nullptr;
    cuMemcpyHtoD = nullptr;
    cuMemcpyDtoH = nullptr;
    cuMemsetD8 = nullptr;
    cuEventCreate = nullptr;
    cuEventRecord = nullptr;
    cuEventSynchronize = nullptr;
    cuEventElapsedTime = nullptr;
    cuEventDestroy = nullptr;
    cuStreamCreate = nullptr;
    cuStreamDestroy = nullptr;
    cuDeviceTotalMem = nullptr;
    cuDeviceComputeCapability = nullptr;
    cuDeviceGetAttribute = nullptr;
    cuGetErrorString = nullptr;

    nvmlShutdown();
    nvmlInit = nullptr;
    nvmlShutdown = nullptr;
    nvmlDeviceGetHandleByIndex = nullptr;
    nvmlDeviceGetUtilizationRates = nullptr;
    nvmlDeviceGetTemperature = nullptr;
    nvmlDeviceGetMemoryInfo = nullptr;

    closeLibrary(cudaHandle);
    closeLibrary(nvmlHandle);
    cudaHandle = nullptr;
    nvmlHandle = nullptr;
  }
}
