#include "hip_backend.hpp"
#include "../shared/shared.hpp"
#include "modules/hip_kernels.hpp"
#include <algorithm>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <vector>

HIPBackend::hipInit_t HIPBackend::hipInit = nullptr;
HIPBackend::hipDeviceReset_t HIPBackend::hipDeviceReset = nullptr;
HIPBackend::hipSetDevice_t HIPBackend::hipSetDevice = nullptr;
HIPBackend::hipGetDeviceCount_t HIPBackend::hipGetDeviceCount = nullptr;
HIPBackend::hipGetDevice_t HIPBackend::hipGetDevice = nullptr;
HIPBackend::hipGetDeviceProperties_t HIPBackend::hipGetDeviceProperties = nullptr;
HIPBackend::hipMalloc_t HIPBackend::hipMalloc = nullptr;
HIPBackend::hipFree_t HIPBackend::hipFree = nullptr;
HIPBackend::hipMemcpy_t HIPBackend::hipMemcpy = nullptr;
HIPBackend::hipMemset_t HIPBackend::hipMemset = nullptr;
// Events + Streams
HIPBackend::hipEventCreate_t HIPBackend::hipEventCreate = nullptr;
HIPBackend::hipEventDestroy_t HIPBackend::hipEventDestroy = nullptr;
HIPBackend::hipEventRecord_t HIPBackend::hipEventRecord = nullptr;
HIPBackend::hipEventSynchronize_t HIPBackend::hipEventSynchronize = nullptr;
HIPBackend::hipEventElapsedTime_t HIPBackend::hipEventElapsedTime = nullptr;
HIPBackend::hipStreamCreate_t HIPBackend::hipStreamCreate = nullptr;
HIPBackend::hipStreamDestroy_t HIPBackend::hipStreamDestroy = nullptr;
// Module (like HIP driver API)
HIPBackend::hipModuleLoadData_t HIPBackend::hipModuleLoadData = nullptr;
HIPBackend::hipModuleUnload_t HIPBackend::hipModuleUnload = nullptr;
HIPBackend::hipModuleGetFunction_t HIPBackend::hipModuleGetFunction = nullptr;
HIPBackend::hipModuleLaunchKernel_t HIPBackend::hipModuleLaunchKernel = nullptr;
// Error
HIPBackend::hipGetErrorString_t HIPBackend::hipGetErrorString = nullptr;
// RSMI
HIPBackend::rsmi_init_t HIPBackend::rsmi_init = nullptr;
HIPBackend::rsmi_shut_down_t HIPBackend::rsmi_shut_down = nullptr;
HIPBackend::rsmi_dev_temp_metric_get_t HIPBackend::rsmi_dev_temp_metric_get = nullptr;
HIPBackend::rsmi_dev_memory_total_get_t HIPBackend::rsmi_dev_memory_total_get = nullptr;
HIPBackend::rsmi_dev_memory_usage_get_t HIPBackend::rsmi_dev_memory_usage_get = nullptr;
HIPBackend::rsmi_dev_name_get_t HIPBackend::rsmi_dev_name_get = nullptr;
HIPBackend::rsmi_dev_busy_percent_get_t HIPBackend::rsmi_dev_busy_percent_get = nullptr;

#define HIP_ERR(call)                                                                                                                                \
  do {                                                                                                                                               \
    hipError_t err = call;                                                                                                                           \
    if (err != hipSuccess) {                                                                                                                         \
      std::cerr << HIP << "HIP error at " << __FILE__ << ":" << __LINE__ << ": " << hipGetErrorString(err) << "\n";                                  \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

#define RSMI_ERR(call)                                                                                                                               \
  do {                                                                                                                                               \
    HIPBackend::rsmi_status_t err = call;                                                                                                            \
    if (err != 0) {                                                                                                                                  \
      std::cerr << HIP << "RSMI error at " << __FILE__ << ":" << __LINE__ << ": " << err << "\n";                                                    \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

bool HIPBackend::gpuUtilizationSafe(int dev) {
  unsigned int utilization;
  RSMI_ERR(rsmi_dev_busy_percent_get(dev, &utilization));
  if (utilization > 7) {
    std::cout << HIP << "Skipping benchmark on this device due to high utilization (" << RED << utilization << "%" << RESET << ")\n";
    return false;
  }
  return true;
}

bool HIPBackend::memUtilizationSafe(int dev) {
  unsigned long usedMemory = 0;
  unsigned long totalMemory = 0;
  RSMI_ERR(rsmi_dev_memory_usage_get(dev, rsmi_memory_type_t::RSMI_MEM_TYPE_VRAM, &usedMemory));
  RSMI_ERR(rsmi_dev_memory_total_get(dev, rsmi_memory_type_t::RSMI_MEM_TYPE_VRAM, &totalMemory));
  // This bench will use rougly 2 GB of memory. Check if the available VRAM is sufficient.
  const unsigned long long requiredMem = 2ull * 1024 * 1024 * 1024;
  unsigned long memoryFree = totalMemory - usedMemory;
  if (totalMemory < requiredMem) {
    std::cout << HIP << "Skipping benchmark on this device due to insufficient total memory (" << RED << (totalMemory / (1024 * 1024))
              << "mb/2048mb required" << RESET << ")\n";
    return false;
  }
  double usagePercent = (double)usedMemory / (double)totalMemory * 100.0;
  if (usagePercent > 25) {
    std::cout << HIP << "Skipping benchmark on this device due to high memory usage (" << RED << std::fixed << std::setprecision(2) << usagePercent
              << "%" << RESET << ")\n";
    return false;
  }
  std::string_view memColor;
  if (usagePercent < 20.0) {
    memColor = GREEN;
  } else if (usagePercent < 25.0) {
    memColor = "\033[33m"; // Yellow
  } else {
    memColor = RED; // Should never be reached, but might
  }
  std::cout << HIP << "GPU Memory: " << memColor << (usedMemory / (1024 * 1024)) << " MB / " << (totalMemory / (1024 * 1024)) << " MB (" << std::fixed
            << std::setprecision(2) << usagePercent << "%" << RESET << ")\n";
  return true;
}

long HIPBackend::getAndPrintTemperature(int dev) {
  long temp = 0;
  RSMI_ERR(rsmi_dev_temp_metric_get(dev, rsmi_temperature_type_t::RSMI_TEMP_TYPE_EDGE, rsmi_temperature_metric_t::RSMI_TEMP_CURRENT, &temp));
  temp /= 1000; // Millidegrees to degrees. WHY AMD?
  std::string_view tempColor;
  if (temp < 50) {
    tempColor = GREEN;
  } else if (temp < 75) {
    tempColor = "\033[33m"; // Yellow
  } else {
    tempColor = RED;
  }
  std::cout << HIP << "GPU Temperature before benchmark: " << tempColor << temp << RESET << " C\n";
  return temp;
}

// Returns true if the benchmark was "slow", and should be skipped.
bool HIPBackend::slowBenchmarks(float linearSetTime, float linearMultiplyTime) {
  constexpr static const float slowMSThreshold = 50.0f;
  if (linearSetTime > slowMSThreshold || linearMultiplyTime > slowMSThreshold) {
    // These words are randomly selected to make sure that the user is paying attention! Seriously, these tests
    // may actually take forever, so the user better know what they're in for.
    static const char* confirmWords[] = {"YES", "HIP", "CONTINUE", "YEAH", "SURE", "GOAHEAD", "FINE", "WHYNOT", "AFFIRMATIVE", "LETSGO", "OKAY"};
    const int randIdx = static_cast<int>(time(nullptr)) % (std::size(confirmWords));
    using namespace std::string_literals;
    const std::string message = "The previous test benchmarks either took a very long time or did not complete at all. "
                                "This may indicate a hardware, driver, or other issue. Continuing to the full test suite may "
                                "take an excessively long time or fail. To proceed, please type '"s +
                                confirmWords[randIdx] + "'. All other responses will be treated as a 'no'.: ";
    wrapped_print(message, std::string(RED) + "[HIP] ");
    std::string userInput;
    std::cin >> userInput;
    if (!stringsRoughlyMatch(userInput, confirmWords[randIdx])) {
      std::cout << HIP << "Aborting further benchmarks on this device.\n";
      return true;
    }
    return false;
  }
  return false;
}

void HIPBackend::prepareDeviceForBenchmarking(int dev) {
  // Take the ptx file compiled at build time and load it.
  HIP_ERR(hipSetDevice(dev));

  hipDeviceProp_t prop;
  hipGetDeviceProperties(&prop, dev);
  std::cout << HIP << "Running benches on '" << prop.name << "'\n";
  if (!gpuUtilizationSafe(dev))
    return;

  if (!memUtilizationSafe(dev))
    return;

  // Temperature before (Fun metric, why not)
  getAndPrintTemperature(dev);
  // All is well. Let's go!
  hipModule_t module = nullptr;
  HIP_ERR(hipModuleLoadData(&module, (void*)hip_kernels_hsaco));
  // Get the kernel functions
  hipFunction_t linearSetKernel;
  hipFunction_t linearMultiplyKernel;
  HIP_ERR(hipModuleGetFunction(&linearSetKernel, module, "linearSetKernel"));
  HIP_ERR(hipModuleGetFunction(&linearMultiplyKernel, module, "linearMultiplyKernel"));
  unsigned int threadsPerBlock = std::clamp(prop.maxThreadsPerBlock, 128, 1024);
  std::cout << HIP << "Running simple tests...\n";
  float linearSetTime = runLinearSetBenchmark(threadsPerBlock, linearSetKernel);
  float linearMultiplyTime = runLinearMultiplyBenchmark(threadsPerBlock, linearMultiplyKernel);

  // Check to see if those first two rather simple tests took a while.
  // If the GPU is running these tests slowly,
  // prompt the user if they would really like to continune with the more intensive tests.
  // ! FOR REFERENCE: my 9070 XT completes both tests in 6 ms. So if it seriously takes this long, something is up.
  if (slowBenchmarks(linearSetTime, linearMultiplyTime)) {
    HIP_ERR(hipModuleUnload(module));
    HIP_ERR(hipDeviceReset());
    return;
  }
  std::cout << HIP << "All set. Starting full test suite...\n";
  hipFunction_t fmaKernel, intThroughputKernel, sharedMemoryKernel, sgemmKernel;
  HIP_ERR(hipModuleGetFunction(&fmaKernel, module, "fmaKernel"));
  HIP_ERR(hipModuleGetFunction(&intThroughputKernel, module, "integerThroughputKernel"));
  HIP_ERR(hipModuleGetFunction(&sharedMemoryKernel, module, "sharedMemoryKernel"));
  HIP_ERR(hipModuleGetFunction(&sgemmKernel, module, "sgemmKernel"));
  runFmaBenchmark(threadsPerBlock, fmaKernel);
  runIntegerThroughputBenchmark(threadsPerBlock, intThroughputKernel);
  runSharedMemoryBenchmark(threadsPerBlock, sharedMemoryKernel);
  runSgemmBenchmark(threadsPerBlock, sgemmKernel);

  HIP_ERR(hipModuleUnload(module));
  HIP_ERR(hipDeviceReset());
}

void HIPBackend::runBenchmark() {
  hipInit(0);
  rsmi_init(0);

  int deviceCount = 0;
  HIP_ERR(hipGetDeviceCount(&deviceCount));
  for (int dev = 0; dev < deviceCount; ++dev) {
    prepareDeviceForBenchmarking(dev);
  }
}

#define HIP_BENCHMARK_KERNEL(kernelFunc, blocks, threadsPerBlock, args, milliseconds)                                                                \
  hipStream_t stream;                                                                                                                                \
  HIP_ERR(hipStreamCreate(&stream));                                                                                                                 \
  hipEvent_t startEvent, stopEvent;                                                                                                                  \
  HIP_ERR(hipEventCreate(&startEvent));                                                                                                              \
  HIP_ERR(hipEventCreate(&stopEvent));                                                                                                               \
  HIP_ERR(hipEventRecord(startEvent, stream));                                                                                                       \
  HIP_ERR(hipModuleLaunchKernel(kernelFunc, blocks, 1, 1, threadsPerBlock, 1, 1, 0, stream, args, nullptr));                                         \
  HIP_ERR(hipEventRecord(stopEvent, stream));                                                                                                        \
  HIP_ERR(hipEventSynchronize(stopEvent));                                                                                                           \
  HIP_ERR(hipEventElapsedTime(&milliseconds, startEvent, stopEvent));                                                                                \
  HIP_ERR(hipEventDestroy(startEvent));                                                                                                              \
  HIP_ERR(hipEventDestroy(stopEvent));                                                                                                               \
  HIP_ERR(hipStreamDestroy(stream));

float HIPBackend::runLinearSetBenchmark(unsigned int threadsPerBlock, HIPBackend::hipFunction_t linearSetFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  std::cout << HIP << "1) Linear Set (~" << N / 1000000 << "M elements)..." << std::flush;

  float* h_data = new float[N];
  hipDeviceptr_t d_data = 0;
  HIP_ERR(hipMalloc(&d_data, N * sizeof(float)));

  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_data};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "1) Linear Set (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  HIP_BENCHMARK_KERNEL(linearSetFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << HIP << "1) Linear Set (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  HIP_ERR(hipMemcpy(h_data, d_data, N * sizeof(float), hipMemcpyDeviceToHost));

  bool valid = true;
  for (unsigned long long i = 0ull; i < N; ++i) {
    if (h_data[i] != static_cast<float>(i)) {
      valid = false;
      std::cerr << "Data verification failed at index " << i << ": expected " << i << ", got " << h_data[i] << "\n";
      break;
    }
  }
  std::cout << "\r" << HIP << "1) Linear Set (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  HIP_ERR(hipFree(d_data));
  delete[] h_data;
  return valid ? milliseconds : 0.0f;
}

float HIPBackend::runLinearMultiplyBenchmark(unsigned int threadsPerBlock, HIPBackend::hipFunction_t linearMultiplyFunc) {
  constexpr const unsigned long long N = (1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats (since two inputs)
  std::cout << HIP << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Preparing..." << std::flush;
  float* h_in1 = new float[N];
  float* h_in2 = new float[N];
  float* h_out = new float[N];
  for (unsigned long long i = 0ull; i < N; ++i) {
    h_in1[i] = static_cast<float>(i);
    h_in2[i] = static_cast<float>(i) / 2;
  }
  hipDeviceptr_t d_in1 = 0, d_in2 = 0, d_out = 0;
  HIP_ERR(hipMalloc(&d_in1, N * sizeof(float)));
  HIP_ERR(hipMalloc(&d_in2, N * sizeof(float)));
  HIP_ERR(hipMalloc(&d_out, N * sizeof(float)));

  HIP_ERR(hipMemcpy(d_in1, h_in1, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERR(hipMemcpy(d_in2, h_in2, N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERR(hipMemset(d_out, 0, N * sizeof(float)));
  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_in1, &d_in2, &d_out};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  HIP_BENCHMARK_KERNEL(linearMultiplyFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << HIP << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  HIP_ERR(hipMemcpy(h_out, d_out, N * sizeof(float), hipMemcpyDeviceToHost));

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
  std::cout << "\r" << HIP << "2) Linear Multiply (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  HIP_ERR(hipFree(d_in1));
  HIP_ERR(hipFree(d_in2));
  HIP_ERR(hipFree(d_out));
  delete[] h_in1;
  delete[] h_in2;
  delete[] h_out;
  return valid ? milliseconds : 0.0f;
}

float HIPBackend::runFmaBenchmark(unsigned int threadsPerBlock, HIPBackend::hipFunction_t fmaFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  constexpr const unsigned int totalIterations = 3000;
  std::cout << HIP << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Preparing..." << std::flush;
  float* h_out = new float[N];
  hipDeviceptr_t d_out = 0;
  HIP_ERR(hipMalloc(&d_out, N * sizeof(float)));
  HIP_ERR(hipMemset(d_out, 0, N * sizeof(float)));

  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_out, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..." << std::flush;
  HIP_BENCHMARK_KERNEL(fmaFunc, blocks, threadsPerBlock, args, milliseconds);

  std::cout << "\r" << HIP << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET;
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  HIP_ERR(hipFree(d_out));
  delete[] h_out;
  return milliseconds;
}

float HIPBackend::runIntegerThroughputBenchmark(unsigned int threadsPerBlock, HIPBackend::hipFunction_t intThroughputFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(unsigned int); // 2GB worth of uints
  constexpr const unsigned int totalIterations = 5000;
  std::cout << HIP << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << std::flush;
  // No host side preparation needed.
  hipDeviceptr_t d_out = 0;
  HIP_ERR(hipMalloc(&d_out, N * sizeof(unsigned int)));
  HIP_ERR(hipMemset(d_out, 0, N * sizeof(unsigned int)));
  unsigned long long blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_out, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  HIP_BENCHMARK_KERNEL(intThroughputFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << HIP << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Verifying..."
            << std::flush;
  std::cout << "\r" << HIP << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << GREEN
            << " PASSED" << RESET;
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  HIP_ERR(hipFree(d_out));
  return milliseconds;
}

float HIPBackend::runSharedMemoryBenchmark(unsigned int threadsPerBlock, HIPBackend::hipFunction_t sharedMemoryFunc) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  constexpr const unsigned int totalIterations = 2000;
  std::cout << HIP << "5) Shared Memory Benchmark (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)..." << std::flush;

  float* h_data = new float[N];
  hipDeviceptr_t d_data = 0;
  HIP_ERR(hipMalloc(&d_data, N * sizeof(float)));

  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_data, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "5) Shared Memory Benchmark (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  HIP_BENCHMARK_KERNEL(sharedMemoryFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << HIP << "5) Shared Memory Benchmark (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Verifying..."
            << std::flush;
  HIP_ERR(hipMemcpy(h_data, d_data, N * sizeof(float), hipMemcpyDeviceToHost));

  std::cout << "\r" << HIP << "5) Shared Memory Benchmark (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  HIP_ERR(hipFree(d_data));
  delete[] h_data;
  return milliseconds;
}

float HIPBackend::runSgemmBenchmark(unsigned int threadsPerBlock, hipFunction_t sgemmFunc) {
  constexpr const unsigned long long N = 1024; // 1024*1024 matrices
  constexpr const unsigned int totalIterations = 5000;
  std::cout << HIP << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations << " iterations)..." << std::flush;
  float* h_A = new float[N * N];
  float* h_B = new float[N * N];
  float* h_C = new float[N * N];
  for (unsigned long long i = 0; i < N * N; ++i) {
    h_A[i] = static_cast<float>(i % 100) / 100.0f;
    h_B[i] = static_cast<float>((i + 50) % 100) / 100.0f;
    h_C[i] = 0.0f;
  }
  hipDeviceptr_t d_A = 0, d_B = 0, d_C = 0;
  HIP_ERR(hipMalloc(&d_A, N * N * sizeof(float)));
  HIP_ERR(hipMalloc(&d_B, N * N * sizeof(float)));
  HIP_ERR(hipMalloc(&d_C, N * N * sizeof(float)));
  HIP_ERR(hipMemcpy(d_A, h_A, N * N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERR(hipMemcpy(d_B, h_B, N * N * sizeof(float), hipMemcpyHostToDevice));
  HIP_ERR(hipMemset(d_C, 0, N * N * sizeof(float)));
  unsigned int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  void* args[] = {&d_A, &d_B, &d_C, (void*)&N, (void*)&totalIterations};
  float milliseconds = 0;
  std::cout << "\r" << HIP << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations << " iterations)... Running..."
            << std::flush;
  HIP_BENCHMARK_KERNEL(sgemmFunc, blocks, threadsPerBlock, args, milliseconds);
  std::cout << "\r" << HIP << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  HIP_ERR(hipFree(d_A));
  HIP_ERR(hipFree(d_B));
  HIP_ERR(hipFree(d_C));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return milliseconds;
}

void HIPBackend::shutdown() {
  if (rsmi_shut_down != nullptr)
    rsmi_shut_down();
  rsmi_shut_down = nullptr;

  // Turn all functions into nullptr
  rsmi_dev_temp_metric_get = nullptr;
  rsmi_dev_memory_total_get = nullptr;
  rsmi_dev_memory_usage_get = nullptr;
  rsmi_dev_name_get = nullptr;
  rsmi_dev_busy_percent_get = nullptr;
  hipInit = nullptr;
  hipGetDeviceCount = nullptr;
  hipGetDevice = nullptr;
  hipGetDeviceProperties = nullptr;
  hipDeviceReset = nullptr;
  hipMalloc = nullptr;
  hipFree = nullptr;
  hipMemcpy = nullptr;
  hipEventCreate = nullptr;
  hipEventDestroy = nullptr;
  hipEventRecord = nullptr;
  hipEventSynchronize = nullptr;
  hipEventElapsedTime = nullptr;
  hipStreamCreate = nullptr;
  hipStreamDestroy = nullptr;
  hipModuleLoadData = nullptr;
  hipModuleUnload = nullptr;
  hipModuleGetFunction = nullptr;
  hipModuleLaunchKernel = nullptr;
  hipMemset = nullptr;
  hipGetErrorString = nullptr;

  closeLibrary(hipHandle);
  closeLibrary(rsmiHandle);
  hipHandle = nullptr;
  rsmiHandle = nullptr;
}
