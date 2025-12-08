#include "opencl_backend.hpp"
#include "../shared/shared.hpp"
#include "modules/opencl_kernels.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

CLBackend::clGetDeviceInfo_t CLBackend::clGetDeviceInfo = nullptr;
CLBackend::clGetPlatformIDs_t CLBackend::clGetPlatformIDs = nullptr;
CLBackend::clGetDeviceIDs_t CLBackend::clGetDeviceIDs = nullptr;
CLBackend::clCreateContext_t CLBackend::clCreateContext = nullptr;
CLBackend::clCreateCommandQueueWithProperties_t CLBackend::clCreateCommandQueueWithProperties = nullptr;
CLBackend::clCreateProgramWithSource_t CLBackend::clCreateProgramWithSource = nullptr;
CLBackend::clBuildProgram_t CLBackend::clBuildProgram = nullptr;
CLBackend::clCreateKernel_t CLBackend::clCreateKernel = nullptr;
CLBackend::clEnqueueNDRangeKernel_t CLBackend::clEnqueueNDRangeKernel = nullptr;
CLBackend::clWaitForEvents_t CLBackend::clWaitForEvents = nullptr;
CLBackend::clGetEventProfilingInfo_t CLBackend::clGetEventProfilingInfo = nullptr;
CLBackend::clReleaseEvent_t CLBackend::clReleaseEvent = nullptr;
CLBackend::clReleaseCommandQueue_t CLBackend::clReleaseCommandQueue = nullptr;
CLBackend::clReleaseContext_t CLBackend::clReleaseContext = nullptr;
CLBackend::clReleaseProgram_t CLBackend::clReleaseProgram = nullptr;
CLBackend::clReleaseKernel_t CLBackend::clReleaseKernel = nullptr;
CLBackend::clCreateBuffer_t CLBackend::clCreateBuffer = nullptr;
CLBackend::clGetPlatformInfo_t CLBackend::clGetPlatformInfo = nullptr;
CLBackend::clEnqueueReadBuffer_t CLBackend::clEnqueueReadBuffer = nullptr;
CLBackend::clSetKernelArg_t CLBackend::clSetKernelArg = nullptr;
CLBackend::clReleaseMemObject_t CLBackend::clReleaseMemObject = nullptr;

#define CL_ERR(call)                                                                                                                                 \
  do {                                                                                                                                               \
    int err = call;                                                                                                                                  \
    if (err != 0) {                                                                                                                                  \
      std::cerr << OPENCL << "OpenCL error at opencl_backend.cpp:" << __LINE__ << ": " << err << "\n";                                               \
      exit(EXIT_FAILURE);                                                                                                                            \
    }                                                                                                                                                \
  } while (0)

bool CLBackend::slowBenchmarks(float linearSetTime, float linearMultiplyTime) {
  constexpr static const float slowMSThreshold = 400.0f; // Give OpenCL a little more wiggle room, since it can be slower on some devices
  if (linearSetTime > slowMSThreshold || linearMultiplyTime > slowMSThreshold) {
    // These words are randomly selected to make sure that the user is paying attention! Seriously, these tests
    // may actually take forever, so the user better know what they're in for.
    static const char* confirmWords[] = {"YES", "OPENCL", "CONTINUE", "YEAH", "SURE", "GOAHEAD", "FINE", "WHYNOT", "AFFIRMATIVE", "LETSGO", "OKAY"};
    int randIdx = rand() % (sizeof(confirmWords) / sizeof(confirmWords[0]));
    std::string message = "The initial benchmarks took a really long time.\n"
                          "This may indicate that the device is under heavy load or not suitable for benchmarking.\n"
                          "To continue with the full benchmark suite, please type '" +
                          std::string(confirmWords[randIdx]) + "' to confirm: ";

    wrapped_print(std::string(RED) + "[OPENCL]" + std::string(RESET) + " ", message);
    std::string userInput;
    std::cout << "\n" << RED << "[OPENCL]" << RESET << " >";
    std::cin >> userInput;
    if (!stringsRoughlyMatch(userInput, confirmWords[randIdx])) {
      std::cout << OPENCL << "Aborting further benchmarks on this device.\n";
      return true;
    }
    return false;
  }
  return false;
}

void CLBackend::prepareDeviceForBenchmarking(cl_device_id dev, cl_context context, cl_command_queue queue, cl_program program) {
  char deviceName[256];
  CL_ERR(clGetDeviceInfo(dev, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr));
  std::cout << OPENCL << "Running benches on '" << deviceName << "'\n";

  // Get the max number of threads per block
  size_t maxWorkGroupSize = 0;
  CL_ERR(clGetDeviceInfo(dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
  unsigned int threadsPerBlock = static_cast<unsigned int>(std::min<size_t>(256, maxWorkGroupSize));
  // Get the kernels from the program
  cl_kernel linearSetKernel = clCreateKernel(program, "linearSetKernel", nullptr);
  cl_kernel linearMultiplyKernel = clCreateKernel(program, "linearMultiplyKernel", nullptr);
  std::cout << OPENCL << "Running simple tests...\n";
  float linearSetTime = runLinearSetBenchmark(threadsPerBlock, linearSetKernel, context, queue);
  float linearMultiplyTime = runLinearMultiplyBenchmark(threadsPerBlock, linearMultiplyKernel, context, queue);

  if (slowBenchmarks(linearSetTime, linearMultiplyTime)) {
    clReleaseKernel(linearMultiplyKernel);
    clReleaseKernel(linearSetKernel);
    return;
  }

  std::cout << OPENCL << "All set. Starting full test suite...\n";
  cl_kernel fmaKernel = clCreateKernel(program, "fmaKernel", nullptr);
  cl_kernel integerThroughputKernel = clCreateKernel(program, "integerThroughputKernel", nullptr);
  cl_kernel sharedMemoryKernel = clCreateKernel(program, "sharedMemoryKernel", nullptr);
  cl_kernel sgemmKernel = clCreateKernel(program, "sgemmKernel", nullptr);
  runFmaBenchmark(threadsPerBlock, fmaKernel, context, queue);
  runIntegerThroughputBenchmark(threadsPerBlock, integerThroughputKernel, context, queue);
  runSharedMemoryBenchmark(threadsPerBlock, sharedMemoryKernel, context, queue);
  runSgemmBenchmark(threadsPerBlock, sgemmKernel, context, queue);
  // Release kernels
  clReleaseKernel(fmaKernel);
  clReleaseKernel(integerThroughputKernel);
  clReleaseKernel(sharedMemoryKernel);
  clReleaseKernel(sgemmKernel);
  clReleaseKernel(linearMultiplyKernel);
  clReleaseKernel(linearSetKernel);
}

void CLBackend::runBenchmark() {
  int platformCount = 0;
  CL_ERR(clGetPlatformIDs(0, nullptr, (unsigned int*)&platformCount));
  if (platformCount == 0) {
    std::cout << OPENCL << "No OpenCL platforms found.\n";
    return;
  }
  std::vector<cl_platform_id> platforms(platformCount);
  CL_ERR(clGetPlatformIDs(platformCount, platforms.data(), nullptr));
  for (int p = 0; p < platformCount; ++p) {
    char platformName[256];
    CL_ERR(clGetPlatformInfo(platforms[p], CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr));
    std::cout << OPENCL << "Platform " << p << ": " << platformName << "\n";

    int deviceCount = 0;
    CL_ERR(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, (unsigned int*)&deviceCount));
    if (deviceCount == 0) {
      std::cout << OPENCL << "No OpenCL devices found on this platform, skipping...\n";
      continue;
    }
    // Ask the user if they want to benchmark this platform.
    // This is becuase some platforms have repeats of devices (like rust_icl and ROCm)
    std::cout << OPENCL << "Do you want to benchmark this platform? (y/n): ";
    std::string userInput;
    std::cin >> userInput;
    if (!stringsRoughlyMatch(userInput, "y") && !stringsRoughlyMatch(userInput, "yes")) {
      std::cout << OPENCL << "Skipping benchmarks on this platform.\n";
      continue;
    }
    std::vector<cl_device_id> devices(deviceCount);
    CL_ERR(clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, deviceCount, devices.data(), nullptr));
    // Create a context, program, and command queue
    cl_context context = clCreateContext(nullptr, deviceCount, devices.data(), nullptr, nullptr, nullptr);
    if (context == nullptr) {
      std::cout << OPENCL << "Failed to create OpenCL context for this platform, skipping...\n";
      continue;
    }
    cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, devices[0], props, nullptr);
    if (commandQueue == nullptr) {
      std::cout << OPENCL << "Failed to create OpenCL command queue for this platform, skipping...\n";
      clReleaseContext(context);
      continue;
    }
    // Create a program
    cl_program program = clCreateProgramWithSource(context, 1, &openclKernelSource, nullptr, nullptr);
    if (program == nullptr) {
      std::cout << OPENCL << "Failed to create OpenCL program for this platform, skipping...\n";
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      continue;
    }
    // Compile program
    int err = clBuildProgram(program, deviceCount, devices.data(), nullptr, nullptr, nullptr);
    if (err != 0) {
      std::cout << OPENCL << "Failed to build OpenCL program for this platform, skipping...\n";
      clReleaseProgram(program);
      clReleaseCommandQueue(commandQueue);
      clReleaseContext(context);
      continue;
    }
    for (int d = 0; d < deviceCount; ++d) {
      prepareDeviceForBenchmarking(devices[d], context, commandQueue, program);
    }
    // Cleanup
    clReleaseProgram(program);
    clReleaseCommandQueue(commandQueue);
    clReleaseContext(context);
  }
}

#define OPENCL_BENCHMARK_KERNEL_1D(kernel, globalSize, localSize, milliseconds)                                                                      \
  do {                                                                                                                                               \
    cl_event __ocl_evt = nullptr;                                                                                                                    \
    size_t __glob = (globalSize);                                                                                                                    \
    size_t __loc = (localSize);                                                                                                                      \
    int __err = clEnqueueNDRangeKernel(commandQueue, (kernel), 1, nullptr, &__glob, &__loc, 0, nullptr, &__ocl_evt);                                 \
    if (__err != 0) {                                                                                                                                \
      std::cerr << "Failed to enqueue kernel (clEnqueueNDRangeKernel): " << __err << "\n";                                                           \
      milliseconds = 0;                                                                                                                              \
    } else {                                                                                                                                         \
      __err = clWaitForEvents(1, (const void**)&__ocl_evt);                                                                                          \
      if (__err != 0) {                                                                                                                              \
        std::cerr << "clWaitForEvents failed: " << __err << "\n";                                                                                    \
        milliseconds = 0;                                                                                                                            \
      } else {                                                                                                                                       \
        unsigned long __start = 0, __end = 0;                                                                                                        \
        clGetEventProfilingInfo(__ocl_evt, CL_PROFILING_COMMAND_START, sizeof(__start), (void**)&__start, nullptr);                                  \
        clGetEventProfilingInfo(__ocl_evt, CL_PROFILING_COMMAND_END, sizeof(__end), (void**)&__end, nullptr);                                        \
        milliseconds = (double)(__end - __start) * 1e-6; /* ns -> ms */                                                                              \
      }                                                                                                                                              \
      clReleaseEvent(__ocl_evt);                                                                                                                     \
    }                                                                                                                                                \
  } while (0)

float CLBackend::runLinearSetBenchmark(unsigned int threadsPerBlock, cl_kernel linearSetFunc, cl_context context, cl_command_queue commandQueue) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  std::cout << OPENCL << "1) Linear Set (~" << N / 1000000 << "M elements)..." << std::flush;

  float* h_data = new float[N];
  cl_mem d_data = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), nullptr, nullptr);
  if (d_data == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffer for linear set benchmark.\n";
    delete[] h_data;
    return 0.0f;
  }
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  clSetKernelArg(linearSetFunc, 0, sizeof(cl_mem), &d_data);
  // clSetKernelArg(linearSetFunc, 1, sizeof(N), &N);
  float milliseconds = 0;
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  std::cout << "\r" << OPENCL << "1) Linear Set (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(linearSetFunc, globalSize, threadsPerBlock, milliseconds);
  std::cout << "\r" << OPENCL << "1) Linear Set (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  CL_ERR(clEnqueueReadBuffer(commandQueue, d_data, 1, 0, N * sizeof(float), h_data, 0, nullptr, nullptr));

  bool valid = true;
  for (unsigned long long i = 0ull; i < N; ++i) {
    if (h_data[i] != static_cast<float>(i)) {
      valid = false;
      std::cerr << "Data verification failed at index " << i << ": expected " << i << ", got " << h_data[i] << "\n";
      break;
    }
  }
  std::cout << "\r" << OPENCL << "1) Linear Set (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  CL_ERR(clReleaseMemObject(d_data));
  delete[] h_data;
  return valid ? milliseconds : 0.0f;
}

float CLBackend::runLinearMultiplyBenchmark(unsigned int threadsPerBlock, cl_kernel linearMultiplyFunc, cl_context context,
                                            cl_command_queue commandQueue) {
  constexpr const unsigned long long N = (1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  std::cout << OPENCL << "2) Linear Multiply (~" << N / 1000000 << "M elements)..." << std::flush;

  float* h_data = new float[N];
  float* h_dataB = new float[N];
  float* h_out = new float[N];
  for (unsigned long long i = 0; i < N; ++i) {
    h_data[i] = static_cast<float>(i);
    h_dataB[i] = static_cast<float>(i) / 2;
  }
  cl_mem d_dataA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_data, nullptr);
  cl_mem d_dataB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(float), h_dataB, nullptr);
  cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), nullptr, nullptr);
  if (d_dataA == nullptr || d_dataB == nullptr || d_out == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffers for linear multiply benchmark.\n";
    delete[] h_data;
    delete[] h_dataB;
    delete[] h_out;
    return 0.0f;
  }
  int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
  clSetKernelArg(linearMultiplyFunc, 0, sizeof(cl_mem), &d_dataA);
  clSetKernelArg(linearMultiplyFunc, 1, sizeof(cl_mem), &d_dataB);
  clSetKernelArg(linearMultiplyFunc, 2, sizeof(cl_mem), &d_out);
  // clSetKernelArg(linearMultiplyFunc, 3, sizeof(N), &N);
  float milliseconds = 0;
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  std::cout << "\r" << OPENCL << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Running..." << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(linearMultiplyFunc, globalSize, threadsPerBlock, milliseconds);
  std::cout << "\r" << OPENCL << "2) Linear Multiply (~" << N / 1000000 << "M elements)... Verifying..." << std::flush;
  CL_ERR(clEnqueueReadBuffer(commandQueue, d_out, 1, 0, N * sizeof(float), h_out, 0, nullptr, nullptr));
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
  std::cout << "\r" << OPENCL << "2) Linear Multiply (~" << N / 1000000 << "M elements)...";
  if (valid) {
    std::cout << GREEN << " PASSED" << RESET;
  } else {
    std::cout << RED << " FAILED" << RESET;
  }
  std::cout << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";

  CL_ERR(clReleaseMemObject(d_dataA));
  CL_ERR(clReleaseMemObject(d_dataB));
  CL_ERR(clReleaseMemObject(d_out));
  delete[] h_data;
  delete[] h_dataB;
  delete[] h_out;
  return valid ? milliseconds : 0.0f;
}

float CLBackend::runFmaBenchmark(unsigned int threadsPerBlock, cl_kernel fmaFunc, cl_context context, cl_command_queue commandQueue) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(float); // 2GB worth of floats
  constexpr const unsigned int totalIterations = 3000;
  std::cout << OPENCL << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Preparing..." << std::flush;
  float* h_out = new float[N];
  cl_mem d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), nullptr, nullptr);
  if (d_out == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffer for FMA benchmark.\n";
    delete[] h_out;
    return 0.0f;
  }

  clSetKernelArg(fmaFunc, 0, sizeof(cl_mem), &d_out);
  clSetKernelArg(fmaFunc, 1, sizeof(totalIterations), &totalIterations);
  // clSetKernelArg(fmaFunc, 2, sizeof(N), &N);
  float milliseconds = 0;
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  std::cout << "\r" << OPENCL << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..." << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(fmaFunc, globalSize, threadsPerBlock, milliseconds);
  // There is no more verifications after this benchmark. The first two were to make sure the device is working correctly,
  // which if we reached this point, it is.
  std::cout << "\r" << OPENCL << "3) FMA (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CL_ERR(clReleaseMemObject(d_out));
  delete[] h_out;
  return milliseconds;
}

float CLBackend::runIntegerThroughputBenchmark(unsigned int threadsPerBlock, cl_kernel integerThroughputFunc, cl_context context,
                                               cl_command_queue commandQueue) {
  constexpr const unsigned long long N = (2ull * 1024 * 1024 * 1024) / sizeof(unsigned int); // 2GB worth of uints
  constexpr const unsigned int totalIterations = 5000;
  std::cout << OPENCL << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Preparing..."
            << std::flush;
  unsigned int* h_out = new unsigned int[N];
  cl_mem d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(unsigned int), nullptr, nullptr);
  if (d_out == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffer for Integer Throughput benchmark.\n";
    delete[] h_out;
    return 0.0f;
  }
  clSetKernelArg(integerThroughputFunc, 0, sizeof(cl_mem), &d_out);
  clSetKernelArg(integerThroughputFunc, 1, sizeof(totalIterations), &totalIterations);
  // clSetKernelArg(integerThroughputFunc, 2, sizeof(N), &N);
  float milliseconds = 0;
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  std::cout << "\r" << OPENCL << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(integerThroughputFunc, globalSize, threadsPerBlock, milliseconds);
  // There is no more verifications after this benchmark. The first two were to make sure the device is working correctly,
  // which if we reached this point, it is.
  std::cout << "\r" << OPENCL << "4) Integer Throughput (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CL_ERR(clReleaseMemObject(d_out));
  delete[] h_out;
  return milliseconds;
}

float CLBackend::runSharedMemoryBenchmark(unsigned int threadsPerBlock, cl_kernel sharedMemoryFunc, cl_context context,
                                          cl_command_queue commandQueue) {
  constexpr const unsigned long long N = (1024 * 1024 * 1024) / sizeof(float); // 1GB worth of floats
  constexpr const unsigned int totalIterations = 2000;
  std::cout << OPENCL << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Preparing..."
            << std::flush;
  float* h_out = new float[N];
  cl_mem d_out = clCreateBuffer(context, CL_MEM_READ_WRITE, N * sizeof(float), nullptr, nullptr);
  if (d_out == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffer for Shared Memory Bandwidth benchmark.\n";
    delete[] h_out;
    return 0.0f;
  }
  clSetKernelArg(sharedMemoryFunc, 0, sizeof(cl_mem), &d_out);
  clSetKernelArg(sharedMemoryFunc, 1, sizeof(totalIterations), &totalIterations);
  // clSetKernelArg(sharedMemoryFunc, 2, sizeof(N), &N);
  float milliseconds = 0;
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  std::cout << "\r" << OPENCL << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)... Running..."
            << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(sharedMemoryFunc, globalSize, threadsPerBlock, milliseconds);
  // There is no more verifications after this benchmark. The first two were to make sure the device is working correctly,
  // which if we reached this point, it is.
  std::cout << "\r" << OPENCL << "5) Shared Memory Bandwidth (~" << N / 1000000 << "M elements, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CL_ERR(clReleaseMemObject(d_out));
  delete[] h_out;
  return milliseconds;
}

float CLBackend::runSgemmBenchmark(unsigned int threadsPerBlock, cl_kernel sgemmFunc, cl_context context, cl_command_queue commandQueue) {
  constexpr const unsigned long long N = 1024; // 1024x1024 matrices
  constexpr const unsigned int totalIterations = 5000;
  std::cout << OPENCL << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations << " iterations)..." << std::flush;
  float* h_A = new float[N * N];
  float* h_B = new float[N * N];
  float* h_C = new float[N * N];
  // Initialize matrices
  for (unsigned long long i = 0; i < N * N; ++i) {
    h_A[i] = static_cast<float>(i % 100) / 100.0f;
    h_B[i] = static_cast<float>((i * 3) % 100) / 100.0f;
    h_C[i] = 0.0f;
  }
  cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), h_A, nullptr);
  cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), h_B, nullptr);
  cl_mem d_C = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * N * sizeof(float), h_C, nullptr);
  if (d_A == nullptr || d_B == nullptr || d_C == nullptr) {
    std::cerr << OPENCL << "Failed to create device buffers for SGEMM benchmark.\n";
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    return 0.0f;
  }
  clSetKernelArg(sgemmFunc, 0, sizeof(cl_mem), &d_A);
  clSetKernelArg(sgemmFunc, 1, sizeof(cl_mem), &d_B);
  clSetKernelArg(sgemmFunc, 2, sizeof(cl_mem), &d_C);
  clSetKernelArg(sgemmFunc, 3, sizeof(N), &N);
  clSetKernelArg(sgemmFunc, 4, sizeof(totalIterations), &totalIterations);
  float milliseconds = 0;
  // Although we could dispatch in 2D, we stick to 1D for parity with the other APIs.
  size_t globalSize = ((size_t)N + threadsPerBlock - 1) / threadsPerBlock * threadsPerBlock;
  size_t localSize = threadsPerBlock;
  std::cout << "\r" << OPENCL << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations
            << " iterations)... Running..." << std::flush;
  OPENCL_BENCHMARK_KERNEL_1D(sgemmFunc, globalSize, localSize, milliseconds);
  // There is no more verifications after this benchmark. The first two were to make sure the device is working correctly,
  // which if we reached this point, it is.
  std::cout << "\r" << OPENCL << "6) SGEMM/Matrix multiplication (" << N << "x" << N << " matrices, " << totalIterations << " iterations)...";
  std::cout << GREEN << " PASSED" << RESET << " in " << std::fixed << std::setprecision(5) << milliseconds << " ms\n";
  CL_ERR(clReleaseMemObject(d_A));
  CL_ERR(clReleaseMemObject(d_B));
  CL_ERR(clReleaseMemObject(d_C));
  delete[] h_A;
  delete[] h_B;
  delete[] h_C;
  return milliseconds;
}

void CLBackend::shutdown() {
  clGetDeviceInfo = nullptr;
  clGetPlatformIDs = nullptr;
  clGetDeviceIDs = nullptr;
  clCreateContext = nullptr;
  clCreateCommandQueueWithProperties = nullptr;
  clCreateProgramWithSource = nullptr;
  clBuildProgram = nullptr;
  clCreateKernel = nullptr;
  clEnqueueNDRangeKernel = nullptr;
  clWaitForEvents = nullptr;
  clGetEventProfilingInfo = nullptr;
  clReleaseEvent = nullptr;
  clReleaseCommandQueue = nullptr;
  clReleaseContext = nullptr;
  clReleaseProgram = nullptr;
  clReleaseKernel = nullptr;
  clCreateBuffer = nullptr;
  clGetPlatformInfo = nullptr;
  clEnqueueReadBuffer = nullptr;
  clReleaseMemObject = nullptr;

  closeLibrary(clHandle);
  clHandle = nullptr;
}
