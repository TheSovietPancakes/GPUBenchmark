#include "backends/cuda_backend.hpp"
#include "backends/hip_backend.hpp"
#include "backends/opencl_backend.hpp"
#include "backends/vulkan_backend.hpp"
#include "shared/shared.hpp"
#include <iostream>

int main() {
  std::cout << ORCHESTRATOR << "GPU Benchmark starting...\n";

  // CUDA
  if (CudaBackend::init()) {
    CudaBackend::runBenchmark();
    CudaBackend::shutdown();
  }

  // HIP
  if (HIPBackend::init()) {
    HIPBackend::runBenchmark();
    HIPBackend::shutdown();
  }

  // // Vulkan
  // VulkanBackend vk;
  // if (vk.init()) {
  //     vk.runBenchmark();
  //     vk.shutdown();
  // }

  // OpenCL
  if (CLBackend::init()) {
      CLBackend::runBenchmark();
      CLBackend::shutdown();
  }

  std::cout << "All benchmarks done.\n";
  return 0;
}
