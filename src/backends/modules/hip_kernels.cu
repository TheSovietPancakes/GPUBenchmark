#include <cmath>
#include <hip/hip_runtime.h>

extern "C" __global__ void linearSetKernel(float* data) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  data[idx] = static_cast<float>(idx);
}

extern "C" __global__ void linearMultiplyKernel(const float* A, const float* B, float* out) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    out[idx] = A[idx] * B[idx];
}

extern "C" __global__ void fmaKernel(float* out, const unsigned int ITERATIONS) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

  float x = 1.0f + idx * 0.0001f; // Cannot be 1 becuase 1 * 1 = 1 :(
  float y = 1.00001f;

  for (unsigned int i = 0; i < ITERATIONS; ++i) {
    x = std::fma(x, y, 1.0f);
  }

  out[idx] = x;
}

extern "C" __global__ void sharedMemoryKernel(float* out, const unsigned int ITERATIONS) {
  __shared__ float tile[4096];
  int tid = threadIdx.x;
  if (tid < 4096) {
    tile[tid] = tid;
    __syncthreads();
  
    for (int i = 0; i < ITERATIONS; i++) {
      tile[tid] = fmaf(tile[tid], 1.0001f, 1.0f);
    }
    __syncthreads();
    out[tid + blockIdx.x * blockDim.x] = tile[tid];
  }
}

extern "C" __global__ void integerThroughputKernel(unsigned int* out, const unsigned int ITERATIONS) {
  unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int v = idx;

  // Better performance (Although technically this is a benchmark so optimization doesn't matter)
  #pragma unroll 1
  for (unsigned int i = 0u; i < ITERATIONS; ++i) {
    v ^= v << 13;
    v ^= v >> 17;
    v ^= v << 5;
  }

  out[idx] = v;
}

extern "C" __global__ void sgemmKernel(const float* A, const float* B, float* C, const unsigned long long N, const unsigned int ITERATIONS) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < N && col < N) {
    float value = 0.0f;
    for (unsigned int iter = 0; iter < ITERATIONS; ++iter) {
      value = 0.0f; // Reset value for each iteration
      for (int k = 0; k < N; ++k) {
        value += A[row * N + k] * B[k * N + col];
      }
    }
    C[row * N + col] = value;
  }
}