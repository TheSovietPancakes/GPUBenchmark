__kernel void linearSetKernel(__global float* out) {
  uint idx = get_global_id(0);
  out[idx] = (float)idx;
}

__kernel void linearMultiplyKernel(__global const float* A, __global const float* B, __global float* out) {
    uint idx = get_global_id(0); // linear global index
    out[idx] = A[idx] * B[idx];
}

__kernel void fmaKernel(__global float* out,
                        const uint ITERATIONS) {
    uint idx = get_global_id(0);
    float x = 1.0f + idx * 0.0001f;
    float y = 1.00001f;
    
    for (uint i = 0; i < ITERATIONS; ++i) {
        x = fma(x, y, 1.0f); // OpenCL has fma
    }
    
    out[idx] = x;
}

__kernel void sharedMemoryKernel(__global float* out, const uint ITERATIONS) {
    __local float tile[4096];
    uint tid = get_local_id(0);
    
    if (tid < 4096) {
        tile[tid] = tid;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint i = 0; i < ITERATIONS; i++) {
            tile[tid] = fma(tile[tid], 1.0001f, 1.0f);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        uint globalIdx = get_group_id(0) * get_local_size(0) + tid;
        out[globalIdx] = tile[tid];
    }
}

__kernel void integerThroughputKernel(__global uint* out, const uint ITERATIONS) {
    uint idx = get_global_id(0);
    uint v = idx;

    for (uint i = 0; i < ITERATIONS; ++i) {
        v ^= v << 13;
        v ^= v >> 17;
        v ^= v << 5;
    }

    out[idx] = v;
}

__kernel void sgemmKernel(__global const float* A, __global const float* B, __global float* C, const ulong N, const uint reps) {
    int row = get_global_id(1);
    int col = get_global_id(0);

    if (row < N && col < N) {
        float value = 0.0f;
        for (ulong r = 0; r < reps; ++r) {
            value = 0.0f;
            for (int k = 0; k < N; ++k) {
                value += A[row * N + k] * B[k * N + col];
            }
        }
        C[row * N + col] = value;
    }
}

