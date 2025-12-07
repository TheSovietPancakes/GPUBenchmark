#pragma once

#include <cstdint>
#include <stddef.h>

namespace HIPBackend {
  static void* hipHandle = nullptr;
  static void* rsmiHandle = nullptr;

  bool init();
  bool gpuUtilizationSafe(int dev);
  bool memUtilizationSafe(int dev);
  long getAndPrintTemperature(int dev);
  bool slowBenchmarks(float linearSetTime, float linearMultiplyTime);
  void prepareDeviceForBenchmarking(int dev);
  void runBenchmark();
  void shutdown();
  void closeLibrary(void *handle);

  typedef struct hipEvent* hipEvent_t;
  typedef struct hipStream* hipStream_t;
  typedef struct hipModule* hipModule_t;
  typedef struct hipFunction* hipFunction_t;
  typedef void* hipDeviceptr_t;

  float runLinearSetBenchmark(unsigned int threadsPerBlock, hipFunction_t linearSetFunc);
  float runLinearMultiplyBenchmark(unsigned int threadsPerBlock, hipFunction_t linearMultiplyFunc);
  float runFmaBenchmark(unsigned int threadsPerBlock, hipFunction_t fmaFunc);
  float runIntegerThroughputBenchmark(unsigned int threadsPerBlock, hipFunction_t integerThroughputFunc);
  float runSharedMemoryBenchmark(unsigned int threadsPerBlock, hipFunction_t sharedMemoryFunc);
  // ------------------------
  // HIP typedefs
  // ------------------------
  typedef enum hipError_t {
    hipSuccess = 0,
    hipErrorInvalidValue = 1,
    hipErrorOutOfMemory = 2,
    hipErrorNotInitialized = 3,
    hipErrorDeinitialized = 4,
    /* ... MANY MORE ... */
    hipErrorUnknown = 999
  } hipError_t;
  typedef enum hipMemcpyKind {
    hipMemcpyHostToHost = 0,
    hipMemcpyHostToDevice = 1,
    hipMemcpyDeviceToHost = 2,
    hipMemcpyDeviceToDevice = 3,
    hipMemcpyDefault = 4
  } hipMemcpyKind;
  typedef struct hipUUID {
    char bytes[16];
  } hipUUID;
  typedef struct {
    // 32-bit Atomics
    unsigned hasGlobalInt32Atomics : 1;
    unsigned hasGlobalFloatAtomicExch : 1;
    unsigned hasSharedInt32Atomics : 1;
    unsigned hasSharedFloatAtomicExch : 1;
    unsigned hasFloatAtomicAdd : 1;
    // 64-bit Atomics
    unsigned hasGlobalInt64Atomics : 1;
    unsigned hasSharedInt64Atomics : 1;
    // Doubles
    unsigned hasDoubles : 1;
    // Warp cross-lane operations
    unsigned hasWarpVote : 1;
    unsigned hasWarpBallot : 1;
    unsigned hasWarpShuffle : 1;
    unsigned hasFunnelShift : 1;
    // Sync
    unsigned hasThreadFenceSystem : 1;
    unsigned hasSyncThreadsExt : 1;
    // Misc
    unsigned hasSurfaceFuncs : 1;
    unsigned has3dGrid : 1;
    unsigned hasDynamicParallelism : 1;
  } hipDeviceArch_t;

  typedef struct hipDeviceProp_t {
    char name[256];
    hipUUID uuid;
    char luid[8];
    unsigned int luidDeviceNodeMask;
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int regsPerBlock;
    int warpSize;
    size_t memPitch;
    int maxThreadsPerBlock;
    int maxThreadsDim[3];
    int maxGridSize[3];
    int clockRate;
    size_t totalConstMem;
    int major;
    int minor;
    size_t textureAlignment;
    size_t texturePitchAlignment;
    int deviceOverlap;
    int multiProcessorCount;
    int kernelExecTimeoutEnabled;
    int integrated;
    int canMapHostMemory;
    int computeMode;
    int maxTexture1D;
    int maxTexture1DMipmap;
    int maxTexture1DLinear;
    int maxTexture2D[2];
    int maxTexture2DMipmap[2];
    int maxTexture2DLinear[3];
    int maxTexture2DGather[2];
    int maxTexture3D[3];
    int maxTexture3DAlt[3];
    int maxTextureCubemap;
    int maxTexture1DLayered[2];
    int maxTexture2DLayered[3];
    int maxTextureCubemapLayered[2];
    int maxSurface1D;
    int maxSurface2D[2];
    int maxSurface3D[3];
    int maxSurface1DLayered[2];
    int maxSurface2DLayered[3];
    int maxSurfaceCubemap;
    int maxSurfaceCubemapLayered[2];
    size_t surfaceAlignment;
    int concurrentKernels;
    int ECCEnabled;
    int pciBusID;
    int pciDeviceID;
    int pciDomainID;
    int tccDriver;
    int asyncEngineCount;
    int unifiedAddressing;
    int memoryClockRate;
    int memoryBusWidth;
    int l2CacheSize;
    int persistingL2CacheMaxSize;
    int maxThreadsPerMultiProcessor;
    int streamPrioritiesSupported;
    int globalL1CacheSupported;
    int localL1CacheSupported;
    size_t sharedMemPerMultiprocessor;
    int regsPerMultiprocessor;
    int managedMemory;
    int isMultiGpuBoard;
    int multiGpuBoardGroupID;
    int hostNativeAtomicSupported;
    int singleToDoublePrecisionPerfRatio;
    int pageableMemoryAccess;
    int concurrentManagedAccess;
    int computePreemptionSupported;
    int canUseHostPointerForRegisteredMem;
    int cooperativeLaunch;
    int cooperativeMultiDeviceLaunch;
    size_t sharedMemPerBlockOptin;
    int pageableMemoryAccessUsesHostPageTables;
    int directManagedMemAccessFromHost;
    int maxBlocksPerMultiProcessor;
    int accessPolicyMaxWindowSize;
    size_t reservedSharedMemPerBlock;
    int hostRegisterSupported;
    int sparseHipArraySupported;
    int hostRegisterReadOnlySupported;
    int timelineSemaphoreInteropSupported;
    int memoryPoolsSupported;
    int gpuDirectRDMASupported;
    unsigned int gpuDirectRDMAFlushWritesOptions;
    int gpuDirectRDMAWritesOrdering;
    unsigned int memoryPoolSupportedHandleTypes;
    int deferredMappingHipArraySupported;
    int ipcEventSupported;
    int clusterLaunch;
    int unifiedFunctionPointers;
    int reserved[63];

    int hipReserved[32];

    /* HIP Only struct members */
    char gcnArchName[256];
    size_t maxSharedMemoryPerMultiProcessor;
    int clockInstructionRate;
    hipDeviceArch_t arch;
    unsigned int* hdpMemFlushCntl;
    unsigned int* hdpRegFlushCntl;
    int cooperativeMultiDeviceUnmatchedFunc;
    int cooperativeMultiDeviceUnmatchedGridDim;
    int cooperativeMultiDeviceUnmatchedBlockDim;
    int cooperativeMultiDeviceUnmatchedSharedMem;
    int isLargeBar;
    int asicRevision;
  } hipDeviceProp_t;

  typedef enum {
    RSMI_STATUS_SUCCESS = 0,
    RSMI_STATUS_INVALID_ARGS = 1,
    RSMI_STATUS_NOT_SUPPORTED = 2,
    RSMI_STATUS_PERMISSION = 3,
    RSMI_STATUS_OUT_OF_RESOURCES = 4,
    RSMI_STATUS_FILE_ERROR = 5,
    RSMI_STATUS_IO_ERROR = 6,
    RSMI_STATUS_TIMEOUT = 7,
    RSMI_STATUS_NO_DATA = 8,
    RSMI_STATUS_INSUFFICIENT_SIZE = 9,
    RSMI_STATUS_UNKNOWN_ERROR = 10
  } rsmi_status_t;
  typedef enum {
    RSMI_TEMP_CURRENT = 0,
    RSMI_TEMP_MAX = 1,
    RSMI_TEMP_MIN = 2,
    RSMI_TEMP_MAX_HYST = 3,
    RSMI_TEMP_MIN_HYST = 4,
    RSMI_TEMP_CRITICAL = 5,
    RSMI_TEMP_CRIT_HYST = 6,
    RSMI_TEMP_EMERGENCY = 7,
    RSMI_TEMP_EMERG_HYST = 8
  } rsmi_temperature_metric_t;
  typedef enum { RSMI_TEMP_TYPE_EDGE, RSMI_TEMP_TYPE_JUNCTION, RSMI_TEMP_TYPE_VRAM } rsmi_temperature_type_t;
  typedef enum { RSMI_MEM_TYPE_VRAM = 0, RSMI_MEM_TYPE_VIS_VRAM = 1, RSMI_MEM_TYPE_GTT = 2 } rsmi_memory_type_t;
  typedef hipError_t (*hipInit_t)(unsigned int);
  typedef hipError_t (*hipDeviceReset_t)();
  typedef hipError_t (*hipSetDevice_t)(int);
  typedef hipError_t (*hipGetDeviceCount_t)(int*);
  typedef hipError_t (*hipGetDevice_t)(int*);
  typedef hipError_t (*hipGetDeviceProperties_t)(hipDeviceProp_t*, int);
  typedef hipError_t (*hipMalloc_t)(void**, size_t);
  typedef hipError_t (*hipFree_t)(void*);
  typedef hipError_t (*hipMemcpy_t)(void*, const void*, size_t, hipMemcpyKind);
  typedef hipError_t (*hipMemset_t)(void*, int, size_t);
  // Events + Streams
  typedef hipError_t (*hipEventCreate_t)(hipEvent_t*);
  typedef hipError_t (*hipEventDestroy_t)(hipEvent_t);
  typedef hipError_t (*hipEventRecord_t)(hipEvent_t, hipStream_t);
  typedef hipError_t (*hipEventSynchronize_t)(hipEvent_t);
  typedef hipError_t (*hipEventElapsedTime_t)(float*, hipEvent_t, hipEvent_t);
  typedef hipError_t (*hipStreamCreate_t)(hipStream_t*);
  typedef hipError_t (*hipStreamDestroy_t)(hipStream_t);
  // Module (like CUDA driver API)
  typedef hipError_t (*hipModuleLoadData_t)(hipModule_t*, const void*);
  typedef hipError_t (*hipModuleUnload_t)(hipModule_t);
  typedef hipError_t (*hipModuleGetFunction_t)(hipFunction_t*, hipModule_t, const char*);
  typedef hipError_t (*hipModuleLaunchKernel_t)(hipFunction_t, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                                unsigned int, hipStream_t, void**, void**);
  // Error
  typedef const char* (*hipGetErrorString_t)(hipError_t);

  // ------------------------
  // RSMI typedefs
  // ------------------------
  typedef rsmi_status_t (*rsmi_init_t)(uint64_t);
  typedef rsmi_status_t (*rsmi_shut_down_t)();
  typedef rsmi_status_t (*rsmi_dev_temp_metric_get_t)(uint32_t, rsmi_temperature_type_t, rsmi_temperature_metric_t, int64_t*);
  typedef rsmi_status_t (*rsmi_dev_memory_total_get_t)(uint32_t, rsmi_memory_type_t, uint64_t*);
  typedef rsmi_status_t (*rsmi_dev_memory_usage_get_t)(uint32_t, rsmi_memory_type_t, uint64_t*);
  typedef rsmi_status_t (*rsmi_dev_name_get_t)(uint32_t, char*, size_t);
  typedef rsmi_status_t (*rsmi_dev_busy_percent_get_t)(uint32_t, uint32_t*);

  // ------------------------
  // Static function pointers
  // ------------------------
  extern hipInit_t hipInit;
  extern hipDeviceReset_t hipDeviceReset;
  extern hipSetDevice_t hipSetDevice;
  extern hipGetDeviceCount_t hipGetDeviceCount;
  extern hipGetDevice_t hipGetDevice;
  extern hipGetDeviceProperties_t hipGetDeviceProperties;
  extern hipMalloc_t hipMalloc;
  extern hipFree_t hipFree;
  extern hipMemcpy_t hipMemcpy;
  extern hipMemset_t hipMemset;
  // Events + Streams
  extern hipEventCreate_t hipEventCreate;
  extern hipEventDestroy_t hipEventDestroy;
  extern hipEventRecord_t hipEventRecord;
  extern hipEventSynchronize_t hipEventSynchronize;
  extern hipEventElapsedTime_t hipEventElapsedTime;
  extern hipStreamCreate_t hipStreamCreate;
  extern hipStreamDestroy_t hipStreamDestroy;
  // Module (like HIP driver API)
  extern hipModuleLoadData_t hipModuleLoadData;
  extern hipModuleUnload_t hipModuleUnload;
  extern hipModuleGetFunction_t hipModuleGetFunction;
  extern hipModuleLaunchKernel_t hipModuleLaunchKernel;
  // Error
  extern hipGetErrorString_t hipGetErrorString;
  // RSMI
  extern rsmi_init_t rsmi_init;
  extern rsmi_shut_down_t rsmi_shut_down;
  extern rsmi_dev_temp_metric_get_t rsmi_dev_temp_metric_get;
  extern rsmi_dev_memory_total_get_t rsmi_dev_memory_total_get;
  extern rsmi_dev_memory_usage_get_t rsmi_dev_memory_usage_get;
  extern rsmi_dev_name_get_t rsmi_dev_name_get;
  extern rsmi_dev_busy_percent_get_t rsmi_dev_busy_percent_get;
};