#pragma once

#include <cstddef>
namespace CudaBackend {
  static void* cudaHandle = nullptr;
  static void* nvmlHandle = nullptr;

  bool init();
  bool gpuUtilizationSafe(void* nvmlDevice);
  bool memUtilizationSafe(void* nvmlDevice);
  unsigned int getAndPrintTemperature(void* nvmlDevice);
  bool slowBenchmarks(float linearSetTime, float linearMultiplyTime);
  void prepareDeviceForBenchmarking(int dev);
  void runBenchmark();
  void shutdown();
  void closeLibrary(void* handle);

  float runLinearSetBenchmark(unsigned int threadsPerBlock, void* kernel);
  float runLinearMultiplyBenchmark(unsigned int threadsPerBlock, void* kernel);
  float runFmaBenchmark(unsigned int threadsPerBlock, void* fmaFunc);
  float runIntegerThroughputBenchmark(unsigned int threadsPerBlock, void* kernel);
  float runSharedMemoryBenchmark(unsigned int threadsPerBlock, void* kernel);

  typedef void* CUfunction;
  typedef void* CUmodule;
  typedef void* CUcontext;
  typedef void* CUstream;
  typedef void* CUevent;
  typedef size_t CUdeviceptr;
  typedef int CUdevice;
  typedef enum { nvmlSuccess = 0 } nvmlReturn_t;
  typedef enum { CUDA_SUCCESS = 0 } CUresult;
  typedef enum {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
  } cudaMemcpyKind;
  typedef void* nvmlDevice_t;
  struct nvmlUtilization_t {
    unsigned int gpu;
    unsigned int memory;
  };
  struct nvmlMemory_t {
    unsigned long long total;
    unsigned long long free;
    unsigned long long used;
  };
  struct cudaUUID_t {
    char bytes[16];
  };
  struct cudaDeviceProp {
    char name[256];                             /**< ASCII string identifying device */
    cudaUUID_t uuid;                            /**< 16-byte unique identifier */
    char luid[8];                               /**< 8-byte locally unique identifier. Value is undefined on TCC and non-Windows platforms */
    unsigned int luidDeviceNodeMask;            /**< LUID device node mask. Value is undefined on TCC and non-Windows platforms */
    size_t totalGlobalMem;                      /**< Global memory available on device in bytes */
    size_t sharedMemPerBlock;                   /**< Shared memory available per block in bytes */
    int regsPerBlock;                           /**< 32-bit registers available per block */
    int warpSize;                               /**< Warp size in threads */
    size_t memPitch;                            /**< Maximum pitch in bytes allowed by memory copies */
    int maxThreadsPerBlock;                     /**< Maximum number of threads per block */
    int maxThreadsDim[3];                       /**< Maximum size of each dimension of a block */
    int maxGridSize[3];                         /**< Maximum size of each dimension of a grid */
    size_t totalConstMem;                       /**< Constant memory available on device in bytes */
    int major;                                  /**< Major compute capability */
    int minor;                                  /**< Minor compute capability */
    size_t textureAlignment;                    /**< Alignment requirement for textures */
    size_t texturePitchAlignment;               /**< Pitch alignment requirement for texture references bound to pitched memory */
    int multiProcessorCount;                    /**< Number of multiprocessors on device */
    int integrated;                             /**< Device is integrated as opposed to discrete */
    int canMapHostMemory;                       /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
    int maxTexture1D;                           /**< Maximum 1D texture size */
    int maxTexture1DMipmap;                     /**< Maximum 1D mipmapped texture size */
    int maxTexture2D[2];                        /**< Maximum 2D texture dimensions */
    int maxTexture2DMipmap[2];                  /**< Maximum 2D mipmapped texture dimensions */
    int maxTexture2DLinear[3];                  /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
    int maxTexture2DGather[2];                  /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
    int maxTexture3D[3];                        /**< Maximum 3D texture dimensions */
    int maxTexture3DAlt[3];                     /**< Maximum alternate 3D texture dimensions */
    int maxTextureCubemap;                      /**< Maximum Cubemap texture dimensions */
    int maxTexture1DLayered[2];                 /**< Maximum 1D layered texture dimensions */
    int maxTexture2DLayered[3];                 /**< Maximum 2D layered texture dimensions */
    int maxTextureCubemapLayered[2];            /**< Maximum Cubemap layered texture dimensions */
    int maxSurface1D;                           /**< Maximum 1D surface size */
    int maxSurface2D[2];                        /**< Maximum 2D surface dimensions */
    int maxSurface3D[3];                        /**< Maximum 3D surface dimensions */
    int maxSurface1DLayered[2];                 /**< Maximum 1D layered surface dimensions */
    int maxSurface2DLayered[3];                 /**< Maximum 2D layered surface dimensions */
    int maxSurfaceCubemap;                      /**< Maximum Cubemap surface dimensions */
    int maxSurfaceCubemapLayered[2];            /**< Maximum Cubemap layered surface dimensions */
    size_t surfaceAlignment;                    /**< Alignment requirements for surfaces */
    int concurrentKernels;                      /**< Device can possibly execute multiple kernels concurrently */
    int ECCEnabled;                             /**< Device has ECC support enabled */
    int pciBusID;                               /**< PCI bus ID of the device */
    int pciDeviceID;                            /**< PCI device ID of the device */
    int pciDomainID;                            /**< PCI domain ID of the device */
    int tccDriver;                              /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
    int asyncEngineCount;                       /**< Number of asynchronous engines */
    int unifiedAddressing;                      /**< Device shares a unified address space with the host */
    int memoryBusWidth;                         /**< Global memory bus width in bits */
    int l2CacheSize;                            /**< Size of L2 cache in bytes */
    int persistingL2CacheMaxSize;               /**< Device's maximum l2 persisting lines capacity setting in bytes */
    int maxThreadsPerMultiProcessor;            /**< Maximum resident threads per multiprocessor */
    int streamPrioritiesSupported;              /**< Device supports stream priorities */
    int globalL1CacheSupported;                 /**< Device supports caching globals in L1 */
    int localL1CacheSupported;                  /**< Device supports caching locals in L1 */
    size_t sharedMemPerMultiprocessor;          /**< Shared memory available per multiprocessor in bytes */
    int regsPerMultiprocessor;                  /**< 32-bit registers available per multiprocessor */
    int managedMemory;                          /**< Device supports allocating managed memory on this system */
    int isMultiGpuBoard;                        /**< Device is on a multi-GPU board */
    int multiGpuBoardGroupID;                   /**< Unique identifier for a group of devices on the same multi-GPU board */
    int hostNativeAtomicSupported;              /**< Link between the device and the host supports native atomic operations */
    int pageableMemoryAccess;                   /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
    int concurrentManagedAccess;                /**< Device can coherently access managed memory concurrently with the CPU */
    int computePreemptionSupported;             /**< Device supports Compute Preemption */
    int canUseHostPointerForRegisteredMem;      /**< Device can access host registered memory at the same virtual address as the CPU */
    int cooperativeLaunch;                      /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
    size_t sharedMemPerBlockOptin;              /**< Per device maximum shared memory per block usable by special opt in */
    int pageableMemoryAccessUsesHostPageTables; /**< Device accesses pageable memory via the host's page tables */
    int directManagedMemAccessFromHost;         /**< Host can directly access managed memory on the device without migration. */
    int maxBlocksPerMultiProcessor;             /**< Maximum number of resident blocks per multiprocessor */
    int accessPolicyMaxWindowSize;              /**< The maximum value of ::cudaAccessPolicyWindow::num_bytes. */
    size_t reservedSharedMemPerBlock;           /**< Shared memory reserved by CUDA driver per block in bytes */
    int hostRegisterSupported;                  /**< Device supports host memory registration via ::cudaHostRegister. */
    int sparseCudaArraySupported;               /**< 1 if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays, 0 otherwise */
    int hostRegisterReadOnlySupported; /**< Device supports using the ::cudaHostRegister flag cudaHostRegisterReadOnly to register memory that must be
                                          mapped as read-only to the GPU */
    int timelineSemaphoreInteropSupported; /**< External timeline semaphore interop is supported on the device */
    int memoryPoolsSupported;              /**< 1 if the device supports using the cudaMallocAsync and cudaMemPool family of APIs, 0 otherwise */
    int gpuDirectRDMASupported;            /**< 1 if the device supports GPUDirect RDMA APIs, 0 otherwise */
    unsigned int gpuDirectRDMAFlushWritesOptions; /**< Bitmask to be interpreted according to the ::cudaFlushGPUDirectRDMAWritesOptions enum */
    int gpuDirectRDMAWritesOrdering;              /**< See the ::cudaGPUDirectRDMAWritesOrdering enum for numerical values */
    unsigned int memoryPoolSupportedHandleTypes;  /**< Bitmask of handle types supported with mempool-based IPC */
    int deferredMappingCudaArraySupported;        /**< 1 if the device supports deferred mapping CUDA arrays and CUDA mipmapped arrays */
    int ipcEventSupported;                        /**< Device supports IPC Events. */
    int clusterLaunch;                            /**< Indicates device supports cluster launch */
    int unifiedFunctionPointers;                  /**< Indicates device supports unified pointers */
    int deviceNumaConfig;                         /**< NUMA configuration of a device: value is of type ::cudaDeviceNumaConfig enum */
    int deviceNumaId;                             /**< NUMA node ID of the GPU memory */
    int mpsEnabled;                               /**< Indicates if contexts created on this device will be shared via MPS */
    int hostNumaId;                               /**< NUMA ID of the host node closest to the device or -1 when system does not support NUMA */
    unsigned int gpuPciDeviceID;                  /**< The combined 16-bit PCI device ID and 16-bit PCI vendor ID */
    unsigned int gpuPciSubsystemID;               /**< The combined 16-bit PCI subsystem ID and 16-bit PCI subsystem vendor ID */
    int hostNumaMultinodeIpcSupported;            /**< 1 if the device supports HostNuma location IPC between nodes in a multi-node system. */
    int reserved[56];                             /**< Reserved for future use */
  };

  typedef CUresult (*cuInit_t)(unsigned int);
  typedef CUresult (*cuMemAlloc_t)(CUdeviceptr*, size_t);
  typedef CUresult (*cuMemFree_t)(CUdeviceptr);
  typedef CUresult (*cuDeviceGetCount_t)(int*);
  typedef CUresult (*cuDeviceGet_t)(CUdevice*, int);
  typedef CUresult (*cuDeviceGetName_t)(char*, int, CUdevice);
  typedef CUresult (*cuCtxCreate_t)(CUcontext*, unsigned int, CUdevice);
  typedef CUresult (*cuCtxDestroy_t)(CUcontext);
  typedef CUresult (*cuCtxSynchronize_t)();
  typedef CUresult (*cuModuleLoadData_t)(CUmodule*, const void*);
  typedef CUresult (*cuModuleUnload_t)(CUmodule);
  typedef CUresult (*cuModuleGetFunction_t)(CUfunction*, CUmodule, const char*);
  typedef CUresult (*cuLaunchKernel_t)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                                                    unsigned int, unsigned int, CUstream, void**, void**);
  typedef CUresult (*cuMemcpyHtoD_t)(CUdeviceptr, const void*, size_t);
  typedef CUresult (*cuMemcpyDtoH_t)(void*, CUdeviceptr, size_t);
  typedef CUresult (*cuMemsetD8_t)(CUdeviceptr, unsigned char, size_t);
  typedef CUresult (*cuEventCreate_t)(CUevent*, unsigned int);
  typedef CUresult (*cuEventRecord_t)(CUevent, CUstream);
  typedef CUresult (*cuEventSynchronize_t)(CUevent);
  typedef CUresult (*cuEventElapsedTime_t)(float*, CUevent, CUevent);
  typedef CUresult (*cuEventDestroy_t)(CUevent);
  typedef CUresult (*cuStreamCreate_t)(CUstream*, unsigned int);
  typedef CUresult (*cuStreamDestroy_t)(CUstream);
  typedef CUresult (*cuDeviceTotalMem_t)(size_t*, CUdevice);
  typedef CUresult (*cuDeviceComputeCapability_t)(int*, int*, CUdevice);
  typedef CUresult (*cuDeviceGetAttribute_t)(int*, int, CUdevice);
  typedef const char* (*cuGetErrorString_t)(CUresult, const char**);

  // ------------------------
  // NVML typedefs
  // ------------------------
  typedef nvmlReturn_t (*nvmlInit_t)();
  typedef nvmlReturn_t (*nvmlShutdown_t)();
  typedef nvmlReturn_t (*nvmlDeviceGetHandleByIndex_t)(unsigned int, nvmlDevice_t*);
  typedef nvmlReturn_t (*nvmlDeviceGetUtilizationRates_t)(nvmlDevice_t, nvmlUtilization_t*);
  typedef nvmlReturn_t (*nvmlDeviceGetTemperature_t)(nvmlDevice_t, unsigned int, unsigned int*);
  typedef nvmlReturn_t (*nvmlDeviceGetMemoryInfo_t)(nvmlDevice_t, nvmlMemory_t*);

  extern cuInit_t cuInit;
  extern cuMemAlloc_t cuMemAlloc;
  extern cuMemFree_t cuMemFree;
  extern cuDeviceGetCount_t cuDeviceGetCount;
  extern cuDeviceGet_t cuDeviceGet;
  extern cuDeviceGetName_t cuDeviceGetName;
  extern cuCtxCreate_t cuCtxCreate;
  extern cuCtxDestroy_t cuCtxDestroy;
  extern cuCtxSynchronize_t cuCtxSynchronize;
  extern cuModuleLoadData_t cuModuleLoadData;
  extern cuModuleUnload_t cuModuleUnload;
  extern cuModuleGetFunction_t cuModuleGetFunction;
  extern cuLaunchKernel_t cuLaunchKernel;
  extern cuMemcpyHtoD_t cuMemcpyHtoD;
  extern cuMemcpyDtoH_t cuMemcpyDtoH;
  extern cuMemsetD8_t cuMemsetD8;
  extern cuEventCreate_t cuEventCreate;
  extern cuEventRecord_t cuEventRecord;
  extern cuEventSynchronize_t cuEventSynchronize;
  extern cuEventElapsedTime_t cuEventElapsedTime;
  extern cuEventDestroy_t cuEventDestroy;
  extern cuStreamCreate_t cuStreamCreate;
  extern cuStreamDestroy_t cuStreamDestroy;
  extern cuDeviceTotalMem_t cuDeviceTotalMem;
  extern cuDeviceComputeCapability_t cuDeviceComputeCapability;
  extern cuDeviceGetAttribute_t cuDeviceGetAttribute;
  
  extern cuGetErrorString_t cuGetErrorString;
  
  // NVML
  extern nvmlInit_t nvmlInit;
  extern nvmlShutdown_t nvmlShutdown;
  extern nvmlDeviceGetHandleByIndex_t nvmlDeviceGetHandleByIndex;
  extern nvmlDeviceGetUtilizationRates_t nvmlDeviceGetUtilizationRates;
  extern nvmlDeviceGetTemperature_t nvmlDeviceGetTemperature;
  extern nvmlDeviceGetMemoryInfo_t nvmlDeviceGetMemoryInfo;
};