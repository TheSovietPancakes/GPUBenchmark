#pragma once

#include <cstddef>
#include <stdint.h>
namespace CLBackend {
static void* clHandle = nullptr;

typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef void* cl_event;
typedef unsigned long cl_queue_properties;

bool init();
bool slowBenchmarks(float linearSetTime, float linearMultiplyTime);
void prepareDeviceForBenchmarking(cl_device_id dev);
void runBenchmark();
void shutdown();

float runLinearSetBenchmark(unsigned int threadsPerBlock, cl_kernel linearSetFunc, cl_context context, cl_command_queue commandQueue);
float runLinearMultiplyBenchmark(unsigned int threadsPerBlock, cl_kernel linearMultiplyFunc, cl_context context, cl_command_queue commandQueue);
float runFmaBenchmark(unsigned int threadsPerBlock, cl_kernel fmaFunc, cl_context context, cl_command_queue commandQueue);
float runIntegerThroughputBenchmark(unsigned int threadsPerBlock, cl_kernel integerThroughputFunc, cl_context context, cl_command_queue commandQueue);
float runSharedMemoryBenchmark(unsigned int threadsPerBlock, cl_kernel sharedMemoryFunc, cl_context context, cl_command_queue commandQueue);
float runSgemmBenchmark(unsigned int threadsPerBlock, cl_kernel sgemmFunc, cl_context context, cl_command_queue commandQueue);

#define CL_DEVICE_GLOBAL_MEM_SIZE 0x101F
#define CL_DEVICE_GLOBAL_MEM_CACHE_SIZE 0x101E
#define CL_DEVICE_NAME 0x102B
#define CL_DEVICE_MAX_WORK_GROUP_SIZE 0x1004
#define CL_PLATFORM_NAME 0x0902
#define CL_DEVICE_TYPE_ALL 0xFFFFFFFF
#define CL_QUEUE_PROPERTIES 0x1093
#define CL_QUEUE_PROFILING_ENABLE (1 << 1)
#define CL_PROFILING_COMMAND_START 0x1282
#define CL_PROFILING_COMMAND_END 0x1283
#define CL_MEM_READ_WRITE (1 << 0)
#define CL_MEM_READ_ONLY (1 << 2)
#define CL_MEM_COPY_HOST_PTR (1 << 5)
#define CL_MEM_WRITE_ONLY (1 << 1)


typedef int (*clGetDeviceInfo_t)(cl_device_id, unsigned int, size_t, void*, size_t*);
typedef cl_kernel (*clCreateKernel_t)(cl_program, const char*, int*);
typedef int (*clGetPlatformIDs_t)(unsigned int, cl_platform_id*, unsigned int*);
typedef int (*clGetDeviceIDs_t)(cl_platform_id, unsigned long, unsigned int, cl_device_id*, unsigned int*);
typedef cl_context (*clCreateContext_t)(const long int*, unsigned int, const cl_device_id*, void(*)(const char*, const void*, size_t, void*), void*,
                                        int*);
typedef cl_command_queue (*clCreateCommandQueueWithProperties_t)(cl_context, cl_device_id, const cl_queue_properties*, int*);
typedef cl_program (*clCreateProgramWithSource_t)(cl_context, unsigned int, const char**, const size_t*, int*);
typedef int (*clBuildProgram_t)(cl_program, unsigned int, const cl_device_id*, const char*, void(*), const void*);
typedef int (*clEnqueueNDRangeKernel_t)(cl_command_queue, cl_kernel, unsigned int, const size_t*, const size_t*, const size_t*, unsigned int,
                                        const void*, void**);
typedef int (*clWaitForEvents_t)(unsigned int, const void**);
typedef int (*clGetEventProfilingInfo_t)(const void*, unsigned int, size_t, void*, size_t*);
typedef int (*clReleaseEvent_t)(void*);
typedef int (*clReleaseCommandQueue_t)(cl_command_queue);
typedef int (*clReleaseContext_t)(cl_context);
typedef int (*clReleaseProgram_t)(cl_program);
typedef int (*clReleaseKernel_t)(cl_kernel);
typedef int (*clGetPlatformInfo_t)(cl_platform_id, unsigned int, size_t, void*, size_t*);
typedef cl_mem (*clCreateBuffer_t)(cl_context, unsigned long, size_t, void*, int*);
typedef int (*clEnqueueReadBuffer_t)(cl_command_queue, cl_mem, unsigned int, size_t, size_t, void*, unsigned int, const void*, void**);
typedef int (*clReleaseMemObject_t)(cl_mem);
typedef int (*clSetKernelArg_t)(cl_kernel, unsigned int, size_t, const void*);

extern clGetDeviceInfo_t clGetDeviceInfo;
extern clGetPlatformIDs_t clGetPlatformIDs;
extern clGetDeviceIDs_t clGetDeviceIDs;
extern clCreateContext_t clCreateContext;
extern clCreateCommandQueueWithProperties_t clCreateCommandQueueWithProperties;
extern clCreateProgramWithSource_t clCreateProgramWithSource;
extern clBuildProgram_t clBuildProgram;
extern clCreateKernel_t clCreateKernel;
extern clEnqueueNDRangeKernel_t clEnqueueNDRangeKernel;
extern clWaitForEvents_t clWaitForEvents;
extern clGetEventProfilingInfo_t clGetEventProfilingInfo;
extern clReleaseEvent_t clReleaseEvent;
extern clReleaseCommandQueue_t clReleaseCommandQueue;
extern clReleaseContext_t clReleaseContext;
extern clReleaseProgram_t clReleaseProgram;
extern clReleaseKernel_t clReleaseKernel;
extern clCreateBuffer_t clCreateBuffer;
extern clGetPlatformInfo_t clGetPlatformInfo;
extern clEnqueueReadBuffer_t clEnqueueReadBuffer;
extern clSetKernelArg_t clSetKernelArg;
extern clReleaseMemObject_t clReleaseMemObject;

} // namespace CLBackend