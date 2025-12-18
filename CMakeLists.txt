cmake_minimum_required(VERSION 3.25)
project(GPUBenchmark VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SOURCES
  src/main.cpp
  src/shared/shared.cpp
  src/backends/cuda_backend.cpp
  src/backends/hip_backend.cpp
  src/backends/vulkan_backend.cpp
  src/backends/opencl_backend.cpp
  src/backends/opengl_backend.cpp
)

if(UNIX AND NOT APPLE)
  set(SOURCES ${SOURCES}
    src/backends/linux/cuda_backend.cpp
    src/backends/linux/hip_backend.cpp
    src/backends/linux/opencl_backend.cpp
    src/backends/linux/shared.cpp
  )
elseif(APPLE)
  set(SOURCES ${SOURCES}
    src/backends/macos/cuda_backend.cpp
    src/backends/macos/hip_backend.cpp
    src/backends/macos/opencl_backend.cpp
    src/backends/macos/shared.cpp
  )
elseif(WIN32)
  set(SOURCES ${SOURCES}
    src/backends/windows/cuda_backend.cpp
    src/backends/windows/hip_backend.cpp
    src/backends/windows/opencl_backend.cpp
    src/backends/windows/shared.cpp
  )
endif()

add_executable(gpumark ${SOURCES})
find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

if(UNIX)
  target_link_libraries(gpumark PRIVATE dl)
endif()

target_link_libraries(gpumark PRIVATE Threads::Threads EGL GL glfw)

# Optimization/stack protector
target_compile_options(gpumark PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang>:-O2 -fstack-protector-strong>
  $<$<CXX_COMPILER_ID:MSVC>:/O2 /GS>
)

set(CMAKE_SKIP_RPATH TRUE)
# Run on changes to backend/hip_kernels.cu:
find_program(HIPCC_EXECUTABLE hipcc)

if(HIPCC_EXECUTABLE)
  set(HIP_KERNELS_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/hip_kernels.cu)
  set(HIP_KERNELS_OUT ${CMAKE_BINARY_DIR}/hip_kernels.hsaco)
  set(HIP_KERNELS_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/hip_kernels.hpp)

  add_custom_command(
    OUTPUT ${HIP_KERNELS_HPP}
    COMMAND ${HIPCC_EXECUTABLE} -fgpu-rdc --genco ${HIP_KERNELS_SRC} -o ${HIP_KERNELS_OUT}
    COMMAND ${CMAKE_COMMAND}
    -DINPUT=${HIP_KERNELS_OUT}
    -DOUTPUT=${HIP_KERNELS_HPP}
    -DSYMBOL=hip_kernels_hsaco
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedBinary.cmake
    DEPENDS ${HIP_KERNELS_SRC}
    COMMENT "Building HIP kernels"
  )

  add_custom_target(hip_kernels ALL DEPENDS ${HIP_KERNELS_HPP})
  add_dependencies(gpumark hip_kernels)
else()
  # Generate a dummy hip_kernels.hpp to avoid build errors
  set(HIP_KERNELS_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/hip_kernels.hpp)
  file(WRITE ${HIP_KERNELS_HPP} "// Dummy hip_kernels.hpp generated because hipcc was not found.\n")
  file(APPEND ${HIP_KERNELS_HPP} "const unsigned char hip_kernels_hsaco[] = {};\n")
endif()


find_program(NVCC_EXECUTABLE nvcc)
if(NVCC_EXECUTABLE)

  set(CUDA_KERNELS_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/cuda_kernels.cu)
  set(CUDA_KERNELS_OUT ${CMAKE_BINARY_DIR}/cuda_kernels.ptx)
  set(CUDA_KERNELS_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/cuda_kernels.hpp)

  add_custom_command(
    OUTPUT ${CUDA_KERNELS_HPP}
    COMMAND ${NVCC_EXECUTABLE} -ptx ${CUDA_KERNELS_SRC} -o ${CUDA_KERNELS_OUT}
    COMMAND ${CMAKE_COMMAND}
    -DINPUT=${CUDA_KERNELS_OUT}
    -DOUTPUT=${CUDA_KERNELS_HPP}
    -DSYMBOL=cudaKernels_ptx
    -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
    DEPENDS ${CUDA_KERNELS_SRC}
  )


  add_custom_target(cuda_kernels ALL DEPENDS ${CUDA_KERNELS_HPP})
  add_dependencies(gpumark cuda_kernels)
else()
  # Generate a dummy cuda_kernels.hpp to avoid build errors
  set(CUDA_KERNELS_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/cuda_kernels.hpp)
  file(WRITE ${CUDA_KERNELS_HPP} "// Dummy cuda_kernels.hpp generated because nvcc was not found.\n")
  file(APPEND ${CUDA_KERNELS_HPP} "const char cudaKernels_ptx[] = \"\";\n")
endif()

# OpenCL kernels are not precompiled, but we can still track changes
set(OPENCL_KERNELS_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/opencl_kernels.cl)
set(OPENCL_KERNELS_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/opencl_kernels.hpp)

# OpenCL copy
add_custom_command(
  OUTPUT ${OPENCL_KERNELS_HPP}
  COMMAND ${CMAKE_COMMAND}
  -DINPUT=${OPENCL_KERNELS_SRC}
  -DOUTPUT=${OPENCL_KERNELS_HPP}
  -DSYMBOL=opencl_kernels_cl
  -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${OPENCL_KERNELS_SRC}
)
add_custom_target(opencl_kernels ALL DEPENDS ${OPENCL_KERNELS_HPP})

# OpenGL copies
# 1. triangleVer.glsl -> triangleVer.hpp
set(OPENGL_TRIANGLE_VERT_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/triangleVer.glsl)
set(OPENGL_TRIANGLE_VERT_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/triangleVer.hpp)
add_custom_command(
  OUTPUT ${OPENGL_TRIANGLE_VERT_HPP}
  COMMAND ${CMAKE_COMMAND}
  -DINPUT=${OPENGL_TRIANGLE_VERT_SRC}
  -DOUTPUT=${OPENGL_TRIANGLE_VERT_HPP}
  -DSYMBOL=triangleVert_src
  -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${OPENGL_TRIANGLE_VERT_SRC}
)
add_custom_target(opengl_triangle_vert ALL DEPENDS ${OPENGL_TRIANGLE_VERT_HPP})
add_dependencies(gpumark opengl_triangle_vert)
# 2. triangleFrag.glsl -> triangleFrag.hpp
set(OPENGL_TRIANGLE_FRAG_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/triangleFrag.glsl)
set(OPENGL_TRIANGLE_FRAG_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/triangleFrag.hpp)
add_custom_command(
  OUTPUT ${OPENGL_TRIANGLE_FRAG_HPP}
  COMMAND ${CMAKE_COMMAND}
  -DINPUT=${OPENGL_TRIANGLE_FRAG_SRC}
  -DOUTPUT=${OPENGL_TRIANGLE_FRAG_HPP}
  -DSYMBOL=triangleFrag_src
  -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${OPENGL_TRIANGLE_FRAG_SRC}
)
add_custom_target(opengl_triangle_frag ALL DEPENDS ${OPENGL_TRIANGLE_FRAG_HPP})
add_dependencies(gpumark opengl_triangle_frag)
# 3. aluHeavyFrag.glsl -> aluHeavyFrag.hpp
set(OPENGL_ALUHEAVY_FRAG_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/aluHeavyFrag.glsl)
set(OPENGL_ALUHEAVY_FRAG_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/aluHeavyFrag.hpp)
add_custom_command(
  OUTPUT ${OPENGL_ALUHEAVY_FRAG_HPP}
  COMMAND ${CMAKE_COMMAND}
  -DINPUT=${OPENGL_ALUHEAVY_FRAG_SRC}
  -DOUTPUT=${OPENGL_ALUHEAVY_FRAG_HPP}
  -DSYMBOL=aluHeavyFrag_src
  -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${OPENGL_ALUHEAVY_FRAG_SRC}
)
add_custom_target(opengl_aluheavy_frag ALL DEPENDS ${OPENGL_ALUHEAVY_FRAG_HPP})
add_dependencies(gpumark opengl_aluheavy_frag)
# 3. memHeavyFrag.glsl -> memHeavyFrag.hpp
set(OPENGL_MEMHEAVY_FRAG_SRC ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/memHeavyFrag.glsl)
set(OPENGL_MEMHEAVY_FRAG_HPP ${CMAKE_SOURCE_DIR}/src/backends/modules/gl_backends/memHeavyFrag.hpp)
add_custom_command(
  OUTPUT ${OPENGL_MEMHEAVY_FRAG_HPP}
  COMMAND ${CMAKE_COMMAND}
  -DINPUT=${OPENGL_MEMHEAVY_FRAG_SRC}
  -DOUTPUT=${OPENGL_MEMHEAVY_FRAG_HPP}
  -DSYMBOL=memHeavyFrag_src
  -P ${CMAKE_SOURCE_DIR}/cmake/EmbedText.cmake
  DEPENDS ${OPENGL_MEMHEAVY_FRAG_SRC}
)
add_custom_target(opengl_memheavy_frag ALL DEPENDS ${OPENGL_MEMHEAVY_FRAG_HPP})
add_dependencies(gpumark opengl_memheavy_frag)

add_dependencies(gpumark opencl_kernels)
