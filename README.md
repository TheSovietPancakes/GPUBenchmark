# GPUBenchmark (GPUMark)

## Overview

GPUBenchmark is a cross-platform GPU benchmarking tool written in C++. It supports many different GPUGPU backends, including:

- NVIDIA CUDA
- AMD HIP (though it SAYS it works on NVIDIA too...)
- OpenCL
- Vulkan

The benchmark suite includes a variety of tests to evaluate GPU performance and capabilities. This one executable can run all kinds of tests on all kinds of GPUs, provided you have the right drivers installed.

## Build Instructions

This project uses CMake for building, but realistically you could use literally any C++ compiler you'd like. For this project, however, a CMakeLists.txt file already exists
so we will use that for now.

Please note that all libraries are loaded dynamically, at runtime. This means that you can still compile and run the program even if you don't have all of the prerequisites installed.
This also means that if you want to add or remove a library later, you do not have to recompile the code. (Although you probably should!)

### Prerequisites

CMake version 3.25 or higher.
A C++17 compiler.
CUDA/NVML Drivers for NVIDIA CUDA support.
HIP/RSMI Drivers for AMD HIP support.
OpenCL for OpenCL backend support.
Vulkan SDK for Vulkan backend support.

### Steps to install Prerequisites

#### Windows

Note: When installing all of these, make sure that they are added to PATH where applicable.

1. **CMake**: Download and install CMake from the [official website](https://cmake.org/download/).
2. **CUDA**: Download and install the CUDA Toolkit from the [NVIDIA website](https://developer.nvidia.com/cuda-downloads).
3. **HIP**: Download and install the AMD ROCm toolkit from the [AMD website](https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html).
4. **Vulkan SDK**: Download and install the Vulkan SDK from the [LunarG website](https://vulkan.lunarg.com/sdk/home).
5. **OpenCL**: OpenCL is typically included with your GPU drivers. Ensure you have the latest drivers installed for your GPU.

#### macOS

Apple and NVIDIA hate each other, and modern AMD GPU support is non-existent in favor of modern Apple Silicon on Metal. This means that the only backends that work is Vulkan and OpenCL (which is now deprecated, but still available).

This project attempts to use CUDA/HIP drivers, but realistically they never will. But just know I was thinking of you Mac users!

OpenCL support on Apple Silicon still exists though, so that backend should work fine (despite it being version 1.2). OpenCL is automatically included with macOS.

#### Debian-based

```bash
sudo apt update
sudo apt install cmake build-essential nvidia-cuda-toolkit rocm-dev vulkan-sdk
```

#### Fedora-based

```bash
sudo dnf install cmake gcc-c++ cuda vulkan-sdk rocm-dev
```

#### Arch-based

(I, SovietPancakes, use Arch btw, so I can confirm that this works. I kind of BS'd the other ones tho ngl)

```bash
sudo pacman -Syu cmake gcc cuda rocm-hip-sdk vulkan-icd-loader vulkan-headers ocl-icd opencl-headers
```

For Vulkan suport, you may need to install additional packages depending on your GPU vendor:
Source: [Arch Linux Wiki](https://wiki.archlinux.org/title/Vulkan)

- AMD: `vulkan-radeon` (or `lib32-vulkan-radeon`)
- Intel: `vulkan-intel` (or `lib32-vulkan-intel`)
- NVIDIA: there are two implementations:
    `nvidia-utils` (or `lib32-nvidia-utils`) - NVIDIA proprietary
    `vulkan-nouveau` (or `lib32-vulkan-nouveau`) - NVK (part of Mesa project)

For OpenCL support, you may need to install additional packages depending on your GPU vendor:
Source: [Arch Linux Wiki](https://wiki.archlinux.org/title/general-purpose_computing_on_graphics_processing_units)

- AMD: `rocm-opencl-runtime`, `rocm-opencl-sdk`
- Intel: `intel-compute-runtime`
- NVIDIA: `opencl-nvidia`

### Building the Project with CMake

1. Get a copy of the source code. You can use the Git CLI or the big fat "Code" button on GitHub.

```bash
git clone https://github.com/TheSovietPancakes/GPUBenchmark.git
cd GPUBenchmark
```

2. Setup CMake build files and compile the project.

```bash
mkdir build
cmake -S . -B build
cmake --build build
```

3. Assuming you did not get any errors, the final executable will be in the `build` directory.
Run it via:

```bash
# On Windowsa
.\build\gpumark.exe

# On everything else
./build/gpumark
```

Note that the executable must be run from the terminal. There is no GUI.

## Usage

The program ignores all command line arguments. Follow the instructions in the terminal after launching the program to use.
