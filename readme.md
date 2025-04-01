# OpenCL Basics

## Overview
This repository provides fundamental examples and implementations to help understand and utilize OpenCL for parallel computing. The code demonstrates how to set up an OpenCL environment, create kernels, manage memory, and execute basic computations on the GPU.

## Features
- Initialization of OpenCL platforms and devices
- Creating and building OpenCL programs
- Memory management using OpenCL buffers
- Executing kernels for basic computations
- Gaussian blur implementation using OpenCL

## Prerequisites
To run the OpenCL programs in this repository, ensure you have the following:
- A system with an OpenCL-compatible GPU
- OpenCL SDK installed for NVIDIA CUDA Toolkit
- A C/C++ compiler - gcc
- CMake 

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/rahulsud48/opencl_basics.git
   cd opencl_basics
   ```
2. Compile the code (example using `g++`):
   ```sh
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the executable:
   ```sh
   ./run
   ```
## Sample Execution Log
```
######### Platform Information ################
OpenCL Platform Details:
Name    : NVIDIA CUDA
Vendor  : NVIDIA Corporation
Version : OpenCL 3.0 CUDA 11.4.364

######### Device Information ################
Device Name           : OpenCL 3.0 CUDA
Device Vendor         : OpenCL 3.0 CUDA
Device Version        : OpenCL 3.0 CUDA
Max Compute Units     : 68
Global Memory Size    : 11016 MB
Max Work Group Size   : 1024

######### Device Profiling ################
Data Write Time                                     : 0.000081 seconds
Kernel Execution Time                               : 0.000065 seconds
Data Read Time                                      : 0.000081 seconds
Time taken for Gaussian blur on device              : 0.000227 seconds

######### Host Profiling ################
Time taken for Gaussian blur on host                : 0.139576 seconds

######### Comparison: Device vs Host ################
Gaussian Blur output array matches for both Device and Host.
Device is 613.639561 times faster than Host
```

## Author
Rahul Sud ([GitHub](https://github.com/rahulsud48))

