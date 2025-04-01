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
- A system with an OpenCL-compatible GPU or CPU
- OpenCL SDK installed (e.g., Intel OpenCL SDK, AMD APP SDK, or NVIDIA CUDA Toolkit)
- A C/C++ compiler (GCC, Clang, or MSVC)
- CMake (optional, for build automation)

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
   g++ -o gaussian_blur gaussian_blur.cpp -lOpenCL
   ```
3. Run the executable:
   ```sh
   ./run
   ```

## Author
Rahul Sud ([GitHub](https://github.com/rahulsud48))

