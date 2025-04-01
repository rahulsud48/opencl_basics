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
   g++ -o gaussian_blur gaussian_blur.cpp -lOpenCL
   ```
3. Run the executable:
   ```sh
   ./gaussian_blur
   ```

## File Structure
```
opencl_basics/
├── src/              # Source code for OpenCL programs
├── include/          # Header files
├── kernels/          # OpenCL kernel files
├── examples/         # Example programs demonstrating OpenCL usage
└── README.md         # Project documentation
```

## Usage
- Modify the kernel files inside the `kernels/` directory to experiment with different parallel computing techniques.
- Use the example programs to understand memory management and performance tuning.

## Troubleshooting
- Ensure that your GPU drivers and OpenCL SDK are correctly installed.
- Use `clinfo` to check available OpenCL devices:
  ```sh
  clinfo
  ```
- If encountering build errors, verify that the OpenCL library path is correctly linked.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Rahul Sud ([GitHub](https://github.com/rahulsud48))

