cmake_minimum_required(VERSION 3.1...3.31)
project(run LANGUAGES C)
find_package(OpenCL REQUIRED)

add_executable(${PROJECT_NAME} gaussian_blur.c)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenCL::OpenCL)
