#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

#define STRING_BUFFER_LEN 1024

//-------------------
// 1. Kernel Code
//-------------------
const char* kernelSource =
"__kernel void vector_add(__global const float* A,          \n"
"                         __global const float* B,          \n"
"                         __global float* C,                \n"
"                         __local float* localA,            \n"
"                         __local float* localB)            \n"
"{                                                          \n"
"    int global_id = get_global_id(0);                      \n"
"    int local_id  = get_local_id(0);                       \n"
"    int group_size = get_local_size(0);                    \n"
"                                                          \n"
"    // Load from global memory into local memory           \n"
"    localA[local_id] = A[global_id];                       \n"
"    localB[local_id] = B[global_id];                       \n"
"                                                          \n"
"    // Synchronize to make sure data is in local memory    \n"
"    barrier(CLK_LOCAL_MEM_FENCE);                          \n"
"                                                          \n"
"    // Perform the addition                                \n"
"    C[global_id] = localA[local_id] + localB[local_id];    \n"
"}                                                          \n";



int main()
{
    //------------------------------------------------------
    // 2. Initialize data on the HOST
    //------------------------------------------------------
    int N = 1024; // Number of elements
    float* A = (float*)malloc(N * sizeof(float));
    float* B = (float*)malloc(N * sizeof(float));
    float* C = (float*)malloc(N * sizeof(float));

    // Initialize A and B
    for (int i = 0; i < N; i++) {
        A[i] = (float)i;
        B[i] = (float)i;
    }

    //------------------------------------------------------
    // 3. Platform and device setup
    //------------------------------------------------------
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numPlatforms, numDevices;
    cl_int err;

    // Get the platform
    err = clGetPlatformIDs(1, &platform, &numPlatforms);
    if (err != CL_SUCCESS) {
        printf("Failed to find an OpenCL platform!\n");
        return -1;
    }

    // Get a GPU device (or CPU if no GPU)
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);
    if (err != CL_SUCCESS) {
        printf("Failed to get an OpenCL GPU device!\n");
        return -1;
    }

    //------------------------------------------------------
    // 4. Create a context and command queue
    //------------------------------------------------------
    //This binds the context to the specified device, allowing memory allocation, kernel execution, and command queue creation.
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    if (!context) {
        printf("Failed to create OpenCL context!\n");
        return -1;
    }

    /*  Creates a command queue to submit tasks to the GPU/CPU/NPU. Binds the queue to the specified device inside the context.
        Commands like memory transfer, kernel execution, and synchronization will be sent through this queue. */
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    if (!queue) {
        printf("Failed to create command queue!\n");
        clReleaseContext(context);
        return -1;
    }

    //------------------------------------------------------
    // 5. Create memory buffers on the DEVICE
    //------------------------------------------------------
    // Allocating memory for variables in the Global Memory --> VRAM
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,  N * sizeof(float), NULL, &err);
    cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, &err);

    //------------------------------------------------------
    // 6. Write data from HOST to DEVICE
    //------------------------------------------------------
    err = clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, N * sizeof(float), A, 0, NULL, NULL);
    err = clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, N * sizeof(float), B, 0, NULL, NULL);

    //------------------------------------------------------
    // 7. Build the program and create the kernel
    //------------------------------------------------------
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, &err);
    if (!program) {
        printf("Failed to create CL program from source.\n");
        return -1;
    }

    err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        // Print build error log
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        printf("Build Log:\n%s\n", log);
        return -1;
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    if (!kernel || err != CL_SUCCESS) {
        printf("Failed to create kernel.\n");
        return -1;
    }

    //------------------------------------------------------
    // 8. Set kernel arguments
    //------------------------------------------------------
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferC);

    // We'll allocate local memory for localA and localB.
    // Let's define the local size (work-group size).
    size_t localSize = 64;
    err |= clSetKernelArg(kernel, 3, localSize * sizeof(float), NULL);  // localA
    err |= clSetKernelArg(kernel, 4, localSize * sizeof(float), NULL);  // localB

    if (err != CL_SUCCESS) {
        printf("Failed to set kernel arguments.\n");
        return -1;
    }

    //------------------------------------------------------
    // 9. Execute the kernel
    //------------------------------------------------------
    size_t globalSize = N;
    cl_event kernel_event;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize, 0, NULL, &kernel_event);
    if (err != CL_SUCCESS) {
        printf("Failed to enqueue kernel.\n");
        return -1;
    }

    // Wait for the kernel to finish
    clWaitForEvents(1, &kernel_event);

    //------------------------------------------------------
    // 10. Read the result from DEVICE to HOST
    //------------------------------------------------------
    err = clEnqueueReadBuffer(queue, bufferC, CL_TRUE, 0, N * sizeof(float), C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("Failed to read bufferC.\n");
        return -1;
    }

    //------------------------------------------------------
    // 11. Print the result (first 10 elements)
    //------------------------------------------------------
    printf("First 10 results:\n");
    for (int i = 0; i < 10; i++) {
        printf("C[%d] = %f\n", i, C[i]);
    }

    //------------------------------------------------------
    // 12. Cleanup
    //------------------------------------------------------
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferC);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(A);
    free(B);
    free(C);

    return 0;
}
