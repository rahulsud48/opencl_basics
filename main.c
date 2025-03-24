#include <stdio.h>
#include <stdlib.h>
#include <CL/cl.h>

// OpenCL Kernel (Vector Addition)
const char *kernelSource = 
"__kernel void vec_add(__global int* A, __global int* B, __global int* C, int n) {\n"
"    int id = get_global_id(0);\n"
"    if (id < n) {\n"
"        C[id] = A[id] + B[id];\n"
"    }\n"
"}\n";

#define N 1024  // Define vector size

int main() {
    int A[N], B[N], C[N];
    int numElements = N;  // Use a variable instead of macro
    
    for (int i = 0; i < N; i++) {
        A[i] = i;
        B[i] = i * 2;
    }

    // 1. Get Platform and Device
    cl_platform_id platform;
    cl_device_id device;
    cl_uint numPlatforms, numDevices;
    
    clGetPlatformIDs(1, &platform, &numPlatforms);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &numDevices);

    // 2. Create Context
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);

    // 3. Create Command Queue (FIXED: Using OpenCL 2.0+ API)
    cl_command_queue_properties properties[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, properties, NULL);

    // 4. Create Buffers
    cl_mem d_A = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, A, NULL);
    cl_mem d_B = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int) * N, B, NULL);
    cl_mem d_C = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(int) * N, NULL, NULL);

    // 5. Build Kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "vec_add", NULL);

    // 6. Set Kernel Arguments (FIXED: Using proper variable)
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
    clSetKernelArg(kernel, 3, sizeof(int), &numElements); // Using a proper variable instead of a macro

    // 7. Execute Kernel
    size_t globalSize = N;
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // 8. Read Results
    clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, sizeof(int) * N, C, 0, NULL, NULL);

    // 9. Display Results
    printf("Result: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", C[i]);  // Print first 10 elements
    }
    printf("...\n");

    // 10. Cleanup
    clReleaseMemObject(d_A);
    clReleaseMemObject(d_B);
    clReleaseMemObject(d_C);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
