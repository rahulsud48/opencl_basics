#include <stdio.h>
#include <CL/cl.h>

// OpenCL kernel
const char *kernelSource =
    "__kernel void hello_kernel(__global char* output) {"
    "   const char message[] = \"Hello, World!\\n\";"
    "   int tid = get_global_id(0);"
    "   output[tid] = message[tid];"
    "}";

int main() {
    // Step 1: Get Platform and Device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Step 2: Create OpenCL Context and Command Queue
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

    // Step 3: Create Memory Buffer
    char output[14] = {0}; // To store "Hello, World!\n"
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(output), NULL, NULL);

    // Step 4: Compile Kernel
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "hello_kernel", NULL);

    // Step 5: Set Kernel Arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &outputBuffer);

    // Step 6: Enqueue Kernel Execution
    size_t globalSize = sizeof(output);
    clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL, NULL);

    // Step 7: Ensure all commands finish
    clFinish(queue); // Wait for all enqueued tasks (kernel execution) to complete

    // Step 8: Read and Print Output
    clEnqueueReadBuffer(queue, outputBuffer, CL_TRUE, 0, sizeof(output), output, 0, NULL, NULL);
    printf("%s", output);

    // Step 9: Cleanup
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseMemObject(outputBuffer);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
