#include <stdio.h>
#include <CL/cl.h>

#define STRING_BUFFER_LEN 1024

// OpenCL kernel
const char *kernelSource =
    "__kernel void hello_kernel(__global char* output) {"
    "   const char message[] = \"Hello, World!\\n\";"
    "   int tid = get_global_id(0);"
    "   output[tid] = message[tid];"
    "}";
    
void printDeviceInfo(cl_device_id device) {
    char buffer[STRING_BUFFER_LEN];
    cl_ulong globalMemSize;
    size_t maxWorkGroupSize;
    cl_uint computeUnits;

    // Get and print device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, STRING_BUFFER_LEN, buffer, NULL);
    printf("Device Name: %s\n", buffer);

    // Get and print device vendor
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, STRING_BUFFER_LEN, buffer, NULL);
    printf("Device Vendor: %s\n", buffer);

    // Get and print device version
    clGetDeviceInfo(device, CL_DEVICE_VERSION, STRING_BUFFER_LEN, buffer, NULL);
    printf("Device Version: %s\n", buffer);

    // Get and print max compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    printf("Max Compute Units: %u\n", computeUnits);

    // Get and print global memory size
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
    printf("Global Memory Size: %lu MB\n", globalMemSize / (1024 * 1024));

    // Get and print max work group size
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);
    printf("Max Work Group Size: %zu\n", maxWorkGroupSize);
}

int main() {
    // Step 1: Get Platform and Device
    cl_platform_id platform;
    cl_device_id device;
    clGetPlatformIDs(1, &platform, NULL);
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    // Print device information
    printDeviceInfo(device);

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
