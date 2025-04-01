// Standard includes
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// OpenCL Include
#include <CL/cl.h>

#include "commons.h"



// Function to read kernel source from file
char* read_kernel_source(const char* filename) 
{
    FILE* file = fopen(filename, "r");
    if (!file) 
    {
        fprintf(stderr, "Error: Could not open kernel file %s\n", filename);
        return NULL;
    }

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* source = (char*)malloc(length + 1);
    if (!source) 
    {
        fprintf(stderr, "Error: Could not allocate memory for kernel source\n");
        fclose(file);
        return NULL;
    }

    fread(source, 1, length, file);
    source[length] = '\0'; // Null-terminate the string

    fclose(file);
    return source;
}



// Main Code
int main()
{
    //------------------------------------------------------
    // 2. Initialize data on the HOST
    //------------------------------------------------------
    // Image dimensions
    int image_width = 1024;
    int image_height = 1024;
    int image_channels = 1; // Grayscale image

    // Allocate memory for the noisy image
    unsigned char* noisy_image = (unsigned char*)malloc(image_width * image_height * image_channels * sizeof(unsigned char));
    unsigned char* blurred_image_host = (unsigned char*)malloc(image_width * image_height * image_channels * sizeof(unsigned char));
    unsigned char* blurred_image_device = (unsigned char*)malloc(image_width * image_height * image_channels * sizeof(unsigned char));
    
    // Generate the input image
    generate_noisy_image(noisy_image, image_width, image_height, image_channels);



    //------------------------------------------------------
    // 3. Platform and device setup
    //------------------------------------------------------
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);


    // Max Work GroupSize of the device
    size_t max_work_group_size;
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_work_group_size), &max_work_group_size, NULL);

    print_platform_details(platform);
    printDeviceInfo(device);


    //------------------------------------------------------
    // 4. Create a context and command queue
    //------------------------------------------------------
    // Create context and command queue with profiling enabled
    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, NULL);

    //------------------------------------------------------
    // 5. Create memory buffers on the DEVICE
    //------------------------------------------------------
    // Create buffers for input and output
    cl_mem input_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY, image_width * image_height * image_channels * sizeof(cl_uchar), NULL, NULL);
    cl_mem output_buffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, image_width * image_height * image_channels * sizeof(cl_uchar), NULL, NULL);
    cl_mem kernel_buffer = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(gaussian_kernel), (void*)gaussian_kernel, NULL);

    //------------------------------------------------------
    // 6. Write data from HOST to DEVICE
    //------------------------------------------------------
    cl_event write_event;
    clEnqueueWriteBuffer(queue, input_buffer, CL_FALSE, 0, image_width * image_height * image_channels * sizeof(cl_uchar), noisy_image, 0, NULL, &write_event);

    //------------------------------------------------------
    // 7. Build the program and create the kernel
    //------------------------------------------------------
    // Read kernel source from file
    const char* kernel_filename = "../gaussian_blur.cl";
    char* kernel_source = read_kernel_source(kernel_filename);
    if (!kernel_source) 
    {
        fprintf(stderr, "Error: Could not read kernel source from file %s\n", kernel_filename);
        return -1;
    }

    // Compile the OpenCL program
    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, NULL, NULL);
    clBuildProgram(program, 1, &device, NULL, NULL, NULL);
    // Check for build errors
    cl_build_status status;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_STATUS, sizeof(status), &status, NULL);
    if (status != CL_BUILD_SUCCESS) 
    {
        char log[4096];
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(log), log, NULL);
        fprintf(stderr, "Error: Kernel build failed:\n%s\n", log);
        return -1;
    }
    //------------------------------------------------------
    // 8. Set kernel arguments
    //------------------------------------------------------
    cl_kernel kernel = clCreateKernel(program, "gaussian_blur", NULL);
    int kernel_radius = 2;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &input_buffer);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &output_buffer);
    clSetKernelArg(kernel, 2, sizeof(int), &image_width);
    clSetKernelArg(kernel, 3, sizeof(int), &image_height);
    clSetKernelArg(kernel, 4, sizeof(cl_mem), &kernel_buffer);
    clSetKernelArg(kernel, 5, sizeof(int), &kernel_radius); // Kernel radius (5x5 kernel has radius 2)

    //------------------------------------------------------
    // 9. Execute the kernel
    //------------------------------------------------------
    size_t global_work_size[2] = {image_width, image_height};
    size_t local_work_size[2] = {16, 16};      // Number of work-items per workgroup
    cl_event kernel_event;
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global_work_size, local_work_size, 1, &write_event, &kernel_event);
    
    
    //------------------------------------------------------
    // 10. Read the result from DEVICE to HOST
    //------------------------------------------------------
    cl_event read_event;
    clEnqueueReadBuffer(queue, output_buffer, CL_FALSE, 0, image_width * image_height * image_channels * sizeof(cl_uchar), blurred_image_device, 1, &kernel_event, &read_event);
    // Wait for read operation to complete
    clWaitForEvents(1, &read_event);


    //------------------------------------------------------
    // 11. Profiling the Device
    //------------------------------------------------------
    // Profiling: Calculate execution times
    printf("\n######### Device Profiling ################\n");
    cl_ulong start, end;
    double write_time_sec, kernel_time_sec, read_time_sec, total_time_sec_device;
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(write_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    write_time_sec = (end - start) * 1e-9; // Convert from nanoseconds to seconds


    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(kernel_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    kernel_time_sec = (end - start) * 1e-9; // Convert from nanoseconds to seconds
    

    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
    clGetEventProfilingInfo(read_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
    read_time_sec = (end - start) * 1e-9; // Convert from nanoseconds to seconds
    
    // Total Time (Sum of all phases)
    total_time_sec_device = write_time_sec + kernel_time_sec + read_time_sec;
    printf("Data Write Time                                     : %f seconds\n", write_time_sec);
    printf("Kernel Execution Time                               : %f seconds\n", kernel_time_sec);
    printf("Data Read Time                                      : %f seconds\n", read_time_sec); 
    printf("Time taken for Gaussian blur on device              : %f seconds\n", total_time_sec_device);

    // Start time measurement
    printf("\n######### Host Profiling ################\n");
    clock_t start_time = clock();
    // Apply Gaussian blur on host
    gaussian_blur_host(noisy_image, blurred_image_host, image_width, image_height, gaussian_kernel, kernel_radius);
    // End time measurement
    clock_t end_time = clock();
    // Calculate the time taken
    double total_time_sec_host = (double)(end_time - start_time) / CLOCKS_PER_SEC;
    printf("Time taken for Gaussian blur on host                : %f seconds\n", total_time_sec_host);    

    printf("\n######### Comparison: Device vs Host ################\n");
    are_arrays_equal(blurred_image_host, blurred_image_device, image_width * image_height * image_channels);     
    printf("Device is %f times faster than Host \n\n\n\n", (total_time_sec_host/total_time_sec_device));    

    // Clean up
    clReleaseMemObject(input_buffer);
    clReleaseMemObject(output_buffer);
    clReleaseMemObject(kernel_buffer);
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
    free(noisy_image);
    free(blurred_image_device);
    free(blurred_image_host);
    free(kernel_source);

    return 0;
}
