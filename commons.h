#define STRING_BUFFER_LEN 1024

// Gaussian Blur Kernel (5X5)
const float gaussian_kernel[25] = 
{
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.023792, 0.094907, 0.150342, 0.094907, 0.023792,
    0.015019, 0.059912, 0.094907, 0.059912, 0.015019,
    0.003765, 0.015019, 0.023792, 0.015019, 0.003765
};

// Function to generate a noisy image
void generate_noisy_image(unsigned char* image, int width, int height, int channels) 
{
    srand(time(NULL)); // Seed the random number generator
    for (int i = 0; i < width * height * channels; i++) 
    {
        image[i] = rand() % 256; // Add random noise (grains)
    }
}

void gaussian_blur_host(unsigned char* input, unsigned char* output, int width, int height, const float* mkernel, int kernel_radius) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;

            for (int ky = -kernel_radius; ky <= kernel_radius; ky++) {
                for (int kx = -kernel_radius; kx <= kernel_radius; kx++) {
                    int ix = x + kx;
                    int iy = y + ky;

                    // Handle boundary conditions
                    ix = (ix < 0) ? 0 : (ix >= width) ? width - 1 : ix;
                    iy = (iy < 0) ? 0 : (iy >= height) ? height - 1 : iy;

                    float pixel = input[iy * width + ix];
                    float weight = mkernel[(ky + kernel_radius) * (2 * kernel_radius + 1) + (kx + kernel_radius)];
                    sum += pixel * weight;
                }
            }

            // Clamp the result between 0 and 255 and store it in the output image
            // output[y * width + x] = (unsigned char)((sum < 0.0f) ? 0 : ((sum > 255.0f) ? 255 : sum));
            output[(y * width) + x] = (unsigned char)sum;
        }
    }
}

void are_arrays_equal(unsigned char* array1, unsigned char* array2, size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (array1[i] != array2[i]) {

            printf("Array1[%ld]: %d == Array2[%ld]: %d", i, array1[i], i, array2[i]);
            printf("The arrays are different.\n");
        }
    }
    printf("Gaussian Blur output array matches for both Device and Host.\n");
}

void printDeviceInfo(cl_device_id device) {
    char buffer[STRING_BUFFER_LEN];
    cl_ulong globalMemSize;
    size_t maxWorkGroupSize;
    cl_uint computeUnits;

    printf("\n######### Device Information ################\n");
    // Get and print device name
    clGetDeviceInfo(device, CL_DEVICE_NAME, STRING_BUFFER_LEN, buffer, NULL);
    

    // Get and print device vendor
    clGetDeviceInfo(device, CL_DEVICE_VENDOR, STRING_BUFFER_LEN, buffer, NULL);
    

    // Get and print device version
    clGetDeviceInfo(device, CL_DEVICE_VERSION, STRING_BUFFER_LEN, buffer, NULL);
    

    // Get and print max compute units
    clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, NULL);
    

    // Get and print global memory size
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMemSize), &globalMemSize, NULL);
    

    // Get and print max work group size
    clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, NULL);


    printf("Device Name           : %s\n", buffer);
    printf("Device Vendor         : %s\n", buffer);
    printf("Device Version        : %s\n", buffer);
    printf("Max Compute Units     : %u\n", computeUnits);
    printf("Global Memory Size    : %lu MB\n", globalMemSize / (1024 * 1024));
    printf("Max Work Group Size   : %zu\n", maxWorkGroupSize);
}

void print_platform_details(cl_platform_id platform)
{
    printf("\n######### Platform Information ################\n");
    // Get platform details
    char platformName[1024], platformVendor[1024], platformVersion[1024];

    clGetPlatformInfo(platform, CL_PLATFORM_NAME, sizeof(platformName), platformName, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, sizeof(platformVendor), platformVendor, NULL);
    clGetPlatformInfo(platform, CL_PLATFORM_VERSION, sizeof(platformVersion), platformVersion, NULL);

    // Print platform details
    printf("OpenCL Platform Details:\n");
    printf("Name    : %s\n", platformName);
    printf("Vendor  : %s\n", platformVendor);
    printf("Version : %s\n", platformVersion);
}