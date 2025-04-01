__kernel void gaussian_blur(__global const uchar* input, __global uchar* output, int width, int height, __constant float* mkernel, int kernel_radius) 
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x >= width || y >= height) 
		return;

    float sum = 0.0f;
    for (int ky = -kernel_radius; ky <= kernel_radius; ky++) 
	{
        for (int kx = -kernel_radius; kx <= kernel_radius; kx++) 
		{
            int ix = x + kx;
            int iy = y + ky;

            // Handle boundary conditions
            if (ix < 0) ix = 0;
            if (iy < 0) iy = 0;
            if (ix >= width) ix = width - 1;
            if (iy >= height) iy = height - 1;

            float pixel = input[(iy * width) + ix];
            float weight = mkernel[(ky + kernel_radius) * ((2 * kernel_radius) + 1) + (kx + kernel_radius)];
            sum += pixel * weight;
        }
    }
    output[(y * width) + x] = (uchar)sum;
}