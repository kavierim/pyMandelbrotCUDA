__global__ void mandelbrot(float *result, int width, int height, float xmin, float xmax, float ymin, float ymax)
{
    // get grid index
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    // check if grid index is within image bounds
    if (gidx >= width || gidy >= height)
        return;

    // calculate the real and imaginary parts
    float real = xmin + (xmax - xmin) * gidx / (width - 1);
    float imag = ymin + (ymax - ymin) * gidy / (height - 1);

    // initialize 
    float x = 0, y = 0;
    float value = 0;

    // iterate until z goes outside the circle of radius 2
    while (x * x + y * y <= 4 && value < 255)
    {
        // calculate
        float x_new = x * x - y * y + real;
        y = 2 * x * y + imag;
        x = x_new;
        
        // increment value
        value++;
    }

    // convert value to log scale
    value = logf(value); // alternatively, value = value

    // store the result
    result[gidy * width + gidx] = value;
}