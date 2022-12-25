__global__ void mandelbrot(double *result, int width, int height, double xmin, double xmax, double ymin, double ymax, int maxIterations)
{
    // get grid index
    int gidx = blockIdx.x * blockDim.x + threadIdx.x;
    int gidy = blockIdx.y * blockDim.y + threadIdx.y;

    // check if grid index is within image bounds
    if (gidx >= width || gidy >= height)
        return;

    // calculate the real and imaginary parts
    double real = xmin + (xmax - xmin) * gidx / (width - 1);
    double imag = ymin + (ymax - ymin) * gidy / (height - 1);

    // initialize 
    double x = 0, y = 0;
    double value = 0;

    // iterate until z goes outside the circle of radius 2
    while (x * x + y * y <= 4 && value < maxIterations)
    {
        // calculate
        double x_new = x * x - y * y + real;
        y = 2 * x * y + imag;
        x = x_new;
        
        // increment value
        value++;
    }

    // convert value to log scale
    value = logf(value); // alternatively, value = value
    // value = blockIdx.x * blockIdx.y * value;

    // store the result
    result[gidy * width + gidx] = value;
}