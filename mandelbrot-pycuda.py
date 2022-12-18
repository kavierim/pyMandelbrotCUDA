import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pycuda.compiler import SourceModule
#import time
import logging


# include the kernel code from a separate file
with open('mandelbrot_kernel.cu', 'r') as f:
    kernel_code = f.read()



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('Program Started')

# Compile the kernel code
logging.info('Compiling kernel code')
mod = SourceModule(kernel_code)



# Get a reference to the kernel function
mandelbrot_kernel = mod.get_function("mandelbrot")
logging.info('Kernel code compiled')

# Set the width and height of the image
width, height = 1600, 1200

# Set the minimum and maximum values for the real and imaginary axes
xmin, xmax = -2.0, 1.0
ymin, ymax = -1.0, 1.0

# Create an array to store the result
result = np.empty((height, width), dtype=np.float32)

# Allocate memory on the GPU for the input and output data
result_gpu = drv.mem_alloc(result.nbytes)

# Transfer the input data to the GPU
drv.memcpy_htod(result_gpu, result)

# Execute the kernel function on the GPU
logging.info('Executing kernel function')
mandelbrot_kernel(result_gpu, np.int32(width), np.int32(height), np.float32(xmin), np.float32(xmax), np.float32(ymin), np.float32(ymax), block=(16, 16, 1), grid=(width // 16, height // 16))
logging.info('Kernel function executed')

# Copy the result back to the host
drv.memcpy_dtoh(result, result_gpu)

# Save the result to a file
# result = result.astype(np.uint8)
# result.tofile("mandelbrot.raw")

# Save data to excel file
# df = pd.DataFrame(result)
# df.to_excel("mandelbrot.xlsx")

# Display result in log scale
plt.imshow((result))
# change the color map to temperature
plt.set_cmap('afmhot')
plt.show()

