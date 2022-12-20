# Python class to generate a Mandelbrot set

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import numpy as np

class Mandelbrot:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.result = np.empty((height, width), dtype=np.float32)
        self.result_gpu = drv.mem_alloc(self.result.nbytes)

        # include the kernel code from a separate file
        with open('mandelbrot_kernel.cu', 'r') as f:
            kernel_code = f.read()

        # Compile the kernel code
        mod = SourceModule(kernel_code)

         # Get a reference to the kernel function
        self.mandelbrot_kernel = mod.get_function("mandelbrot")

    def generate(self, xmin, xmax, ymin, ymax):
        
        # Execute the kernel function on the GPU
        self.mandelbrot_kernel(self.result_gpu, np.int32(self.width), np.int32(self.height), np.float32(xmin), np.float32(xmax), np.float32(ymin), np.float32(ymax), block=(16, 16, 1), grid=(self.width // 16, self.height // 16))

        # Copy the result back to the host
        drv.memcpy_dtoh(self.result, self.result_gpu)

        return self.result
