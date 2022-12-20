import mandelbrot
import numpy as np
import matplotlib.pyplot as plt


m = mandelbrot.Mandelbrot(1600, 1200)
result = m.generate(-2.0, 1.0, -1.0, 1.0)

# Display result in log scale
plt.imshow((result))
# change the color map to temperature
plt.set_cmap('afmhot')
plt.show()

