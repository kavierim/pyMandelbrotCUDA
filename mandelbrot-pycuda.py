import mandelbrot
import numpy as np
from PIL import Image as Img
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import keyboard

# Set the centre point as np.double
centre_point = np.double((-0.74, 0.1))
zoom = np.double(1.0)
width = 800
height = 640

maxIterations = 100

# Create a Mandelbrot object
m = mandelbrot.Mandelbrot(width, height)

# Calculate xmin, xmax, ymin, ymax
xmin = np.double(centre_point[0] - 2.0 / zoom)
xmax = np.double(centre_point[0] + 2.0 / zoom)
ymin = np.double(centre_point[1] - 1.5 / zoom)
ymax = np.double(centre_point[1] + 1.5 / zoom)

# Initial results for the Mandelbrot set
result = m.generate(xmin, xmax, ymin, ymax, maxIterations)

# Create a figure and plot the initial Mandelbrot set
fig, ax = plt.subplots()
im = ax.imshow(result, extent=[xmin, xmax, ymin, ymax], interpolation='none', cmap='afmhot') 
# show scale   
plt.colorbar(im)

#show the plot, non blocking
plt.show(block=False)
 
# while loop until space is pressed
while not keyboard.is_pressed('space'):
    # store start time
    start = time.time()

    # if page up or page down is pressed, change the max iterations
    if keyboard.is_pressed('page up'):
        maxIterations = maxIterations + 10
    if keyboard.is_pressed('page down'):
        maxIterations = maxIterations - 10
        if maxIterations < 10:
            maxIterations = 10
  

    # zoom if + ir - is pressed
    if keyboard.is_pressed('+'):
        zoom = np.double(zoom * 1.05)
    if keyboard.is_pressed('-'):
        zoom = np.double(zoom * 0.95)

    #if arrow key is pressed, chnage the centre point
    if keyboard.is_pressed('left'):
        centre_point = (centre_point[0] - 0.1 / zoom, centre_point[1])
    if keyboard.is_pressed('right'):
        centre_point = (centre_point[0] + 0.1 / zoom, centre_point[1])
    if keyboard.is_pressed('down'):
        centre_point = (centre_point[0], centre_point[1] + 0.1 / zoom)
    if keyboard.is_pressed('up'):
        centre_point = (centre_point[0], centre_point[1] - 0.1 / zoom)
    if keyboard.is_pressed('esc'):
        break

    # zoom in on the mandelbrot set
    #zoom = zoom * 1.05
    # Calculate xmin, xmax, ymin, ymax
    xmin = np.double(centre_point[0] - 2.0 / zoom) 
    xmax = np.double(centre_point[0] + 2.0 / zoom)
    ymin = np.double(centre_point[1] - 1.5 / zoom)
    ymax = np.double(centre_point[1] + 1.5 / zoom)

    # Generate the Mandelbrot set
    result = m.generate(xmin, xmax, ymin, ymax, maxIterations)

    #
    im.set_data(result)
    im.set_extent([xmin, xmax, ymin, ymax])
    fig.canvas.draw()
    fig.canvas.flush_events()
    #update scale
    im.set_clim(vmin=result.min(), vmax=result.max())

    # store end time
    end = time.time()
    # print the time taken to generate the mandelbrot set
    print("Refresh rate: ", 1 / (end - start), "Hz", "Zoom: ", zoom, "Centre point: ", centre_point, "Max iterations: ", maxIterations)









