import mandelbrot
import numpy as np
from PIL import Image as Img
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
import keyboard



# Set the centre point as np.double 
centre_point = np.double((-1.0, 0.0))
zoom = np.double(1.0)
width = 800
height = 640

maxIterations = 100

def on_mouse_move(event):
    #print('Mouse move event: ', event.x, event.y)
    #set the centre point to the mouse position
    global centre_point
    global xmin, xmax, ymin, ymax

    # if mouse button event, change the centre point
    if event.button == 1:
        # check that xdata and ydata are valid numbers
        if event.xdata != None and event.ydata != None:
            centre_point = np.double((event.xdata, event.ydata ))

    # if mouse wheel is used, change the zoom
    if event.button == 'up':
        global zoom
        zoom = np.double(zoom * 1.20)
    if event.button == 'down':
        zoom = np.double(zoom * 0.80)
    
    # zoom in on the mandelbrot set
    #zoom = zoom * 1.05
    # Calculate xmin, xmax, ymin, ymax
    xmin = np.double(centre_point[0] - 2.0 / zoom) 
    xmax = np.double(centre_point[0] + 2.0 / zoom)
    ymin = np.double(centre_point[1] - 1.5 / zoom)
    ymax = np.double(centre_point[1] + 1.5 / zoom)


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
im = ax.imshow(result, extent=[xmin, xmax, ymax, ymin], interpolation='none') #, cmap='afmhot') 
# show scale   
plt.colorbar(im)

#show the plot, non blocking
plt.show(block=False)

#create a connection to the mouse move event
cid = fig.canvas.mpl_connect('button_press_event', on_mouse_move)
# mouse wheel event
cid = fig.canvas.mpl_connect('scroll_event', on_mouse_move)


# while loop until space is pressed
while not keyboard.is_pressed('space'):
    # store start time
    start = time.time()

    # if page up or page down is pressed, change the max iterations
    if keyboard.is_pressed('+'):
        maxIterations = maxIterations + 10
    if keyboard.is_pressed('-'):
        maxIterations = maxIterations - 10
        if maxIterations < 10:
            maxIterations = 10
  

    # Generate the Mandelbrot set
    result = m.generate(xmin, xmax, ymin, ymax, maxIterations)

    #
    im.set_data(result)
    im.set_extent([xmin, xmax, ymax, ymin])
    fig.canvas.draw()
    fig.canvas.flush_events()
    #update scale
    im.set_clim(vmin=result.min(), vmax=result.max())

    # store end time
    end = time.time()
    # print the time taken to generate the mandelbrot set
    print("Refresh rate: ", 1 / (end - start), "Hz", "Zoom: ", zoom, "Centre point: ", centre_point, "Max iterations: ", maxIterations)

    







