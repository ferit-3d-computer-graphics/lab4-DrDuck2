import k3d
import numpy as np
import time
import matplotlib.pyplot as plt

plot = k3d.plot()
size = 32 
center = size // 2
current_surface = None

x, y, z = np.indices((size, size, size))

n_frames = 30
radius_range = np.linspace(3, 15, n_frames)

cmap = plt.cm.jet
colors = []
for i in range(n_frames):
    rgba = cmap(i/n_frames, bytes=True)  
    hex_color = f'{rgba[0]:02x}{rgba[1]:02x}{rgba[2]:02x}'  
    colors.append(int(hex_color, 16))  

plot.display()

for frame in range(n_frames):
    radius = radius_range[frame]
    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    field = np.where(r < radius, 1.0, 0.0).astype(np.float32)
    
    new_surface = k3d.marching_cubes(
        field,
        level=0.5,
        color=colors[frame],
        wireframe=False,
        flat_shading=True,
        bounds=[0, size, 0, size, 0, size]
    )
    
    if current_surface:
        plot -= current_surface
    plot += new_surface
    current_surface = new_surface
    
    time.sleep(0.1)