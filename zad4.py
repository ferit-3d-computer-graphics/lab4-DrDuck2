import k3d
import numpy as np

size = 32
x, y, z = np.meshgrid(np.linspace(-1.5, 1.5, size),
                     np.linspace(-1.5, 1.5, size),
                     np.linspace(-1.5, 1.5, size),
                     indexing='ij')

r = np.sqrt(x**2 + y**2 + z**2)
field = np.exp(-r**2).astype(np.float32)  

custom_cmap = [
    [0.0, 0.0, 1.0, 1.0],  
    [1.0, 0.0, 0.0, 1.0]   
]

plot = k3d.plot()
surface = k3d.marching_cubes(
    scalar_field=field,
    level=0.5,
    attribute=r.astype(np.float32).flatten(), 
    color_map=custom_cmap,
    color_range=[r.min(), r.max()],           
    wireframe=False,
    flat_shading=False,
    bounds=[-1.5, 1.5, -1.5, 1.5, -1.5, 1.5] 
)

plot += surface
plot.display()

