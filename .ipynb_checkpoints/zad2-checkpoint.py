import k3d
import numpy as np

size = 32
x, y, z = np.indices((size, size, size))
center = size // 2
r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
field = np.exp(-r**2 / (size/4)**2)  
field = (field - field.min()) / (field.max() - field.min()) 

plot = k3d.plot()

surfaces = [
    (0.3, 0xFF0000),  # Red
    (0.5, 0x00FF00),  # Green
    (0.7, 0x0000FF)   # Blue
]

for level, color in surfaces:
    surface = k3d.marching_cubes(
        field.astype(np.float32),
        level=level,
        color=color,
        wireframe=False,
        flat_shading=True,
        opacity=0.7,
        bounds=[0, size, 0, size, 0, size]
    )
    plot += surface

plot.grid = [0, 0, 0, size, size, size]
plot.camera = [60, 60, 60, center, center, center, 0, 0, 1]
plot.display()
