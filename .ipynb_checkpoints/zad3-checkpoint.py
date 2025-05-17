import k3d
import numpy as np
import time

# 1. Set up initial parameters
size = 32  # Grid resolution
center = size // 2
plot = k3d.plot()
current_surface = None

# 2. Create coordinate grid
x, y, z = np.indices((size, size, size))

# 3. Animation parameters
n_frames = 30
radius_range = np.linspace(3, 12, n_frames)  # Radius in grid units

# 4. Main animation loop
for frame, radius in enumerate(radius_range):
    # Generate scalar field (growing sphere)
    r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
    field = np.where(r < radius, 1.0, 0.0).astype(np.float32)
    
    # Extract isosurface using Marching Cubes
    new_surface = k3d.marching_cubes(
        field,
        level=0.5,
        color=k3d.helpers.map_color(frame, [0, n_frames], k3d.colormaps.basic_color_maps.Jet),
        wireframe=False,
        flat_shading=True
    )
    
    # Update display
    if current_surface:
        plot -= current_surface
    plot += new_surface
    current_surface = new_surface
    
    # Add slight delay for animation
    time.sleep(0.1)

# 5. Display the final plot with interactive controls
plot.display()
