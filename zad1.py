import k3d
import numpy as np

def create_scalar_field(size=32, field_type='gaussian'):
    x, y, z = np.indices((size, size, size))
    center = size // 2
    
    if field_type == 'gaussian':
        r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
        field = np.exp(-r**2 / (2*(size/8)**2))
        
    elif field_type == 'sine_wave':
        field = np.sin(x*0.5) * np.cos(y*0.5) * np.sin(z*0.5)
        
    elif field_type == 'noise':
        field = np.random.rand(size, size, size)
    
    return (field - field.min()) / (field.max() - field.min() + 1e-8)

def visualize_field(field, resolution):
    plot = k3d.plot()
    volume = k3d.volume(
        field.astype(np.float32),
        color_map=k3d.colormaps.basic_color_maps.Jet,
        bounds=[0, resolution, 0, resolution, 0, resolution],
        samples=128
    )
    plot += volume
    plot.display()

configurations = [
    (16, 'gaussian'),
    (32, 'sine_wave'),
    (64, 'noise')
]

for resolution, field_type in configurations:
    print(f"Visualizing {field_type} at {resolution}³ resolution...")
    field = create_scalar_field(resolution, field_type)
    visualize_field(field, resolution)
