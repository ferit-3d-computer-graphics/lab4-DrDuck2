import k3d
import numpy as np

size = 32
x, y, z = np.indices((size, size, size))
center = size // 2
r = np.sqrt((x - center)**2 + (y - center)**2 + (z - center)**2)
field = (r / r.max()).astype(np.float32)  

level = 0.5
vertices, faces = k3d.marching_cubes_export(field, level)
vertices = vertices.astype(np.float32)
faces = faces.astype(np.uint32)

vertex_distances = np.sqrt(
    (vertices[:, 0] - center)**2 + 
    (vertices[:, 1] - center)**2 + 
    (vertices[:, 2] - center)**2
)

dist_min, dist_max = vertex_distances.min(), vertex_distances.max()
normalized_dist = (vertex_distances - dist_min) / (dist_max - dist_min)

colors = k3d.helpers.map_color(normalized_dist, k3d.colormaps.paraview_color_maps.Plasma)

plot = k3d.plot()
mesh = k3d.mesh(
    vertices, 
    faces,
    colors=colors,
    wireframe=False,
    flat_shading=False
)
plot += mesh
plot.display()
