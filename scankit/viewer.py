"""`scan view` and `scan measure`: show a reconstruction, measure distances on it."""

from __future__ import annotations

import numpy as np
import open3d as o3d


def view(path: str):
    g = o3d.io.read_triangle_mesh(path)
    if len(g.triangles):
        g.compute_vertex_normals()
    else:
        g = o3d.io.read_point_cloud(path)
    o3d.visualization.draw_geometries([g], window_name=path, mesh_show_back_face=True)


def measure(path: str):
    """Shift + left click to pick points (shift + right click undoes), Q to finish. Prints the
    distances between consecutive points in mm. Use a known length to derive
    --depth-scale-correction (= true length / measured length)."""
    geom = o3d.io.read_point_cloud(path)
    if len(geom.points) == 0 or path.endswith("mesh.ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        geom = o3d.geometry.PointCloud(mesh.vertices)
        geom.colors = mesh.vertex_colors
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window("measure: shift+click points, Q to finish", 1280, 900)
    vis.add_geometry(geom)
    vis.run()
    vis.destroy_window()
    pts = np.asarray(geom.points)[vis.get_picked_points()]
    for k in range(len(pts) - 1):
        print(f"P{k} -> P{k + 1}: {np.linalg.norm(pts[k + 1] - pts[k]) * 1000:.1f} mm")
    if len(pts) > 2:
        print(f"P0 -> P{len(pts) - 1}: {np.linalg.norm(pts[-1] - pts[0]) * 1000:.1f} mm")
