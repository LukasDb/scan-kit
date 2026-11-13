"""TSDF integration of all frames with their final poses, and mesh clean-up."""

from __future__ import annotations

import time

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .config import Config
from .frames import FrameSource


def integrate(src: FrameSource, frame_poses: list, cfg: Config, dev, log):
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=cfg.voxel_size, block_resolution=16, block_count=cfg.block_count, device=dev)
    K = o3c.Tensor(src.K)
    t0 = time.time()
    empty = 0
    for k, (i, T) in enumerate(frame_poses):
        d, c = src.load(i, cfg.depth_max)
        # (almost) nothing within [depth_min, depth_max): Open3D aborts when no block is touched,
        # e.g. for a single pixel at exactly depth_max
        if np.count_nonzero(d) < 100:
            empty += 1
            continue
        depth = o3d.t.geometry.Image(d).to(dev)
        color = o3d.t.geometry.Image(np.ascontiguousarray(c)).to(dev)
        extr = o3c.Tensor(np.linalg.inv(T))
        blocks = vbg.compute_unique_block_coordinates(depth, K, extr, src.depth_scale,
                                                      cfg.depth_max, cfg.trunc_voxel_multiplier)
        vbg.integrate(blocks, depth, color, K, extr, src.depth_scale, cfg.depth_max,
                      cfg.trunc_voxel_multiplier)
        if k % 100 == 0:
            log(f"  integrating {k + 1}/{len(frame_poses)}")
    if empty:
        log(f"  {empty} frames skipped: no depth within {cfg.depth_min}-{cfg.depth_max} m (see --depth-max)")
    log(f"  integration {time.time() - t0:.0f}s")
    return vbg


def clean_mesh(mesh: o3d.geometry.TriangleMesh, cfg: Config):
    if len(mesh.triangles) == 0:
        return mesh
    ids, counts, _ = mesh.cluster_connected_triangles()
    ids, counts = np.asarray(ids), np.asarray(counts)
    if cfg.keep_largest_only:
        remove = ids != counts.argmax()
    else:
        remove = counts[ids] < cfg.min_cluster_fraction * counts.max()
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return mesh
