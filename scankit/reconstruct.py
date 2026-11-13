"""`scan reconstruct`: offline reconstruction of a scan folder.

Pipeline (geared towards metric/dimensional accuracy rather than fine detail):
  1. Frame selection: stride, range, blur rejection (per-segment sharpness percentile),
     frames that have depth in --depth-dir.
  2. Depth cleaning: range clamp, flying-pixel / depth-discontinuity removal,
     optional global depth-scale correction (calibrate with an object of known size).
  3. Poses: COLMAP structure-from-motion (scankit/colmap.py, default) or dense tracking with
     fragments, loop closure and pose-graph optimisation (scankit/fragments.py).
  4. Gravity alignment with the IMU, if the scan has imu.csv (world +Z = up).
  5. TSDF integration of all frames with the final poses -> mesh + point cloud.

Every run writes its full config next to the outputs, so different settings can be
compared on the same scan:

    scan reconstruct scans/room -o scans/room/recon/fine --voxel-size 0.003 --depth-max 1.2
    scan reconstruct scans/room --config scans/room/recon/fine/config.json --stride 2
    scan reconstruct scans/room --depth-dir depth_raftstereo   # depth from scan depth
    scan reconstruct --print-config                          # show defaults
"""

from __future__ import annotations

import dataclasses
import json
import os
import time

import numpy as np
import open3d as o3d
import open3d.core as o3c

from .config import Config
from .frames import FrameSource, select_frames
from .imu import gravity_alignment
from .scanfile import ScanReader
from .tsdf import clean_mesh, integrate

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def add_arguments(p):
    p.add_argument("scan", nargs="?", help="scan folder")
    p.add_argument("-o", "--out", help="output directory (default: <scan>/recon/<timestamp>)")
    p.add_argument("--config", help="JSON config file (CLI flags override it)")
    p.add_argument("--view", action="store_true", help="show the result when done")
    p.add_argument("--print-config", action="store_true")
    for f in dataclasses.fields(Config):
        flag = "--" + f.name.replace("_", "-")
        if f.type in ("bool", bool):
            p.add_argument(flag, type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=None,
                           metavar="BOOL")
        elif f.name == "segments":
            p.add_argument(flag, type=int, nargs="*", default=None)
        else:
            typ = {"int": int, "float": float, "str": str}.get(f.type if isinstance(f.type, str) else f.type.__name__)
            p.add_argument(flag, type=typ, default=None)


def make_config(args) -> Config:
    cfg = Config()
    if args.config:
        with open(args.config) as fh:
            data = json.load(fh)
        data = data.get("config", data)
        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)
    for f in dataclasses.fields(Config):
        v = getattr(args, f.name, None)
        if v is not None:
            setattr(cfg, f.name, v)
    return cfg


def get_device(name: str) -> o3c.Device:
    if name == "auto":
        return o3c.Device("CUDA:0" if o3c.cuda.is_available() else "CPU:0")
    return o3c.Device(name)


def reconstruct(scan_path: str, out_dir: str, cfg: Config, log=print):
    dev = get_device(cfg.device)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as fh:
        json.dump({"scan": os.path.abspath(scan_path), "config": dataclasses.asdict(cfg)}, fh, indent=2)
    r = ScanReader(scan_path, cfg.depth_dir)
    src = FrameSource(r, cfg)
    t_start = time.time()
    log(f"[1/5] selecting frames ({len(r)} total, mode={r.info.mode}, depth={cfg.depth_dir}, device={dev})")
    frames = select_frames(r, cfg)
    log(f"  {len(frames)} frames selected")
    if len(frames) < cfg.min_fragment_frames:
        raise SystemExit("Not enough frames.")

    report_extra = {}
    if cfg.pose_method == "colmap":
        from .colmap import colmap_poses
        if len(frames) > cfg.colmap_max_images:
            frames = frames[np.round(np.linspace(0, len(frames) - 1, cfg.colmap_max_images)).astype(int)]
        log(f"[2-4/5] COLMAP structure-from-motion on {len(frames)} frames")
        frame_poses, report_extra = colmap_poses(src, frames, os.path.join(scan_path, "colmap"), log,
                                                 cfg.colmap_matcher,
                                                 cfg.colmap_max_features, reuse=cfg.colmap_cache)
    else:
        from .fragments import track_and_optimize
        frame_poses = track_and_optimize(src, frames, cfg, dev, out_dir, log)
    imu = r.imu() if cfg.gravity_align else None
    if imu is not None:
        A, imu_info = gravity_alignment(r.timestamp, r.segment, frame_poses, imu, log)
        report_extra.update(imu_info)
        if A is not None:
            frame_poses = [(i, A @ T) for i, T in frame_poses]
    np.savez(os.path.join(out_dir, "poses.npz"), frames=np.array([i for i, _ in frame_poses]),
             poses=np.stack([T for _, T in frame_poses]), intrinsics=src.K)

    log(f"[5/5] TSDF integration of {len(frame_poses)} frames (voxel {cfg.voxel_size * 1000:.1f} mm)")
    vbg = integrate(src, frame_poses, cfg, dev, log)
    # Open3D 0.19 mesh and point cloud extraction are exact (tested on a synthetic plane); only
    # ray_cast is offset by about half a voxel (see tracking.py)
    mesh = vbg.extract_triangle_mesh(weight_threshold=cfg.weight_threshold).to_legacy()
    mesh = clean_mesh(mesh, cfg)
    mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.ply"), mesh)
    if cfg.save_pointcloud:
        pcd = vbg.extract_point_cloud(weight_threshold=cfg.weight_threshold).to_legacy()
        o3d.io.write_point_cloud(os.path.join(out_dir, "pointcloud.ply"), pcd)

    report = {
        "frames_total": len(r), "frames_selected": int(len(frames)), "frames_used": len(frame_poses),
        "frames_used_ratio": round(len(frame_poses) / max(1, len(frames)), 3),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
        "seconds": round(time.time() - t_start, 1),
        "pose_method": cfg.pose_method, **report_extra,
    }
    if len(mesh.vertices):
        aabb = mesh.get_axis_aligned_bounding_box()
        obb = mesh.get_minimal_oriented_bounding_box() if hasattr(mesh, "get_minimal_oriented_bounding_box") \
            else mesh.get_oriented_bounding_box()
        report["aabb_extent_m"] = np.round(aabb.get_extent(), 4).tolist()
        report["obb_extent_m"] = np.round(np.sort(obb.extent)[::-1], 4).tolist()
    with open(os.path.join(out_dir, "report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    log(json.dumps(report, indent=2))
    log(f"done -> {out_dir}")
    return mesh


def run(args):
    cfg = make_config(args)
    if args.print_config:
        print(json.dumps(dataclasses.asdict(cfg), indent=2))
        return
    if not args.scan:
        raise SystemExit("scan folder required")
    out = args.out or os.path.join(args.scan, "recon", time.strftime("%Y%m%d_%H%M%S"))
    mesh = reconstruct(args.scan, out, cfg)
    if args.view:
        o3d.visualization.draw_geometries([mesh], window_name=out)
