"""`scan export-o3d`: export a scan folder to the dataset layout of Open3D's reference reconstruction system
(examples/python/reconstruction_system in the Open3D repo, matching your Open3D version).

    scan export-o3d scans/<scan> out_dir [--stride 1] [--segment 0] [--depth-dir depth]
    python <open3d>/examples/python/reconstruction_system/run_system.py \
        --config out_dir/config.json --make --register --refine --integrate

Useful as a reference to compare against `scan reconstruct`.
Note: Open3D's system assumes one continuous sequence; export one segment at a time.
"""

import json
import os

import cv2
import numpy as np
import open3d as o3d

from .scanfile import ScanReader


def add_arguments(p):
    p.add_argument("scan")
    p.add_argument("out")
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--segment", type=int, default=None, help="only export this segment")
    p.add_argument("--depth-dir", default="depth", help="depth | depth_raftstereo | ... (scan depth)")
    p.add_argument("--depth-max", type=float, default=1.5)
    p.add_argument("--voxel-size", type=float, default=0.01)


def run(a):

    r = ScanReader(a.scan, a.depth_dir)
    info = r.info
    idx = np.arange(len(r))
    if a.segment is not None:
        idx = idx[r.segment == a.segment]
    elif len(np.unique(r.segment)) > 1:
        print("WARNING: scan has several segments; Open3D's system treats them as one sequence. "
              "Use --segment N.")
    idx = idx[::a.stride]
    os.makedirs(os.path.join(a.out, "color"), exist_ok=True)
    os.makedirs(os.path.join(a.out, "depth"), exist_ok=True)
    for k, i in enumerate(idx):
        cv2.imwrite(os.path.join(a.out, "color", f"{k:06d}.jpg"), cv2.cvtColor(r.color(i), cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(a.out, "depth", f"{k:06d}.png"), r.depth(i))
    K = info.intrinsics
    intr = o3d.camera.PinholeCameraIntrinsic(info.width, info.height, K[0, 0], K[1, 1], K[0, 2], K[1, 2])
    o3d.io.write_pinhole_camera_intrinsic(os.path.join(a.out, "intrinsic.json"), intr)
    out = os.path.abspath(a.out)
    cfg = {
        "name": os.path.basename(a.scan.rstrip("/")), "path_dataset": out,
        "path_intrinsic": os.path.join(out, "intrinsic.json"),
        "depth_scale": 1.0 / info.depth_unit_m, "max_depth": a.depth_max,
        "voxel_size": a.voxel_size, "max_depth_diff": 0.07,
        "preference_loop_closure_odometry": 0.1, "preference_loop_closure_registration": 5.0,
        "tsdf_cubic_size": 3.0, "icp_method": "color", "global_registration": "ransac",
        "python_multi_threading": True,
    }
    with open(os.path.join(a.out, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    print(f"exported {len(idx)} frames to {a.out}")
