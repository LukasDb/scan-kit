"""`scan depth`: depth from the recorded rectified stereo pairs of a scan, for any camera.

Needs a scan with pair/ (ZED Mini, OAK mono mode).
Writes <scan>/depth_<method>/000000.png: 16-bit mm, aligned with images/, the same format as
the camera depth in depth/. Frames that already have a depth file are skipped (resume).

Row correction (default on): imperfect rectification leaves the two views vertically offset,
on the OAK-D by up to 4 px at the image edges. The static offset field is estimated from SIFT
matches over the scan and removed from the other view before matching, so the vertical-flow
check keeps the image borders. The fit is stored in params.json.

    scan depth scans/room                          # RAFT-Stereo on CUDA / MPS / CPU
    scan depth scans/room --stride 3 --scale 0.5   # faster
    scan depth scans/room --method raft            # RAFT optical flow (slower)
    scan depth scans/room --method sgbm            # classical baseline
    scan reconstruct scans/room --depth-dir depth_raftstereo
"""

from __future__ import annotations

import json
import os
import time

import cv2
import numpy as np

from .scanfile import ScanReader
from .stereo import METHODS, make_matcher, row_correction, stereo_depth


def add_arguments(p):
    p.add_argument("scan")
    p.add_argument("--method", default="raftstereo", choices=METHODS)
    p.add_argument("--out", help="output folder name in the scan (default: depth_<method>)")
    p.add_argument("--stride", type=int, default=1, help="every N-th frame")
    p.add_argument("--scale", type=float, default=1.0, help="image scale for matching (0.5 = 4x faster)")
    p.add_argument("--iters", type=int, help="refinement iterations (default: raftstereo 7, raft 20)")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu (raftstereo, raft)")
    p.add_argument("--depth-min", type=float, default=0.1, help="m")
    p.add_argument("--depth-max", type=float, default=5.0, help="m")
    p.add_argument("--lr-tol", type=float, default=1.0, help="left-right check tolerance [px]")
    p.add_argument("--row-correction", type=lambda s: s.lower() in ("1", "true", "yes", "on"), default=True,
                   metavar="BOOL", help="remove the static vertical misalignment of the pair (default on)")
    p.add_argument("--overwrite", action="store_true")


def run(a):

    r = ScanReader(a.scan)
    if not r.stereo:
        raise SystemExit(f"{a.scan} has no stereo pair (scan.json has no 'stereo'). Record with the "
                         "ZED Mini, or OAK mono mode.")
    ref_is_left = r.stereo["pair_side"] == "right"
    fx, baseline = float(r.info.intrinsics[0, 0]), float(r.stereo["baseline_m"])
    out = os.path.join(a.scan, a.out or f"depth_{a.method}")
    os.makedirs(out, exist_ok=True)
    opts = {k: v for k, v in vars(a).items() if k not in ("func", "command")}
    params = {**opts, "fx": fx, "baseline_m": baseline, "reference_view": "left" if ref_is_left else "right"}
    maps = None
    if a.row_correction:
        sample = [i for i in np.linspace(0, len(r) - 1, 25).astype(int) if r.pair(i) is not None]
        mx, my, info = row_correction([(r.color(i), r.pair(i)) for i in sample], ref_is_left,
                                      (r.info.width, r.info.height))
        maps = (mx, my)
        params["row_correction"] = info
        print(f"row correction from {len(sample)} frames, {info['inliers']} matches: |dy| median "
              f"{info['before_median_abs_dy_px']} -> {info['after_median_abs_dy_px']} px, "
              f"field {info['field_min_max_px'][0]:+.2f}..{info['field_min_max_px'][1]:+.2f} px")
    with open(os.path.join(out, "params.json"), "w") as fh:
        json.dump(params, fh, indent=1)
    print(f"{len(r)} frames, baseline {baseline * 1000:.2f} mm, fx {fx:.1f}, reference view "
          f"{params['reference_view']}, method {a.method} -> {out}")

    num_disp = int(fx * a.scale * baseline / a.depth_min) + 16
    matcher = make_matcher(a.method, a.device, a.iters, min(num_disp, 256))
    todo = [i for i in range(0, len(r), a.stride)
            if a.overwrite or not os.path.exists(os.path.join(out, r.names[i][:-4] + ".png"))]
    t0, missing = time.time(), 0
    for k, i in enumerate(todo):
        other = r.pair(i)
        if other is None:
            missing += 1
            continue
        if maps is not None:
            other = cv2.remap(other, *maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        z = stereo_depth(matcher, r.color(i), other, ref_is_left, fx, baseline, a.depth_min, a.depth_max,
                         a.scale, a.lr_tol)
        if not cv2.imwrite(os.path.join(out, r.names[i][:-4] + ".png"), z):
            raise RuntimeError(f"writing depth for frame {i} failed")
        if k % 10 == 0 or k == len(todo) - 1:
            dt = (time.time() - t0) / (k + 1)
            print(f"  {k + 1}/{len(todo)}  valid {(z > 0).mean() * 100:.0f}%  {dt:.2f} s/frame  "
                  f"eta {dt * (len(todo) - k - 1) / 60:.1f} min", flush=True)
    if missing:
        print(f"  {missing} frames have no pair image and were skipped")
    print(f"done -> {out}   (scan reconstruct {a.scan} --depth-dir {os.path.basename(out)})")
