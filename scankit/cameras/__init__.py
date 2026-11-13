"""Camera drivers for `scan capture`.

Each driver runs in its own thread and calls on_frame(depth_mm, texture, ts, pair), where pair
is the other rectified stereo view or None. After start() it has K, dist, w, h, device_info,
calib_json and stereo ({"pair_side", "baseline_m"} or None).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaptureSettings:
    mode: str = "mono"               # mono | rgb (OAK-D, OAK-D Lite) | zed (ZED Mini)
    resolution: str = "full"         # OAK: full | half;  ZED (per eye): 2K | FHD | HD | VGA
    camera_fps: float = 30.0
    save_fps: float = 15.0
    depth_min: float = 0.2           # m
    depth_max: float = 4.0           # m, stored range (reconstruction clips tighter)
    confidence: int = 55             # stereo confidence 0-255, DepthAI v3: HIGHER = stricter
    exposure_limit_us: int = 8000    # cap auto-exposure to limit motion blur
    min_valid_fraction: float = 0.05  # skip frames with less valid depth
    preview: bool = True
    preview_voxel: float = 0.01
    record_delay_s: float = 5.0      # countdown between pressing Record and saving frames


RESOLUTIONS = {"oak": ["full", "half"], "zed": ["HD", "VGA", "FHD", "2K"]}


def make_camera(s: CaptureSettings, on_frame, on_error, on_imu):
    if s.mode == "zed":
        from .zed import ZedCamera
        return ZedCamera(s, on_frame, on_error, on_imu)
    from .oak import Camera
    return Camera(s, on_frame, on_error, on_imu)
