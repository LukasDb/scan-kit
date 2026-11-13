"""Frame selection, depth cleaning and back-projection."""

from __future__ import annotations

import os

import cv2
import numpy as np

from .config import Config
from .scanfile import ScanReader


def clean_depth(depth: np.ndarray, depth_min_mm: float, depth_max_mm: float,
                edge_rel: float, erode_px: int) -> np.ndarray:
    """Range clamp + removal of flying pixels at depth discontinuities and hole borders."""
    d = depth.astype(np.float32)
    valid = (d >= depth_min_mm) & (d <= depth_max_mm)
    if edge_rel > 0:
        k = np.ones((3, 3), np.uint8)
        dmax = cv2.dilate(np.where(valid, d, 0), k)
        dmin = cv2.erode(np.where(valid, d, 1e7).astype(np.float32), k)
        valid &= (dmax - dmin) <= edge_rel * d
    if erode_px > 0:
        valid = cv2.erode(valid.astype(np.uint8), np.ones((2 * erode_px + 1,) * 2, np.uint8)) > 0
    out = depth.copy()
    out[~valid] = 0
    return out


class FrameSource:
    """Loads, undistorts and cleans frames from a scan file."""

    def __init__(self, reader: ScanReader, cfg: Config):
        self.r = reader
        self.cfg = cfg
        info = reader.info
        self.K = info.intrinsics.astype(np.float64)
        self.w, self.h = info.width, info.height
        # depth_scale in Open3D = raw units per metre; apply correction here (free).
        self.depth_scale = 1.0 / (info.depth_unit_m * cfg.depth_scale_correction)
        D = np.array(info.dist_coeffs, np.float64)
        need = D.size and np.abs(D).max() > 1e-6
        self.maps = None
        if cfg.undistort == "on" or (cfg.undistort == "auto" and need):
            if D.size:
                self.maps = cv2.initUndistortRectifyMap(self.K, D, None, self.K, (self.w, self.h), cv2.CV_32FC1)

    def mm(self, metres: float) -> float:
        return metres * self.depth_scale

    def load(self, i: int, depth_max: float):
        d = self.r.depth(i)
        c = self.r.color(i)
        if self.maps is not None:
            d = cv2.remap(d, *self.maps, cv2.INTER_NEAREST)
            c = cv2.remap(c, *self.maps, cv2.INTER_LINEAR)
        d = clean_depth(d, self.mm(self.cfg.depth_min), self.mm(depth_max),
                        self.cfg.edge_rel_threshold, self.cfg.edge_erode_px)
        return d, c


def backproject(depth_u16: np.ndarray, K: np.ndarray, depth_scale: float, stride: int = 2,
                return_pixels: bool = False):
    """Depth image -> Nx3 points in the camera frame (numpy; avoids Open3D image ops that
    are not implemented in the macOS CPU build)."""
    d = depth_u16[::stride, ::stride].astype(np.float32) / depth_scale
    v, u = np.nonzero(d)
    z = d[v, u]
    u = u * stride
    v = v * stride
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], 1)
    return (pts, v, u) if return_pixels else pts


def select_frames(r: ScanReader, cfg: Config) -> np.ndarray:
    n = len(r)
    end = n if cfg.end < 0 else min(cfg.end, n)
    idx = np.arange(cfg.start, end)
    idx = idx[[os.path.exists(r.depth_path(i)) for i in idx]]     # `scan depth --stride` skips frames
    if cfg.segments:
        idx = idx[np.isin(r.segment[idx], cfg.segments)]
    keep = np.ones(len(idx), bool)
    if cfg.blur_reject_rel > 0:
        # relative to a rolling median so that only isolated blurry frames are dropped
        for s in np.unique(r.segment[idx]):
            m = np.nonzero(r.segment[idx] == s)[0]
            sh = r.sharpness[idx][m]
            med = np.array([np.median(sh[max(0, k - 7):k + 8]) for k in range(len(sh))])
            keep[m] &= sh >= cfg.blur_reject_rel * med
    idx = idx[keep]
    if cfg.stride > 1:
        out = []
        for s in np.unique(r.segment[idx]):
            out.append(idx[r.segment[idx] == s][::cfg.stride])
        idx = np.sort(np.concatenate(out))
    return idx
