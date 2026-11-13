"""Scan folder format (COLMAP-ready).

<scan>/
  scan.json            metadata: mode, size, intrinsics, depth unit, device, calibration,
                       capture settings, COLMAP camera, stereo (optional)
  frames.csv           index,name,timestamp,segment,sharpness (one row per frame)
  cameras.txt          COLMAP text-format camera (PINHOLE, factory intrinsics)
  images/000000.jpg    texture (reference view): rectified right mono (mode=mono),
                       undistorted RGB (mode=rgb), rectified left color (mode=zed)
  depth/000000.png     16-bit depth in mm from the camera, pixel-aligned with images/
  pair/000000.jpg      optional: the other rectified stereo view (scan.json "stereo" says
                       which side and the baseline). Input for `scan depth`.
  depth_<method>/      optional: depth computed by `scan depth` (same format as depth/)
  imu.csv              optional: timestamp,gx,gy,gz,ax,ay,az  (s, rad/s, m/s^2, raw IMU axes,
                       same clock as frames.csv)
  colmap/              created by `scan reconstruct` (database.db, sparse/0)
  recon/<run>/         reconstruction results

Open it in COLMAP with images/ as image folder and camera model PINHOLE + the params in
cameras.txt (single camera, no refinement).

    scan info scans/<scan>      # summary
"""

from __future__ import annotations

import json
import os
import time
import warnings
from dataclasses import dataclass

import cv2
import numpy as np

FORMAT = "oak-scan-folder"
VERSION = 1


def sharpness(gray: np.ndarray) -> float:
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def frame_name(i: int) -> str:
    return f"{i:06d}"


@dataclass
class ScanInfo:
    mode: str
    width: int
    height: int
    intrinsics: np.ndarray
    dist_coeffs: list
    depth_unit_m: float
    num_frames: int
    attrs: dict


class ScanWriter:
    """Append frames to a scan folder. Not thread-safe; use from one thread."""

    def __init__(self, path: str, mode: str, width: int, height: int,
                 intrinsics: np.ndarray, dist_coeffs=None, device_info: dict | None = None,
                 calibration_json: str | None = None, stereo: dict | None = None,
                 jpeg_quality: int = 95, png_compression: int = 3):
        self.path = path.rstrip("/")
        self.jpeg_quality = jpeg_quality
        self.png_compression = png_compression
        meta_path = os.path.join(self.path, "scan.json")
        os.makedirs(os.path.join(self.path, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.path, "depth"), exist_ok=True)
        self.csv_path = os.path.join(self.path, "frames.csv")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                self.meta = json.load(fh)
            m = self.meta
            if (m["mode"], m["width"], m["height"]) != (mode, width, height):
                raise ValueError(
                    f"Existing scan is mode={m['mode']} {m['width']}x{m['height']}, "
                    f"camera is mode={mode} {width}x{height}. Use the same mode to append.")
            if not np.allclose(np.array(m["intrinsics"]), intrinsics, rtol=0.01):
                print("WARNING: camera intrinsics differ from the stored ones by >1% "
                      "(different device or calibration?). Stored intrinsics are kept.")
            rows = _read_csv(self.csv_path)
            self.n = int(rows["index"].max()) + 1 if len(rows["index"]) else 0
            self.segment = int(rows["segment"].max()) + 1 if len(rows["segment"]) else 0
        else:
            K = np.asarray(intrinsics, float)
            self.meta = {
                "format": FORMAT, "version": VERSION,
                "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "mode": mode, "width": width, "height": height, "depth_unit_m": 0.001,
                "intrinsics": K.tolist(),
                "dist_coeffs": list(map(float, dist_coeffs)) if dist_coeffs is not None else [],
                "device": device_info or {}, "calibration": calibration_json,
                "colmap_camera": {"model": "PINHOLE",
                                  "params": [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]},
            }
            with open(os.path.join(self.path, "cameras.txt"), "w") as fh:
                fh.write("# Camera list with one line of data per camera:\n"
                         "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
                         f"1 PINHOLE {width} {height} {K[0, 0]:.6f} {K[1, 1]:.6f} {K[0, 2]:.6f} {K[1, 2]:.6f}\n")
            with open(self.csv_path, "w") as fh:
                fh.write("index,name,timestamp,segment,sharpness\n")
            self.n = 0
            self.segment = 0
        if stereo:
            self.meta["stereo"] = {**stereo, "pair_dir": "pair"}
            os.makedirs(os.path.join(self.path, "pair"), exist_ok=True)
        self._write_meta()
        self._has_frames_in_segment = False
        self.bytes_written = 0          # this session (for the GUI)
        self.csv = open(self.csv_path, "a")
        self.imu_csv = None

    def _write_meta(self):
        tmp = os.path.join(self.path, "scan.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(self.meta, fh, indent=1)
        os.replace(tmp, os.path.join(self.path, "scan.json"))

    @property
    def num_frames(self) -> int:
        return self.n

    def new_segment(self):
        if self._has_frames_in_segment:
            self.segment += 1
            self._has_frames_in_segment = False

    def set_capture_settings(self, settings: dict):
        self.meta["capture_settings"] = settings
        self._write_meta()

    def _imwrite(self, folder, name, img, params):
        path = os.path.join(self.path, folder, name)
        if not cv2.imwrite(path, img, params):
            raise RuntimeError(f"writing {folder}/{name} failed")
        self.bytes_written += os.path.getsize(path)

    def append(self, depth_mm: np.ndarray, texture: np.ndarray, timestamp: float,
               sharp: float | None = None, pair: np.ndarray | None = None):
        name = frame_name(self.n)
        jpg = [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
        self._imwrite("images", name + ".jpg", texture, jpg)
        self._imwrite("depth", name + ".png", depth_mm.astype(np.uint16),
                      [cv2.IMWRITE_PNG_COMPRESSION, self.png_compression])
        if pair is not None:
            self._imwrite("pair", name + ".jpg", pair, jpg)
        sh = sharpness(texture) if sharp is None else sharp
        # the csv row is written last: a frame only "exists" once all its files are complete
        self.csv.write(f"{self.n},{name}.jpg,{timestamp:.6f},{self.segment},{sh:.2f}\n")
        self.csv.flush()
        self.n += 1
        self._has_frames_in_segment = True

    def append_imu(self, rows: np.ndarray):
        """rows: Nx7 (timestamp, gx, gy, gz, ax, ay, az)."""
        if self.imu_csv is None:
            p = os.path.join(self.path, "imu.csv")
            new = not os.path.exists(p)
            self.imu_csv = open(p, "a")
            if new:
                self.imu_csv.write("timestamp,gx,gy,gz,ax,ay,az\n")
        np.savetxt(self.imu_csv, rows, fmt=["%.6f"] + ["%.5f"] * 6, delimiter=",")

    def close(self):
        for f in (self.csv, self.imu_csv):
            if f:
                f.close()
        self.csv = self.imu_csv = None


def _read_csv(path):
    cols = {"index": [], "name": [], "timestamp": [], "segment": [], "sharpness": []}
    if os.path.exists(path):
        with open(path) as fh:
            next(fh, None)
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) != 5:
                    continue        # torn last line after a crash
                cols["index"].append(int(parts[0]))
                cols["name"].append(parts[1])
                cols["timestamp"].append(float(parts[2]))
                cols["segment"].append(int(parts[3]))
                cols["sharpness"].append(float(parts[4]))
    return {k: np.array(v) for k, v in cols.items()}


class ScanReader:
    def __init__(self, path: str, depth_dir: str = "depth"):
        self.path = path.rstrip("/")
        with open(os.path.join(self.path, "scan.json")) as fh:
            m = json.load(fh)
        rows = _read_csv(os.path.join(self.path, "frames.csv"))
        self.names = list(rows["name"])
        self.segment = rows["segment"].astype(np.int32)
        self.timestamp = rows["timestamp"].astype(np.float64)
        self.sharpness = rows["sharpness"].astype(np.float32)
        self.image_dir = os.path.join(self.path, "images")
        self.depth_dir = os.path.join(self.path, depth_dir)
        if not os.path.isdir(self.depth_dir):
            raise FileNotFoundError(f"no depth folder {self.depth_dir}")
        self.stereo = m.get("stereo")
        self.info = ScanInfo(mode=m["mode"], width=int(m["width"]), height=int(m["height"]),
                             intrinsics=np.array(m["intrinsics"]), dist_coeffs=m.get("dist_coeffs", []),
                             depth_unit_m=float(m["depth_unit_m"]), num_frames=len(self.names), attrs=m)

    def __len__(self):
        return self.info.num_frames

    def depth_path(self, i: int) -> str:
        return os.path.join(self.depth_dir, self.names[i][:-4] + ".png")

    def depth(self, i: int) -> np.ndarray:
        d = cv2.imread(self.depth_path(i), cv2.IMREAD_UNCHANGED)
        if d is None:
            raise IOError(f"missing depth for frame {i}")
        return d

    def color(self, i: int) -> np.ndarray:
        """Returns RGB uint8 HxWx3 (grayscale replicated in mono mode)."""
        img = cv2.imread(os.path.join(self.image_dir, self.names[i]), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def pair(self, i: int) -> np.ndarray | None:
        """The other rectified stereo view as RGB, or None if this frame has none."""
        if not self.stereo:
            return None
        img = cv2.imread(os.path.join(self.path, self.stereo["pair_dir"], self.names[i]), cv2.IMREAD_COLOR)
        return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def imu(self) -> np.ndarray | None:
        """Nx7 (timestamp, gx, gy, gz, ax, ay, az), sorted by time, or None."""
        p = os.path.join(self.path, "imu.csv")
        if not os.path.exists(p):
            return None
        with warnings.catch_warnings():         # a torn last line is expected after a crash
            warnings.simplefilter("ignore")
            a = np.atleast_2d(np.genfromtxt(p, delimiter=",", skip_header=1, invalid_raise=False))
        a = a[(a.shape[1] == 7) & np.isfinite(a).all(1)] if a.size else a     # torn last line
        return a[np.argsort(a[:, 0], kind="stable")] if len(a) else None


def _size(path):
    return sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs)


def summarize(path: str) -> str:
    r = ScanReader(path)
    i = r.info
    segs, counts = np.unique(r.segment, return_counts=True)
    dur = (r.timestamp[-1] - r.timestamp[0]) if len(r) > 1 else 0
    extra = [d for d in sorted(os.listdir(path)) if d.startswith("depth_")]
    imu = r.imu()
    return (f"{path}: mode={i.mode} {i.width}x{i.height}, {len(r)} frames, "
            f"{len(segs)} segment(s) {dict(zip(segs.tolist(), counts.tolist()))}, "
            f"{_size(path) / 1e6:.1f} MB, span {dur:.0f}s\n"
            f"fx={i.intrinsics[0, 0]:.2f} fy={i.intrinsics[1, 1]:.2f} "
            f"cx={i.intrinsics[0, 2]:.2f} cy={i.intrinsics[1, 2]:.2f}\n"
            f"stereo pair: {r.stereo or 'no'}\n"
            f"IMU: {f'{len(imu)} samples' if imu is not None else 'no'}\n"
            f"extra depth: {', '.join(extra) or 'none'}")
