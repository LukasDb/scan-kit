"""ZED Mini without the ZED SDK (no CUDA): UVC video, factory calibration, rectification, IMU.

Video  The camera is a UVC webcam that sends left|right side by side. It needs a USB 3 link.
       The USB-C plug on the camera side is directional: if only the IMU enumerates, flip it.
Calib  Factory calibration per serial number from calib.stereolabs.com, cached in calib/.
       R = Rodrigues(RX, CV, RZ), T = (Baseline, TY, TZ), as in zed-open-capture(-mac).
       Distortion: files with k1..k4 and no p1/p2 are Kannala-Brandt (OpenCV fisheye); on
       the unit tested this halves the row error vs. radial-tangential (which zed-open-capture
       uses, ignoring k4). Files with p1/p2 are radial-tangential k1 k2 p1 p2 k3.
       `scan zed-check` compares the conventions on live frames.
IMU    The camera MCU is a HID device (2b03:f681). Report layout and scale factors are from
       Stereolabs' zed-open-capture (sensorcapture_def.hpp). The HID serial number is the
       camera serial number.

Depth  Live depth is OpenCV SGBM on half-resolution images, for the preview and the frame
       gate only. The rectified pair is recorded, run `scan depth` for the real depth.

    scan devices      # serial, calibration, rectified camera, IMU rate
    scan zed-check    # rectification check on live frames
"""

from __future__ import annotations

import configparser
import glob
import os
import struct
import sys
import threading
import time
import urllib.request

import cv2
import numpy as np

from . import CaptureSettings

VENDOR_ID = 0x2B03
PID_IMU = 0xF681                 # ZED Mini MCU (zed-open-capture: SL_USB_PROD_MCU_ZEDM_REV_1)
REP_ID_STREAM = 0x32             # feature report: enable/disable sensor stream
IMU_FMT = struct.Struct("<BBQ6hBBIh")
ACC_SCALE = 9.8189 * 8.0 / 32768.0           # m/s^2 per LSB
GYRO_SCALE = np.deg2rad(1000.0 / 32768.0)    # rad/s per LSB
TS_SCALE = 39062.5e-9                        # s per timestamp tick

RESOLUTIONS = {"2K": (2208, 1242), "FHD": (1920, 1080), "HD": (1280, 720), "VGA": (672, 376)}
CALIB_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "calib")
CALIB_URL = "https://calib.stereolabs.com/?SN={sn}"


# --------------------------------------------------------------------------- calibration

def load_calibration(serial: str | None) -> str:
    """Factory calibration file text. Downloaded once per serial number into calib/."""
    os.makedirs(CALIB_DIR, exist_ok=True)
    if serial is None:
        files = glob.glob(os.path.join(CALIB_DIR, "SN*.conf"))
        if len(files) != 1:
            raise RuntimeError("ZED serial number unknown (IMU not found) and calib/ does not hold "
                               "exactly one SN*.conf. Put your camera's calibration file there.")
        path = files[0]
    else:
        path = os.path.join(CALIB_DIR, f"SN{serial}.conf")
        if not os.path.exists(path):
            with urllib.request.urlopen(CALIB_URL.format(sn=serial), timeout=15) as resp:
                text = resp.read().decode()
            if "[LEFT_CAM_HD]" not in text:
                raise RuntimeError(f"no calibration for ZED serial {serial} on calib.stereolabs.com")
            with open(path, "w") as fh:
                fh.write(text)
    with open(path) as fh:
        return fh.read()


class Rectifier:
    """Rectification maps for one resolution. Rectified views share K; depth = fx * B / d."""

    def __init__(self, conf_text: str, resolution: str = "HD", model: str = "auto",
                 rotation: str = "as_given"):
        c = configparser.ConfigParser()
        c.read_string(conf_text)
        self.size = RESOLUTIONS[resolution]

        def cam(side):
            s = c[f"{side}_CAM_{resolution}"]
            K = np.array([[float(s["fx"]), 0, float(s["cx"])], [0, float(s["fy"]), float(s["cy"])], [0, 0, 1]])
            return K, s
        Kl, sl = cam("LEFT")
        Kr, sr = cam("RIGHT")
        if model == "auto":
            model = "fisheye" if "k4" in sl and "p1" not in sl else "radtan"
        self.model = model
        if model == "radtan":
            dist = lambda s: np.array([float(s.get(k, 0)) for k in ("k1", "k2", "p1", "p2", "k3")])
        else:   # Kannala-Brandt k1..k4 (OpenCV fisheye)
            dist = lambda s: np.array([float(s.get(k, 0)) for k in ("k1", "k2", "k3", "k4")])
        Dl, Dr = dist(sl), dist(sr)
        st = c["STEREO"]
        r = resolution
        rvec = np.array([float(st[f"RX_{r}"]), float(st[f"CV_{r}"]), float(st[f"RZ_{r}"])])
        Rz = cv2.Rodrigues(rvec)[0]
        C = np.array([float(st["Baseline"]), float(st.get(f"TY_{r}", st.get("TY", 0))),
                      float(st.get(f"TZ_{r}", st.get("TZ", 0)))])
        # OpenCV wants x_right = R x_left + T. The sign of T only flips the sign of P2[0, 3].
        R = Rz if rotation == "as_given" else Rz.T
        T = C if rotation == "as_given" else -R @ C
        if model == "radtan":
            R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(Kl, Dl, Kr, Dr, self.size, R, T.reshape(3, 1),
                                                        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
            init = cv2.initUndistortRectifyMap
        else:
            R1, R2, P1, P2, _ = cv2.fisheye.stereoRectify(Kl, Dl, Kr, Dr, self.size, R, T.reshape(3, 1),
                                                          cv2.CALIB_ZERO_DISPARITY, self.size, 0.0, 1.0)
            init = cv2.fisheye.initUndistortRectifyMap
        self.map_l = init(Kl, Dl, R1, P1, self.size, cv2.CV_16SC2)
        self.map_r = init(Kr, Dr, R2, P2, self.size, cv2.CV_16SC2)
        self.K = P1[:3, :3].copy()
        self.baseline_m = abs(P2[0, 3] / P2[0, 0]) / 1000.0

    def __call__(self, frame_sbs: np.ndarray):
        w = self.size[0]
        left = cv2.remap(frame_sbs[:, :w], *self.map_l, cv2.INTER_LINEAR)
        right = cv2.remap(frame_sbs[:, w:], *self.map_r, cv2.INTER_LINEAR)
        return left, right


def row_error(left: np.ndarray, right: np.ndarray) -> tuple[float, float, int]:
    """Row error of SIFT matches between rectified views: (median |dy|, 90th percentile |dy|, n).
    0 = perfect rectification. SIFT keypoints are sub-pixel, ORB ones are not."""
    sift = cv2.SIFT_create(4000)
    kl, dl = sift.detectAndCompute(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), None)
    kr, dr = sift.detectAndCompute(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY), None)
    if dl is None or dr is None:
        return float("nan"), float("nan"), 0
    ms = cv2.BFMatcher(cv2.NORM_L2).knnMatch(dl, dr, k=2)
    good = [m[0] for m in ms if len(m) == 2 and m[0].distance < 0.7 * m[1].distance]
    dy = np.array([kl[m.queryIdx].pt[1] - kr[m.trainIdx].pt[1] for m in good])
    dx = np.array([kl[m.queryIdx].pt[0] - kr[m.trainIdx].pt[0] for m in good])
    ok = (np.abs(dy) < 8) & (dx > 0)
    if not ok.sum():
        return float("nan"), float("nan"), 0
    a = np.abs(dy[ok])
    return float(np.median(a)), float(np.percentile(a, 90)), int(ok.sum())


# --------------------------------------------------------------------------- IMU

class ZedImu:
    """Reads the IMU in a thread. on_samples(Nx7): host-clock time [s], gyro [rad/s], accel [m/s^2]."""

    def __init__(self, on_samples=None):
        import hid
        devs = [d for d in hid.enumerate(VENDOR_ID) if d["product_id"] == PID_IMU]
        if not devs:
            raise RuntimeError("ZED Mini IMU (HID 2b03:f681) not found")
        self.dev = hid.device()
        self.dev.open_path(devs[0]["path"])
        self.serial = devs[0]["serial_number"] or self.dev.get_serial_number_string()
        self.on_samples = on_samples
        self.offset = None          # host - device clock [s], running minimum = lowest latency
        self.temperature = float("nan")
        self.last_gyro = np.zeros(3)
        self.count = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.dev.send_feature_report([REP_ID_STREAM, 1])
        self.thread.start()

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=2)
        try:
            self.dev.send_feature_report([REP_ID_STREAM, 0])
        finally:
            self.dev.close()

    def _run(self):
        batch = []
        while not self._stop.is_set():
            buf = self.dev.read(64, 100)
            # report id byte is 1 in zed-open-capture; macOS hidapi delivers it as 0
            if len(buf) < IMU_FMT.size or buf[0] not in (0, 1):
                continue
            _, not_valid, ts, gx, gy, gz, ax, ay, az, _, _, _, temp = IMU_FMT.unpack_from(bytes(buf))
            if not_valid:
                continue
            t_dev = ts * TS_SCALE
            off = time.monotonic() - t_dev
            # shortcut: constant clock offset, no drift model. Error grows ~ppm of the scan length.
            self.offset = off if self.offset is None else min(self.offset, off)
            g = np.array([gx, gy, gz]) * GYRO_SCALE
            self.last_gyro = g
            self.temperature = temp * 0.01
            self.count += 1
            batch.append((t_dev + self.offset, *g, ax * ACC_SCALE, ay * ACC_SCALE, az * ACC_SCALE))
            if len(batch) >= 40:
                if self.on_samples:
                    self.on_samples(np.array(batch))
                batch = []


# --------------------------------------------------------------------------- video

def open_video(resolution: str, fps: float, index: int = -1) -> cv2.VideoCapture:
    """First video device that delivers side-by-side frames of the requested resolution."""
    w, h = RESOLUTIONS[resolution]
    api = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    for i in ([index] if index >= 0 else range(6)):
        cap = cv2.VideoCapture(i, api)
        if not cap.isOpened():
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2 * w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        ok, frame = cap.read()
        if ok and frame.shape[:2] == (h, 2 * w):
            return cap
        cap.release()
    raise RuntimeError(f"no ZED video device with {2 * w}x{h} frames. The ZED Mini needs USB 3: "
                       f"plug it in directly and flip the USB-C plug on the camera side.")


class ZedCamera:
    """Same interface as cameras.oak.Camera, plus on_imu(Nx7 samples)."""

    def __init__(self, s: CaptureSettings, on_frame, on_error, on_imu=None):
        self.s = s
        self.on_frame, self.on_error, self.on_imu = on_frame, on_error, on_imu
        self.dist = []
        self.error = None
        self.imu = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout=20.0):
        s = self.s
        try:
            self.imu = ZedImu(self.on_imu)
            serial = self.imu.serial
        except Exception as e:  # noqa: BLE001  video works without the IMU
            print(f"ZED IMU not available: {e}")
            self.imu, serial = None, None
        self.calib_json = load_calibration(serial)
        self.rect = Rectifier(self.calib_json, s.resolution)
        self.w, self.h = self.rect.size
        self.K = self.rect.K
        self.stereo = {"pair_side": "right", "baseline_m": float(self.rect.baseline_m)}
        self.cap = open_video(s.resolution, s.camera_fps)
        self.device_info = {"product": "ZED Mini", "serial": serial, "resolution": s.resolution,
                            "distortion_model": self.rect.model, "imu": self.imu is not None,
                            "usb": "UVC"}
        if self.imu:
            self.imu.start()
        self.thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("ZED did not deliver frames (timeout).")
        if self.error:
            raise RuntimeError(self.error)

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)
        if self.imu:
            self.imu.stop()
        self.cap.release()

    def live_depth(self, left, right):
        """Quick SGBM depth in mm on half-resolution images."""
        from ..stereo import sgbm_disparity
        sc = 0.5 if self.w > 700 else 1.0
        fx = self.K[0, 0] * sc
        num = min(256, fx * self.rect.baseline_m / max(self.s.depth_min, 0.05))
        small = lambda im: cv2.resize(im, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        d = sgbm_disparity(small(left), small(right), int(num))
        z = np.zeros(d.shape, np.float32)
        z[d > 0] = fx * self.rect.baseline_m * 1000.0 / d[d > 0]
        z[(z < self.s.depth_min * 1000) | (z > self.s.depth_max * 1000)] = 0
        return np.rint(cv2.resize(z, (self.w, self.h), interpolation=cv2.INTER_NEAREST)).astype(np.uint16)

    def _run(self):
        try:
            while not self._stop.is_set():
                ok, frame = self.cap.read()
                ts = time.monotonic()
                if not ok:
                    raise RuntimeError("ZED video stream stopped")
                left, right = self.rect(frame)
                self._ready.set()
                self.on_frame(self.live_depth(left, right), left, ts, right)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self._ready.set()
            self.on_error(self.error)


# --------------------------------------------------------------------------- diagnostics

def info():
    imu = ZedImu()
    print(f"serial {imu.serial}")
    conf = load_calibration(imu.serial)
    for res in RESOLUTIONS:
        r = Rectifier(conf, res)
        print(f"{res:4s} {r.size[0]}x{r.size[1]}  {r.model}  rectified fx={r.K[0, 0]:.2f} "
              f"cx={r.K[0, 2]:.2f} cy={r.K[1, 2]:.2f}  baseline={r.baseline_m * 1000:.2f} mm")
    rows = []
    imu.on_samples = rows.append
    imu.start()
    time.sleep(2.0)
    imu.stop()
    a = np.concatenate(rows) if rows else np.zeros((0, 7))
    if len(a):
        print(f"IMU {len(a) / 2:.0f} Hz, |accel| {np.linalg.norm(a[:, 4:7].mean(0)):.3f} m/s^2, "
              f"accel {np.round(a[:, 4:7].mean(0), 3)}, gyro bias {np.round(np.degrees(a[:, 1:4].mean(0)), 3)} deg/s, "
              f"{imu.temperature:.1f} C")
    else:
        print("IMU: no samples")


def check(resolution="HD"):
    """Rectify live frames with each distortion model / rotation convention and compare."""
    try:
        serial = ZedImu().serial
    except Exception:  # noqa: BLE001
        serial = None
    conf = load_calibration(serial)
    cap = open_video(resolution, 15)
    for _ in range(10):
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("no frame")
    print("row error |dy| of feature matches (lower is better; good rectification: median < 0.3 px):")
    for model in ("radtan", "fisheye"):
        for rot in ("as_given", "inverse"):
            r = Rectifier(conf, resolution, model, rot)
            med, p90, n = row_error(*r(frame))
            print(f"  {model:8s} {rot:9s} median {med:.3f} px  p90 {p90:.3f} px  ({n} matches)")
    r = Rectifier(conf, resolution)
    left, right = r(frame)
    out = os.path.abspath("zed_check.jpg")
    vis = np.hstack([left, right])
    for y in range(0, vis.shape[0], 40):
        cv2.line(vis, (0, y), (vis.shape[1], y), (0, 255, 0), 1)
    cv2.imwrite(out, vis)
    print(f"default: {r.model}, rectified pair with row lines -> {out}")
