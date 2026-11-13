"""OAK-D / OAK-D Lite (DepthAI v3): on-device stereo depth with rectified mono or undistorted
RGB texture, the rectified stereo pair, and the IMU when the device has a supported one.

The image size follows the mono sensor: OV7251 (OAK-D Lite) 640x480, OV9282 (OAK-D) 1280x800.
resolution="half" halves it, which is needed for 30 fps over a USB 2 link.

Rectification: for lenses under 85 deg FOV DepthAI rectifies with a 3x3 homography, which does
not correct lens distortion, and its on-device mesh does not fix it either (OAK-D: rows off by
up to 4 px at the image edges). The driver computes the rectification mesh on the host with
OpenCV from the device calibration and uploads it: rows then agree to ~0.3 px, and the on-chip
depth covers ~80% instead of ~50% of the image.
"""

from __future__ import annotations

import datetime
import json
import threading
import time

import cv2
import depthai as dai
import numpy as np

from . import CaptureSettings

MONO_SIZES = {"OV7251": (640, 480), "OV9282": (1280, 800)}
BNO086_MIN_FW = (3, 9)          # older BNO086 firmware crashes DepthAI 3.12 ("not supported")


def _imu_supported(dev) -> tuple[bool, str]:
    name = dev.getConnectedIMU()
    if name in ("", "NONE"):
        return False, "no IMU"
    if name == "BNO086":
        fw = str(dev.getIMUFirmwareVersion())
        if tuple(int(x) for x in fw.split(".")[:2]) < BNO086_MIN_FW:
            return False, f"BNO086 firmware {fw} is too old for DepthAI {dai.__version__} (needs a firmware update)"
    return True, name


def rectification_mesh(calib, w: int, h: int, step: int = 16):
    """Left/right mesh bytes for StereoDepth.loadMeshData: for every output grid node (step px)
    the (y, x) source pixel, from the full distortion model. The rectified camera is K_right,
    as on the device."""
    socks = (dai.CameraBoardSocket.CAM_B, dai.CameraBoardSocket.CAM_C)
    K = [np.array(calib.getCameraIntrinsics(s, w, h)) for s in socks]
    D = [np.array(calib.getDistortionCoefficients(s)) for s in socks]
    R = [np.array(calib.getStereoLeftRectificationRotation()), np.array(calib.getStereoRightRectificationRotation())]
    out = []
    for k, d, r in zip(K, D, R):
        # one pixel larger so that the last grid node (x = w, y = h) exists
        mx, my = cv2.initUndistortRectifyMap(k, d, r, K[1], (w + 1, h + 1), cv2.CV_32FC1)
        out.append(np.stack([my[::step, ::step], mx[::step, ::step]], -1).astype(np.float32).tobytes())
    return out


class Camera:
    """Runs the DepthAI pipeline in a thread, calls on_frame(depth_mm, texture_bgr_or_gray, ts, pair)
    and on_imu(Nx7 samples) on the device clock."""

    def __init__(self, s: CaptureSettings, on_frame, on_error, on_imu=None):
        self.s = s
        self.on_frame, self.on_error, self.on_imu = on_frame, on_error, on_imu
        self.K = None
        self.dist = []
        self.device_info = {}
        self.calib_json = None
        self.w = self.h = None
        self.stereo = None
        self._stop = threading.Event()
        self._ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self, timeout=20.0):
        self.thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("Camera did not deliver frames (timeout).")
        if self.error:
            raise RuntimeError(self.error)

    def stop(self):
        self._stop.set()
        self.thread.join(timeout=5)

    error = None

    def _build(self, p: dai.Pipeline):
        s = self.s
        dev = p.getDefaultDevice()
        calib = dev.readCalibration()
        try:
            self.calib_json = json.dumps(calib.eepromToJson())
        except Exception:
            self.calib_json = None
        sensors = {c.socket: c.sensorName for c in dev.getConnectedCameraFeatures()}
        mono = sensors.get(dai.CameraBoardSocket.CAM_B)
        if mono not in MONO_SIZES:
            raise RuntimeError(f"unsupported mono sensor {mono}")
        mw, mh = MONO_SIZES[mono]
        if s.resolution == "half":
            mw, mh = mw // 2, mh // 2
        # rgb mode: depth is aligned to a 4:3 RGB output of the mono width
        self.w, self.h = (mw, mw * 3 // 4) if s.mode == "rgb" else (mw, mh)
        imu_ok, imu_info = _imu_supported(dev)
        if not imu_ok:
            print(f"OAK IMU not used: {imu_info}")
        self.device_info = {"name": dev.getDeviceName(), "product": dev.getProductName(),
                            "mxid": dev.getDeviceId() if hasattr(dev, "getDeviceId") else "",
                            "usb": str(dev.getUsbSpeed()), "depthai": dai.__version__,
                            "mono_sensor": mono, "imu": imu_info}

        left = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_B)
        right = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_C)
        for cam in (left, right):
            cam.initialControl.setAutoExposureLimit(int(s.exposure_limit_us))
        stereo = p.create(dai.node.StereoDepth).build(
            left.requestOutput((mw, mh), fps=s.camera_fps),
            right.requestOutput((mw, mh), fps=s.camera_fps),
            presetMode=dai.node.StereoDepth.PresetMode.ACCURACY)
        stereo.setLeftRightCheck(True)
        stereo.setExtendedDisparity(False)
        stereo.setSubpixel(True)
        stereo.setSubpixelFractionalBits(5)
        mesh_l, mesh_r = rectification_mesh(calib, mw, mh)
        stereo.loadMeshData(list(mesh_l), list(mesh_r))
        stereo.setMeshStep(16, 16)
        cfg = stereo.initialConfig
        cfg.setConfidenceThreshold(int(s.confidence))
        pp = cfg.postProcessing
        pp.thresholdFilter.minRange = int(s.depth_min * 1000)
        pp.thresholdFilter.maxRange = int(s.depth_max * 1000)
        pp.temporalFilter.enable = False          # smears depth while the camera moves
        pp.speckleFilter.enable = True
        pp.speckleFilter.speckleRange = 28       # max supported with default HW resources
        pp.decimationFilter.decimationFactor = 1

        sync = p.create(dai.node.Sync)
        sync.setSyncThreshold(datetime.timedelta(milliseconds=int(500 / s.camera_fps)))
        if s.mode == "rgb":
            rgb = p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
            lens = calib.getLensPosition(dai.CameraBoardSocket.CAM_A)
            if lens:
                rgb.initialControl.setManualFocus(int(lens))   # calibration is only valid at this focus
            rgb.initialControl.setAutoExposureLimit(int(s.exposure_limit_us))
            tex = rgb.requestOutput((self.w, self.h), dai.ImgFrame.Type.NV12, fps=s.camera_fps,
                                    enableUndistortion=True)
            tex.link(stereo.inputAlignTo)
        else:
            stereo.setDepthAlign(dai.StereoDepthConfig.AlgorithmControl.DepthAlign.RECTIFIED_RIGHT)
            tex = stereo.rectifiedRight
            # the rectified left view is the stereo pair (input for `scan depth`)
            stereo.rectifiedLeft.link(sync.inputs["pair"])
            b = calib.getBaselineDistance(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B, False)
            self.stereo = {"pair_side": "left", "baseline_m": abs(b) / 100.0}
        stereo.depth.link(sync.inputs["depth"])
        tex.link(sync.inputs["tex"])
        qi = None
        if imu_ok and self.on_imu:
            imu = p.create(dai.node.IMU)
            imu.enableIMUSensor([dai.IMUSensor.ACCELEROMETER_RAW, dai.IMUSensor.GYROSCOPE_RAW], 200)
            imu.setBatchReportThreshold(10)
            imu.setMaxBatchReports(20)
            qi = imu.out.createOutputQueue(maxSize=50, blocking=False)
        return sync.out.createOutputQueue(maxSize=4, blocking=False), qi

    def _imu_rows(self, qi):
        rows = []
        for msg in qi.tryGetAll():
            for pk in msg.packets:
                g, a = pk.gyroscope, pk.acceleroMeter
                rows.append((g.getTimestampDevice().total_seconds(), g.x, g.y, g.z, a.x, a.y, a.z))
        if rows:
            self.on_imu(np.array(rows))

    def _run(self):
        try:
            with dai.Pipeline() as p:
                q, qi = self._build(p)
                p.start()
                while p.isRunning() and not self._stop.is_set():
                    if qi is not None:
                        self._imu_rows(qi)
                    msg = q.tryGet()
                    if msg is None:
                        time.sleep(0.002)
                        continue
                    dmsg, tmsg = msg["depth"], msg["tex"]
                    depth = dmsg.getFrame()
                    texture = tmsg.getCvFrame()
                    if self.K is None:
                        tr = dmsg.getTransformation()
                        self.K = np.array(tr.getIntrinsicMatrix(), np.float64)
                        # rectified (mono) / undistorted (rgb) outputs are pinhole
                        self.dist = []
                        if depth.shape != (self.h, self.w):
                            raise RuntimeError(f"unexpected depth size {depth.shape}, expected {(self.h, self.w)}")
                        self._ready.set()
                    ts = dmsg.getTimestampDevice().total_seconds()
                    pair = msg["pair"].getCvFrame() if self.stereo else None
                    self.on_frame(depth, texture, ts, pair)
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
            self._ready.set()
            self.on_error(self.error)


def info():
    devs = dai.Device.getAllAvailableDevices()
    if not devs:
        print("OAK: none found")
        return
    with dai.Device() as dev:
        sensors = {c.socket: c.sensorName for c in dev.getConnectedCameraFeatures()}
        mono = sensors.get(dai.CameraBoardSocket.CAM_B)
        speed = dev.getUsbSpeed()
        b = dev.readCalibration().getBaselineDistance(dai.CameraBoardSocket.CAM_C, dai.CameraBoardSocket.CAM_B, False)
        print(f"{dev.getProductName()} ({dev.getDeviceId()}): mono {mono} {MONO_SIZES.get(mono)}, "
              f"color {sensors.get(dai.CameraBoardSocket.CAM_A)}, baseline {abs(b) * 10:.1f} mm, "
              f"IR projector {'yes' if dev.getIrDrivers() else 'no'}, IMU: {_imu_supported(dev)[1]}, USB {speed.name}")
        if speed in (dai.UsbSpeed.HIGH, dai.UsbSpeed.FULL, dai.UsbSpeed.LOW):
            print("  USB 2 link: full resolution runs at ~8 fps. Use a USB 3 cable/port, or resolution 'half'.")
