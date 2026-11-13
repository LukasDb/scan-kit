"""`scan capture`: capture GUI (Tkinter, Open3D preview window) for the OAK-D, OAK-D Lite and ZED Mini.

- Live texture + depth view, optional live 3D preview (dense tracking) to check coverage.
- Records depth + texture (+ the rectified stereo pair, + IMU) into a COLMAP-ready scan folder
  (see scankit/scanfile.py). Re-opening an existing folder appends to it.
- Pressing Record starts a 5 s countdown before frames are saved (press again to cancel).
- Pause/resume adds a new segment.

Modes
  mono : OAK. Depth in the rectified right camera frame, global-shutter grayscale texture.
         Best OAK geometry. Saves the rectified left view as stereo pair.  [default]
  rgb  : OAK. Depth aligned to the (undistorted, focus-locked) RGB camera; colored.
  zed  : ZED Mini over UVC + HID (no ZED SDK). Rectified left color texture, rectified right
         view as stereo pair, IMU. Live depth is coarse: run `scan depth` afterwards.
"""

from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict

import cv2
import numpy as np
import open3d as o3d
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from .cameras import RESOLUTIONS, CaptureSettings, make_camera
from .preview import Preview
from .recorder import Recorder
from .scanfile import ScanWriter, sharpness

PREVIEW_W = 640                 # px per view in the live preview (texture | depth)

# --------------------------------------------------------------------------- GUI
# Tkinter control window + separate Open3D (GLFW) window for the 3D preview.
# (Open3D's own Filament GUI segfaults on recent macOS, the GLFW visualizer is fine.)

class App:
    def __init__(self, path: str | None):
        self.s = CaptureSettings()
        self.cam = None
        self.imu_rate = None            # latest gyro |w| [deg/s] (ZED)
        self.rec: Recorder | None = None
        self.preview: Preview | None = None
        self.recording = False
        self.countdown_end = None       # time.time() at which a pending recording starts
        self.last_save = -1e9
        self.last_disp = 0.0
        self.save_times: list[float] = []
        self.disp_slot = None           # latest (image, stats) from the camera thread
        self.prev_slot = None           # latest (pcd, T, ok) from the preview thread
        self.cam_error = None
        self.cam_starting = False
        self.vis = None
        self.vis_has_map = False
        self.photo = None

        self.root = tk.Tk()
        self.root.title("Scanner")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<space>", lambda e: self.on_record() if self.cam else None)
        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="ns")
        self.img_label = ttk.Label(self.root)
        self.img_label.grid(row=0, column=1, sticky="nsew")
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        r = 0
        ttk.Label(left, text="Scan folder (appended if it exists)").grid(row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.path_var = tk.StringVar(value=path or self._new_name())
        self.path_entry = ttk.Entry(left, textvariable=self.path_var, width=42)
        self.path_entry.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        fb = ttk.Frame(left)
        fb.grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 8))
        self.btn_browse = ttk.Button(fb, text="Browse...", command=self.on_browse)
        self.btn_browse.pack(side="left")
        self.btn_newname = ttk.Button(fb, text="New name", command=lambda: self.path_var.set(self._new_name()))
        self.btn_newname.pack(side="left")
        r += 1

        ttk.Label(left, text="Camera settings (applied on start)", font=("", 0, "bold")).grid(
            row=r, column=0, columnspan=2, sticky="w")
        r += 1
        self.mode_var = tk.StringVar(value=self.s.mode)
        ttk.Label(left, text="Mode").grid(row=r, column=0, sticky="w")
        self.mode_cb = ttk.Combobox(left, textvariable=self.mode_var, values=["mono", "rgb", "zed"],
                                    state="readonly", width=8)
        self.mode_cb.grid(row=r, column=1, sticky="w")
        r += 1
        self.res_var = tk.StringVar(value=self.s.resolution)
        ttk.Label(left, text="Resolution").grid(row=r, column=0, sticky="w")
        self.res_cb = ttk.Combobox(left, textvariable=self.res_var, state="readonly", width=8)
        self.res_cb.grid(row=r, column=1, sticky="w")
        self.mode_cb.bind("<<ComboboxSelected>>", lambda e: self._update_resolutions())
        self._update_resolutions()
        r += 1
        self.vars = {}
        self.cam_widgets = [self.mode_cb, self.res_cb]
        for key, label, lo, hi, inc, cam_setting in (
                ("camera_fps", "Camera FPS", 5, 60, 5, True),
                ("depth_min", "Depth min [m]", 0.1, 5, 0.05, True),
                ("depth_max", "Depth max [m]", 0.3, 10, 0.1, True),
                ("confidence", "Confidence (higher=stricter)", 0, 255, 5, True),
                ("exposure_limit_us", "Max exposure [us]", 500, 33000, 500, True),
                ("save_fps", "Save FPS", 1, 60, 1, False)):
            if key == "save_fps":
                ttk.Separator(left).grid(row=r, column=0, columnspan=2, sticky="ew", pady=6)
                r += 1
            v = tk.StringVar(value=str(getattr(self.s, key)))
            ttk.Label(left, text=label).grid(row=r, column=0, sticky="w")
            sp = ttk.Spinbox(left, textvariable=v, from_=lo, to=hi, increment=inc, width=8)
            sp.grid(row=r, column=1, sticky="w")
            self.vars[key] = v
            if cam_setting:
                self.cam_widgets.append(sp)
            r += 1
        self.preview_var = tk.BooleanVar(value=self.s.preview)
        ttk.Checkbutton(left, text="Live 3D preview (tracking)", variable=self.preview_var).grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(4, 8))
        r += 1

        self.btn_cam = ttk.Button(left, text="Start camera", command=self.on_camera)
        self.btn_cam.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        self.btn_rec = ttk.Button(left, text="Record  [space]", command=self.on_record, state="disabled")
        self.btn_rec.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        self.btn_close = ttk.Button(left, text="Close file", command=self.close_file, state="disabled")
        self.btn_close.grid(row=r, column=0, columnspan=2, sticky="ew")
        r += 1
        ttk.Button(left, text="Reset 3D preview", command=self.on_reset_preview).grid(
            row=r, column=0, columnspan=2, sticky="ew")
        r += 1

        self.status = tk.StringVar(value="camera stopped")
        self.stats = tk.StringVar()
        self.track = tk.StringVar()
        for var in (self.status, self.stats, self.track):
            ttk.Label(left, textvariable=var, justify="left", wraplength=320).grid(
                row=r, column=0, columnspan=2, sticky="w", pady=(6, 0))
            r += 1
        self.track_label = left.grid_slaves(row=r - 1)[0]
        ttk.Label(left, foreground="gray", justify="left", text=(
            "Tips: move slowly, stay ~0.4-1.2 m from surfaces,\n"
            "revisit earlier areas (loop closure), avoid\n"
            "textureless or shiny surfaces. Pausing starts a\n"
            "new segment: resume with overlap to what is\n"
            "already scanned.")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self._show(np.zeros((480, 2 * PREVIEW_W, 3), np.uint8))
        self.root.after(30, self.tick)

    @staticmethod
    def _new_name():
        return os.path.abspath(os.path.join("scans", time.strftime("scan_%Y%m%d_%H%M%S")))

    def _show(self, rgb):
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.img_label.configure(image=self.photo)

    # ---- settings
    def _update_resolutions(self):
        values = RESOLUTIONS["zed" if self.mode_var.get() == "zed" else "oak"]
        self.res_cb.configure(values=values)
        if self.res_var.get() not in values:
            self.res_var.set(values[0])

    def read_settings(self):
        s = self.s
        try:
            s.mode = self.mode_var.get()
            s.resolution = self.res_var.get()
            s.camera_fps = float(self.vars["camera_fps"].get())
            s.depth_min = float(self.vars["depth_min"].get())
            s.depth_max = float(self.vars["depth_max"].get())
            s.confidence = int(float(self.vars["confidence"].get()))
            s.exposure_limit_us = int(float(self.vars["exposure_limit_us"].get()))
            s.save_fps = float(self.vars["save_fps"].get())
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e))
            return False
        s.preview = self.preview_var.get()
        return True

    def set_cam_widgets(self, enabled):
        for w in self.cam_widgets:
            w.configure(state=("readonly" if isinstance(w, ttk.Combobox) else "normal") if enabled else "disabled")

    # ---- actions
    def on_browse(self):
        """Pick an existing scan folder (to append) or a parent folder for a new scan."""
        start = os.path.dirname(self.path_var.get().rstrip("/")) or os.getcwd()
        p = filedialog.askdirectory(title="Existing scan folder, or folder to create a new scan in",
                                    initialdir=start, mustexist=False)
        if not p:
            return
        if os.path.exists(os.path.join(p, "scan.json")):
            self.path_var.set(p)
        else:
            self.path_var.set(os.path.join(p, time.strftime("scan_%Y%m%d_%H%M%S")))

    def on_camera(self):
        if self.cam is not None:
            self.stop_camera()
            return
        if self.cam_starting or not self.read_settings():
            return
        self.cam_starting = True
        self.status.set("starting camera...")
        self.btn_cam.configure(state="disabled")

        def start():
            cam = make_camera(self.s, self.on_frame, self.on_cam_error, self.on_imu)
            try:
                cam.start()
                self.cam = cam
            except Exception as e:  # noqa: BLE001
                self.cam_error = str(e)
            self.cam_starting = False
        threading.Thread(target=start, daemon=True).start()

    def _cam_started(self):
        self.btn_cam.configure(text="Stop camera", state="normal")
        self.btn_rec.configure(state="normal")
        self.set_cam_widgets(False)
        K, info = self.cam.K, self.cam.device_info
        self.status.set(f"camera running ({self.s.mode}), fx={K[0, 0]:.1f} fy={K[1, 1]:.1f}\n"
                        f"{info.get('product', '')}  {info.get('usb', '')}")

    def on_cam_error(self, err):
        self.cam_error = err      # handled in tick() on the Tk thread

    def stop_camera(self):
        self.countdown_end = None
        if self.recording:
            self.on_record()
        if self.cam:
            cam, self.cam = self.cam, None
            cam.stop()
        self.imu_rate = None
        if self.preview:
            self.preview.close()
            self.preview = None
        self.btn_cam.configure(text="Start camera", state="normal")
        self.btn_rec.configure(state="disabled")
        self.set_cam_widgets(True)
        self.status.set("camera stopped")

    def open_writer(self):
        path = self.path_var.get()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        c = self.cam
        w = ScanWriter(path, self.s.mode, c.w, c.h, c.K, c.dist, c.device_info, c.calib_json, c.stereo)
        w.set_capture_settings(asdict(self.s))
        self.rec = Recorder(w)
        self.btn_close.configure(state="normal")
        for wdg in (self.path_entry, self.btn_browse, self.btn_newname):
            wdg.configure(state="disabled")

    def close_file(self):
        self.countdown_end = None
        if self.recording:
            self.on_record()
        if self.rec:
            rec, self.rec = self.rec, None
            rec.close()
            self.stats.set(f"file closed: {os.path.basename(self.path_var.get())}")
        self.btn_close.configure(state="disabled")
        for wdg in (self.path_entry, self.btn_browse, self.btn_newname):
            wdg.configure(state="normal")

    def on_record(self):
        """Record -> countdown -> recording -> pause. Pressing during the countdown cancels."""
        if self.recording:
            self.recording = False
            self.btn_rec.configure(text="Record  [space]")
            return
        if self.countdown_end is not None:
            self.countdown_end = None
            self.btn_rec.configure(text="Record  [space]")
            return
        if not self.read_settings():
            return
        try:
            self._resume = self.rec is not None
            if self.rec is None:
                self.open_writer()      # open now so errors (e.g. mode mismatch) show immediately
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Cannot open scan", str(e))
            return
        if self.s.record_delay_s <= 0:
            self._start_recording()
        else:
            self.countdown_end = time.time() + self.s.record_delay_s

    def _start_recording(self):
        self.countdown_end = None
        if self._resume:
            self.rec.put("segment")
        if self.s.preview and self.preview is None:
            self.preview = Preview(self.cam.K, self.cam.w, self.cam.h, self.s, self.on_preview)
        self.recording = True
        self.btn_rec.configure(text="Pause  [space]")

    def on_reset_preview(self):
        if self.preview:
            self.preview.reset_flag = True
        if self.vis:
            self.vis.clear_geometries()
            self.vis_has_map = False

    # ---- worker-thread callbacks (no Tk calls here)
    def on_frame(self, depth, tex, ts, pair=None):
        now = time.time()
        valid = float((depth > 0).mean())
        sharp = sharpness(tex)
        # rate limit on device timestamps with half a camera frame of tolerance, so that
        # save_fps == camera_fps really keeps every frame
        min_dt = 1.0 / max(self.s.save_fps, 0.1) - 0.5 / max(self.s.camera_fps, 1.0)
        if self.recording and self.rec and ts - self.last_save >= min_dt:
            if valid >= self.s.min_valid_fraction:
                self.last_save = ts
                self.rec.put((depth, tex, ts, sharp, pair))
                self.save_times = [t for t in self.save_times if now - t < 2.0] + [now]
                if self.preview:
                    self.preview.push(depth, tex)
        if now - self.last_disp > 1 / 20:
            self.last_disp = now
            # fixed preview size, built here: full-resolution Tk images are slow
            sc = PREVIEW_W / tex.shape[1]
            size = (PREVIEW_W, round(tex.shape[0] * sc))
            tex = cv2.resize(tex, size, interpolation=cv2.INTER_AREA)
            depth = cv2.resize(depth, size, interpolation=cv2.INTER_NEAREST)
            rgb = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB) if tex.ndim == 2 else cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
            dmin, dmax = self.s.depth_min * 1000, self.s.depth_max * 1000
            dn = np.clip((depth.astype(np.float32) - dmin) / (dmax - dmin) * 255, 0, 255).astype(np.uint8)
            dc = cv2.applyColorMap(255 - dn, cv2.COLORMAP_TURBO)
            dc[depth == 0] = 0
            img = np.ascontiguousarray(np.hstack([rgb, cv2.cvtColor(dc, cv2.COLOR_BGR2RGB)]))
            if self.recording:
                cv2.circle(img, (20, 20), 9, (255, 40, 40), -1)
            cd = self.countdown_end
            if cd is not None:
                cw, ch = img.shape[1] // 4, img.shape[0] // 2
                left = max(0.0, cd - time.time())
                txt = str(int(np.ceil(left))) if left > 0 else "0"
                cv2.putText(img, txt, (cw - 60, ch + 60), cv2.FONT_HERSHEY_SIMPLEX, 6,
                            (255, 60, 60), 14, cv2.LINE_AA)
                cv2.putText(img, "recording starts in", (cw - 190, ch - 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 60, 60), 3, cv2.LINE_AA)
            self.disp_slot = (img, valid, sharp)

    def on_imu(self, rows):
        self.imu_rate = float(np.degrees(np.linalg.norm(rows[-1, 1:4])))
        if self.recording and self.rec:
            self.rec.put(("imu", rows))

    def on_preview(self, pcd, T, ok):
        old = self.prev_slot
        if pcd is None and old is not None and old[0] is not None:
            pcd = old[0]          # don't drop a pending map update
        self.prev_slot = (pcd, T, ok)

    # ---- Tk main-thread loop
    def tick(self):
        try:
            if self.countdown_end is not None:
                left = self.countdown_end - time.time()
                if left <= 0:
                    self._start_recording()
                else:
                    self.btn_rec.configure(text=f"Cancel  (starts in {int(np.ceil(left))} s)")
            if self.cam_error:
                err, self.cam_error = self.cam_error, None
                self.stop_camera()
                messagebox.showerror("Camera error", err)
            if self.cam and str(self.btn_cam["text"]) == "Start camera":
                self._cam_started()
            slot, self.disp_slot = self.disp_slot, None
            if slot:
                img, valid, sharp = slot
                self._show(img)
                st = f"valid depth {valid * 100:.0f}%   sharpness {sharp:.0f}"
                if self.imu_rate is not None:
                    st += f"   rotation {self.imu_rate:.0f} deg/s"
                rec = self.rec
                if rec:
                    size = getattr(rec.w, "bytes_written", 0) / 1e6
                    st += (f"\nframes {rec.w.num_frames}   segment {rec.w.segment}"
                           f"\n{len(self.save_times) / 2:.1f} fps saved   queue {rec.q.qsize()}   dropped {rec.dropped}"
                           f"\n{size:.0f} MB this session   {os.path.basename(rec.w.path)}")
                self.stats.set(st)
                if self.preview:
                    self.track.set(self.preview.status)
                    self.track_label.configure(foreground="red" if "LOST" in self.preview.status else "")
            self._update_vis()
        except Exception as e:  # noqa: BLE001  keep the UI alive
            print("UI error:", e)
        self.root.after(30, self.tick)

    def _update_vis(self):
        upd, self.prev_slot = self.prev_slot, None
        if upd is not None and self.vis is None:
            self.vis = o3d.visualization.Visualizer()
            self.vis.create_window("3D preview", 960, 720)
            opt = self.vis.get_render_option()
            opt.point_size = 2.0
            opt.background_color = np.array([0.12, 0.12, 0.14])
            self.vis_has_map = False
        if self.vis is None:
            return
        if upd is not None:
            pcd, T, ok = upd
            if pcd is not None and len(pcd.points):
                if self.vis_has_map:
                    self.vis.remove_geometry(self.vis_map, reset_bounding_box=False)
                self.vis.add_geometry(pcd, reset_bounding_box=not self.vis_has_map)
                if not self.vis_has_map:
                    vc = self.vis.get_view_control()
                    vc.set_front([0, -0.3, -1])
                    vc.set_up([0, -1, 0])
                self.vis_map = pcd
                self.vis_has_map = True
            if self.cam is not None:
                fr = o3d.geometry.LineSet.create_camera_visualization(self.cam.w, self.cam.h, self.cam.K, np.linalg.inv(T), 0.1)
                fr.paint_uniform_color([0.1, 0.9, 0.2] if ok else [1, 0.1, 0.1])
                if getattr(self, "vis_cam", None) is not None:
                    self.vis.remove_geometry(self.vis_cam, reset_bounding_box=False)
                self.vis.add_geometry(fr, reset_bounding_box=False)
                self.vis_cam = fr
        if not self.vis.poll_events():       # user closed the preview window
            self.vis.destroy_window()
            self.vis = None
            self.vis_cam = None
            return
        self.vis.update_renderer()

    def on_close(self):
        self.recording = False
        if self.cam:
            self.cam.stop()
        if self.preview:
            self.preview.close()
        if self.rec:
            self.rec.close()
            self.rec = None
        if self.vis:
            self.vis.destroy_window()
        self.root.destroy()

    def run(self):
        self.root.mainloop()
