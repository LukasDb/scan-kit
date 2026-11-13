"""Live dense tracking on recorded frames during capture, only to visualise coverage."""

from __future__ import annotations

import threading
import time

import cv2

from .frames import clean_depth
from .tracking import ModelTracker


class Preview:
    """Live dense tracking on recorded frames, only to visualise coverage."""

    def __init__(self, K, w, h, s, on_update):
        self.K, self.s, self.on_update = K, s, on_update
        self.tracker = ModelTracker(K, w, h, s.preview_voxel, 1000.0, s.depth_min, s.depth_max,
                                    depth_diff=0.07, min_fitness=0.3, max_step=0.15,
                                    trunc_mult=6.0, block_count=60000)
        self.slot = None
        self.lock = threading.Lock()
        self.ev = threading.Event()
        self._stop = threading.Event()
        self.reset_flag = False
        self.status = "idle"
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def push(self, depth, tex):
        with self.lock:
            self.slot = (depth, tex)
        self.ev.set()

    def _run(self):
        last_pub = 0
        lost = 0
        while not self._stop.is_set():
            if not self.ev.wait(0.2):
                continue
            self.ev.clear()
            with self.lock:
                item, self.slot = self.slot, None
            if item is None:
                continue
            if self.reset_flag:
                self.tracker.reset()
                self.reset_flag = False
                lost = 0
            depth, tex = item
            d = clean_depth(depth, self.s.depth_min * 1000, self.s.depth_max * 1000, 0.03, 1)
            rgb = cv2.cvtColor(tex, cv2.COLOR_GRAY2RGB) if tex.ndim == 2 else cv2.cvtColor(tex, cv2.COLOR_BGR2RGB)
            ok = self.tracker.add(d, rgb)
            lost = 0 if ok else lost + 1
            self.status = (f"tracking  fit={self.tracker.last_fitness:.2f}" if ok else
                           f"LOST ({lost}) - move back to a scanned area")
            now = time.time()
            if now - last_pub > 1.0 and self.tracker.n > 0:
                last_pub = now
                pcd = self.tracker.pointcloud(weight_threshold=2.0)
                self.on_update(pcd, self.tracker.T.copy(), ok)
            else:
                self.on_update(None, self.tracker.T.copy(), ok)

    def close(self):
        self._stop.set()
        self.ev.set()
        self.t.join(timeout=2)
