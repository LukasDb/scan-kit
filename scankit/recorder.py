"""Writer thread for capture: decouples image encoding from the camera loop."""

from __future__ import annotations

import queue
import threading

from .scanfile import ScanWriter


class Recorder:
    """put(frame tuple) / put("segment") / put(("imu", Nx7 rows)); writes them in order."""

    def __init__(self, writer: ScanWriter):
        self.w = writer
        self.q: queue.Queue = queue.Queue(maxsize=300)
        self.dropped = 0
        self.written = 0
        self._stop = threading.Event()
        self.t = threading.Thread(target=self._run, daemon=True)
        self.t.start()

    def put(self, item):
        try:
            self.q.put_nowait(item)
        except queue.Full:
            self.dropped += 1

    def _run(self):
        while not (self._stop.is_set() and self.q.empty()):
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            if item == "segment":
                self.w.new_segment()
                continue
            if isinstance(item[0], str):
                self.w.append_imu(item[1])
                continue
            self.w.append(*item)
            self.written += 1

    def close(self):
        self._stop.set()
        self.t.join()
        self.w.close()
