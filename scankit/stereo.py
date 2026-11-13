"""Disparity and depth from a rectified stereo pair, for any camera.

Backends
  raftstereo : RAFT-Stereo realtime (vendored in third_party/, weights downloaded once to
         ~/.cache/scankit). Default: most accurate and fastest of the three on desk3.
  raft : RAFT optical flow (torchvision weights, trained on synthetic + real data). The
         horizontal flow between rectified views is the disparity, the vertical flow is a
         quality signal. Runs on CUDA, Apple MPS or CPU.
  sgbm : OpenCV semi-global block matching (CPU, fast, noisier, holes on weak texture).

Both compute the disparity of each view and keep only pixels where the two agree
(left-right check). Disparity d is in pixels and positive: x_left - x_right = d.
"""

from __future__ import annotations

import os

import cv2
import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img


def sgbm_disparity(left: np.ndarray, right: np.ndarray, num_disp: int = 128, block: int = 5) -> np.ndarray:
    """Disparity of the left view in pixels (0 = no match)."""
    num_disp = int(np.ceil(max(num_disp, 16) / 16) * 16)
    m = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block,
                              P1=8 * block * block, P2=32 * block * block, uniquenessRatio=10,
                              speckleWindowSize=100, speckleRange=2, disp12MaxDiff=1,
                              mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    d = m.compute(_gray(left), _gray(right)).astype(np.float32) / 16.0
    d[d <= 0] = 0
    return d


class SGBM:
    def __init__(self, num_disp: int = 128):
        self.num_disp = num_disp

    def both(self, left, right):
        """Returns (disparity of left, disparity of right, None)."""
        dl = sgbm_disparity(left, right, self.num_disp)
        # the right view's disparity: mirror both images, the right view becomes the left one
        dr = sgbm_disparity(right[:, ::-1].copy(), left[:, ::-1].copy(), self.num_disp)[:, ::-1]
        return dl, np.ascontiguousarray(dr), None


class RAFT:
    def __init__(self, device: str = "auto", iters: int = 20):
        import torch
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.torch, self.device, self.iters = torch, device, iters
        weights = Raft_Large_Weights.C_T_SKHT_V2
        self.model = raft_large(weights=weights).eval().to(device)
        self.transforms = weights.transforms()

    def both(self, left, right):
        """Returns (disparity of left, disparity of right, (|vertical flow| of left, of right))."""
        torch = self.torch
        h, w = left.shape[:2]

        def t(img):
            img = np.repeat(img[..., None], 3, 2) if img.ndim == 2 else img
            return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)[None]
        a, b = t(left), t(right)
        img1, img2 = self.transforms(torch.cat([a, b]), torch.cat([b, a]))
        ph, pw = (-h) % 8, (-w) % 8
        pad = lambda x: torch.nn.functional.pad(x, (0, pw, 0, ph), mode="replicate")
        with torch.no_grad():
            flow = self.model(pad(img1).to(self.device), pad(img2).to(self.device),
                              num_flow_updates=self.iters)[-1]
        flow = flow[..., :h, :w].float().cpu().numpy()
        # left->right flow is -d, right->left flow is +d
        return -flow[0, 0], flow[1, 0], (np.abs(flow[0, 1]), np.abs(flow[1, 1]))


RAFT_STEREO_ZIP = ("https://www.dropbox.com/scl/fi/5khx1bhz84dapi8vtwapg/models.zip"
                   "?rlkey=ggddrn1du1iiq6mgc2dsdpmwi&dl=1")
RAFT_STEREO_CKPT = "raftstereo-realtime.pth"


def _raft_stereo_weights() -> str:
    """Path of the realtime checkpoint. Downloads the release zip (~200 MB) once, keeps only it."""
    import io
    import urllib.request
    import zipfile
    path = os.path.join(os.path.expanduser("~/.cache/scankit"), RAFT_STEREO_CKPT)
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        print(f"downloading RAFT-Stereo weights -> {path}", flush=True)
        with urllib.request.urlopen(RAFT_STEREO_ZIP, timeout=60) as resp:
            z = zipfile.ZipFile(io.BytesIO(resp.read()))
        name = next(n for n in z.namelist() if n.endswith(RAFT_STEREO_CKPT) and "__MACOSX" not in n)
        with open(path + ".tmp", "wb") as fh:
            fh.write(z.read(name))
        os.replace(path + ".tmp", path)
    return path


class RAFTStereo:
    """RAFT-Stereo, realtime variant (1D correlation, 1/8 resolution GRUs). On desk3 against
    COLMAP points: 0.93% depth error vs 0.96% for RAFT, at 1/3 of the runtime."""

    def __init__(self, device: str = "auto", iters: int = 7):
        import types

        import torch
        from .third_party.raft_stereo import RAFTStereo as Net
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        args = types.SimpleNamespace(
            hidden_dims=[128] * 3, corr_implementation="reg", shared_backbone=True, corr_levels=4,
            corr_radius=4, n_downsample=3, context_norm="batch", slow_fast_gru=True, n_gru_layers=2,
            mixed_precision=False)
        net = Net(args)
        sd = torch.load(_raft_stereo_weights(), map_location="cpu")
        net.load_state_dict({k.removeprefix("module."): v for k, v in sd.items()})
        self.torch, self.device, self.iters = torch, device, iters
        self.model = net.eval().to(device)

    def both(self, left, right):
        """Returns (disparity of left, disparity of right, None)."""
        torch = self.torch
        h, w = left.shape[:2]

        def t(img):
            img = np.repeat(img[..., None], 3, 2) if img.ndim == 2 else img
            return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()
        # one batch: (left, right), and the mirrored pair whose "left" view is the right one
        a = torch.stack([t(left), t(right[:, ::-1])])
        b = torch.stack([t(right), t(left[:, ::-1])])
        pad = lambda x: torch.nn.functional.pad(x, (0, (-w) % 32, 0, (-h) % 32), mode="replicate")
        with torch.no_grad():
            _, flow = self.model(pad(a).to(self.device), pad(b).to(self.device), iters=self.iters,
                                 test_mode=True)
        d = -flow[:, 0, :h, :w].float().cpu().numpy()       # flow of the left view is -d
        return d[0], np.ascontiguousarray(d[1][:, ::-1]), None


METHODS = ("raftstereo", "raft", "sgbm")


def make_matcher(method: str, device: str = "auto", iters: int | None = None, num_disp: int = 128):
    if method == "raftstereo":
        return RAFTStereo(device, iters or 7)
    if method == "raft":
        return RAFT(device, iters or 20)
    if method == "sgbm":
        return SGBM(num_disp)
    raise ValueError(f"unknown stereo method {method!r} ({' | '.join(METHODS)})")


def lr_check(d_ref: np.ndarray, d_other: np.ndarray, ref_is_left: bool, tol: float) -> np.ndarray:
    """True where the other view's disparity at the matched pixel agrees with d_ref."""
    h, w = d_ref.shape
    xs = np.arange(w, dtype=np.float32)[None, :] + (-d_ref if ref_is_left else d_ref)
    ys = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, 1)
    back = cv2.remap(d_other.astype(np.float32), xs.astype(np.float32), ys, cv2.INTER_NEAREST,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=-1)
    return (back >= 0) & (np.abs(back - d_ref) <= tol)


def stereo_depth(matcher, ref: np.ndarray, other: np.ndarray, ref_is_left: bool, fx: float,
                 baseline_m: float, depth_min: float = 0.1, depth_max: float = 10.0,
                 scale: float = 1.0, lr_tol: float = 1.0, max_vflow: float = 1.0) -> np.ndarray:
    """Depth in mm (uint16, 0 = invalid) of the reference view of a rectified pair."""
    h, w = ref.shape[:2]
    if scale != 1.0:
        ref = cv2.resize(ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        other = cv2.resize(other, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    left, right = (ref, other) if ref_is_left else (other, ref)
    dl, dr, vflow = matcher.both(left, right)
    d_ref, d_other = (dl, dr) if ref_is_left else (dr, dl)
    valid = (d_ref > 0.5) & lr_check(d_ref, d_other, ref_is_left, lr_tol * scale)
    if vflow is not None:       # rows are aligned after rectification: vertical flow = mismatch
        valid &= vflow[0 if ref_is_left else 1] <= max_vflow * scale
    z = np.zeros(d_ref.shape, np.float32)
    z[valid] = fx * scale * baseline_m * 1000.0 / d_ref[valid]
    z[(z < depth_min * 1000) | (z > depth_max * 1000)] = 0
    if scale != 1.0:
        z = cv2.resize(z, (w, h), interpolation=cv2.INTER_NEAREST)
    return np.clip(np.rint(z), 0, 65535).astype(np.uint16)


def _poly(x, y, deg=4):
    return np.stack([x ** i * y ** j for i in range(deg + 1) for j in range(deg + 1 - i)], -1)


def row_correction(pairs, ref_is_left: bool, size, deg: int = 4):
    """Static row misalignment of a stereo rig, estimated from SIFT matches over several frames.

    pairs: [(ref, other)] images. Returns (map_x, map_y, info). cv2.remap(other, map_x, map_y)
    moves every pixel of the other view onto the row of its match in the reference view.
    A degree-4 polynomial over the image is fitted robustly to dy = y_ref - y_other.
    """
    w, h = size
    sift = cv2.SIFT_create(4000)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    pts = []
    for ref, other in pairs:
        kr, dr = sift.detectAndCompute(_gray(ref), None)
        ko, do = sift.detectAndCompute(_gray(other), None)
        if dr is None or do is None:
            continue
        for m in bf.knnMatch(do, dr, k=2):
            if len(m) < 2 or m[0].distance > 0.7 * m[1].distance:
                continue
            (xo, yo), (xr, yr) = ko[m[0].queryIdx].pt, kr[m[0].trainIdx].pt
            disp = (xr - xo) if ref_is_left else (xo - xr)      # x_left - x_right
            if 1 < disp < 300 and abs(yr - yo) < 15:
                pts.append((xo, yo, yr - yo))
    p = np.array(pts)
    if len(p) < 500:
        raise RuntimeError(f"row correction: only {len(p)} feature matches")
    A = _poly(p[:, 0] / w * 2 - 1, p[:, 1] / h * 2 - 1, deg)
    keep = np.ones(len(p), bool)
    for _ in range(5):              # robust fit: drop matches far from the smooth field
        c = np.linalg.lstsq(A[keep], p[keep, 2], rcond=None)[0]
        res = p[:, 2] - A @ c
        sd = 1.4826 * np.median(np.abs(res[keep]))
        keep = np.abs(res) < max(3 * sd, 0.3)
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    field = (_poly(xs / w * 2 - 1, ys / h * 2 - 1, deg) @ c).astype(np.float32)
    info = {"matches": int(len(p)), "inliers": int(keep.sum()),
            "fit_residual_px": round(float(sd), 3),
            "before_median_abs_dy_px": round(float(np.median(np.abs(p[keep, 2]))), 3),
            "after_median_abs_dy_px": round(float(np.median(np.abs(res[keep]))), 3),
            "field_min_max_px": [round(float(field.min()), 2), round(float(field.max()), 2)],
            "coeffs": c.round(5).tolist(), "degree": deg}
    return xs, ys - field, info
