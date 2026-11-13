"""Camera poses from COLMAP structure-from-motion (pycolmap), scaled to metres with stereo depth.

SfM solves all poses jointly (bundle adjustment over every matched image pair), so loops are
closed implicitly and there is no incremental drift. Its scale is arbitrary; we recover it
from the median ratio between stereo depth and SfM depth at the triangulated keypoints.
"""

from __future__ import annotations

import os
import shutil
import time

import numpy as np
import pycolmap


def _cam_from_world(img) -> np.ndarray:
    cfw = img.cam_from_world() if callable(img.cam_from_world) else img.cam_from_world
    T = np.eye(4)
    T[:3, :4] = cfw.matrix()
    return T


def colmap_poses(src, frames, workdir: str, log=print, matcher: str = "exhaustive",
                 max_features: int = 4096, seq_overlap: int = 20, reuse: bool = True):
    """Returns (list[(frame, world_T_cam)], info dict). `src` is a reconstruct.FrameSource."""
    r, K = src.r, src.K
    img_dir = r.image_dir
    db = os.path.join(workdir, "database.db")
    sparse = os.path.join(workdir, "sparse")
    names = [r.names[i] for i in frames]
    stamp = os.path.join(workdir, "frames.txt")
    key = f"{matcher} {max_features} {seq_overlap} " + " ".join(map(str, frames))
    if not (reuse and os.path.exists(stamp) and open(stamp).read() == key):
        for pth in (db, stamp):
            if os.path.exists(pth):
                os.remove(pth)
        shutil.rmtree(sparse, ignore_errors=True)
    os.makedirs(workdir, exist_ok=True)

    recs = {}
    if os.path.isdir(os.path.join(sparse, "0")):
        recs = {0: pycolmap.Reconstruction(os.path.join(sparse, "0"))}
        log(f"  reusing cached COLMAP model in {workdir}")
    else:
        t0 = time.time()
        if os.path.exists(db):
            os.remove(db)
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "PINHOLE"
        reader.camera_params = f"{K[0, 0]},{K[1, 1]},{K[0, 2]},{K[1, 2]}"
        ext = pycolmap.FeatureExtractionOptions()
        ext.sift.max_num_features = max_features
        pycolmap.extract_features(db, img_dir, image_names=names, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=ext)
        log(f"  features: {time.time() - t0:.0f}s")
        t0 = time.time()
        if matcher == "exhaustive":
            pycolmap.match_exhaustive(db)
        else:
            pairing = pycolmap.SequentialPairingOptions()
            pairing.overlap = seq_overlap
            pairing.quadratic_overlap = True
            pycolmap.match_sequential(db, pairing_options=pairing)
        log(f"  matching ({matcher}): {time.time() - t0:.0f}s")
        t0 = time.time()
        opts = pycolmap.IncrementalPipelineOptions()
        opts.ba_refine_focal_length = False        # factory calibration is better than self-calibration
        opts.ba_refine_principal_point = False
        opts.ba_refine_extra_params = False
        os.makedirs(sparse, exist_ok=True)
        recs = pycolmap.incremental_mapping(db, img_dir, sparse, opts)
        log(f"  mapping: {time.time() - t0:.0f}s")
        with open(stamp, "w") as fh:
            fh.write(key)
    if not recs:
        raise RuntimeError("COLMAP could not reconstruct the scan (too few matches?)")
    rec = max(recs.values(), key=lambda x: x.num_reg_images())
    if len(recs) > 1:
        log(f"  WARNING: COLMAP produced {len(recs)} separate models; using the largest "
            f"({rec.num_reg_images()} images), sizes {[x.num_reg_images() for x in recs.values()]}")

    # metric scale: stereo depth / SfM depth at triangulated keypoints
    name_to_frame = {n: i for i, n in zip(frames, names)}
    poses, ratios = {}, []
    dmin, dmax = src.mm(src.cfg.depth_min), src.mm(src.cfg.depth_max)
    for img in rec.images.values():
        if img.name not in name_to_frame:
            continue
        fi = name_to_frame[img.name]
        T_cw = _cam_from_world(img)
        poses[fi] = np.linalg.inv(T_cw)
        depth = src.load(fi, src.cfg.depth_max)[0]
        for p2 in img.points2D:
            if not p2.has_point3D():
                continue
            X = rec.points3D[p2.point3D_id].xyz
            zc = (T_cw[:3, :3] @ X + T_cw[:3, 3])[2]
            u, v = int(round(p2.xy[0])), int(round(p2.xy[1]))
            if zc <= 0 or not (0 <= u < src.w and 0 <= v < src.h):
                continue
            zs = float(depth[v, u])
            if dmin < zs < dmax:
                ratios.append(zs / src.depth_scale / zc)
    if len(ratios) < 100:
        raise RuntimeError("Too few stereo-depth/SfM correspondences to recover scale")
    ratios = np.array(ratios)
    scale = float(np.median(ratios))
    mad = float(np.median(np.abs(ratios / scale - 1)))
    out = []
    for fi in sorted(poses):
        T = poses[fi].copy()
        T[:3, 3] *= scale
        out.append((fi, T))
    info = {"registered": len(out), "input": len(frames), "scale": scale,
            "scale_samples": int(len(ratios)), "scale_rel_mad": round(mad, 4),
            "mean_reproj_error_px": round(float(rec.compute_mean_reprojection_error()), 3),
            "points3D": int(rec.num_points3D())}
    log(f"  registered {len(out)}/{len(frames)} frames, reprojection error "
        f"{info['mean_reproj_error_px']} px, scale {scale:.4f} m/unit from {len(ratios)} points "
        f"(rel. MAD {mad * 100:.1f}%)")
    return out, info
