"""Frame-to-model dense tracking (TSDF raycast + colored ICP, ORB/PnP visual odometry).
Used by the capture preview and by the `tracking` pose method."""

from __future__ import annotations

import cv2
import numpy as np
import open3d as o3d
import open3d.core as o3c

from .frames import backproject

reg = o3d.pipelines.registration


def rot_deg(T):
    return float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1))))


class ModelTracker:
    """Frame-to-model tracking.

    1. Initial guess from visual odometry (ORB features + PnP RANSAC against the last tracked
       frame, using its depth). Robust to fast motion, frame gaps and planar scenes.
    2. Refinement with colored ICP (geometry + image intensity) against a raycast of the TSDF
       model. The photometric term stops ICP from sliding along planes (floors, walls) or
       thin structures where geometry alone is degenerate.
    3. Sanity checks: model overlap, speed, and agreement between ICP and confident VO.
    Rejected frames leave the model untouched; the caller can simply skip them.
    """

    def __init__(self, K, w, h, voxel_size, depth_scale, depth_min, depth_max,
                 depth_diff=0.05, min_fitness=0.35, max_step=0.08, trunc_mult=6.0,
                 block_count=40000, device=None, raycast_scale=0.5,
                 use_color=True, use_vo=True, lambda_geometric=0.968,
                 vo_check_trans=0.05, vo_check_deg=5.0):
        self.dev = device or o3c.Device("CPU:0")
        self.K = np.asarray(K, np.float64)
        self.Kt = o3c.Tensor(self.K)
        self.rs = raycast_scale
        Kr = self.K.copy()
        Kr[:2] *= raycast_scale
        self.Kr = o3c.Tensor(Kr)
        self.w, self.h = w, h
        self.p = dict(voxel_size=voxel_size, depth_scale=depth_scale, depth_min=depth_min,
                      depth_max=depth_max, depth_diff=depth_diff, min_fitness=min_fitness,
                      max_step=max_step, trunc=trunc_mult, block_count=block_count,
                      use_color=use_color, use_vo=use_vo, lambda_geometric=lambda_geometric,
                      vo_check_trans=vo_check_trans, vo_check_deg=vo_check_deg)
        self.orb = cv2.ORB_create(2000, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.reset()

    def reset(self, T_init=None):
        p = self.p
        self.vbg = o3d.t.geometry.VoxelBlockGrid(
            attr_names=("tsdf", "weight", "color"),
            attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
            attr_channels=((1), (1), (3)),
            voxel_size=p["voxel_size"], block_resolution=16, block_count=p["block_count"], device=self.dev)
        self.T = np.eye(4) if T_init is None else T_init.copy()
        self.blocks = None
        self.n = 0
        self.prev = None            # (keypoints, descriptors, depth, pose, gray) of last tracked frame
        self.n_vo = 0               # frames where the VO pose was preferred over ICP
        self.last_fitness = 1.0
        self.last_reason = ""
        self.last_vo_inliers = 0

    # -- model
    def _integrate(self, depth_u16, color_u8):
        p = self.p
        depth = o3d.t.geometry.Image(np.ascontiguousarray(depth_u16)).to(self.dev)
        color = o3d.t.geometry.Image(np.ascontiguousarray(color_u8)).to(self.dev)
        extr = o3c.Tensor(np.linalg.inv(self.T))
        self.blocks = self.vbg.compute_unique_block_coordinates(depth, self.Kt, extr, p["depth_scale"],
                                                                p["depth_max"], p["trunc"])
        self.vbg.integrate(self.blocks, depth, color, self.Kt, extr, p["depth_scale"],
                           p["depth_max"], p["trunc"])

    def _model_cloud(self, T):
        p = self.p
        w, h = int(self.w * self.rs), int(self.h * self.rs)
        attrs = ["vertex", "normal", "color"] if p["use_color"] else ["vertex", "normal"]
        r = self.vbg.ray_cast(self.blocks, self.Kr, o3c.Tensor(np.linalg.inv(T)), w, h,
                              attrs, p["depth_scale"], p["depth_min"], p["depth_max"],
                              1.0, p["trunc"])
        v = r["vertex"].cpu().numpy().reshape(-1, 3)
        nrm = r["normal"].cpu().numpy().reshape(-1, 3)
        ok = np.isfinite(v).all(1) & (np.abs(v).sum(1) > 0) & (np.abs(nrm).sum(1) > 0.5)
        # ray_cast returns camera-frame vertices/normals -> world frame
        # and remove Open3D's ray_cast offset of about half a voxel (mesh/point cloud extraction has none)
        R, t = T[:3, :3], T[:3, 3]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(
            v[ok].astype(np.float64) @ R.T + t - 0.5 * p["voxel_size"]))
        pc.normals = o3d.utility.Vector3dVector(nrm[ok].astype(np.float64) @ R.T)
        if p["use_color"]:
            col = r["color"].cpu().numpy().reshape(-1, 3)[ok].astype(np.float64)
            pc.colors = o3d.utility.Vector3dVector(np.clip(col, 0, 1))
        return pc.voxel_down_sample(p["voxel_size"])

    # -- visual odometry
    def _features(self, gray):
        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def _vo(self, kp, des):
        """PnP of the previous frame's 3D keypoints in the current image -> world_T_cam."""
        self.last_vo_inliers = 0
        if self.prev is None or des is None or self.prev[1] is None or len(kp) < 20:
            return None
        pkp, pdes, pdepth, pT, _ = self.prev
        matches = self.bf.knnMatch(pdes, des, k=2)
        obj, img = [], []
        K, ds = self.K, self.p["depth_scale"]
        for m in matches:
            if len(m) < 2 or m[0].distance > 0.8 * m[1].distance:
                continue
            u, v = pkp[m[0].queryIdx].pt
            z = pdepth[int(round(v)) % self.h, int(round(u)) % self.w] / ds
            if z <= 0:
                continue
            obj.append(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z))
            img.append(kp[m[0].trainIdx].pt)
        if len(obj) < 15:
            return None
        obj = np.array(obj, np.float64)
        img = np.array(img, np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=300,
                                                 reprojectionError=2.0, confidence=0.999)
        if not ok or inl is None or len(inl) < 15:
            return None
        inl = inl[:, 0]
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], img[inl], K, None, rvec, tvec)
        T_cp = np.eye(4)
        T_cp[:3, :3] = cv2.Rodrigues(rvec)[0]
        T_cp[:3, 3] = tvec[:, 0]
        self.last_vo_inliers = len(inl)
        return pT @ np.linalg.inv(T_cp)

    # -- main entry
    def add(self, depth_u16: np.ndarray, color_u8: np.ndarray, max_step: float | None = None) -> bool:
        """Track + integrate a frame. Returns False (and leaves the model untouched) if rejected;
        the reason is in self.last_reason."""
        p = self.p
        if (depth_u16 > 0).mean() < 0.03:
            self.last_fitness, self.last_reason = 0.0, "too little depth"
            return False
        gray = cv2.cvtColor(color_u8, cv2.COLOR_RGB2GRAY) if color_u8.ndim == 3 else color_u8
        kp, des = self._features(gray) if p["use_vo"] else (None, None)
        if self.n == 0:
            self._integrate(depth_u16, color_u8)
            self.prev = (kp, des, depth_u16, self.T.copy(), gray)
            self.n = 1
            return True

        T_vo = self._vo(kp, des) if p["use_vo"] else None
        T = T_vo if T_vo is not None else self.T
        pts, vv, uu = backproject(depth_u16, self.K, p["depth_scale"], 2, return_pixels=True)
        src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
        if p["use_color"]:
            g = gray[vv, uu].astype(np.float64) / 255.0
            src.colors = o3d.utility.Vector3dVector(np.repeat(g[:, None], 3, 1))
        src = src.voxel_down_sample(p["voxel_size"])
        tgt = self._model_cloud(self.T)
        if len(tgt.points) < 300 or len(src.points) < 300:
            self.last_fitness, self.last_reason = 0.0, "no model overlap"
            return False

        crit = lambda it: reg.ICPConvergenceCriteria(relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=it)
        levels = ((p["depth_diff"], 15), (p["voxel_size"] * 2.5, 10), (p["voxel_size"] * 1.5, 10))
        try:
            for dist, it in levels:
                if p["use_color"]:
                    est = reg.TransformationEstimationForColoredICP(p["lambda_geometric"], reg.TukeyLoss(k=dist))
                    T = reg.registration_colored_icp(src, tgt, dist, T, est, crit(it)).transformation
                else:
                    est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=dist))
                    T = reg.registration_icp(src, tgt, dist, T, est, crit(it)).transformation
        except RuntimeError as e:
            self.last_fitness, self.last_reason = 0.0, f"icp error: {e}"
            return False
        # overlap with the model at the coarse threshold (robust to stereo noise at range)
        fit = reg.evaluate_registration(src, tgt, p["depth_diff"], T).fitness
        self.last_fitness = float(fit)
        step = np.linalg.inv(self.T) @ T
        if fit < p["min_fitness"]:
            self.last_reason = f"overlap {fit:.2f}"
            return False
        if np.linalg.norm(step[:3, 3]) > (max_step or p["max_step"]):
            self.last_reason = f"jump {np.linalg.norm(step[:3, 3]) * 1000:.0f} mm"
            return False
        self.last_reason = ""
        if T_vo is not None and self.last_vo_inliers >= 40:
            dev = np.linalg.inv(T_vo) @ T
            if np.linalg.norm(dev[:3, 3]) > p["vo_check_trans"] or rot_deg(dev) > p["vo_check_deg"]:
                # Geometry is ambiguous here (planes, thin parts) or the model is locally off.
                # Keep whichever pose explains the image better w.r.t. the previous frame.
                e_icp = self._photo_err(gray, T)
                e_vo = self._photo_err(gray, T_vo)
                if e_vo < e_icp:
                    fit_vo = reg.evaluate_registration(src, tgt, p["depth_diff"], T_vo).fitness
                    if fit_vo < p["min_fitness"]:
                        self.last_reason = f"ICP/VO disagree, VO pose off-model ({fit_vo:.2f})"
                        return False
                    T = T_vo
                    self.n_vo += 1
                    self.last_reason = "vo"
        self.T = T
        self._integrate(depth_u16, color_u8)
        self.prev = (kp, des, depth_u16, T.copy(), gray)
        self.n += 1
        return True

    def _photo_err(self, gray, T):
        """Median intensity error of the previous frame warped into this one with pose T."""
        _, _, pdepth, pT, pgray = self.prev
        pts, v, u = backproject(pdepth, self.K, self.p["depth_scale"], 4, return_pixels=True)
        rel = np.linalg.inv(T) @ pT
        pc = pts @ rel[:3, :3].T + rel[:3, 3]
        m = pc[:, 2] > 0.05
        K = self.K
        uu = K[0, 0] * pc[m, 0] / pc[m, 2] + K[0, 2]
        vv = K[1, 1] * pc[m, 1] / pc[m, 2] + K[1, 2]
        ok = (uu >= 0) & (uu < self.w - 1) & (vv >= 0) & (vv < self.h - 1)
        if ok.sum() < 100:
            return np.inf
        a = pgray[v[m][ok], u[m][ok]].astype(np.float32)
        b = gray[np.round(vv[ok]).astype(int), np.round(uu[ok]).astype(int)].astype(np.float32)
        b *= np.median(a) / max(np.median(b), 1.0)      # auto-exposure changes
        return float(np.median(np.abs(a - b)))

    def pointcloud(self, weight_threshold=1.0) -> o3d.geometry.PointCloud:
        pc = self.vbg.extract_point_cloud(weight_threshold=weight_threshold).to_legacy()
        return pc
