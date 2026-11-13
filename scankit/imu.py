"""Gravity alignment of a reconstruction with the recorded IMU (imu.csv).

The calibration file does not give the IMU axes relative to the camera, so they are estimated
from the scan. Between two frames, the camera rotation (from the poses) and the gyro-integrated
IMU rotation are the same rotation seen in two frames: r_cam = R_ci r_imu for their rotation
vectors. All frame pairs together give R_ci (Kabsch), and a small time offset between the frame
and IMU clocks is searched for. The accelerometer, averaged around each frame and rotated into
the world, then gives the up direction. The output world has +Z up.
"""

from __future__ import annotations

import cv2
import numpy as np


def _exp_batch(w: np.ndarray) -> np.ndarray:
    """Rotation matrices for N rotation vectors."""
    th = np.linalg.norm(w, axis=1)
    k = w / np.maximum(th, 1e-12)[:, None]
    Kx = np.zeros((len(w), 3, 3))
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 2] = -k[:, 2], k[:, 1], -k[:, 0]
    Kx -= Kx.transpose(0, 2, 1)
    s, c = np.sin(th)[:, None, None], np.cos(th)[:, None, None]
    return np.eye(3) + s * Kx + (1 - c) * Kx @ Kx


def _rotvec(R: np.ndarray) -> np.ndarray:
    return cv2.Rodrigues(R)[0][:, 0]


def integrate_gyro(imu: np.ndarray) -> np.ndarray:
    """Orientation world_R_imu (arbitrary start) at every IMU sample."""
    t = imu[:, 0]
    dt = np.diff(t, prepend=t[0])
    dt = np.clip(dt, 0, 5 * np.median(dt[1:]))      # recording pauses: no integration over gaps
    inc = _exp_batch(imu[:, 1:4] * dt[:, None])
    Q = np.empty_like(inc)
    R = np.eye(3)
    for k in range(len(inc)):
        R = R @ inc[k]
        Q[k] = R
    return Q


def _kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """R minimising sum |a_i - R b_i|^2."""
    U, _, Vt = np.linalg.svd(b.T @ a)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def gravity_alignment(times: np.ndarray, segments: np.ndarray, frame_poses: list, imu: np.ndarray,
                      log=print, max_offset: float = 0.15):
    """Returns (4x4 world rotation that puts gravity on -Z, info dict) or (None, info)."""
    Q = integrate_gyro(imu)
    ti = imu[:, 0]
    at = lambda t: np.clip(np.searchsorted(ti, t), 0, len(ti) - 1)
    pairs = []
    for (i1, T1), (i2, T2) in zip(frame_poses[:-1], frame_poses[1:]):
        if segments[i1] != segments[i2] or not 0 < times[i2] - times[i1] < 1.0:
            continue
        rc = _rotvec(T1[:3, :3].T @ T2[:3, :3])
        if np.degrees(np.linalg.norm(rc)) > 2.0:
            pairs.append((times[i1], times[i2], rc))
    info = {"imu_samples": int(len(imu)), "imu_pairs": len(pairs)}
    if len(pairs) < 15:
        log(f"  IMU: only {len(pairs)} usable frame pairs (need 15 with >2 deg rotation), not aligning")
        return None, info

    a = np.array([p[2] for p in pairs])
    best = None
    for off in np.arange(-max_offset, max_offset + 1e-9, 0.005):
        b = np.array([_rotvec(Q[at(t1 + off)].T @ Q[at(t2 + off)]) for t1, t2, _ in pairs])
        R = _kabsch(a, b)
        err = np.degrees(np.linalg.norm(a - b @ R.T, axis=1))
        score = float(np.median(err))
        if best is None or score < best[0]:
            best = (score, off, R)
    score, off, R_ci = best
    info.update({"imu_time_offset_s": round(float(off), 3), "imu_rot_residual_deg": round(score, 3),
                 "R_cam_imu": np.round(R_ci, 5).tolist()})
    if score > 1.5:
        log(f"  IMU: camera/IMU rotations disagree (median {score:.1f} deg), not aligning")
        return None, info

    ups = []
    for i, T in frame_poses:
        m = np.abs(ti - (times[i] + off)) < 0.2
        if m.sum() >= 10:
            f = imu[m, 4:7].mean(0)                  # specific force ~ +g (up) when not accelerating
            ups.append(T[:3, :3] @ R_ci @ f)
    ups = np.array(ups)
    g = float(np.median(np.linalg.norm(ups, axis=1)))
    u = ups.sum(0)
    u /= np.linalg.norm(u)
    spread = float(np.degrees(np.median(np.arccos(np.clip(ups @ u / np.linalg.norm(ups, axis=1), -1, 1)))))
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, z)
    ang = np.arctan2(np.linalg.norm(axis), u @ z)
    A = np.eye(4)
    A[:3, :3] = cv2.Rodrigues(axis / max(np.linalg.norm(axis), 1e-12) * ang)[0]
    info.update({"gravity_m_s2": round(g, 3), "gravity_spread_deg": round(spread, 2)})
    log(f"  IMU: camera/IMU rotation residual {score:.2f} deg, time offset {off * 1000:.0f} ms, "
        f"|g| {g:.2f} m/s^2, up direction spread {spread:.1f} deg -> world +Z is up")
    return A, info
