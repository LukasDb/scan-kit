"""Pose method `tracking`: fragments of tracked frames, fragment registration with loop
closure, and global pose-graph optimisation."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import open3d as o3d

from .config import Config
from .frames import FrameSource
from .tracking import ModelTracker, rot_deg

reg = o3d.pipelines.registration


@dataclass
class Fragment:
    frames: list            # scan frame indices
    poses: list             # fragment_T_camera (4x4) per frame
    linked_prev: bool       # first frame == previous fragment's last frame
    segment: int
    cloud: o3d.geometry.PointCloud | None = None
    fpfh: object = None
    keyframes: list = field(default_factory=list)   # [Keyframe]


@dataclass
class Keyframe:
    frame: int
    pose: np.ndarray        # fragment_T_camera
    p3: np.ndarray          # Nx3 keypoints in camera frame (valid depth only)
    p2: np.ndarray          # Nx2 pixel coordinates (same keypoints)
    desc: np.ndarray        # NxD ORB descriptors


def make_keyframe(frame, pose, kp, des, depth, K, depth_scale, max_kp=1000):
    if des is None or not len(kp):
        return None
    order = np.argsort([-k.response for k in kp])[:max_kp]
    p2 = np.array([kp[j].pt for j in order], np.float64)
    ui = np.clip(np.round(p2[:, 0]).astype(int), 0, depth.shape[1] - 1)
    vi = np.clip(np.round(p2[:, 1]).astype(int), 0, depth.shape[0] - 1)
    z = depth[vi, ui].astype(np.float64) / depth_scale
    ok = z > 0
    p3 = np.stack([(p2[:, 0] - K[0, 2]) * z / K[0, 0], (p2[:, 1] - K[1, 2]) * z / K[1, 1], z], 1)
    if ok.sum() < 30:
        return None
    return Keyframe(frame, pose.copy(), p3[ok], p2[ok], des[order][ok])


_BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def visual_match(a: Fragment, b: Fragment, K, min_inliers=40):
    """Best keyframe-to-keyframe PnP between two fragments -> (T_b_a, inliers) or None."""
    best = None
    for ka in a.keyframes:
        for kb in b.keyframes:
            ms = _BF.knnMatch(ka.desc, kb.desc, k=2)
            good = [m[0] for m in ms if len(m) == 2 and m[0].distance < 0.75 * m[1].distance]
            if len(good) < min_inliers or (best and len(good) <= best[1]):
                continue
            obj = ka.p3[[m.queryIdx for m in good]]
            img = kb.p2[[m.trainIdx for m in good]]
            ok, rvec, tvec, inl = cv2.solvePnPRansac(obj, img, K, None, iterationsCount=500,
                                                     reprojectionError=3.0, confidence=0.999)
            if not ok or inl is None or len(inl) < min_inliers or (best and len(inl) <= best[1]):
                continue
            T_cb_ca = np.eye(4)
            T_cb_ca[:3, :3] = cv2.Rodrigues(rvec)[0]
            T_cb_ca[:3, 3] = tvec[:, 0]
            best = (kb.pose @ T_cb_ca @ np.linalg.inv(ka.pose), len(inl))
    return best


def make_fragments(src: FrameSource, frames: np.ndarray, cfg: Config, dev, log) -> list[Fragment]:
    r = src.r
    tracker = ModelTracker(src.K, src.w, src.h, cfg.track_voxel_size, src.depth_scale,
                           cfg.depth_min, cfg.track_depth_max, cfg.track_depth_diff,
                           cfg.track_min_fitness, 0.1, cfg.trunc_voxel_multiplier,
                           device=dev, use_color=cfg.track_use_color, use_vo=cfg.track_use_vo)
    fragments: list[Fragment] = []
    cur: Fragment | None = None
    lost = 0
    skipped = 0
    t0 = time.time()

    def add_kf(f: Fragment, i):
        if tracker.prev is None:
            return
        kp, des, depth, _, _ = tracker.prev
        kf = make_keyframe(int(i), tracker.T, kp, des, depth, src.K, src.depth_scale)
        if kf is not None:
            f.keyframes.append(kf)

    def finish(f: Fragment):
        if f is None or len(f.frames) < cfg.min_fragment_frames:
            if f is not None and len(f.frames) > 2:
                log(f"  dropping short fragment ({len(f.frames)} frames)")
            return False
        pcd = tracker.pointcloud(weight_threshold=2.0)
        f.cloud = pcd
        fragments.append(f)
        return True

    prev_seg = None
    last_ok_i = None
    n_skip = 0              # consecutive rejected frames
    skipped_total = 0
    for k, i in enumerate(frames):
        seg = int(r.segment[i])
        d, c = src.load(i, cfg.track_depth_max)
        dt = (r.timestamp[i] - r.timestamp[last_ok_i]) if last_ok_i is not None else 0.0
        max_step = max(0.03, cfg.track_max_speed * dt)
        if cur is not None and seg != prev_seg:
            finish(cur)
            cur = None
        if cur is not None and len(cur.frames) >= cfg.fragment_size:
            # close fragment; the next one starts at its last frame (shared frame link)
            last_i, last_d, last_c = cur.frames[-1], cur._last_d, cur._last_c
            ok = finish(cur)
            tracker.reset()
            tracker.add(last_d, last_c)
            cur = Fragment([last_i], [np.eye(4)], linked_prev=ok, segment=seg)
            add_kf(cur, last_i)
            cur._last_d, cur._last_c = last_d, last_c
        if cur is None:
            tracker.reset()
            if not tracker.add(d, c):
                skipped += 1
                prev_seg = seg
                continue
            cur = Fragment([int(i)], [np.eye(4)], linked_prev=False, segment=seg)
            cur._last_d, cur._last_c = d, c
            last_ok_i, n_skip = i, 0
            add_kf(cur, i)
        elif tracker.add(d, c, max_step):
            cur.frames.append(int(i))
            cur.poses.append(tracker.T.copy())
            cur._last_d, cur._last_c = d, c
            last_ok_i, n_skip = i, 0
            if (len(cur.frames) - 1) % cfg.keyframe_every == 0:
                add_kf(cur, i)
        elif n_skip < cfg.track_max_skip:
            # drop this frame, keep tracking against the model from the last good frame
            n_skip += 1
            skipped_total += 1
        else:
            lost += 1
            if len(cur.frames) >= cfg.min_fragment_frames:
                log(f"  tracking lost at frame {i} ({tracker.last_reason}) -> new fragment")
            n_skip = 0
            finish(cur)
            tracker.reset()
            cur = None
            if tracker.add(d, c):
                cur = Fragment([int(i)], [np.eye(4)], linked_prev=False, segment=seg)
                cur._last_d, cur._last_c = d, c
                last_ok_i = i
                add_kf(cur, i)
        prev_seg = seg
        if k % 50 == 0:
            log(f"  tracking {k + 1}/{len(frames)}  fragments={len(fragments)}  "
                f"{(k + 1) / max(time.time() - t0, 1e-6):.1f} fps")
    finish(cur)
    for f in fragments:
        for a in ("_last_d", "_last_c"):
            f.__dict__.pop(a, None)
    if skipped_total:
        log(f"  {skipped_total} frames rejected by tracking checks and skipped")
    if skipped:
        log(f"  {skipped} frames skipped: <3% valid depth after cleaning (in [depth_min, track_depth_max])")
    log(f"  {len(fragments)} fragments, {lost} tracking losses, {time.time() - t0:.0f}s")
    return fragments


def prepare_cloud(f: Fragment, voxel: float):
    pcd = f.cloud.voxel_down_sample(voxel)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 2.5, max_nn=30))
    f.cloud = pcd
    f.fpfh = reg.compute_fpfh_feature(pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=voxel * 5, max_nn=100))


def multiscale_icp(src, tgt, voxel, init):
    T = init
    res = None
    for scale, iters in ((2.0, 50), (1.0, 30), (0.5, 20)):
        dist = voxel * scale * 1.5
        res = reg.registration_icp(src, tgt, dist, T, reg.TransformationEstimationPointToPlane(),
                                   reg.ICPConvergenceCriteria(max_iteration=iters))
        T = res.transformation
    info = reg.get_information_matrix_from_point_clouds(src, tgt, voxel * 0.75, T)
    return T, info, res


def overlap_ok(info, src, tgt, thr):
    return info[5, 5] / max(1, min(len(src.points), len(tgt.points))) >= thr


def register_fragments(frags: list[Fragment], cfg: Config, log, K=None):
    v = cfg.reg_voxel_size
    for f in frags:
        prepare_cloud(f, v)
    edges = []   # (s, t, T_t_s, info, uncertain)
    n = len(frags)
    for s in range(n - 1):
        t = s + 1
        if not frags[t].linked_prev:
            continue
        T_init = np.linalg.inv(frags[s].poses[-1])   # T_t_s via the shared frame
        if cfg.adjacent_icp:
            T, info, _ = multiscale_icp(frags[s].cloud, frags[t].cloud, v, T_init)
            # guard against ICP sliding on degenerate geometry
            if np.linalg.norm(T[:3, 3] - T_init[:3, 3]) > 3 * v:
                T = T_init
                info = reg.get_information_matrix_from_point_clouds(frags[s].cloud, frags[t].cloud, v * 0.75, T)
        else:
            T = T_init
            info = reg.get_information_matrix_from_point_clouds(frags[s].cloud, frags[t].cloud, v * 0.75, T)
        edges.append((s, t, T, info, False))

    if not (cfg.loop_closure and n > 1):
        return edges
    # Tracked chains (fragments connected by shared-frame links) give a reliable prior for
    # every pair inside the chain; loop closures there only refine drift and must stay close
    # to the prior. This rejects false matches in repetitive / symmetric scenes.
    comps = components(n, edges)
    comp_of, prior = {}, {}
    for ci, comp in enumerate(comps):
        poses = init_poses(sorted(comp), [e for e in edges if e[0] in comp])
        for i in comp:
            comp_of[i], prior[i] = ci, poses[i]
    linked = {(s, s + 1) for s, t, *_ in edges}
    same = [(s, t) for s in range(n) for t in range(s + 1, n)
            if comp_of[s] == comp_of[t] and (s, t) not in linked]
    cross = [(s, t) for s in range(n) for t in range(s + 1, n) if comp_of[s] != comp_of[t]]
    log(f"  {len(comps)} tracked chain(s); testing {len(same)} in-chain and {len(cross)} cross-chain pairs")

    def verify(a, b, T_init):
        """ICP-refine a candidate; must keep enough overlap and stay close to the candidate."""
        T, info, _ = multiscale_icp(a.cloud, b.cloud, v, T_init)
        d = np.linalg.inv(T_init) @ T
        if not overlap_ok(info, a.cloud, b.cloud, cfg.loop_min_overlap):
            return None
        if np.linalg.norm(d[:3, 3]) > 0.1 or rot_deg(d) > 6:
            return None
        return T, info

    found_v = found_g = 0
    for s, t in same:
        a, b = frags[s], frags[t]
        T0 = np.linalg.inv(prior[t]) @ prior[s]
        res = None
        vm = visual_match(a, b, K, cfg.loop_min_inliers) if K is not None else None
        if vm is not None:
            d = np.linalg.inv(T0) @ vm[0]
            if np.linalg.norm(d[:3, 3]) <= cfg.loop_max_correction_visual and rot_deg(d) <= 30:
                res = verify(a, b, vm[0])
                found_v += res is not None
        if res is None and reg.evaluate_registration(a.cloud, b.cloud, v * 4, T0).fitness >= 0.15:
            T, info, _ = multiscale_icp(a.cloud, b.cloud, v, T0)
            d = np.linalg.inv(T0) @ T
            if (overlap_ok(info, a.cloud, b.cloud, cfg.loop_min_overlap)
                    and np.linalg.norm(d[:3, 3]) <= cfg.loop_max_correction
                    and rot_deg(d) <= cfg.loop_max_correction_deg):
                res = (T, info)
                found_g += 1
        if res is not None:
            edges.append((s, t, res[0], res[1], True))
    log(f"  in-chain loop closures: {found_v} visual, {found_g} geometric")

    if cross:
        o3d.utility.random.seed(cfg.seed)
        cand = {}
        for s, t in cross:
            a, b = frags[s], frags[t]
            vm = visual_match(a, b, K, cfg.loop_min_inliers) if K is not None else None
            if vm is not None:
                res = verify(a, b, vm[0])
                if res is not None:
                    C = prior[t] @ res[0] @ np.linalg.inv(prior[s])
                    cand.setdefault((comp_of[s], comp_of[t]), []).append((s, t, res[0], res[1], C))
                    continue
            dist = v * 1.5
            res = reg.registration_ransac_based_on_feature_matching(
                a.cloud, b.cloud, a.fpfh, b.fpfh, True, dist,
                reg.TransformationEstimationPointToPoint(False), 3,
                [reg.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                 reg.CorrespondenceCheckerBasedOnDistance(dist),
                 reg.CorrespondenceCheckerBasedOnNormal(np.deg2rad(30))],
                reg.RANSACConvergenceCriteria(cfg.ransac_iterations, cfg.ransac_confidence))
            if res.fitness < 0.1:
                continue
            T, info, _ = multiscale_icp(a.cloud, b.cloud, v, res.transformation)
            if not overlap_ok(info, a.cloud, b.cloud, cfg.loop_min_overlap):
                continue
            # chain-to-chain transform implied by this match
            C = prior[t] @ T @ np.linalg.inv(prior[s])
            key = (comp_of[s], comp_of[t])
            cand.setdefault(key, []).append((s, t, T, info, C))
        joined = 0
        for key, cs in cand.items():
            # keep the largest set of mutually consistent matches (consensus between chains)
            best = []
            for c in cs:
                agree = [o for o in cs if np.linalg.norm(np.linalg.inv(c[4]) @ o[4] - np.eye(4)) < 0.1
                         and np.linalg.norm((np.linalg.inv(c[4]) @ o[4])[:3, 3]) < cfg.loop_max_correction]
                if len(agree) > len(best):
                    best = agree
            if len(best) >= min(cfg.cross_min_support, len(cs)) and (len(best) >= cfg.cross_min_support or
                                                                       len(cs) == 1):
                for s, t, T, info, _ in best:
                    edges.append((s, t, T, info, True))
                joined += 1
        log(f"  {joined} chain pair(s) joined by global registration")
    return edges


def components(n, edges):
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for s, t, *_ in edges:
        parent[find(s)] = find(t)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def init_poses(nodes, edges):
    """Spanning-tree initialisation (certain edges first) of world_T_fragment."""
    poses = {nodes[0]: np.eye(4)}
    adj = {}
    for s, t, T, info, unc in sorted(edges, key=lambda e: e[4]):
        adj.setdefault(s, []).append((t, np.linalg.inv(T)))   # pose_t = pose_s @ inv(T_t_s)
        adj.setdefault(t, []).append((s, T))                  # pose_s = pose_t @ T_t_s
    stack = [nodes[0]]
    while stack:
        a = stack.pop(0)
        for b, rel in adj.get(a, []):
            if b not in poses:
                poses[b] = poses[a] @ rel
                stack.append(b)
    return poses


def optimize(frags, edges, cfg: Config, log):
    n = len(frags)
    nframes = lambda comp: sum(len(frags[i].frames) for i in comp)
    comps = sorted(components(n, edges), key=nframes, reverse=True)
    if len(comps) > 1:
        log(f"  WARNING: {len(comps)} disconnected parts (frames: {[nframes(c) for c in comps]}); "
            f"keeping the largest. Overlap more when resuming after a pause.")
    for _ in range(2):
        keep = sorted(comps[0])
        remap = {o: k for k, o in enumerate(keep)}
        E = [(remap[s], remap[t], T, I, u) for s, t, T, I, u in edges if s in remap and t in remap]
        sub = [frags[i] for i in keep]
        poses = init_poses(list(range(len(sub))), E)
        pg = reg.PoseGraph()
        for i in range(len(sub)):
            pg.nodes.append(reg.PoseGraphNode(poses[i]))
        for s, t, T, I, u in E:
            pg.edges.append(reg.PoseGraphEdge(s, t, T, I, uncertain=u))
        if len(sub) > 1:
            opt = reg.GlobalOptimizationOption(
                max_correspondence_distance=cfg.reg_voxel_size * 1.5,
                edge_prune_threshold=cfg.edge_prune_threshold,
                preference_loop_closure=cfg.preference_loop_closure, reference_node=0)
            with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Error):
                reg.global_optimization(pg, reg.GlobalOptimizationLevenbergMarquardt(),
                                        reg.GlobalOptimizationConvergenceCriteria(), opt)
        surviving = [(e.source_node_id, e.target_node_id) for e in pg.edges]
        n_loop = sum(1 for e in pg.edges if e.uncertain)
        sub_comps = sorted(components(len(sub), [(s, t) for s, t in surviving]),
                           key=lambda c: sum(len(sub[i].frames) for i in c), reverse=True)
        if len(sub_comps) == 1:
            log(f"  pose graph: {len(sub)} fragments, {len(E)} edges, {n_loop} loop closures kept")
            return sub, [np.asarray(nd.pose) for nd in pg.nodes]
        log(f"  pruning disconnected the graph into {len(sub_comps)} parts; re-optimising largest")
        comps = [[keep[i] for i in sub_comps[0]]]
    return sub, [np.asarray(nd.pose) for nd in pg.nodes]


def track_and_optimize(src, frames, cfg: Config, dev, out_dir, log):
    log("[2/5] building fragments (frame-to-model tracking)")
    frags = make_fragments(src, frames, cfg, dev, log)
    if not frags:
        raise SystemExit("Tracking failed everywhere - check depth range / scan quality.")
    if cfg.save_fragments:
        for k, f in enumerate(frags):
            o3d.io.write_point_cloud(os.path.join(out_dir, f"fragment_{k:03d}.ply"), f.cloud)

    log("[3/5] registering fragments")
    edges = register_fragments(frags, cfg, log, src.K)

    log("[4/5] global pose graph optimisation")
    frags, frag_poses = optimize(frags, edges, cfg, log)
    frame_poses, seen = [], set()
    for f, P in zip(frags, frag_poses):
        for i, T in zip(f.frames, f.poses):
            if i not in seen:
                seen.add(i)
                frame_poses.append((i, P @ T))
    frame_poses.sort(key=lambda x: x[0])
    return frame_poses
