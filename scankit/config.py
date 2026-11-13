"""Reconstruction settings. Every field is also a `scan reconstruct` flag (--field-name)."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # frame selection
    start: int = 0
    end: int = -1                       # exclusive, -1 = all
    stride: int = 1
    segments: list = field(default_factory=list)   # empty = all segments
    blur_reject_rel: float = 0.6        # drop frames with sharpness < X * local median (0 = off)
    # depth
    depth_min: float = 0.2              # m
    depth_max: float = 1.5              # m, used for integration (stereo error grows ~z^2)
    track_depth_max: float = 2.5        # m, used for tracking (more structure helps)
    depth_dir: str = "depth"            # depth (camera) | depth_raftstereo, depth_raft, depth_sgbm (scan depth)
    depth_scale_correction: float = 1.0  # multiply all depths (measure a known length to calibrate)
    edge_rel_threshold: float = 0.03    # drop pixels whose 3x3 depth range > X * depth
    edge_erode_px: int = 0              # extra erosion of valid-depth borders (kills thin regions)
    undistort: str = "auto"             # auto|on|off (only matters if distortion stored)
    # poses
    pose_method: str = "colmap"         # colmap (SfM + bundle adjustment) | tracking (dense SLAM)
    colmap_matcher: str = "exhaustive"  # exhaustive (robust, O(n^2)) | sequential (long scans)
    colmap_max_images: int = 300        # frames are sub-sampled evenly to at most this many
    colmap_max_features: int = 4096
    colmap_cache: bool = True           # reuse <scan>/colmap if frames/settings match
    gravity_align: bool = True          # rotate the result so that +Z is up (needs imu.csv)
    # fragment tracking (pose_method=tracking)
    fragment_size: int = 100            # frames per fragment
    min_fragment_frames: int = 8
    track_voxel_size: float = 0.008     # m
    track_depth_diff: float = 0.05      # m, max correspondence depth difference
    track_min_fitness: float = 0.35     # below -> tracking lost -> new fragment
    track_max_speed: float = 1.0        # m/s, larger jumps between frames = tracking failure
    track_max_skip: int = 8             # consecutive rejected frames skipped before a new fragment
    track_use_color: bool = True        # colored ICP (geometry + intensity)
    track_use_vo: bool = True           # ORB/PnP visual odometry for initialisation + checks
    # registration / loop closure
    reg_voxel_size: float = 0.015       # m, fragment cloud downsampling
    loop_closure: bool = True
    loop_min_overlap: float = 0.3       # fraction of points with correspondences
    loop_max_correction: float = 0.15   # m, max deviation from the tracked prior (same chain)
    loop_max_correction_deg: float = 10.0
    cross_min_support: int = 2          # agreeing matches needed to join separate chains
    keyframe_every: int = 10            # frames between keyframes used for visual loop detection
    loop_min_inliers: int = 40          # PnP inliers for a visual loop candidate
    loop_max_correction_visual: float = 0.6   # m, allowed drift correction for visual loops
    adjacent_icp: bool = True           # refine adjacent links with ICP
    edge_prune_threshold: float = 0.25
    preference_loop_closure: float = 1.0
    ransac_iterations: int = 1000000
    ransac_confidence: float = 0.999
    # final integration
    voxel_size: float = 0.004           # m
    trunc_voxel_multiplier: float = 6.0
    weight_threshold: float = 3.0       # min observations to keep a voxel
    block_count: int = 60000
    # output
    min_cluster_fraction: float = 0.02  # remove mesh islands smaller than X * largest
    keep_largest_only: bool = False
    save_pointcloud: bool = True
    save_fragments: bool = False
    device: str = "auto"                # auto | CPU:0 | CUDA:0
    seed: int = 0
