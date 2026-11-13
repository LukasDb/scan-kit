# Stereo scanner (OAK-D, OAK-D Lite, ZED Mini)

Capture, depth and reconstruction are separate steps, all run through one command, `scan`. Capture writes each scan as a COLMAP-ready folder. Depth and reconstruction can be re-run on that folder with different settings.

## Setup

```sh
conda env create -f environment.yml      # env "scan", installs the `scan` command
conda activate scan
```

Tab completion for commands, options and paths: add this line to `~/.zshrc` (or `~/.bashrc`), after `compinit` in zsh:

```sh
eval "$(register-python-argcomplete scan)"
```

Open3D is pinned to 0.19: the 0.20 macOS wheel links Homebrew's `libusb`. Open3D's Filament GUI segfaults on current macOS, so the capture GUI uses Tkinter, and the 3D preview uses Open3D's GLFW window.

## Commands

```sh
scan devices                                # connected cameras: model, sensors, USB speed, IMU
scan capture [scans/room]                   # capture GUI: new scan, or append to an existing one
scan info scans/room                        # scan summary
scan depth scans/room                       # neural depth from the stereo pairs -> depth_raftstereo/
scan reconstruct scans/room --depth-dir depth_raftstereo --view
scan view scans/room/recon/<run>/mesh.ply
scan measure scans/room/recon/<run>/mesh.ply
scan zed-check                              # ZED Mini rectification check on live frames
scan export-o3d scans/room out/             # Open3D reconstruction_system layout, for comparison
```

`scan <command> -h` lists the options of a command. A typical session:

```sh
scan capture scans/room                     # record (mono and zed modes save the stereo pair)
scan depth scans/room                       # ~0.2 s per 1280x800 frame on Apple MPS
scan reconstruct scans/room --depth-dir depth_raftstereo
scan reconstruct scans/room                 # same, with the camera's own depth, to compare
```

## Scan folder

```
scans/<scan>/
  images/000000.jpg   texture (reference view, COLMAP image folder)
  depth/000000.png    16-bit depth in mm from the camera, pixel-aligned with images/
  pair/000000.jpg     the other rectified stereo view (input for `scan depth`)
  depth_raftstereo/   depth from `scan depth` (same format as depth/)
  imu.csv             IMU samples: time [s], gyro [rad/s], accel [m/s^2], raw IMU axes
  cameras.txt         COLMAP camera (PINHOLE)
  frames.csv          timestamp, segment, sharpness per frame
  scan.json           metadata, calibration, stereo baseline
  colmap/             COLMAP database + sparse model (created by `scan reconstruct`)
  recon/<run>/        mesh.ply, pointcloud.ply, poses.npz, report.json, config.json
```

## Cameras

| | OAK-D Lite | OAK-D | ZED Mini |
|---|---|---|---|
| Stereo | 640×480 mono, global shutter | 1280×800 mono, global shutter | 1280×720 color per eye (HD), rolling shutter |
| Baseline | 75 mm | 75 mm | 63 mm |
| Depth | on device | on device | coarse live SGBM; use `scan depth` |
| IMU | some versions | BNO086 | yes |
| IR projector | no | no | no |

Without a projector, blank walls give no depth. Textured scenes at 0.3–1.5 m work best.

### Capture modes

- **mono** (OAK, default): depth in the rectified right camera frame, global-shutter grayscale texture, and the rectified left view as the stereo pair.
- **rgb** (OAK): depth aligned to the undistorted RGB camera, with focus locked at the calibration position. Gives a colored mesh. No stereo pair.
- **zed** (ZED Mini): rectified left color texture, rectified right view as the stereo pair, IMU.
- **Resolution**: OAK `full` (sensor size) or `half`. ZED `HD`, `VGA`, `FHD` or `2K` per eye.
- Space or the button starts recording after a 5 s countdown (press again to cancel), and pauses while recording. Each resume starts a new segment. Reconstruction links segments by global registration, so resume with overlap.

### USB

Both cameras need a USB 3 link for full speed. `scan devices` prints the link speed.

- **OAK-D over USB 2**: full resolution (1280×800) runs at about 8 fps, half resolution at about 28 fps.
- **ZED Mini over USB 2**: only the IMU enumerates ("ZED-M Hid Device"), no video. The camera LED stays off.
- If the link is USB 2, change the cable or adapter: many USB-C cables and USB-A-to-C adapters are USB 2 only. Plug in directly, without a hub.

### OAK rectification

For lenses under 85° field of view, DepthAI rectifies with a 3×3 homography, which does not correct lens distortion (on the OAK-D the rows of the two views were off by up to 4 px at the image edges). The driver computes the rectification mesh with OpenCV from the device calibration and uploads it. The rows then agree to about 0.3 px, and the on-chip depth covers about 80% of the image instead of about 50%. Scans recorded before this change (recorded before the change) keep the distortion error in their images: `scan depth` removes the vertical part (row correction), the horizontal part remains at the image edges. Re-record such scans.

At full resolution (1280×800) the on-chip stereo limits capture to about 7 fps, also over USB 3. Half resolution runs at about 25 fps.

### OAK IMU

DepthAI 3.12 does not support BNO086 firmware older than 3.9 (the device crashes when the IMU starts). The driver then records without the IMU and prints why. Update the IMU firmware once with `Device.startIMUFirmwareUpdate()` (about 30 s; do not unplug).

### ZED Mini

No ZED SDK or CUDA is needed: video through AVFoundation (UVC), the IMU through HID. The factory calibration is downloaded once by serial number and cached in `calib/` (git-ignored). Give the terminal camera access (System Settings > Privacy & Security > Camera). Run `scan zed-check` once with the camera pointed at a textured scene: it prints the rectification row error for each calibration convention. The default should be lowest, with a median under about 0.3 px. Calibration files with `k1..k4` and no `p1/p2` use the fisheye (Kannala-Brandt) model; on the unit tested this gives a median row error of 0.26 px, against 0.5 px with the radial-tangential model that zed-open-capture uses. Reference: [zed-open-capture-mac](https://github.com/The-Real-Thisas/zed-open-capture-mac).

## Depth from the stereo pair (`scan depth`)

Works for any scan with `pair/`. Before matching, the static vertical misalignment of the pair is estimated from SIFT matches over the scan and removed (`--row-correction`, on by default; the fit is in `params.json`). Only pixels that pass a left-right consistency check are kept. `--stride 3 --scale 0.5` is much faster. The command resumes: frames that already have depth are skipped. `scan reconstruct` uses only frames with depth in `--depth-dir`, so a strided run is fine.

Methods (`--method`, output in `depth_<method>/`):

| method | | s/frame | depth error | disparity σ |
|---|---|---|---|---|
| `raftstereo` (default) | [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo) realtime, vendored in `scankit/third_party/` (MIT), weights downloaded once to `~/.cache/scankit/` | 0.22 | 0.93% | 0.85 px |
| `raft` | RAFT optical flow, torchvision weights; the vertical flow is also checked | 0.67 | 0.96% | 0.88 px |
| `sgbm` | OpenCV semi-global matching, CPU | | | |
| camera | on-chip OAK-D depth (`depth/`) | | 1.13% | 1.02 px |

Measured on one desk scan (OAK-D, 1280×800, Apple M-series MPS) against the 700k COLMAP triangulated points of the scan: median relative depth error and robust sigma of the disparity residual. Part of the error is SfM noise, so the true differences may be larger. The full-size RAFT-Stereo (Middlebury weights) reached 0.88% at 1.35 s per frame; it is not included.

## Reconstruction (`scan reconstruct`)

```sh
scan reconstruct scans/room -o scans/room/recon/fine --voxel-size 0.003 --depth-max 1.2
scan reconstruct scans/room --config scans/room/recon/fine/config.json --stride 2
scan reconstruct --print-config             # all options; every one is also a --flag
```

Pose methods (`--pose-method`):

- **`colmap`** (default): SIFT features, exhaustive matching, and incremental structure-from-motion with bundle adjustment. Intrinsics are fixed to the calibration. Loops close implicitly and there is no drift. Metric scale is the median ratio of stereo depth to SfM depth over all triangulated keypoints; `report.json` lists the scale and its spread (`scale_rel_mad`). Frames are sub-sampled to `--colmap-max-images` (default 300). The COLMAP workspace in `<scan>/colmap/` is reused, so re-running with different integration settings is fast; it can also be opened in the COLMAP GUI. Very long scans can use `--colmap-matcher sequential`.
- **`tracking`**: dense frame-to-model tracking (ORB/PnP visual-odometry initialization, then colored ICP against the TSDF), fragments, loop closure, and pose-graph optimization. Faster to start, but it drifts on long loops.

**Gravity alignment** (`--gravity-align`, on by default): if the scan has `imu.csv`, the camera/IMU rotation and the clock offset are estimated from the poses and the gyro, and the result is rotated so that +Z is up. `report.json` lists the residual (`imu_rot_residual_deg`) and the gravity estimate. The alignment is skipped if the residual is above 1.5°.

Both pose methods finish with TSDF integration of the depth using the final poses.

Accuracy tips:

- **`--depth-max`**: stereo error grows with z². Keep integration depth short (default 1.5 m) and scan closer rather than farther.
- **`--depth-scale-correction`**: stereo cameras often have a small global scale bias. Scan an object of known size, measure it with `scan measure`, then pass `known/measured` (e.g. `1.004`).
- **`report.json`**: lists the mesh AABB/OBB extents as a quick check.

## Code (`scankit/`)

| module | contents |
|---|---|
| `cli.py` | the `scan` command |
| `gui.py`, `recorder.py`, `preview.py` | capture GUI, writer thread, live tracking preview |
| `cameras/oak.py`, `cameras/zed.py` | OAK (DepthAI) and ZED Mini (UVC + HID) drivers |
| `scanfile.py` | scan folder reader/writer |
| `stereo.py`, `depth.py` | stereo matching backends; `scan depth` |
| `third_party/raft_stereo/` | RAFT-Stereo model code (MIT) |
| `reconstruct.py`, `config.py` | `scan reconstruct` and its settings |
| `frames.py` | frame selection, depth cleaning |
| `colmap.py` | COLMAP poses, scaled to metres with stereo depth |
| `tracking.py`, `fragments.py` | dense tracking, fragments, loop closure, pose graph |
| `imu.py` | camera/IMU rotation from the scan, gravity alignment |
| `tsdf.py` | TSDF integration, mesh clean-up |
| `viewer.py`, `export_o3d.py` | `scan view`, `scan measure`; `scan export-o3d` |

Each command imports only its own modules: depthai and pycolmap bundle conflicting glog builds and must not share a process.
