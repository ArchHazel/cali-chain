"""
Stitch depth images from multiple cameras into a single world-frame point cloud.
- Depth: videos/<cam_name>/depth/depth_1.npy (first frame only)
- Camera extrinsics (T_world_cam): output/cam_extrinsics.json
- Intrinsics: optional output/cam_intrinsics.json or default K below
"""

import json
import numpy as np
from pathlib import Path

# Default intrinsics (e.g. from calibration); override via output/cam_intrinsics.json
DEFAULT_K = np.array([
    [1.110516063691994532e03, 0.0, 9.560036883813438635e02],
    [0.0, 1.119431526830967186e03, 4.795592441694046784e02],
    [0.0, 0.0, 1.0],
])

OUT_DIR = Path("output")
VIDEOS_DIR = Path("data/videos")
DEPTH_SUBDIR = "depth"
DEPTH_FILENAME = "depth_1.npy"
# If depth is in millimeters, set to 0.001 to convert to meters
DEPTH_SCALE = 0.001


def load_intrinsics():
    """Load K from output/cam_intrinsics.json if present, else default."""
    p = OUT_DIR / "cam_intrinsics.json"
    if p.exists():
        with open(p) as f:
            data = json.load(f)
        # Support single "K" or per-cam "cam_name": [[...]]
        if "K" in data:
            return np.array(data["K"], dtype=np.float64)
        # Use first cam's K as global if per-cam
        first = next(iter(data.values()))
        return np.array(first, dtype=np.float64)
    return DEFAULT_K.copy()


def depth_to_points(depth, K, T_world_cam, depth_min=0.1, depth_max=10.0, stride=1, depth_scale=1.0):
    """
    Unproject depth map to 3D points in world frame.
    depth: (H, W). Use depth_scale=0.001 if depth is in millimeters.
    Invalid: NaN, <= 0, or outside [depth_min, depth_max] (in meters after scale).
    """
    H, W = depth.shape
    z_raw = np.asarray(depth, dtype=np.float64)
    z_raw = z_raw * depth_scale  # e.g. mm -> m

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    u = np.arange(0, W, stride, dtype=np.float64)
    v = np.arange(0, H, stride, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)
    z = z_raw[vv.astype(int), uu.astype(int)]

    valid = np.isfinite(z) & (z > depth_min) & (z <= depth_max)
    uu, vv, z = uu[valid], vv[valid], z[valid]

    x_cam = (uu - cx) * z / fx
    y_cam = (vv - cy) * z / fy
    ones = np.ones_like(z)
    p_cam = np.stack([x_cam, y_cam, z, ones], axis=1)  # (N, 4)
    p_world = (np.array(T_world_cam, dtype=np.float64) @ p_cam.T).T  # (N, 4)
    return p_world[:, :3]


def main():
    # Load extrinsics
    ext_path = OUT_DIR / "cam_extrinsics.json"
    if not ext_path.exists():
        print("Error: output/cam_extrinsics.json not found. Run reconstruct_cam_tag.py first.")
        return
    with open(ext_path) as f:
        cam_extrinsics = json.load(f)

    K = load_intrinsics()
    all_pts = []
    all_colors = []  # optional: one color per cam for visualization

    for cam_name, T_world_cam in cam_extrinsics.items():
        # Try videos/cam_name/... then videos/<id>/... (e.g. cam8 -> 8, cam4_8 -> 4_8)
        depth_path = VIDEOS_DIR / cam_name / DEPTH_SUBDIR / DEPTH_FILENAME
        if not depth_path.exists() and cam_name.startswith("cam"):
            alt_name = cam_name[3:]  # strip "cam"
            depth_path = VIDEOS_DIR / alt_name / DEPTH_SUBDIR / DEPTH_FILENAME
        if not depth_path.exists():
            print(f"Skip {cam_name}: {depth_path} not found")
            continue
        depth = np.load(depth_path)
        if depth.ndim == 3:
            # (num_frames, H, W) -> use first frame only
            depth = depth[0]
        assert depth.ndim == 2, f"Expected (H,W), got {depth.shape}"

        pts = depth_to_points(depth, K, T_world_cam, stride=2, depth_scale=DEPTH_SCALE)
        if pts.size == 0:
            finite = np.isfinite(depth).sum()
            d_min, d_max = np.nanmin(depth), np.nanmax(depth)
            print(f"  {cam_name}: no valid depth points (depth range {d_min:.1f}–{d_max:.1f}, finite {finite}). Try DEPTH_SCALE=0.001 if depth is in mm.")
            continue
        all_pts.append(pts)
        # Optional: color by cam (simple hue)
        n = len(pts)
        hue = hash(cam_name) % 360 / 360.0
        r = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * hue)))
        g = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 1 / 3))))
        b = int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 2 / 3))))
        all_colors.append(np.tile([r, g, b], (n, 1)))
        print(f"  {cam_name}: {n} points")

    if not all_pts:
        print("No points to save.")
        print("Expected depth at: videos/<cam_name>/depth/depth_1.npy or videos/<id>/depth/depth_1.npy")
        print("  (e.g. videos/cam8/depth/depth_1.npy or videos/8/depth/depth_1.npy)")
        return

    pts_world = np.vstack(all_pts)
    colors = np.vstack(all_colors)

    # Save as PLY (with color)
    ply_path = OUT_DIR / "stitched_pointcloud.ply"
    with open(ply_path, "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex {}\n".format(len(pts_world)))
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for i in range(len(pts_world)):
            x, y, z = pts_world[i]
            r, g, b = colors[i]
            f.write(f"{x} {y} {z} {r} {g} {b}\n")
    print(f"Saved {len(pts_world)} points to {ply_path}")


if __name__ == "__main__":
    main()
