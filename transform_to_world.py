# transform_to_world.py
"""
Transform 3D joint data from camera frame to world frame using cali-chain extrinsics.

Reads:
  - output/cam_extrinsics.json  (T_world_cam 4x4 matrices from reconstruct_cam_tag.py)
  - data/videos/<cam_name>/3d_joint_data.json  (joint positions in camera frame, mm)

Writes:
  - output/<cam_name>/3d_joint_data_world.json  (joint positions in world frame, meters)
"""

import json
import argparse
import numpy as np
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
VIDEOS_DIR = BASE_DIR / "data" / "videos"
OUT_DIR = BASE_DIR / "output"


def load_cam_extrinsics(filepath):
    """Load all camera extrinsics (T_world_cam) from JSON."""
    with open(filepath, "r") as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


def transform_camera_to_world(points_cam_mm, T_world_cam):
    """
    Transform points from camera frame to world frame.

    Args:
        points_cam_mm: [x, y, z] in camera frame (mm)
        T_world_cam: 4x4 transformation matrix (world <- camera)

    Returns:
        [x, y, z] in world frame (meters)
    """
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]

    points_cam_m = np.array(points_cam_mm) / 1000.0
    points_world_m = R @ points_cam_m + t

    return points_world_m


def compute_look_at_point(T_world_cam, distance=1.0):
    """
    Compute a point B that the camera is looking at.

    Camera +Z axis in world = third column of R (from T_world_cam).

    Returns:
        A: camera position in world (meters)
        B: look-at point in world (meters)
    """
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]

    cam_pos = t
    cam_z_in_world = R[:, 2]  # third column = camera +Z in world
    look_at = cam_pos + distance * cam_z_in_world

    return cam_pos, look_at


def process_camera(cam_name, T_world_cam):
    """Process joint data for a single camera. Returns dict or None."""
    # Try to find joint data
    joint_path = VIDEOS_DIR / cam_name / "3d_joint_data.json"
    if not joint_path.exists() and cam_name.startswith("cam"):
        joint_path = VIDEOS_DIR / cam_name[3:] / "3d_joint_data.json"
    if not joint_path.exists():
        print(f"  [SKIP] No 3d_joint_data.json found")
        return None

    with open(joint_path, "r") as f:
        joint_data_cam = json.load(f)

    print(f"  Found {len(joint_data_cam)} frames")

    # Print camera info
    cam_pos, look_at = compute_look_at_point(T_world_cam)
    print(f"  Camera position (world, m): [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]")
    print(f"  Look-at point (world, m):   [{look_at[0]:.3f}, {look_at[1]:.3f}, {look_at[2]:.3f}]")

    # Transform each frame
    joint_data_world = {}
    for frame_name, joint_pos_cam in joint_data_cam.items():
        # note we do this x negation since joint data was calculated on flipped images
        joint_pos_cam[0] = -joint_pos_cam[0]
        joint_pos_world = transform_camera_to_world(joint_pos_cam, T_world_cam)
        joint_data_world[frame_name] = joint_pos_world.tolist()

    # Sanity check
    all_world = np.array(list(joint_data_world.values()))
    if np.any(all_world < -5) or np.any(all_world > 20):
        print(f"  ⚠️  WARNING: Some world coordinates seem unusual. Check extrinsics.")
    else:
        print(f"  ✓ World coordinates look reasonable.")

    return joint_data_world


def main():
    parser = argparse.ArgumentParser(description="Transform joint data from camera to world frame")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Specific camera names to process (default: all in extrinsics)")
    parser.add_argument("--extrinsics", type=str, default=str(OUT_DIR / "cam_extrinsics.json"),
                        help="Path to cam_extrinsics.json")
    args = parser.parse_args()

    print("=" * 60)
    print("  Transform Joint Data to World Coordinates (cali-chain)")
    print("=" * 60)

    # 1. Load extrinsics
    extrinsics_path = Path(args.extrinsics)
    if not extrinsics_path.exists():
        print(f"Error: {extrinsics_path} not found. Run reconstruct_cam_tag.py first.")
        return

    cam_extrinsics = load_cam_extrinsics(extrinsics_path)
    print(f"\nLoaded extrinsics for {len(cam_extrinsics)} cameras: {list(cam_extrinsics.keys())}")

    # 2. Filter cameras if specified
    cameras_to_process = args.cameras if args.cameras else list(cam_extrinsics.keys())

    # 3. Process each camera
    for cam_name in cameras_to_process:
        if cam_name not in cam_extrinsics:
            print(f"\n[SKIP] {cam_name}: not in extrinsics file")
            continue

        print(f"\n--- {cam_name} ---")
        T_world_cam = cam_extrinsics[cam_name]
        result = process_camera(cam_name, T_world_cam)

        if result is not None:
            cam_out_dir = OUT_DIR / cam_name
            cam_out_dir.mkdir(parents=True, exist_ok=True)
            output_path = cam_out_dir / "3d_joint_data_world.json"
            with open(output_path, "w") as f:
                json.dump(result, f, indent=2)
            print(f"  Saved to: {output_path}")

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()