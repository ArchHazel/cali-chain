# visualize_trajectory.py
"""
Visualize joint trajectories in world coordinates for one or more cameras.

Reads:
  - output/<cam_name>/3d_joint_data_world.json  (from transform_to_world.py)

Usage:
  python visualize_trajectory.py --cameras cam8 cam4
  python visualize_trajectory.py                      # all cameras with world data
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from pathlib import Path


# Ground-truth waypoints (x, y) in meters — piecewise axis-aligned path
GT = [
    (1, 1),
    (2, 1),
    (2, 3.5),
    (6, 3.5),
    (2, 3.5),
    (2, 6),
    (3, 6),
    (3, 8),
    (3, 6),
    (7, 6),
    (3, 6),
    (3, 3.5),
]

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "output"


def load_world_joint_data(filepath):
    """Load transformed joint data in world coordinates."""
    with open(filepath, "r") as f:
        data = json.load(f)

    frames = sorted(data.keys())
    positions = np.array([data[frame] for frame in frames])

    return frames, positions


def find_cameras_with_world_data():
    """Scan output/ for camera directories containing 3d_joint_data_world.json."""
    cameras = []
    if not OUT_DIR.exists():
        return cameras
    for cam_dir in sorted(OUT_DIR.iterdir()):
        if cam_dir.is_dir() and (cam_dir / "3d_joint_data_world.json").exists():
            cameras.append(cam_dir.name)
    return cameras


def main():
    parser = argparse.ArgumentParser(description="Visualize joint trajectories in world coordinates")
    parser.add_argument("--cameras", nargs="*", default=None,
                        help="Camera names to visualize (default: all with world data)")
    parser.add_argument("--no-show", action="store_true",
                        help="Don't show interactive plot, just save")
    args = parser.parse_args()

    # Discover cameras
    if args.cameras:
        cameras = args.cameras
    else:
        cameras = find_cameras_with_world_data()
        if not cameras:
            print("No cameras found with world joint data in output/*/3d_joint_data_world.json")
            print("Run transform_to_world.py first.")
            return

    print(f"Visualizing trajectories for: {cameras}\n")

    # Color cycle for multiple cameras
    cam_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cameras), 1)))

    # --- Top-down view (XY) ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    ax_xy = axes[0]
    ax_xy.set_title("Top-Down View (XY Plane)", fontsize=14, fontweight="bold")
    ax_xy.set_xlabel("X (meters)", fontsize=12)
    ax_xy.set_ylabel("Y (meters)", fontsize=12)

    ax_yz = axes[1]
    ax_yz.set_title("Side View (YZ Plane)", fontsize=14, fontweight="bold")
    ax_yz.set_xlabel("Y (meters)", fontsize=12)
    ax_yz.set_ylabel("Z (meters)", fontsize=12)

    # Draw GT trajectory on XY plot
    gt_arr = np.array(GT)
    ax_xy.plot(gt_arr[:, 0], gt_arr[:, 1], "-", color="black", linewidth=2, alpha=0.5, label="GT path", zorder=1)
    ax_xy.plot(gt_arr[0, 0], gt_arr[0, 1], "^", color="black", markersize=10, zorder=2)
    ax_xy.plot(gt_arr[-1, 0], gt_arr[-1, 1], "v", color="black", markersize=10, zorder=2)

    all_positions = []

    for i, cam_name in enumerate(cameras):
        world_path = OUT_DIR / cam_name / "3d_joint_data_world.json"
        if not world_path.exists():
            print(f"  [SKIP] {cam_name}: {world_path} not found")
            continue

        frames, positions = load_world_joint_data(world_path)
        if(len(positions) < 1):
            print(f"  [SKIP] {cam_name}: no data found")
            continue
        all_positions.append(positions)
        color = cam_colors[i]

        avg_pos = positions.mean(axis=0)
        std_pos = positions.std(axis=0)

        print(f"--- {cam_name} ({len(frames)} frames) ---")
        print(f"  X: [{positions[:, 0].min():.2f}, {positions[:, 0].max():.2f}] m")
        print(f"  Y: [{positions[:, 1].min():.2f}, {positions[:, 1].max():.2f}] m")
        print(f"  Z: [{positions[:, 2].min():.2f}, {positions[:, 2].max():.2f}] m")
        print(f"  Avg: [{avg_pos[0]:.3f}, {avg_pos[1]:.3f}, {avg_pos[2]:.3f}] m")
        print(f"  Std: [{std_pos[0]:.3f}, {std_pos[1]:.3f}, {std_pos[2]:.3f}] m\n")

        # XY plot - solid camera color
        ax_xy.plot(positions[:, 0], positions[:, 1], "-", color=color, alpha=0.3, linewidth=1, zorder=3)
        ax_xy.scatter(positions[:, 0], positions[:, 1],
                      color=color, s=15, alpha=0.6, zorder=4)
        ax_xy.plot(positions[0, 0], positions[0, 1], "o", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} start", zorder=5)
        ax_xy.plot(positions[-1, 0], positions[-1, 1], "s", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} end", zorder=5)
        ax_xy.plot(avg_pos[0], avg_pos[1], "D", color=color, markersize=8,
                   markeredgecolor="black", markeredgewidth=1.5, alpha=0.8, zorder=5)

        # YZ plot - solid camera color
        ax_yz.plot(positions[:, 1], positions[:, 2], "-", color=color, alpha=0.3, linewidth=1)
        ax_yz.scatter(positions[:, 1], positions[:, 2],
                      color=color, s=15, alpha=0.6)
        ax_yz.plot(positions[0, 1], positions[0, 2], "o", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} start")
        ax_yz.plot(positions[-1, 1], positions[-1, 2], "s", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} end")

    if not all_positions:
        print("No data to plot.")
        return

    # Auto-scale axes with some padding
    all_pos = np.vstack(all_positions)
    pad = 0.5

    # Include GT points in axis limits
    all_xy = np.vstack([all_pos[:, :2], gt_arr])
    x_min, x_max = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    y_min, y_max = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    z_min, z_max = all_pos[:, 2].min() - pad, all_pos[:, 2].max() + pad

    ax_xy.set_xlim(x_min, x_max)
    ax_xy.set_ylim(y_min, y_max)
    ax_xy.set_aspect("equal")
    # 1m black grid on XY plot
    ax_xy.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_xy.yaxis.set_major_locator(MultipleLocator(1.0))
    ax_xy.grid(True, which="major", color="black", alpha=0.15, linewidth=0.8)
    ax_xy.legend(fontsize=8, loc="upper left")

    ax_yz.set_xlim(y_min, y_max)
    ax_yz.set_ylim(z_min, z_max)
    ax_yz.set_aspect("equal")
    ax_yz.grid(True, alpha=0.3)
    ax_yz.axhline(y=0, color="brown", linestyle="-", lw=1, alpha=0.5, label="Floor")
    ax_yz.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    # Save
    cam_suffix = "_".join(cameras) if len(cameras) <= 4 else f"{len(cameras)}cams"
    output_path = OUT_DIR / f"trajectory_{cam_suffix}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved to: {output_path}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()