"""
Transform 3D joint data from camera frame to world frame and visualize trajectories.

Reads:
  - Extrinsics: output/<ext_session>/extrinsics/cam_extrinsics.json
  - Joint data: data/<data_session>/videos/<cam>/3d_joint_data.json

Writes:
  - Per-camera world joint data: output/<data_session>/world_trajectories/<cam>/3d_joint_data_world.json
  - Trajectory plot: output/<data_session>/world_trajectories/trajectory.png

Usage:
    python -m src.transform_to_world
    python -m src.transform_to_world data.session=calib_4 extrinsics.session=calib_4
"""

import json
import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def load_extrinsics(extrinsics_dir: Path, filename: str) -> dict[str, np.ndarray]:
    ext_path = extrinsics_dir / filename
    if not ext_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {ext_path}")
    with open(ext_path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


def transform_to_world(
    points_cam_mm: list[float],
    T_world_cam: np.ndarray,
    flip_x: bool = True,
) -> np.ndarray:
    """
    Transform [x, y, z] from camera frame (mm) to world frame (meters).
    flip_x: negate X because joint data was computed on horizontally flipped images.
    """
    pts = np.array(points_cam_mm, dtype=np.float64)
    if flip_x:
        pts[0] = -pts[0]
    pts_m = pts / 1000.0

    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return R @ pts_m + t


def find_joint_data(
    cam_key: str,
    videos_dir: Path,
    joint_filename: str,
) -> Path | None:
    """
    Find 3d_joint_data.json for a camera.
    Tries:
      1. videos/<cam_key>/                     (e.g. camHAR6/)
      2. videos/<cam_key without 'cam' prefix>/ (e.g. HAR6/)
    """
    cam_name = cam_key[3:] if cam_key.startswith("cam") else cam_key

    for name in [cam_key, cam_name]:
        p = videos_dir / name / joint_filename
        if p.exists():
            return p

    return None


def process_camera(
    cam_key: str,
    T_world_cam: np.ndarray,
    joint_path: Path,
) -> dict[str, list[float]]:
    """Transform all frames for one camera. Returns {frame_name: [x, y, z] in world}."""
    with open(joint_path) as f:
        joint_data = json.load(f)

    cam_pos = T_world_cam[:3, 3]
    look_dir = T_world_cam[:3, 2]
    log.info(
        f"  Camera pos: [{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]  "
        f"look: [{look_dir[0]:.3f}, {look_dir[1]:.3f}, {look_dir[2]:.3f}]"
    )

    world_data = {}
    for frame_name, joint_pos_cam in joint_data.items():
        world_pos = transform_to_world(joint_pos_cam, T_world_cam)
        world_data[frame_name] = world_pos.tolist()

    # Sanity check
    all_pts = np.array(list(world_data.values()))
    if len(all_pts) > 0:
        if np.any(all_pts < -5) or np.any(all_pts > 20):
            log.warning(f"  Some world coordinates look unusual — check extrinsics")
        else:
            log.info(f"  World coordinates look reasonable")
        log.info(
            f"  {len(world_data)} frames  "
            f"X: [{all_pts[:, 0].min():.2f}, {all_pts[:, 0].max():.2f}]  "
            f"Y: [{all_pts[:, 1].min():.2f}, {all_pts[:, 1].max():.2f}]  "
            f"Z: [{all_pts[:, 2].min():.2f}, {all_pts[:, 2].max():.2f}]"
        )

    return world_data


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def visualize_trajectories(
    cam_trajectories: dict[str, np.ndarray],
    ground_truth: list[list[float]] | None,
    out_path: Path,
):
    """Plot top-down (XY) and side (YZ) views of all camera trajectories."""
    if not cam_trajectories:
        return

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(16, 8))

    cam_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cam_trajectories), 1)))
    all_pts = np.vstack(list(cam_trajectories.values()))
    pad = 0.5

    # --- Top-down XY ---
    ax_xy.set_title("Top-Down View (XY Plane)", fontsize=14, fontweight="bold")
    ax_xy.set_xlabel("X (meters)", fontsize=12)
    ax_xy.set_ylabel("Y (meters)", fontsize=12)

    if ground_truth:
        gt = np.array(ground_truth)
        ax_xy.plot(gt[:, 0], gt[:, 1], "-", color="black", linewidth=2, alpha=0.5, label="GT path", zorder=1)
        ax_xy.plot(gt[0, 0], gt[0, 1], "^", color="black", markersize=10, zorder=2)
        ax_xy.plot(gt[-1, 0], gt[-1, 1], "v", color="black", markersize=10, zorder=2)

    for i, (cam_name, positions) in enumerate(sorted(cam_trajectories.items())):
        color = cam_colors[i]
        ax_xy.plot(positions[:, 0], positions[:, 1], "-", color=color, alpha=0.3, linewidth=1, zorder=3)
        ax_xy.scatter(positions[:, 0], positions[:, 1], color=color, s=15, alpha=0.6, zorder=4)
        ax_xy.plot(positions[0, 0], positions[0, 1], "o", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} start", zorder=5)
        ax_xy.plot(positions[-1, 0], positions[-1, 1], "s", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} end", zorder=5)

    all_xy = all_pts[:, :2]
    if ground_truth:
        all_xy = np.vstack([all_xy, np.array(ground_truth)])
    ax_xy.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax_xy.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)
    ax_xy.set_aspect("equal")
    ax_xy.xaxis.set_major_locator(MultipleLocator(1.0))
    ax_xy.yaxis.set_major_locator(MultipleLocator(1.0))
    ax_xy.grid(True, which="major", color="black", alpha=0.15, linewidth=0.8)
    ax_xy.legend(fontsize=8, loc="upper left")

    # --- Side YZ ---
    ax_yz.set_title("Side View (YZ Plane)", fontsize=14, fontweight="bold")
    ax_yz.set_xlabel("Y (meters)", fontsize=12)
    ax_yz.set_ylabel("Z (meters)", fontsize=12)

    for i, (cam_name, positions) in enumerate(sorted(cam_trajectories.items())):
        color = cam_colors[i]
        ax_yz.plot(positions[:, 1], positions[:, 2], "-", color=color, alpha=0.3, linewidth=1)
        ax_yz.scatter(positions[:, 1], positions[:, 2], color=color, s=15, alpha=0.6)
        ax_yz.plot(positions[0, 1], positions[0, 2], "o", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} start")
        ax_yz.plot(positions[-1, 1], positions[-1, 2], "s", color=color, markersize=10,
                markeredgecolor="black", markeredgewidth=1, label=f"{cam_name} end")

    ax_yz.set_xlim(all_pts[:, 1].min() - pad, all_pts[:, 1].max() + pad)
    ax_yz.set_ylim(all_pts[:, 2].min() - pad, all_pts[:, 2].max() + pad)
    ax_yz.set_aspect("equal")
    ax_yz.grid(True, alpha=0.3)
    ax_yz.axhline(y=0, color="brown", linestyle="-", lw=1, alpha=0.5, label="Floor")
    ax_yz.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved trajectory plot -> {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="trajectory", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Transform to World Pipeline\n{OmegaConf.to_yaml(cfg)}")

    videos_dir = Path(cfg.data.videos_dir)
    extrinsics_dir = Path(cfg.extrinsics.dir)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load extrinsics
    cam_extrinsics = load_extrinsics(extrinsics_dir, cfg.extrinsics.filename)
    log.info(f"Loaded extrinsics for {len(cam_extrinsics)} cameras: {list(cam_extrinsics.keys())}")

    # Process each camera
    cam_trajectories: dict[str, np.ndarray] = {}

    for cam_key, T_world_cam in sorted(cam_extrinsics.items()):
        log.info(f"\n{'='*50}  {cam_key}  {'='*50}")

        joint_path = find_joint_data(cam_key, videos_dir, cfg.data.joint_filename)
        if joint_path is None:
            log.warning(f"[{cam_key}] No joint data found, skipping")
            continue

        log.info(f"  Joint data: {joint_path}")
        world_data = process_camera(cam_key, T_world_cam, joint_path)

        if not world_data:
            log.warning(f"[{cam_key}] No frames in joint data")
            continue

        # Save
        cam_out = out_dir / cam_key
        cam_out.mkdir(parents=True, exist_ok=True)
        out_path = cam_out / "3d_joint_data_world.json"
        with open(out_path, "w") as f:
            json.dump(world_data, f, indent=2)
        log.info(f"  Saved -> {out_path}")

        # Collect for visualization
        positions = np.array(list(world_data.values()))
        if len(positions) > 0:
            cam_trajectories[cam_key] = positions

    # Visualize
    if cam_trajectories:
        gt = OmegaConf.to_container(cfg.ground_truth, resolve=True) if cfg.ground_truth else None
        visualize_trajectories(cam_trajectories, gt, out_dir / "trajectory.png")
    else:
        log.warning("No trajectories to visualize")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()