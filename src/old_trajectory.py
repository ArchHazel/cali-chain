"""
Transform 3D joint data from camera frame to world frame using HAR camera
position + look-at extrinsics (not AprilTag calibration).

Reads:
  - Joint data: data/<data_session>/videos/<cam>/3d_joint_data.json

Writes:
  - Per-camera world joint data: output/<data_session>/world_trajectories_har/<cam>/3d_joint_data_world.json
  - Trajectory plot: output/<data_session>/world_trajectories_har/trajectory.png

Usage:
    python -m tools.trajectory_har
    python -m tools.trajectory_har data.session=traj_0
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
# Hardcoded HAR camera positions
# A = position (meters), B = look-at point (meters)
# ---------------------------------------------------------------------------

HAR_CAMERAS = {
    "HAR1": {"A": [5.03, 8.45, 0.90], "B": [5.73, 8.18, 0.90]},
    "HAR2": {"A": [0.73, 5.71, 0.90], "B": [1.47, 5.80, 0.90]},
    "HAR3": {"A": [1.42, 7.75, 0.90], "B": [2.18, 7.79, 0.90]},
    "HAR4": {"A": [3.72, 0.32, 0.90], "B": [4.27, 0.86, 0.90]},
    "HAR6": {"A": [4.23, 4.36, 0.90], "B": [3.50, 4.03, 0.90]},
    "HAR8": {"A": [5.00, 4.00, 0.90], "B": [5.28, 3.52, 0.90]},
}


# ---------------------------------------------------------------------------
# Extrinsics from look-at
# ---------------------------------------------------------------------------

def build_extrinsic_from_lookat(position: list, lookat: list) -> np.ndarray:
    """
    Build a 4x4 T_world_cam matrix from camera position and look-at point.

    OpenCV camera convention: +Z = forward (into scene), +X = right, +Y = down.
    World convention: Z = up.
    """
    pos = np.array(position, dtype=np.float64)
    target = np.array(lookat, dtype=np.float64)

    # Camera forward = +Z axis in camera frame
    forward = target - pos
    forward = forward / np.linalg.norm(forward)

    # World up
    world_up = np.array([0.0, 0.0, 1.0])

    # Camera right = +X axis
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)

    # Camera down = +Y axis (OpenCV convention: Y points down)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # T_world_cam: columns are camera axes expressed in world frame
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = pos

    return T


def build_har_extrinsics() -> dict[str, np.ndarray]:
    """Build T_world_cam for all HAR cameras from position + look-at."""
    extrinsics = {}
    for cam_id, params in HAR_CAMERAS.items():
        extrinsics[cam_id] = build_extrinsic_from_lookat(params["A"], params["B"])
        log.info(f"[{cam_id}] Built extrinsic: pos={params['A']}, lookat={params['B']}")
    return extrinsics


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------

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
    cam_id: str,
    videos_dir: Path,
    joint_filename: str,
) -> Path | None:
    """Find 3d_joint_data.json for a camera."""
    for name in [cam_id, f"cam{cam_id}"]:
        p = videos_dir / name / joint_filename
        if p.exists():
            return p
    return None


def process_camera(
    cam_id: str,
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
# Ground truth error metrics
# ---------------------------------------------------------------------------

def point_to_segment_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Shortest distance from point p to line segment a-b (all 2D)."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def point_to_polyline_dist(p: np.ndarray, polyline: np.ndarray) -> float:
    """Shortest distance from point p to a polyline defined by ordered waypoints (2D)."""
    min_dist = np.inf
    for i in range(len(polyline) - 1):
        d = point_to_segment_dist(p, polyline[i], polyline[i + 1])
        if d < min_dist:
            min_dist = d
    return min_dist


def compute_gt_errors(
    cam_trajectories: dict[str, np.ndarray],
    ground_truth: list[list[float]],
) -> dict[str, dict]:
    """
    Compute per-camera distance-to-GT-path metrics.

    Uses XY (top-down) projection only, since GT waypoints are 2D.

    Returns dict of cam_id -> {
        mean, median, std, min, max, p90, p95 (all in meters),
        distances: np.ndarray of per-point distances
    }
    """
    gt_poly = np.array(ground_truth, dtype=np.float64)
    results = {}

    for cam_id, positions in sorted(cam_trajectories.items()):
        xy = positions[:, :2]
        dists = np.array([point_to_polyline_dist(pt, gt_poly) for pt in xy])

        results[cam_id] = {
            "mean": float(np.mean(dists)),
            "median": float(np.median(dists)),
            "std": float(np.std(dists)),
            "min": float(np.min(dists)),
            "max": float(np.max(dists)),
            "p90": float(np.percentile(dists, 90)),
            "p95": float(np.percentile(dists, 95)),
            "n_points": len(dists),
            "distances": dists,
        }

    return results


def log_gt_errors(errors: dict[str, dict]):
    """Pretty-print GT error metrics to log."""
    log.info("\n" + "=" * 70)
    log.info("  Ground Truth Path Error (XY, meters)")
    log.info("=" * 70)
    log.info(f"  {'Camera':<10} {'Mean':>7} {'Median':>7} {'Std':>7} "
             f"{'Min':>7} {'Max':>7} {'P90':>7} {'P95':>7} {'N':>6}")
    log.info("-" * 70)
    for cam_id, e in sorted(errors.items()):
        log.info(f"  {cam_id:<10} {e['mean']:7.3f} {e['median']:7.3f} {e['std']:7.3f} "
                 f"{e['min']:7.3f} {e['max']:7.3f} {e['p90']:7.3f} {e['p95']:7.3f} "
                 f"{e['n_points']:6d}")

    # Overall
    all_dists = np.concatenate([e["distances"] for e in errors.values()])
    log.info("-" * 70)
    log.info(f"  {'ALL':<10} {np.mean(all_dists):7.3f} {np.median(all_dists):7.3f} "
             f"{np.std(all_dists):7.3f} {np.min(all_dists):7.3f} {np.max(all_dists):7.3f} "
             f"{np.percentile(all_dists, 90):7.3f} {np.percentile(all_dists, 95):7.3f} "
             f"{len(all_dists):6d}")
    log.info("=" * 70)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_trajectories(
    cam_trajectories: dict[str, np.ndarray],
    ground_truth: list[list[float]] | None,
    out_path: Path,
    gt_errors: dict[str, dict] | None = None,
):
    """Plot top-down (XY) and side (YZ) views of all camera trajectories."""
    if not cam_trajectories:
        return

    fig, (ax_xy, ax_yz) = plt.subplots(1, 2, figsize=(16, 8))

    cam_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cam_trajectories), 1)))
    all_pts = np.vstack(list(cam_trajectories.values()))
    pad = 0.5

    # --- Top-down XY ---
    ax_xy.set_title("Top-Down View (XY Plane) — HAR Extrinsics", fontsize=14, fontweight="bold")
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

        # Build label with error if available
        label_start = f"{cam_name} start"
        label_end = f"{cam_name} end"
        if gt_errors and cam_name in gt_errors:
            e = gt_errors[cam_name]
            label_start += f" (mean={e['mean']:.2f}m)"

        ax_xy.plot(positions[0, 0], positions[0, 1], "o", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=label_start, zorder=5)
        ax_xy.plot(positions[-1, 0], positions[-1, 1], "s", color=color, markersize=10,
                   markeredgecolor="black", markeredgewidth=1, label=label_end, zorder=5)

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

@hydra.main(config_path="../configs", config_name="old_trajectory", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Transform to World (HAR Extrinsics)\n{OmegaConf.to_yaml(cfg)}")

    # Build extrinsics from hardcoded positions
    cam_extrinsics = build_har_extrinsics()

    videos_dir = Path(cfg.data.videos_dir)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine which cameras to process
    cameras = list(cfg.cameras) if "cameras" in cfg else list(HAR_CAMERAS.keys())

    cam_trajectories: dict[str, np.ndarray] = {}

    for cam_id in cameras:
        log.info(f"\n{'='*50}  {cam_id}  {'='*50}")

        T_world_cam = cam_extrinsics.get(cam_id)
        if T_world_cam is None:
            log.warning(f"[{cam_id}] No HAR extrinsics defined, skipping")
            continue

        joint_path = find_joint_data(cam_id, videos_dir, cfg.data.joint_filename)
        if joint_path is None:
            log.warning(f"[{cam_id}] No joint data found, skipping")
            continue

        log.info(f"  Joint data: {joint_path}")
        world_data = process_camera(cam_id, T_world_cam, joint_path)

        if not world_data:
            log.warning(f"[{cam_id}] No frames in joint data")
            continue

        # Save
        cam_out = out_dir / cam_id
        cam_out.mkdir(parents=True, exist_ok=True)
        out_path = cam_out / "3d_joint_data_world.json"
        with open(out_path, "w") as f:
            json.dump(world_data, f, indent=2)
        log.info(f"  Saved -> {out_path}")

        # Collect for visualization
        positions = np.array(list(world_data.values()))
        if len(positions) > 0:
            cam_trajectories[cam_id] = positions

    # Visualize
    if cam_trajectories:
        gt = OmegaConf.to_container(cfg.ground_truth, resolve=True) if cfg.ground_truth else None

        # Compute GT path errors
        gt_errors = None
        if gt:
            gt_errors = compute_gt_errors(cam_trajectories, gt)
            log_gt_errors(gt_errors)

            # Save error metrics to JSON (without the distances array)
            errors_json = {}
            for cam_id, e in gt_errors.items():
                errors_json[cam_id] = {k: v for k, v in e.items() if k != "distances"}
            # Add overall stats
            all_dists = np.concatenate([e["distances"] for e in gt_errors.values()])
            errors_json["ALL"] = {
                "mean": float(np.mean(all_dists)),
                "median": float(np.median(all_dists)),
                "std": float(np.std(all_dists)),
                "min": float(np.min(all_dists)),
                "max": float(np.max(all_dists)),
                "p90": float(np.percentile(all_dists, 90)),
                "p95": float(np.percentile(all_dists, 95)),
                "n_points": len(all_dists),
            }
            with open(out_dir / "gt_path_errors.json", "w") as f:
                json.dump(errors_json, f, indent=2)
            log.info(f"Saved GT errors -> {out_dir / 'gt_path_errors.json'}")

        visualize_trajectories(cam_trajectories, gt, out_dir / "trajectory.png", gt_errors)
    else:
        log.warning("No trajectories to visualize")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()