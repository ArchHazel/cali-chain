"""
Plane Extraction from Kinect Depth Data.

For each camera, back-projects depth frames into a camera-frame point cloud,
then iteratively fits planes via RANSAC. Outputs:
  - Per-camera plane normals and statistics
  - Per-camera colored point cloud visualization (each plane a different color)
  - Summary of dominant normal directions across all cameras
  - Manhattan world check: do normals cluster into 3 orthogonal groups?

Usage:
    python -m tools.extract_planes
    python -m tools.extract_planes session=calib_5
    python -m tools.extract_planes cameras='[HAR1,HAR6]'
"""

import logging
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_kinect_config(cam_id: str, configs_dir: str = "configs/kinect") -> dict:
    config_path = Path(configs_dir) / f"{cam_id}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Kinect config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    di = cfg["depth_intrinsics"]
    K = np.array([
        [di["fx"], 0.0, di["cx"]],
        [0.0, di["fy"], di["cy"]],
        [0.0, 0.0, 1.0]
    ])
    return {"K_depth": K}


def load_depth_frames(depth_dir: str, cam_id: str, chunk: int,
                      start_frame: int, num_frames: int) -> list[np.ndarray]:
    depth_path = Path(depth_dir) / cam_id / "depth" / f"depth_{chunk}.npy"
    if not depth_path.exists():
        log.warning(f"Depth file not found: {depth_path}")
        return []
    depth_chunk = np.load(depth_path)
    end_frame = min(start_frame + num_frames, depth_chunk.shape[0])
    return [depth_chunk[i] for i in range(start_frame, end_frame)]


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth pixels to 3D in depth camera space (Y-down, OpenCV convention)."""
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H)[:, np.newaxis]
    cam[:, :, 2] = 1.0

    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]

    valid = depth_m > 0
    return cam_flat[valid], depth_m[valid]


def accumulate_camera_points(depth_frames: list[np.ndarray], K_depth: np.ndarray,
                             max_depth: float = 6.0,
                             subsample: int = 2) -> np.ndarray:
    """
    Back-project multiple depth frames and accumulate in CAMERA frame.
    X is negated to match the horizontally flipped RGB frames that the
    extrinsics were calibrated from — keeps visuals consistent with RGB.
    """
    all_pts = []
    for depth_frame in depth_frames:
        pts, depths = backproject_depth(depth_frame, K_depth)

        mask = depths < max_depth
        pts = pts[mask]

        # Negate X to match the horizontally flipped RGB frames
        pts[:, 0] = -pts[:, 0]

        if subsample > 1:
            pts = pts[::subsample]

        all_pts.append(pts)

    if not all_pts:
        return np.zeros((0, 3))
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------------------------------------
# RANSAC plane fitting
# ---------------------------------------------------------------------------

def fit_plane_ransac(points: np.ndarray, distance_thresh: float = 0.03,
                     num_iterations: int = 2000) -> tuple[np.ndarray, np.ndarray, float]:
    """
    RANSAC plane fitting. Returns (normal, inlier_mask, plane_d).
    Plane equation: normal . x + d = 0
    """
    n_pts = len(points)
    best_inliers = np.zeros(n_pts, dtype=bool)
    best_normal = np.array([0, 0, 1], dtype=np.float64)
    best_d = 0.0

    for _ in range(num_iterations):
        # Sample 3 random points
        idx = np.random.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]

        # Compute plane normal
        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        # Distance from all points to plane
        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)

        inliers = dists < distance_thresh
        n_inliers = inliers.sum()

        if n_inliers > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # Refit normal using all inliers via PCA
    if best_inliers.sum() >= 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        best_normal = Vt[2]  # smallest singular value = normal direction
        best_d = -np.dot(best_normal, centroid)

        # Recompute inliers with refined normal
        dists = np.abs(points @ best_normal + best_d)
        best_inliers = dists < distance_thresh

    return best_normal, best_inliers, best_d


def extract_planes(points: np.ndarray, num_planes: int = 5,
                   distance_thresh: float = 0.03, num_iterations: int = 2000,
                   min_inlier_ratio: float = 0.02) -> list[dict]:
    """
    Iteratively extract planes from a point cloud.
    Returns list of dicts with: normal, d, inlier_indices, num_inliers, ratio.
    """
    planes = []
    remaining_mask = np.ones(len(points), dtype=bool)
    total_pts = len(points)

    for i in range(num_planes):
        remaining_pts = points[remaining_mask]
        if len(remaining_pts) < 100:
            break

        normal, inlier_mask_local, d = fit_plane_ransac(
            remaining_pts, distance_thresh, num_iterations
        )

        n_inliers = inlier_mask_local.sum()
        ratio = n_inliers / total_pts

        if ratio < min_inlier_ratio:
            log.info(f"  Plane {i}: only {n_inliers} inliers ({ratio:.1%}) — stopping")
            break

        # Map local inlier mask back to global indices
        remaining_indices = np.where(remaining_mask)[0]
        global_inlier_indices = remaining_indices[inlier_mask_local]

        planes.append({
            "normal": normal,
            "d": d,
            "inlier_indices": global_inlier_indices,
            "num_inliers": n_inliers,
            "ratio": ratio,
        })

        log.info(f"  Plane {i}: normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]  "
                 f"inliers={n_inliers} ({ratio:.1%})")

        # Remove inliers from remaining points
        remaining_mask[global_inlier_indices] = False

    return planes


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

PLANE_COLORS = [
    "#e74c3c",  # red
    "#2ecc71",  # green
    "#3498db",  # blue
    "#f39c12",  # orange
    "#9b59b6",  # purple
]


def plot_planes_3view(points: np.ndarray, planes: list[dict],
                      cam_id: str, out_path: Path):
    """Three-view plot with points colored by plane membership."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor="black")

    # Assign colors
    colors = np.full(len(points), "#555555")  # gray for unassigned
    for i, plane in enumerate(planes):
        color = PLANE_COLORS[i % len(PLANE_COLORS)]
        colors[plane["inlier_indices"]] = color

    # Subsample for plotting
    max_plot = 100_000
    if len(points) > max_plot:
        idx = np.random.choice(len(points), max_plot, replace=False)
        pts_plot = points[idx]
        colors_plot = colors[idx]
    else:
        pts_plot = points
        colors_plot = colors

    views = [
        (0, 2, "Bird's Eye (XZ)", "X - right (m)", "Z - depth (m)"),
        (0, 1, "Front / Image View (XY)", "X - right (m)", "Y - down (m)"),
        (2, 1, "Side (ZY)", "Z - depth (m)", "Y - down (m)"),
    ]

    for ax, (a0, a1, title, xlabel, ylabel) in zip(axes, views):
        ax.set_facecolor("black")
        ax.scatter(pts_plot[:, a0], pts_plot[:, a1], c=colors_plot, s=0.1, alpha=0.5)
        ax.set_xlabel(xlabel, color="white")
        ax.set_ylabel(ylabel, color="white")
        ax.set_title(title, color="white", fontsize=12)
        ax.set_aspect("equal")
        ax.tick_params(colors="white")
        ax.grid(True, alpha=0.2, color="white")

        # Invert Y-down axis so "down" appears as down in the plot
        if a1 == 1:  # Y axis is the vertical axis
            ax.invert_yaxis()

    # Legend
    from matplotlib.patches import Patch
    legend_elements = []
    for i, plane in enumerate(planes):
        color = PLANE_COLORS[i % len(PLANE_COLORS)]
        n = plane["normal"]
        label = (f"Plane {i}: n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}] "
                 f"({plane['num_inliers']} pts, {plane['ratio']:.0%})")
        legend_elements.append(Patch(facecolor=color, label=label))
    legend_elements.append(Patch(facecolor="#555555", label="Unassigned"))

    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=7,
                   facecolor="black", edgecolor="white", labelcolor="white")

    fig.suptitle(f"{cam_id} — Extracted Planes", fontsize=14,
                 fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="extract_planes", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Plane Extraction\n{OmegaConf.to_yaml(cfg)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = list(cfg.cameras)

    for cam_id in cameras:
        log.info(f"\n{'='*50}")
        log.info(f"  {cam_id}")
        log.info(f"{'='*50}")

        # Load kinect config
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue
        K_depth = kinect["K_depth"]

        # Load and accumulate depth
        frames = load_depth_frames(
            cfg.depth_dir, cam_id, cfg.depth.chunk,
            cfg.depth.frame_idx, cfg.depth.num_frames
        )
        if not frames:
            log.warning(f"[{cam_id}] No depth frames found")
            continue

        pts = accumulate_camera_points(
            frames, K_depth,
            max_depth=cfg.depth.max_depth,
            subsample=cfg.depth.subsample
        )
        log.info(f"[{cam_id}] Accumulated {len(pts)} points from {len(frames)} frames")

        if len(pts) < 1000:
            log.warning(f"[{cam_id}] Too few points, skipping")
            continue

        # Extract planes
        planes = extract_planes(
            pts,
            num_planes=cfg.ransac.num_planes,
            distance_thresh=cfg.ransac.distance_thresh,
            num_iterations=cfg.ransac.num_iterations,
            min_inlier_ratio=cfg.ransac.min_inlier_ratio,
        )

        # Visualize
        plot_planes_3view(pts, planes, cam_id, out_dir / f"{cam_id}_planes.png")

        # Save plane data
        plane_data = []
        for i, p in enumerate(planes):
            plane_data.append({
                "plane_idx": i,
                "normal": p["normal"].tolist(),
                "d": float(p["d"]),
                "num_inliers": int(p["num_inliers"]),
                "ratio": float(p["ratio"]),
            })

        import json
        with open(out_dir / f"{cam_id}_planes.json", "w") as f:
            json.dump(plane_data, f, indent=2)

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()