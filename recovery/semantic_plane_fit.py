"""
Semantic Plane Fitting.

Uses semantic segmentation masks (from semantic_planes.py) to filter
depth points, then fits planes to wall/floor/ceiling regions only.

Pipeline:
  1. Load semantic masks from output/<session>/semantic/<cam>_<class>_mask.npy
  2. Load depth frames and back-project to 3D (camera frame)
  3. Project depth points into RGB space via DLT
  4. Look up each depth point's semantic label from the mask
  5. Fit planes (via RANSAC) to each structural class separately
  6. Output per-camera plane parameters (normal + rho) and visualization

Prerequisites:
    Run semantic_planes.py first to generate the masks.

Usage:
    python -m tools.semantic_plane_fit
    python -m tools.semantic_plane_fit session=calib_5
    python -m tools.semantic_plane_fit session=calib_5 cameras='[HAR1,HAR6]'
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080

STRUCTURAL_CLASSES = ["wall", "floor", "ceiling"]
STRUCTURAL_COLORS = {
    "wall": "#3498db",
    "floor": "#2ecc71",
    "ceiling": "#e74c3c",
}


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
    corr_path = cfg.get("depth_to_color_correspondences", None)
    return {"K_depth": K, "corr_path": corr_path}


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
# DLT
# ---------------------------------------------------------------------------

def fit_dlt(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
    """Fit a 3x4 DLT projection matrix from 3D depth-camera-space to 2D color pixels."""
    n = len(points_3d)
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth pixels to 3D in depth camera space.
    Y is reversed to match Kinect SDK CameraSpacePoint convention (Y-up).
    This is the same convention used by visualize_depth.py and the DLT correspondences.
    """
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H - 1, -1, -1)[:, np.newaxis]
    cam[:, :, 2] = 1.0

    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]

    valid = depth_m > 0
    return cam_flat[valid], depth_m[valid]


def project_to_color_pixels(pts_3d: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Project 3D depth-camera-space points to color pixel coordinates via DLT.
    Returns (pixel_coords [N, 2], valid_mask [N]).
    Pixel coords are flipped horizontally to match the pre-flipped RGB frames.
    """
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    proj = (P @ pts_h.T).T

    w = proj[:, 2]
    px = (proj[:, 0] / w).astype(np.int32)
    py = (proj[:, 1] / w).astype(np.int32)

    # Flip horizontally to match pre-flipped RGB frames
    px = COLOR_W - 1 - px

    valid = (w > 0) & (px >= 0) & (px < COLOR_W) & (py >= 0) & (py < COLOR_H)
    return np.stack([px, py], axis=1), valid


# ---------------------------------------------------------------------------
# Plane fitting
# ---------------------------------------------------------------------------

def fit_plane_ransac(points: np.ndarray, distance_thresh: float = 0.03,
                     num_iterations: int = 2000) -> tuple[np.ndarray, float, np.ndarray]:
    """
    RANSAC plane fitting with PCA refinement.
    Returns (normal, rho, inlier_mask).
    Normal convention: points toward camera (positive Z preferred).
    """
    n_pts = len(points)
    if n_pts < 10:
        return None, None, None

    best_inliers = np.zeros(n_pts, dtype=bool)
    best_normal = np.array([0, 0, 1], dtype=np.float64)
    best_d = 0.0

    for _ in range(num_iterations):
        idx = np.random.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]

        v1 = p1 - p0
        v2 = p2 - p0
        normal = np.cross(v1, v2)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal = normal / norm

        d = -np.dot(normal, p0)
        dists = np.abs(points @ normal + d)
        inliers = dists < distance_thresh
        if inliers.sum() > best_inliers.sum():
            best_inliers = inliers
            best_normal = normal
            best_d = d

    # PCA refit on inliers
    if best_inliers.sum() >= 3:
        inlier_pts = points[best_inliers]
        centroid = inlier_pts.mean(axis=0)
        centered = inlier_pts - centroid
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        best_normal = Vt[2]
        best_d = -np.dot(best_normal, centroid)

        # Recompute inliers
        dists = np.abs(points @ best_normal + best_d)
        best_inliers = dists < distance_thresh

    # Enforce sign convention: normal should have positive Z (pointing toward camera)
    # This makes rho consistent across sessions
    if best_normal[2] < 0:
        best_normal = -best_normal
        best_d = -best_d

    rho = -best_d  # distance from origin to plane along normal
    return best_normal, rho, best_inliers


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_semantic_planes(points: np.ndarray, labels: np.ndarray,
                         planes: dict, cam_id: str, out_path: Path):
    """Three-view plot with points colored by semantic class + plane fits."""
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor="black")

    WALL_COLORS_VIS = ["#3498db", "#1abc9c", "#9b59b6"]  # blue, teal, purple
    colors = np.full(len(points), "#333333")  # dark gray for unlabeled/other

    # Color floor and ceiling
    colors[labels == "floor"] = STRUCTURAL_COLORS["floor"]
    colors[labels == "ceiling"] = STRUCTURAL_COLORS["ceiling"]

    # Color individual wall planes
    for i in range(10):  # up to 10 walls
        key = f"wall_{i}"
        mask = labels == key
        if mask.any():
            colors[mask] = WALL_COLORS_VIS[i % len(WALL_COLORS_VIS)]

    # Remaining unassigned wall points (not claimed by any wall plane)
    colors[labels == "wall"] = "#555555"

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
        (0, 1, "Front / Image View (XY)", "X - right (m)", "Y - up (m)"),
        (2, 1, "Side (ZY)", "Z - depth (m)", "Y - up (m)"),
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

        # Invert X axis so visualization matches horizontally flipped RGB
        if a0 == 0:  # X is on horizontal axis
            ax.invert_xaxis()

    # Legend
    WALL_COLORS = ["#3498db", "#1abc9c", "#9b59b6"]  # blue, teal, purple for wall_0, wall_1, wall_2
    legend_elements = []
    for key, plane in planes.items():
        if key.startswith("wall_"):
            wall_idx = int(key.split("_")[1])
            color = WALL_COLORS[wall_idx % len(WALL_COLORS)]
        elif key == "floor":
            color = STRUCTURAL_COLORS["floor"]
        elif key == "ceiling":
            color = STRUCTURAL_COLORS["ceiling"]
        else:
            color = "#888888"

        if plane is not None:
            n = plane["normal"]
            rho = plane["rho"]
            count = plane["num_inliers"]
            label = (f"{key}: n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}] "
                     f"ρ={rho:.3f}m ({count} pts)")
        else:
            label = f"{key}: not detected"
        legend_elements.append(Patch(facecolor=color, label=label))

    # Add entries for classes with no planes
    for class_name in ["floor", "ceiling"]:
        if class_name not in planes:
            legend_elements.append(Patch(facecolor=STRUCTURAL_COLORS[class_name],
                                        label=f"{class_name}: not detected"))
    legend_elements.append(Patch(facecolor="#333333", label="Other / unlabeled"))

    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=7,
                   facecolor="black", edgecolor="white", labelcolor="white")

    fig.suptitle(f"{cam_id} — Semantic Plane Fit", fontsize=14,
                 fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="semantic_plane_fit", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Semantic Plane Fitting\n{OmegaConf.to_yaml(cfg)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = list(cfg.cameras)

    for cam_id in cameras:
        log.info(f"\n{'='*50}")
        log.info(f"  {cam_id}")
        log.info(f"{'='*50}")

        # --- Load kinect config (depth intrinsics + DLT correspondences) ---
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue

        K_depth = kinect["K_depth"]
        corr_path = kinect["corr_path"]
        if corr_path is None:
            corr_path = f"data/{cfg.session}/videos/{cam_id}/depth3d_to_color2d_correspondences.npz"

        if not Path(corr_path).exists():
            log.warning(f"[{cam_id}] Correspondences not found: {corr_path}, skipping")
            continue

        corr = np.load(corr_path)
        P_dlt = fit_dlt(corr["points_3d"], corr["points_2d"])

        # --- Load semantic masks ---
        masks = {}
        for class_name in STRUCTURAL_CLASSES:
            mask_path = Path(cfg.semantic_dir) / f"{cam_id}_{class_name}_mask.npy"
            if mask_path.exists():
                masks[class_name] = np.load(mask_path)
                log.info(f"[{cam_id}] Loaded {class_name} mask: "
                         f"{masks[class_name].sum()} pixels ({masks[class_name].mean():.1%})")
            else:
                log.warning(f"[{cam_id}] Mask not found: {mask_path}")

        if not masks:
            log.warning(f"[{cam_id}] No masks found, skipping")
            continue

        # --- Load and accumulate depth frames ---
        frames = load_depth_frames(
            cfg.depth_dir, cam_id, cfg.depth.chunk,
            cfg.depth.frame_idx, cfg.depth.num_frames
        )
        if not frames:
            log.warning(f"[{cam_id}] No depth frames, skipping")
            continue

        all_pts = []
        all_labels = []

        for depth_frame in frames:
            pts_3d, depths = backproject_depth(depth_frame, K_depth)

            # Filter by max depth
            depth_mask = depths < cfg.depth.max_depth
            pts_3d = pts_3d[depth_mask]

            if len(pts_3d) == 0:
                continue

            # Project to color pixels
            pixels, valid = project_to_color_pixels(pts_3d, P_dlt)
            pts_3d = pts_3d[valid]
            pixels = pixels[valid]

            # Look up semantic label for each point
            point_labels = np.full(len(pts_3d), "other", dtype=object)
            for class_name, mask in masks.items():
                px, py = pixels[:, 0], pixels[:, 1]
                on_mask = mask[py, px].astype(bool)
                point_labels[on_mask] = class_name

            all_pts.append(pts_3d)
            all_labels.append(point_labels)

        if not all_pts:
            log.warning(f"[{cam_id}] No valid depth points")
            continue

        all_pts = np.concatenate(all_pts, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        log.info(f"[{cam_id}] Total points: {len(all_pts)}")
        for class_name in STRUCTURAL_CLASSES:
            count = (all_labels == class_name).sum()
            log.info(f"  {class_name}: {count} points ({count/len(all_pts):.1%})")

        # --- Fit planes per class ---
        # Floor and ceiling: single plane each
        # Walls: iterative — fit first wall, remove inliers, fit second wall
        planes = {}
        for class_name in ["floor", "ceiling"]:
            class_pts = all_pts[all_labels == class_name]
            if len(class_pts) < 100:
                log.info(f"  {class_name}: too few points ({len(class_pts)}), skipping")
                planes[class_name] = None
                continue

            normal, rho, inlier_mask = fit_plane_ransac(
                class_pts,
                distance_thresh=cfg.ransac.distance_thresh,
                num_iterations=cfg.ransac.num_iterations,
            )

            if normal is None:
                planes[class_name] = None
                continue

            n_inliers = inlier_mask.sum()
            planes[class_name] = {
                "normal": normal,
                "rho": float(rho),
                "num_inliers": int(n_inliers),
                "inlier_ratio": float(n_inliers / len(class_pts)),
            }
            log.info(f"  {class_name}: normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]  "
                     f"ρ={rho:.3f}m  inliers={n_inliers} ({n_inliers/len(class_pts):.0%})")

        # Walls: iterative extraction
        # Track global indices of wall points so we can relabel them
        wall_global_indices = np.where(all_labels == "wall")[0]
        wall_pts = all_pts[wall_global_indices]
        wall_idx = 0
        remaining_mask = np.ones(len(wall_pts), dtype=bool)

        while remaining_mask.sum() >= 100 and wall_idx < cfg.get("max_walls", 3):
            remaining_pts = wall_pts[remaining_mask]

            normal, rho, inlier_mask = fit_plane_ransac(
                remaining_pts,
                distance_thresh=cfg.ransac.distance_thresh,
                num_iterations=cfg.ransac.num_iterations,
            )

            if normal is None:
                break

            n_inliers = inlier_mask.sum()
            inlier_ratio = n_inliers / len(wall_pts)

            # Stop if this wall plane is too small
            if inlier_ratio < cfg.ransac.get("min_inlier_ratio", 0.02):
                log.info(f"  wall_{wall_idx}: only {n_inliers} inliers ({inlier_ratio:.1%}) — stopping")
                break

            key = f"wall_{wall_idx}"
            planes[key] = {
                "normal": normal,
                "rho": float(rho),
                "num_inliers": int(n_inliers),
                "inlier_ratio": float(inlier_ratio),
            }
            log.info(f"  {key}: normal=[{normal[0]:.3f}, {normal[1]:.3f}, {normal[2]:.3f}]  "
                     f"ρ={rho:.3f}m  inliers={n_inliers} ({inlier_ratio:.0%})")

            # Relabel inlier points in all_labels
            remaining_indices = np.where(remaining_mask)[0]
            inlier_local = remaining_indices[inlier_mask]
            inlier_global = wall_global_indices[inlier_local]
            all_labels[inlier_global] = key

            # Remove inliers from remaining points
            remaining_mask[inlier_local] = False
            wall_idx += 1

        if wall_idx == 0:
            log.info(f"  wall: too few points or no plane found")

        # --- Visualize ---
        plot_semantic_planes(all_pts, all_labels, planes, cam_id,
                            out_dir / f"{cam_id}_semantic_planes.png")

        # --- Save plane parameters ---
        plane_data = {}
        for class_name, plane in planes.items():
            if plane is not None:
                plane_data[class_name] = {
                    "normal": plane["normal"].tolist(),
                    "rho": plane["rho"],
                    "num_inliers": plane["num_inliers"],
                    "inlier_ratio": plane["inlier_ratio"],
                }
        with open(out_dir / f"{cam_id}_planes.json", "w") as f:
            json.dump(plane_data, f, indent=2)
        log.info(f"Saved: {out_dir / f'{cam_id}_planes.json'}")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()