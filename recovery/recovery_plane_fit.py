"""
Semantic Segmentation + Plane Fitting for Recovery Experiments.

For each 'no_tag' segment in the extracted recovery data:
  1. Run OneFormer semantic segmentation on a representative RGB frame
  2. Backproject depth to 3D, project to color via DLT, look up labels
  3. Fit planes (RANSAC) per structural class (wall, floor, ceiling)
  4. Save plane parameters + visualization

Outputs per segment (in each no_tag/ directory):
  - semantic/<class>_mask.npy  (binary masks)
  - semantic/segmentation_vis.png
  - planes.json  (plane parameters with bounded representation)
  - semantic_planes.png  (3-view visualization)

Usage:
    python -m recovery.recovery_plane_fit
    python -m recovery.recovery_plane_fit environment=bathroom
    python -m recovery.recovery_plane_fit environment=living_room segments='[orig,rotation_g01]'
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080

STRUCTURAL_CLASSES_LIST = ["wall", "floor", "ceiling"]

# ADE20K class IDs (0-indexed)
ADE20K_CLASSES = {
    "wall": 0,
    "floor": 3,
    "ceiling": 5,
}

STRUCTURAL_COLORS = {
    "wall": "#3498db",
    "floor": "#2ecc71",
    "ceiling": "#e74c3c",
}


# ---------------------------------------------------------------------------
# Kinect / DLT
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


def fit_dlt(points_3d: np.ndarray, points_2d: np.ndarray) -> np.ndarray:
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
# Depth processing (Y-up, matching existing pipeline)
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H - 1, -1, -1)[:, np.newaxis]  # Y-up
    cam[:, :, 2] = 1.0
    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T
    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]
    valid = depth_m > 0
    return cam_flat[valid], depth_m[valid]


def project_to_color_pixels(pts_3d: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
# Semantic segmentation
# ---------------------------------------------------------------------------

def load_segmentation_model(model_name: str, device: str):
    MODEL_REGISTRY = {
        "mask2former": "facebook/mask2former-swin-large-ade-semantic",
        "oneformer": "shi-labs/oneformer_ade20k_swin_large",
    }
    resolved = MODEL_REGISTRY.get(model_name, model_name)
    log.info(f"Loading model: {resolved}")

    if "oneformer" in resolved.lower():
        from transformers import OneFormerProcessor, OneFormerForUniversalSegmentation
        processor = OneFormerProcessor.from_pretrained(resolved)
        model = OneFormerForUniversalSegmentation.from_pretrained(resolved)
    else:
        from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(resolved)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(resolved)

    model = model.to(device)
    model.eval()
    log.info(f"Model loaded on {device}")
    return model, processor, resolved


def run_segmentation(model, processor, image: Image.Image, device: str,
                     model_name: str) -> np.ndarray:
    if "oneformer" in model_name.lower():
        inputs = processor(images=image, task_inputs=["semantic"], return_tensors="pt")
    else:
        inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]
    return result.cpu().numpy()


def segment_frame(model, processor, model_name: str, device: str,
                  frame_path: Path, out_dir: Path) -> dict[str, np.ndarray]:
    """Run segmentation on one frame, save masks and visualization."""
    image = Image.open(frame_path).convert("RGB")
    seg_map = run_segmentation(model, processor, image, device, model_name)

    sem_dir = out_dir / "semantic"
    sem_dir.mkdir(parents=True, exist_ok=True)

    masks = {}
    for class_name, class_id in ADE20K_CLASSES.items():
        mask = (seg_map == class_id).astype(np.uint8)
        np.save(sem_dir / f"{class_name}_mask.npy", mask)
        masks[class_name] = mask
        log.info(f"    {class_name}: {mask.sum()} pixels ({mask.mean():.1%})")

    # Visualization
    img_np = np.array(image)
    fig, axes = plt.subplots(1, 3, figsize=(24, 6), facecolor="black")
    for ax in axes:
        ax.set_facecolor("black")
        ax.axis("off")

    axes[0].imshow(img_np)
    axes[0].set_title("Original", color="white")

    overlay = img_np.copy().astype(np.float32) / 255.0
    vis_colors = {"wall": (0.2, 0.6, 1.0), "floor": (0.2, 0.8, 0.2), "ceiling": (1.0, 0.4, 0.4)}
    for cn, cid in ADE20K_CLASSES.items():
        m = seg_map == cid
        for c in range(3):
            overlay[:, :, c] = np.where(m, overlay[:, :, c] * 0.5 + vis_colors[cn][c] * 0.5, overlay[:, :, c])
    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Structural Overlay", color="white")

    mask_vis = np.zeros((*seg_map.shape, 3), dtype=np.float32)
    for cn, cid in ADE20K_CLASSES.items():
        m = seg_map == cid
        for c in range(3):
            mask_vis[:, :, c] = np.where(m, vis_colors[cn][c], mask_vis[:, :, c])
    axes[2].imshow(mask_vis)
    axes[2].set_title("Structural Mask", color="white")

    plt.tight_layout()
    plt.savefig(str(sem_dir / "segmentation_vis.png"), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)

    return masks


# ---------------------------------------------------------------------------
# Plane fitting (from semantic_plane_fit.py)
# ---------------------------------------------------------------------------

def fit_plane_ransac(points: np.ndarray, distance_thresh: float = 0.03,
                     num_iterations: int = 2000) -> tuple[np.ndarray, float, np.ndarray]:
    best_inliers = 0
    best_normal = None
    best_d = 0

    n_pts = len(points)
    for _ in range(num_iterations):
        idx = np.random.choice(n_pts, 3, replace=False)
        p0, p1, p2 = points[idx]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        d = np.dot(normal, p0)
        dists = np.abs(points @ normal - d)
        inlier_mask = dists < distance_thresh
        n_inliers = inlier_mask.sum()
        if n_inliers > best_inliers:
            best_inliers = n_inliers
            best_normal = normal
            best_d = d
            best_mask = inlier_mask

    if best_normal is None:
        return None, 0, np.zeros(n_pts, dtype=bool)

    # PCA refinement on inliers
    inlier_pts = points[best_mask]
    centroid = inlier_pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(inlier_pts - centroid, full_matrices=False)
    refined_normal = Vt[2]
    if np.dot(refined_normal, best_normal) < 0:
        refined_normal = -refined_normal
    refined_d = np.dot(refined_normal, centroid)

    # Recompute inliers with refined plane
    dists = np.abs(points @ refined_normal - refined_d)
    refined_mask = dists < distance_thresh

    # Normal toward camera (positive Z preferred)
    if refined_normal[2] < 0:
        refined_normal = -refined_normal
        refined_d = -refined_d

    return refined_normal, refined_d, refined_mask


def compute_plane_bounds(points: np.ndarray, inlier_mask: np.ndarray,
                         normal: np.ndarray) -> dict:
    inlier_pts = points[inlier_mask]
    centroid = inlier_pts.mean(axis=0)
    offsets = inlier_pts - centroid
    normal_component = offsets @ normal
    on_plane = offsets - np.outer(normal_component, normal)
    _, _, Vt = np.linalg.svd(on_plane, full_matrices=False)
    axis_0 = Vt[0]
    axis_1 = Vt[1]
    proj_0 = np.abs(on_plane @ axis_0)
    proj_1 = np.abs(on_plane @ axis_1)
    extent_0 = float(np.percentile(proj_0, 95))
    extent_1 = float(np.percentile(proj_1, 95))
    return {
        "centroid": centroid,
        "axis_0": axis_0,
        "axis_1": axis_1,
        "extent_0": extent_0,
        "extent_1": extent_1,
    }


def fit_planes_for_segment(masks: dict[str, np.ndarray], depth_frames: list[np.ndarray],
                           K_depth: np.ndarray, P_dlt: np.ndarray,
                           max_depth: float, ransac_cfg: dict) -> tuple[dict, np.ndarray, np.ndarray]:
    """
    Backproject depth, label via DLT + masks, fit planes per class.
    Returns (planes_dict, all_pts, all_labels).
    """
    all_pts = []
    all_labels = []

    for depth_frame in depth_frames:
        pts_3d, depths = backproject_depth(depth_frame, K_depth)
        depth_mask = depths < max_depth
        pts_3d = pts_3d[depth_mask]
        if len(pts_3d) == 0:
            continue

        pixels, valid = project_to_color_pixels(pts_3d, P_dlt)
        pts_3d = pts_3d[valid]
        pixels = pixels[valid]

        point_labels = np.full(len(pts_3d), "other", dtype=object)
        for class_name, mask in masks.items():
            px, py = pixels[:, 0], pixels[:, 1]
            on_mask = mask[py, px].astype(bool)
            point_labels[on_mask] = class_name

        all_pts.append(pts_3d)
        all_labels.append(point_labels)

    if not all_pts:
        return {}, np.array([]), np.array([])

    all_pts = np.concatenate(all_pts, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    log.info(f"    Total points: {len(all_pts)}")
    for cn in STRUCTURAL_CLASSES_LIST:
        count = (all_labels == cn).sum()
        log.info(f"      {cn}: {count} ({count/len(all_pts):.1%})")

    # Fit planes
    planes = {}

    # Floor
    floor_pts = all_pts[all_labels == "floor"]
    if len(floor_pts) > 100:
        normal, rho, inlier_mask = fit_plane_ransac(
            floor_pts, ransac_cfg["distance_thresh"], ransac_cfg["num_iterations"])
        if normal is not None:
            bounds = compute_plane_bounds(floor_pts, inlier_mask, normal)
            planes["floor"] = {
                "normal": normal, "rho": float(rho),
                "num_inliers": int(inlier_mask.sum()),
                "inlier_ratio": float(inlier_mask.sum() / len(floor_pts)),
                **bounds,
            }
            log.info(f"    floor: n=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}] "
                     f"ρ={rho:.3f}m ({inlier_mask.sum()} pts)")

    # Ceiling
    ceil_pts = all_pts[all_labels == "ceiling"]
    if len(ceil_pts) > 100:
        normal, rho, inlier_mask = fit_plane_ransac(
            ceil_pts, ransac_cfg["distance_thresh"], ransac_cfg["num_iterations"])
        if normal is not None:
            bounds = compute_plane_bounds(ceil_pts, inlier_mask, normal)
            planes["ceiling"] = {
                "normal": normal, "rho": float(rho),
                "num_inliers": int(inlier_mask.sum()),
                "inlier_ratio": float(inlier_mask.sum() / len(ceil_pts)),
                **bounds,
            }
            log.info(f"    ceiling: n=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}] "
                     f"ρ={rho:.3f}m ({inlier_mask.sum()} pts)")

    # Walls — iterative RANSAC
    wall_pts = all_pts[all_labels == "wall"]
    wall_global_indices = np.where(all_labels == "wall")[0]
    remaining_mask = np.ones(len(wall_pts), dtype=bool)
    wall_idx = 0
    max_walls = ransac_cfg.get("max_walls", 3)

    while wall_idx < max_walls and remaining_mask.sum() > 100:
        remaining_pts = wall_pts[remaining_mask]
        n_remaining = len(remaining_pts)
        normal, rho, inlier_mask = fit_plane_ransac(
            remaining_pts, ransac_cfg["distance_thresh"], ransac_cfg["num_iterations"])

        if normal is None:
            break

        n_inliers = inlier_mask.sum()
        inlier_ratio = n_inliers / n_remaining

        if inlier_ratio < ransac_cfg["min_inlier_ratio"]:
            log.info(f"    wall_{wall_idx}: only {n_inliers} inliers ({inlier_ratio:.1%}) — stopping")
            break

        key = f"wall_{wall_idx}"
        bounds = compute_plane_bounds(remaining_pts, inlier_mask, normal)
        planes[key] = {
            "normal": normal, "rho": float(rho),
            "num_inliers": int(n_inliers),
            "inlier_ratio": float(inlier_ratio),
            **bounds,
        }
        log.info(f"    {key}: n=[{normal[0]:.3f},{normal[1]:.3f},{normal[2]:.3f}] "
                 f"ρ={rho:.3f}m ({n_inliers} pts, {inlier_ratio:.0%}) "
                 f"extents={bounds['extent_0']:.3f}x{bounds['extent_1']:.3f}m")

        # Update labels for visualization
        remaining_indices = np.where(remaining_mask)[0]
        inlier_local = remaining_indices[inlier_mask]
        inlier_global = wall_global_indices[inlier_local]
        all_labels[inlier_global] = key

        remaining_mask[inlier_local] = False
        wall_idx += 1

    return planes, all_pts, all_labels


# ---------------------------------------------------------------------------
# Visualization (3-view plot)
# ---------------------------------------------------------------------------

def plot_semantic_planes(points: np.ndarray, labels: np.ndarray,
                         planes: dict, segment_name: str, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(24, 7), facecolor="black")

    # Flip X for visualization
    points = points.copy()
    points[:, 0] = -points[:, 0]

    WALL_COLORS_VIS = ["#3498db", "#1abc9c", "#9b59b6"]
    colors = np.full(len(points), "#333333")
    colors[labels == "floor"] = STRUCTURAL_COLORS["floor"]
    colors[labels == "ceiling"] = STRUCTURAL_COLORS["ceiling"]
    for i in range(10):
        key = f"wall_{i}"
        mask = labels == key
        if mask.any():
            colors[mask] = WALL_COLORS_VIS[i % len(WALL_COLORS_VIS)]
    colors[labels == "wall"] = "#555555"

    max_plot = 100_000
    if len(points) > max_plot:
        idx = np.random.choice(len(points), max_plot, replace=False)
        pts_plot = points[idx]
        colors_plot = colors[idx]
    else:
        pts_plot = points
        colors_plot = colors

    views = [
        (0, 2, "Bird's Eye (XZ)", "X (m)", "Z - depth (m)"),
        (0, 1, "Front (XY)", "X (m)", "Y - up (m)"),
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

    # Legend
    legend_elements = []
    for key, plane in planes.items():
        if key.startswith("wall_"):
            widx = int(key.split("_")[1])
            color = WALL_COLORS_VIS[widx % len(WALL_COLORS_VIS)]
        elif key == "floor":
            color = STRUCTURAL_COLORS["floor"]
        elif key == "ceiling":
            color = STRUCTURAL_COLORS["ceiling"]
        else:
            color = "#888888"
        n = plane["normal"]
        rho = plane["rho"]
        count = plane["num_inliers"]
        label = (f"{key}: n=[{n[0]:.2f},{n[1]:.2f},{n[2]:.2f}] "
                 f"ρ={rho:.3f}m ({count} pts)")
        legend_elements.append(Patch(facecolor=color, label=label))

    axes[0].legend(handles=legend_elements, loc="upper left", fontsize=7,
                   facecolor="black", edgecolor="white", labelcolor="white")
    fig.suptitle(f"{segment_name} — Semantic Plane Fit", fontsize=14,
                 fontweight="bold", color="white")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight", facecolor="black")
    plt.close(fig)
    log.info(f"    Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="recovery_plane_fit", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Recovery Plane Fitting\n{OmegaConf.to_yaml(cfg)}")

    env_dir = Path(cfg.recovery_dir) / cfg.environment
    if not env_dir.exists():
        log.error(f"Environment not found: {env_dir}")
        return

    # Load kinect config
    kinect = load_kinect_config(cfg.camera_id)
    K_depth = kinect["K_depth"]
    corr_path = kinect["corr_path"]
    if corr_path is None:
        log.error("No DLT correspondences path in kinect config")
        return
    if not Path(corr_path).exists():
        log.error(f"Correspondences not found: {corr_path}")
        return

    corr = np.load(corr_path)
    P_dlt = fit_dlt(corr["points_3d"], corr["points_2d"])
    log.info(f"Loaded DLT from {corr_path}")

    # Load segmentation model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seg_model, seg_processor, seg_model_name = load_segmentation_model(
        cfg.model.name, device)

    # Determine which segments to process
    segment_filter = list(cfg.get("segments", [])) or None

    ransac_cfg = OmegaConf.to_container(cfg.ransac, resolve=True)

    # Process each segment
    results = {}
    for segment_dir in sorted(env_dir.iterdir()):
        if not segment_dir.is_dir():
            continue

        no_tag_dir = segment_dir / "no_tag"
        if not no_tag_dir.exists():
            continue

        if segment_filter and segment_dir.name not in segment_filter:
            continue

        frames_dir = no_tag_dir / "frames"
        depth_dir = no_tag_dir / "depth"

        if not frames_dir.exists() or not depth_dir.exists():
            log.warning(f"  {segment_dir.name}: missing frames/ or depth/")
            continue

        # Check if already processed
        planes_path = no_tag_dir / "planes.json"
        if planes_path.exists() and not cfg.get("force", False):
            log.info(f"  {segment_dir.name}: already processed (use force=true to redo)")
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  {segment_dir.name}/no_tag")
        log.info(f"{'='*60}")

        # --- Step 1: Semantic segmentation ---
        # Use middle frame for segmentation
        frame_files = sorted(frames_dir.glob("*.jpg"))
        if not frame_files:
            log.warning(f"  No frames found")
            continue

        mid_idx = len(frame_files) // 2
        seg_frame = frame_files[mid_idx]
        log.info(f"  Segmenting frame: {seg_frame.name}")

        masks = segment_frame(seg_model, seg_processor, seg_model_name,
                              device, seg_frame, no_tag_dir)

        # --- Step 2: Load depth ---
        depth_path = depth_dir / "depth_1.npy"
        if not depth_path.exists():
            log.warning(f"  No depth file found")
            continue

        depth_chunk = np.load(depth_path)
        n_frames = min(depth_chunk.shape[0], cfg.depth.num_frames)
        # Use frames from the middle of the segment for stability
        start = max(0, depth_chunk.shape[0] // 2 - n_frames // 2)
        depth_frames = [depth_chunk[i] for i in range(start, start + n_frames)]
        log.info(f"  Using {len(depth_frames)} depth frames (of {depth_chunk.shape[0]} total)")

        # --- Step 3: Plane fitting ---
        planes, all_pts, all_labels = fit_planes_for_segment(
            masks, depth_frames, K_depth, P_dlt, cfg.depth.max_depth, ransac_cfg)

        if not planes:
            log.warning(f"  No planes detected")
            continue

        # --- Step 4: Save ---
        # Convert numpy arrays for JSON serialization
        plane_data = {}
        for key, plane in planes.items():
            plane_data[key] = {
                "normal": plane["normal"].tolist(),
                "rho": plane["rho"],
                "num_inliers": plane["num_inliers"],
                "inlier_ratio": plane["inlier_ratio"],
                "centroid": plane["centroid"].tolist(),
                "axis_0": plane["axis_0"].tolist(),
                "axis_1": plane["axis_1"].tolist(),
                "extent_0": plane["extent_0"],
                "extent_1": plane["extent_1"],
            }

        with open(planes_path, "w") as f:
            json.dump(plane_data, f, indent=2)
        log.info(f"  Saved planes -> {planes_path}")

        # Visualization
        if len(all_pts) > 0:
            plot_semantic_planes(all_pts, all_labels, planes,
                                segment_dir.name,
                                no_tag_dir / "semantic_planes.png")

        results[segment_dir.name] = {
            "num_planes": len(planes),
            "plane_labels": list(planes.keys()),
        }

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY")
    log.info(f"{'='*60}")
    for name, res in sorted(results.items()):
        log.info(f"  {name:<35} {res['num_planes']} planes: {res['plane_labels']}")


if __name__ == "__main__":
    main()