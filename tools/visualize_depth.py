"""
Visualize Kinect V2 depth frames projected into the color camera view.

Loads depth intrinsics and correspondence paths from Hydra kinect configs.

Usage:
    python -m src.visualize_depth dataset=calib_3 cam_id=HAR1
    python -m src.visualize_depth dataset=calib_3 cam_id=HAR1 rgb_frame=data/calib_3/frames/HAR1/000000.jpg
    python -m src.visualize_depth dataset=calib_3 cam_id=HAR1 frame_idx=10 chunk=2
"""

import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
import yaml
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080


# ---------------------------------------------------------------------------
# Kinect config loading
# ---------------------------------------------------------------------------

def load_kinect_config(cam_id: str, configs_dir: str = "configs/kinect") -> dict:
    """Load depth intrinsics and correspondence path from kinect config."""
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

    return {
        "K_depth": K,
        "corr_path": cfg.get("depth_to_color_correspondences", None),
    }


# ---------------------------------------------------------------------------
# DLT fitting
# ---------------------------------------------------------------------------

def fit_dlt(points_3d, points_2d):
    """
    Fit a 3x4 projection matrix via DLT that maps 3D depth-camera-space
    points to 2D color image pixels.
    """
    n = len(points_3d)
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]

    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)

    pts_h = np.hstack([points_3d, np.ones((n, 1))])
    proj = (P @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - points_2d, axis=1)
    log.info(f"DLT fit: mean reproj error = {err.mean():.2f} px, max = {err.max():.2f} px")

    return P


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame, K):
    """
    Back-project depth pixels to 3D in depth camera space.
    Y is reversed to match Kinect SDK CameraSpacePoint convention (Y-up).
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

    return cam_flat, depth_m


def project_to_color_dlt(cam_space, depth_m, P):
    """Project 3D depth-camera-space points to color image using the DLT matrix."""
    valid = depth_m > 0
    pts = cam_space[valid]
    depths = depth_m[valid]

    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (P @ pts_h.T).T

    w = proj[:, 2]
    mask = w > 0
    px = (proj[:, 0] / w).astype(np.int32)
    py = (proj[:, 1] / w).astype(np.int32)

    in_bounds = mask & (px >= 0) & (px < COLOR_W) & (py >= 0) & (py < COLOR_H)
    px, py, depths = px[in_bounds], py[in_bounds], depths[in_bounds]

    depth_img = np.full((COLOR_H, COLOR_W), np.inf, dtype=np.float32)
    np.minimum.at(depth_img, (py, px), depths)
    depth_img[depth_img == np.inf] = 0

    return depth_img


def colorize_depth(depth_img, min_depth=0.5, max_depth=6.0):
    valid = depth_img > 0
    normalized = np.zeros_like(depth_img)
    normalized[valid] = 1.0 - np.clip(
        (depth_img[valid] - min_depth) / (max_depth - min_depth), 0, 1
    )
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="visualize_depth", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Visualize Depth\n{OmegaConf.to_yaml(cfg)}")

    cam_id = cfg.cam_id
    session = cfg.dataset.session

    # Load kinect config for this camera
    kinect = load_kinect_config(cam_id)
    K_depth = kinect["K_depth"]
    log.info(f"[{cam_id}] Depth intrinsics: fx={K_depth[0,0]:.1f} fy={K_depth[1,1]:.1f} "
             f"cx={K_depth[0,2]:.1f} cy={K_depth[1,2]:.1f}")

    # Load correspondences
    corr_path = kinect["corr_path"]
    if corr_path is None:
        corr_path = f"data/{session}/videos/{cam_id}/depth3d_to_color2d_correspondences.npz"
    log.info(f"Loading correspondences from {corr_path}")
    corr = np.load(corr_path)
    P = fit_dlt(corr['points_3d'], corr['points_2d'])

    # Load depth
    base = Path(cfg.data.depth_dir) / cam_id / "depth"
    depth_path = base / f"depth_{cfg.chunk}.npy"
    log.info(f"Loading {depth_path}")
    depth_chunk = np.load(depth_path)
    depth_frame = depth_chunk[cfg.frame_idx]
    log.info(f"Depth frame: {depth_frame.shape}, "
             f"non-zero: {(depth_frame > 0).sum()}/{depth_frame.size}")

    # Back-project and project (in native Kinect coordinates)
    cam_space, depth_m = backproject_depth(depth_frame, K_depth)
    depth_in_color = project_to_color_dlt(cam_space, depth_m, P)

    # Flip horizontally to match the pre-flipped RGB frames
    depth_in_color = np.fliplr(depth_in_color)

    colored = colorize_depth(depth_in_color)

    # Save
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{cam_id}_chunk{cfg.chunk}_frame{cfg.frame_idx}"

    cv2.imwrite(str(out_dir / f"depth_color_{tag}.png"), colored)
    log.info(f"Saved: {out_dir / f'depth_color_{tag}.png'}")

    # Overlay on RGB (already pre-flipped during preprocessing)
    rgb_frame_path = cfg.get("rgb_frame", None)
    if rgb_frame_path:
        rgb = cv2.imread(rgb_frame_path)
        if rgb is None:
            log.warning(f"Could not load {rgb_frame_path}")
        else:
            if rgb.shape[:2] != (COLOR_H, COLOR_W):
                rgb = cv2.resize(rgb, (COLOR_W, COLOR_H))

            depth_valid = depth_in_color > 0
            overlay = rgb.copy()
            overlay[depth_valid] = cv2.addWeighted(
                rgb[depth_valid], 1.0 - cfg.alpha,
                colored[depth_valid], cfg.alpha, 0
            )

            cv2.imwrite(str(out_dir / f"depth_overlay_{tag}.png"), overlay)
            log.info(f"Saved overlay: {out_dir / f'depth_overlay_{tag}.png'}")

    # Stats
    valid = depth_in_color[depth_in_color > 0]
    if len(valid):
        log.info(f"Depth range: {valid.min():.2f}m - {valid.max():.2f}m")
        log.info(f"Coverage: {len(valid)}/{COLOR_H * COLOR_W} "
                 f"({100 * len(valid) / (COLOR_H * COLOR_W):.1f}%)")


if __name__ == "__main__":
    main()