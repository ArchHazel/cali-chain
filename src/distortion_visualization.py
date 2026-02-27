"""
Distortion Visualization Pipeline.

Generates two visualizations per calibrated camera:
  1. corner_shift      — Per-frame anaglyph overlay with corner shift vectors
  2. magnitude_heatmap — Per-camera distortion magnitude heatmap

Usage:
    python -m src.distortion_visualization
    python -m src.distortion_visualization dataset=intrinsic_1
    python -m src.distortion_visualization cameras='[HAR2,HAR6]'
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DPI = 150
CHECKERBOARD_SIZE = (7, 10)
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_intrinsics(intrinsics_path: Path) -> dict:
    with open(intrinsics_path) as f:
        data = json.load(f)
    K = np.array(data["k_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64) if "dist_coeffs" in data else None
    return {"K": K, "dist": dist, "raw": data}


def load_camera_frames(cfg: DictConfig) -> dict[str, list[str]]:
    """Load the per-camera frame lists from the intrinsic calibration config."""
    calib_cfg_path = Path(cfg.intrinsics.calibration_config)
    if not calib_cfg_path.exists():
        log.error(f"Intrinsic calibration config not found: {calib_cfg_path}")
        return {}
    calib_cfg = OmegaConf.load(calib_cfg_path)
    cameras = OmegaConf.to_container(calib_cfg.cameras, resolve=True)
    return {str(k): v for k, v in cameras.items()}


def compute_displacement_field(K, dist, size, stride=20):
    """
    For a grid of points in undistorted space, compute where they land in
    distorted space. Returns (pts_undist, pts_dist, displacement).
    """
    w, h = size
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    us = np.arange(stride // 2, w, stride, dtype=np.float64)
    vs = np.arange(stride // 2, h, stride, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    pts_undist = np.stack([uu.ravel(), vv.ravel()], axis=1)

    x_norm = (pts_undist[:, 0] - cx) / fx
    y_norm = (pts_undist[:, 1] - cy) / fy

    pts_3d = np.zeros((len(x_norm), 3), dtype=np.float64)
    pts_3d[:, 0] = x_norm
    pts_3d[:, 1] = y_norm
    pts_3d[:, 2] = 1.0
    pts_dist_2d, _ = cv2.projectPoints(pts_3d, np.zeros(3), np.zeros(3), K, dist)
    pts_dist = pts_dist_2d.reshape(-1, 2)

    displacement = pts_dist - pts_undist
    return pts_undist, pts_dist, displacement


# ---------------------------------------------------------------------------
# Visualizations
# ---------------------------------------------------------------------------


def vis_corner_shift(img_dist, img_undist, K, dist, out_path):
    """
    Overlay distorted and undistorted images using color channels
    (red = distorted, cyan = undistorted) so misalignment shows as
    color fringing. Corner shift vectors are drawn on top.

    Magenta dots = original (distorted) corner positions
    Yellow dots  = undistorted corner positions
    White lines  = shift vectors
    """
    gray_dist = cv2.cvtColor(img_dist, cv2.COLOR_BGR2GRAY)
    gray_undist = cv2.cvtColor(img_undist, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_dist, CHECKERBOARD_SIZE, None)
    if not ret:
        return False

    corners = cv2.cornerSubPix(gray_dist, corners, (11, 11), (-1, -1), CORNER_CRITERIA).reshape(-1, 2)
    corners_undist = cv2.undistortPoints(
        corners.reshape(-1, 1, 2), K, dist, P=K
    ).reshape(-1, 2)

    shifts = np.linalg.norm(corners_undist - corners, axis=1)

    # Anaglyph: red channel from distorted, green+blue (cyan) from undistorted
    # Where images align -> grayscale; where they differ -> color fringing
    # OpenCV is BGR: channel 0=Blue, 1=Green, 2=Red
    canvas = np.zeros_like(img_dist)
    canvas[:, :, 2] = gray_dist       # Red   = distorted
    canvas[:, :, 1] = gray_undist     # Green = undistorted
    canvas[:, :, 0] = gray_undist     # Blue  = undistorted

    # Draw corner shift vectors — colors chosen to contrast against red/cyan
    for i in range(len(corners)):
        pd = tuple(corners[i].astype(int))
        pu = tuple(corners_undist[i].astype(int))
        cv2.line(canvas, pd, pu, (255, 255, 255), 2)             # white lines
        cv2.circle(canvas, pd, 6, (255, 0, 255), -1)             # magenta = distorted
        cv2.circle(canvas, pu, 4, (0, 255, 255), -1)             # yellow  = undistorted

    # Text with dark background for readability
    lines = [
        "Red channel = distorted  |  Cyan channel = undistorted",
        f"Magenta dots = distorted corners  |  Yellow dots = undistorted",
        f"Corner shift: mean={shifts.mean():.1f}px  max={shifts.max():.1f}px",
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale, thickness = 0.7, 2
    y_offset = 30
    for line in lines:
        (tw, th), baseline = cv2.getTextSize(line, font, font_scale, thickness)
        cv2.rectangle(canvas, (15, y_offset - th - 4), (25 + tw, y_offset + baseline + 4), (0, 0, 0), -1)
        cv2.putText(canvas, line, (20, y_offset), font, font_scale, (255, 255, 255), thickness)
        y_offset += th + baseline + 12

    cv2.imwrite(str(out_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return True


def vis_magnitude_heatmap(K, dist, size, out_path):
    """Heatmap of distortion magnitude across the image."""
    w, h = size

    pts_undist, _, displacement = compute_displacement_field(K, dist, (w, h), stride=20)
    magnitudes = np.linalg.norm(displacement, axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    try:
        from scipy.interpolate import griddata
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        mag_map = griddata(pts_undist, magnitudes, (grid_x, grid_y), method="cubic", fill_value=0)
        im = ax.imshow(np.clip(mag_map, 0, None), cmap="inferno")
    except ImportError:
        im = ax.scatter(pts_undist[:, 0], pts_undist[:, 1], c=magnitudes, cmap="inferno", s=2)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
    plt.colorbar(im, ax=ax, label="Displacement (pixels)", shrink=0.8)
    ax.set_title("Distortion Magnitude Heatmap", fontsize=12)
    ax.set_aspect("equal")
    ax.axis("off")

    log.info(f"  Displacement: mean={magnitudes.mean():.2f}px  max={magnitudes.max():.2f}px")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=DPI, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


@hydra.main(config_path="../configs", config_name="distortion_visualization", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Distortion Visualization Pipeline\n{OmegaConf.to_yaml(cfg)}")

    intrinsics_dir = Path(cfg.intrinsics.dir)
    frames_dir = Path(cfg.data.frames_dir)
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load per-camera frame lists from intrinsic calibration config
    camera_frames = load_camera_frames(cfg)
    if not camera_frames:
        log.error("No camera frame lists found")
        return

    # Filter to requested cameras
    if cfg.cameras is not None:
        requested = set(OmegaConf.to_container(cfg.cameras, resolve=True))
        camera_frames = {k: v for k, v in camera_frames.items() if k in requested}

    log.info(f"Cameras: {list(camera_frames.keys())}")

    for cam_id, frame_names in camera_frames.items():
        log.info(f"\n{'='*50}  {cam_id}  {'='*50}")

        # Load intrinsics
        intr_path = intrinsics_dir / cam_id / "intrinsics.json"
        if not intr_path.exists():
            log.warning(f"[{cam_id}] No intrinsics found, skipping")
            continue
        intr = load_intrinsics(intr_path)
        K, dist = intr["K"], intr["dist"]

        if dist is None:
            log.warning(f"[{cam_id}] No distortion coefficients, skipping")
            continue

        # Resolve frame paths
        cam_frames_dir = frames_dir / cam_id
        frame_paths = []
        for fname in frame_names:
            fp = cam_frames_dir / fname
            if fp.exists():
                frame_paths.append(fp)
            else:
                log.warning(f"[{cam_id}] Frame not found: {fp}")

        if not frame_paths:
            log.warning(f"[{cam_id}] No valid frames, skipping")
            continue

        log.info(f"  {len(frame_paths)} frames from intrinsic calibration config")

        cam_out = out_dir / cam_id
        (cam_out / "corner_shift").mkdir(parents=True, exist_ok=True)

        # Per-camera: magnitude heatmap (use first frame for size)
        first_img = cv2.imread(str(frame_paths[0]))
        h, w = first_img.shape[:2]

        try:
            vis_magnitude_heatmap(K, dist, (w, h), cam_out / "magnitude_heatmap.png")
            log.info(f"  Saved magnitude_heatmap.png")
        except Exception as e:
            log.warning(f"  Magnitude heatmap failed: {e}")

        # Per-frame: corner shift
        corner_count = 0
        for frame_path in frame_paths:
            stem = frame_path.stem
            img_dist = cv2.imread(str(frame_path))
            img_undist = cv2.undistort(img_dist, K, dist, None, K)

            if vis_corner_shift(img_dist, img_undist, K, dist, cam_out / "corner_shift" / f"{stem}.jpg"):
                corner_count += 1

        log.info(f"  Per-frame: {corner_count}/{len(frame_paths)} corner_shift")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()