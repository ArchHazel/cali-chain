"""
Intrinsic Calibration Pipeline.

Reads manually selected frames per camera from the Hydra config,
detects checkerboard corners, calibrates, and saves per-camera intrinsics.

When auto_detect is enabled, scans all frames in each camera directory
instead of using the manual frame list.

Usage:
    python -m src.intrinsic_calibration
    python -m src.intrinsic_calibration dataset=intrinsic_1
    python -m src.intrinsic_calibration +auto_detect=true
    python -m src.intrinsic_calibration +auto_detect=true +run_cameras=new
"""

import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

CORNER_REFINE_WIN = (11, 11)
TERM_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
OPTIMAL_ALPHA = 1.0


def calibrate_camera(cfg: DictConfig, cam_id: str, frame_names: list[str]) -> dict | None:
    board_size = tuple(cfg.checkerboard.size)
    square_mm = cfg.checkerboard.square_size_mm
    cam_dir = Path(cfg.data.frames_dir) / cam_id

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    objpoints, imgpoints, used_frames = [], [], []
    image_shape = None

    for fname in frame_names:
        fp = cam_dir / fname
        img = cv2.imread(str(fp))
        if img is None:
            log.warning(f"[{cam_id}] Could not read: {fp}")
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if ret:
            refined = cv2.cornerSubPix(gray, corners, CORNER_REFINE_WIN, (-1, -1), TERM_CRITERIA)
            objpoints.append(objp)
            imgpoints.append(refined)
            used_frames.append(fp)
        else:
            log.warning(f"[{cam_id}] No corners found in {fname}")

    if not objpoints:
        log.warning(f"[{cam_id}] No successful detections. Skipping.")
        return None

    log.info(f"[{cam_id}] Calibrating with {len(objpoints)}/{len(frame_names)} frames...")
    rms, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
    log.info(f"[{cam_id}] RMS reprojection error: {rms:.4f} px")

    result = {
        "camera_name": cam_id,
        "resolution": list(image_shape),
        "fx": float(mtx[0, 0]),
        "fy": float(mtx[1, 1]),
        "cx": float(mtx[0, 2]),
        "cy": float(mtx[1, 2]),
        "k_matrix": mtx.tolist(),
        "dist_coeffs": dist.tolist(),
        "rms_error": float(rms),
        "num_frames_used": len(objpoints),
    }

    cam_out = Path(cfg.output.dir) / cam_id
    cam_out.mkdir(parents=True, exist_ok=True)

    if cfg.output.save_corner_vis:
        corners_dir = cam_out / "corners"
        corners_dir.mkdir(exist_ok=True)
        for fp, imgpts in zip(used_frames, imgpoints):
            vis = cv2.imread(str(fp))
            cv2.drawChessboardCorners(vis, board_size, imgpts, True)
            cv2.imwrite(str(corners_dir / fp.name), vis)

    if cfg.output.save_undistorted:
        undist_dir = cam_out / "undistorted"
        undist_dir.mkdir(exist_ok=True)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, image_shape, OPTIMAL_ALPHA, image_shape
        )
        x, y, w, h = roi
        for fp in used_frames:
            img = cv2.imread(str(fp))
            dst = cv2.undistort(img, mtx, dist, None, new_mtx)
            if w > 0 and h > 0:
                dst = dst[y:y+h, x:x+w]
            cv2.imwrite(str(undist_dir / fp.name), dst)

    return result


def discover_all_frames(frames_dir: Path, cam_id: str, stride: int = 1) -> list[str]:
    """Find all jpg frames in a camera directory, optionally skipping by stride."""
    cam_dir = frames_dir / cam_id
    if not cam_dir.exists():
        return []
    all_frames = sorted(f.name for f in cam_dir.glob("*.jpg"))
    return all_frames[::stride]


@hydra.main(config_path="../configs", config_name="intrinsic_calibration", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Intrinsic Calibration Pipeline\n{OmegaConf.to_yaml(cfg)}")

    auto_detect = True
    cameras = OmegaConf.to_container(cfg.cameras, resolve=True)

    # Filter cameras if run_cameras is specified
    run_filter = cfg.get("run_cameras", None)
    if run_filter:
        if isinstance(run_filter, str):
            selected = [s.strip() for s in run_filter.split(",")]
        else:
            selected = list(run_filter)
        cameras = {k: v for k, v in cameras.items() if str(k) in selected}

    log.info(f"Cameras to calibrate: {list(cameras.keys())}")
    if auto_detect:
        log.info("Auto-detect mode: using ALL frames with detected checkerboards")

    summary = {}
    for cam_id, frame_names in cameras.items():
        cam_id = str(cam_id)
        log.info(f"\n{'='*40}  {cam_id}  {'='*40}")

        if auto_detect:
            stride = 3
            frame_names = discover_all_frames(Path(cfg.data.frames_dir), cam_id, stride)
            log.info(f"[{cam_id}] Auto-detected {len(frame_names)} frames (stride={stride})")

        intrinsics = calibrate_camera(cfg, cam_id, frame_names)
        if intrinsics is None:
            continue

        out_path = Path(cfg.output.dir) / cam_id / "intrinsics.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(intrinsics, f, indent=4)
        log.info(f"[{cam_id}] Saved -> {out_path}")

        summary[cam_id] = {
            "rms_error": intrinsics["rms_error"],
            "num_frames": intrinsics["num_frames_used"],
            "fx": intrinsics["fx"],
            "fy": intrinsics["fy"],
        }

    if summary:
        summary_path = Path(cfg.output.dir) / "calibration_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"\nSummary -> {summary_path}")


if __name__ == "__main__":
    main()