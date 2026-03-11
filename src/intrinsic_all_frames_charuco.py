"""
Intrinsic Calibration Pipeline — CharUco Board.

Uses a CharUco board (5x7 squares, ArUco DICT_4X4) instead of a plain
checkerboard.  CharUco gives sub-pixel corner accuracy even under partial
occlusion because each corner is independently identifiable.

When auto_detect is enabled (default), scans all frames in each camera
directory instead of using the manual frame list.

Usage:
    python -m src.intrinsic_all_frames_charuco
    python -m src.intrinsic_all_frames_charuco dataset=intrinsic_1
    python -m src.intrinsic_all_frames_charuco +run_cameras=new
    python -m src.intrinsic_all_frames_charuco +run_cameras=new +min_corners=6
"""

import json
import logging
from pathlib import Path

import cv2
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# ── CharUco board parameters ─────────────────────────────────────────────
# 5×7 = 5 squares wide, 7 squares tall
BOARD_SQUARES_X = 7
BOARD_SQUARES_Y = 5
ARUCO_DICT = cv2.aruco.DICT_4X4_50

# You'll need to set these to match your physical board.
# square_length > marker_length (marker sits inside the white square).
SQUARE_LENGTH_M = 0.035   # side of a chessboard square  (metres) — from board filename: 35mm
MARKER_LENGTH_M = 0.026   # side of the ArUco marker      (metres) — from board filename: 26mm

# Minimum detected CharUco corners to accept a frame for calibration
MIN_CHARUCO_CORNERS = 10

OPTIMAL_ALPHA = 1.0


# ── Board & detector construction ────────────────────────────────────────

def make_board_and_detector(
    squares_x: int = BOARD_SQUARES_X,
    squares_y: int = BOARD_SQUARES_Y,
    square_length: float = SQUARE_LENGTH_M,
    marker_length: float = MARKER_LENGTH_M,
    aruco_dict_id: int = ARUCO_DICT,
):
    """Create a CharucoBoard and its CharucoDetector (OpenCV ≥ 4.7 API)."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_id)
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length,
        marker_length,
        aruco_dict,
    )
    detector = cv2.aruco.CharucoDetector(board)
    return board, detector


# ── Per-frame detection ──────────────────────────────────────────────────

def detect_charuco_corners(
    img_path: Path,
    detector: cv2.aruco.CharucoDetector,
    min_corners: int = MIN_CHARUCO_CORNERS,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Detect CharUco corners in a single image.

    Returns (charuco_corners, charuco_ids, gray) or (None, None, None).
    charuco_corners: (N, 1, 2) float32  — sub-pixel corner locations
    charuco_ids:     (N, 1)    int32    — which board corner each one is
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    if charuco_corners is None or charuco_ids is None:
        return None, None, None
    if len(charuco_corners) < min_corners:
        return None, None, None

    return charuco_corners, charuco_ids, gray


# ── Calibration ──────────────────────────────────────────────────────────

def calibrate_camera(
    cfg: DictConfig,
    cam_id: str,
    frame_names: list[str],
    board: cv2.aruco.CharucoBoard,
    detector: cv2.aruco.CharucoDetector,
    min_corners: int = MIN_CHARUCO_CORNERS,
) -> dict | None:
    cam_dir = Path(cfg.data.frames_dir) / cam_id

    all_charuco_corners = []
    all_charuco_ids = []
    used_frames: list[Path] = []
    image_shape = None

    for fname in frame_names:
        fp = cam_dir / fname
        corners, ids, gray = detect_charuco_corners(fp, detector, min_corners)
        if corners is None:
            continue
        if image_shape is None:
            image_shape = gray.shape[::-1]  # (W, H)

        all_charuco_corners.append(corners)
        all_charuco_ids.append(ids)
        used_frames.append(fp)

    if not all_charuco_corners:
        log.warning(f"[{cam_id}] No successful CharUco detections. Skipping.")
        return None

    log.info(
        f"[{cam_id}] Calibrating with {len(all_charuco_corners)}/{len(frame_names)} frames …"
    )

    # CharUco calibration — board carries its own 3-D geometry so we don't
    # need to build objpoints manually.
    # Some frames may cause degenerate PnP (near-collinear corners, etc.),
    # so we iteratively remove them until calibration succeeds.
    while True:
        try:
            rms, mtx, dist, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
                charucoCorners=all_charuco_corners,
                charucoIds=all_charuco_ids,
                board=board,
                imageSize=image_shape,
                cameraMatrix=None,
                distCoeffs=None,
            )
            break
        except cv2.error as e:
            if len(all_charuco_corners) <= 10:
                log.error(f"[{cam_id}] Calibration failed with {len(all_charuco_corners)} frames: {e}")
                return None
            # Find the frame with the fewest corners and remove it
            worst_idx = min(range(len(all_charuco_corners)),
                           key=lambda i: len(all_charuco_corners[i]))
            worst_name = used_frames[worst_idx].name
            worst_n = len(all_charuco_corners[worst_idx])
            log.warning(
                f"[{cam_id}] Calibration error, removing {worst_name} "
                f"({worst_n} corners). {len(all_charuco_corners)-1} frames remain."
            )
            all_charuco_corners.pop(worst_idx)
            all_charuco_ids.pop(worst_idx)
            used_frames.pop(worst_idx)

    log.info(f"[{cam_id}] RMS reprojection error: {rms:.4f} px  ({len(used_frames)} frames)")

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
        "num_frames_used": len(all_charuco_corners),
        "board_type": "charuco",
        "board_squares": [BOARD_SQUARES_X, BOARD_SQUARES_Y],
        "aruco_dict": "DICT_4X4_50",
        "square_length_m": SQUARE_LENGTH_M,
        "marker_length_m": MARKER_LENGTH_M,
    }

    # ── Optional outputs ─────────────────────────────────────────────────
    cam_out = Path(cfg.output.dir) / cam_id
    cam_out.mkdir(parents=True, exist_ok=True)

    if cfg.output.save_corner_vis:
        corners_dir = cam_out / "corners"
        corners_dir.mkdir(exist_ok=True)
        for fp, corners, ids in zip(used_frames, all_charuco_corners, all_charuco_ids):
            vis = cv2.imread(str(fp))
            cv2.aruco.drawDetectedCornersCharuco(vis, corners, ids, cornerColor=(0, 0, 255))
            cv2.imwrite(str(corners_dir / fp.name), vis)

    if cfg.output.save_undistorted:
        undist_dir = cam_out / "undistorted"
        undist_dir.mkdir(exist_ok=True)
        new_mtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, image_shape, OPTIMAL_ALPHA, image_shape,
        )
        x, y, w, h = roi
        for fp in used_frames:
            img = cv2.imread(str(fp))
            dst = cv2.undistort(img, mtx, dist, None, new_mtx)
            if w > 0 and h > 0:
                dst = dst[y : y + h, x : x + w]
            cv2.imwrite(str(undist_dir / fp.name), dst)

    return result


# ── Frame discovery ──────────────────────────────────────────────────────

def discover_all_frames(frames_dir: Path, cam_id: str, target_count: int = 400) -> tuple[list[str], int]:
    """Find all jpg frames in a camera directory, auto-selecting stride to hit target_count."""
    cam_dir = frames_dir / cam_id
    if not cam_dir.exists():
        return [], 1
    all_frames = sorted(f.name for f in cam_dir.glob("*.jpg"))
    total = len(all_frames)
    if total == 0:
        return [], 1
    stride = max(1, total // target_count)
    return all_frames[::stride], stride


# ── Entry point ──────────────────────────────────────────────────────────

@hydra.main(config_path="../configs", config_name="intrinsic_calibration", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Intrinsic Calibration Pipeline (CharUco)\n{OmegaConf.to_yaml(cfg)}")

    auto_detect = True
    min_corners = int(cfg.get("min_corners", MIN_CHARUCO_CORNERS))

    board, detector = make_board_and_detector()

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
        log.info("Auto-detect mode: scanning ALL frames for CharUco corners")

    summary = {}
    for cam_id, frame_names in cameras.items():
        cam_id = str(cam_id)
        log.info(f"\n{'=' * 40}  {cam_id}  {'=' * 40}")

        if auto_detect:
            target_count = int(cfg.get("target_frames", 400))
            frame_names, stride = discover_all_frames(Path(cfg.data.frames_dir), cam_id, target_count)
            log.info(f"[{cam_id}] Auto-detected {len(frame_names)} frames (stride={stride}, target={target_count})")

        intrinsics = calibrate_camera(cfg, cam_id, frame_names, board, detector, min_corners)
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