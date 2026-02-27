"""
Intrinsic Calibration Diagnostics.

For each camera, runs leave-one-out analysis to identify problematic
calibration frames. For each frame, calibrates without it and measures:
  - Change in RMS reprojection error
  - Change in focal length (fx, fy)
  - Per-frame reprojection error in the full calibration

Frames are ranked by impact. High-impact frames that *increase* RMS when
included (i.e. RMS drops when they're removed) are likely problematic.

Usage:
    python -m src.calibration_diagnostics
    python -m src.calibration_diagnostics dataset=intrinsic_1
    python -m src.calibration_diagnostics cameras='[new]'
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
CORNER_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


def detect_corners(img_path: Path, board_size: tuple) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Detect and refine checkerboard corners. Returns (corners, gray) or (None, None)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        return None, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return corners, gray


def calibrate_subset(objpoints, imgpoints, image_shape):
    """Run calibration on a subset, return (rms, K, dist) or None on failure."""
    if len(objpoints) < 4:
        return None
    try:
        rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)
        return rms, K, dist
    except cv2.error:
        return None


def per_frame_reprojection_errors(objpoints, imgpoints, K, dist, rvecs, tvecs):
    """Compute mean reprojection error per frame."""
    errors = []
    for i in range(len(objpoints)):
        projected, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(imgpoints[i].reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        errors.append(err.mean())
    return errors


def analyze_camera(cfg: DictConfig, cam_id: str, frame_names: list[str], out_dir: Path):
    board_size = tuple(cfg.checkerboard.size)
    square_mm = cfg.checkerboard.square_size_mm
    cam_dir = Path(cfg.data.frames_dir) / cam_id

    # Prepare object points
    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    # Detect corners in all frames
    all_corners = []  # list of (frame_name, corners)
    image_shape = None

    for fname in frame_names:
        corners, gray = detect_corners(cam_dir / fname, board_size)
        if corners is None:
            log.warning(f"  [{cam_id}] No corners in {fname}, excluding")
            continue
        if image_shape is None:
            image_shape = gray.shape[::-1]
        all_corners.append((fname, corners))

    n = len(all_corners)
    if n < 5:
        log.warning(f"  [{cam_id}] Only {n} frames with corners, need at least 5")
        return

    log.info(f"  {n} frames with detected corners")

    # Full calibration
    objpoints_full = [objp] * n
    imgpoints_full = [c for _, c in all_corners]

    rms_full, K_full, dist_full, rvecs_full, tvecs_full = cv2.calibrateCamera(
        objpoints_full, imgpoints_full, image_shape, None, None
    )
    fx_full, fy_full = K_full[0, 0], K_full[1, 1]

    log.info(f"  Full calibration: RMS={rms_full:.4f}  fx={fx_full:.1f}  fy={fy_full:.1f}")

    # Per-frame reprojection errors
    per_frame_errors = per_frame_reprojection_errors(
        objpoints_full, imgpoints_full, K_full, dist_full, rvecs_full, tvecs_full
    )

    # Leave-one-out analysis
    loo_results = []
    for i in range(n):
        obj_subset = objpoints_full[:i] + objpoints_full[i + 1:]
        img_subset = imgpoints_full[:i] + imgpoints_full[i + 1:]

        result = calibrate_subset(obj_subset, img_subset, image_shape)
        if result is None:
            continue

        rms_loo, K_loo, dist_loo = result
        fx_loo, fy_loo = K_loo[0, 0], K_loo[1, 1]

        loo_results.append({
            "index": i,
            "frame": all_corners[i][0],
            "rms_without": float(rms_loo),
            "rms_delta": float(rms_full - rms_loo),
            "fx_without": float(fx_loo),
            "fy_without": float(fy_loo),
            "fx_delta": float(fx_full - fx_loo),
            "fy_delta": float(fy_full - fy_loo),
            "per_frame_error": float(per_frame_errors[i]),
        })

    # Sort by impact: frames whose removal helps most (largest positive rms_delta)
    loo_results.sort(key=lambda r: r["rms_delta"], reverse=True)

    # Log results
    log.info(f"\n  {'Frame':<16} {'PerFrameErr':>12} {'RMS w/o':>10} {'ΔRMS':>10} {'fx w/o':>10} {'Δfx':>10}")
    log.info(f"  {'-'*70}")
    for r in loo_results:
        flag = " ⚠️" if r["rms_delta"] > 0.01 or abs(r["fx_delta"]) > 20 else ""
        log.info(
            f"  {r['frame']:<16} {r['per_frame_error']:>12.4f} {r['rms_without']:>10.4f} "
            f"{r['rms_delta']:>+10.4f} {r['fx_without']:>10.1f} {r['fx_delta']:>+10.1f}{flag}"
        )

    # Identify problematic frames
    problematic = [r for r in loo_results if r["rms_delta"] > 0.01 or abs(r["fx_delta"]) > 20]
    if problematic:
        log.info(f"\n  ⚠️  {len(problematic)} potentially problematic frame(s):")
        for r in problematic:
            log.info(f"    {r['frame']}: removing it reduces RMS by {r['rms_delta']:.4f}, shifts fx by {r['fx_delta']:+.1f}")
    else:
        log.info(f"\n  ✓ No clearly problematic frames detected")

    # Generate plots
    cam_out = out_dir / cam_id
    cam_out.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    # Top-left: per-frame reprojection error
    ax = axes[0, 0]
    frame_labels = [r["frame"] for r in loo_results]
    pf_errors = [r["per_frame_error"] for r in loo_results]
    colors = ["red" if r["rms_delta"] > 0.01 or abs(r["fx_delta"]) > 20 else "steelblue" for r in loo_results]
    ax.barh(range(n), pf_errors, color=colors, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_xlabel("Mean reprojection error (px)")
    ax.set_title("Per-Frame Reprojection Error")
    ax.axvline(x=rms_full, color="black", linestyle="--", alpha=0.5, label=f"Overall RMS={rms_full:.3f}")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    # Top-right: RMS delta when frame is removed
    ax = axes[0, 1]
    rms_deltas = [r["rms_delta"] for r in loo_results]
    colors_delta = ["red" if d > 0.01 else "green" if d < -0.01 else "gray" for d in rms_deltas]
    ax.barh(range(n), rms_deltas, color=colors_delta, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_xlabel("ΔRMS (positive = removing helps)")
    ax.set_title("Leave-One-Out RMS Impact")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Bottom-left: fx shift when frame is removed
    ax = axes[1, 0]
    fx_deltas = [r["fx_delta"] for r in loo_results]
    colors_fx = ["red" if abs(d) > 20 else "steelblue" for d in fx_deltas]
    ax.barh(range(n), fx_deltas, color=colors_fx, alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_xlabel("Δfx (positive = frame was pulling fx up)")
    ax.set_title("Leave-One-Out fx Impact")
    ax.axvline(x=0, color="black", linewidth=0.5)
    ax.invert_yaxis()

    # Bottom-right: fx values without each frame
    ax = axes[1, 1]
    fx_values = [r["fx_without"] for r in loo_results]
    ax.barh(range(n), fx_values, color="steelblue", alpha=0.8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(frame_labels, fontsize=7)
    ax.set_xlabel("fx without this frame")
    ax.set_title("fx Stability (Leave-One-Out)")
    ax.axvline(x=fx_full, color="red", linestyle="--", alpha=0.7, label=f"Full fx={fx_full:.1f}")
    ax.legend(fontsize=9)
    ax.invert_yaxis()

    fig.suptitle(
        f"{cam_id} — Calibration Diagnostics ({n} frames)\n"
        f"Full: RMS={rms_full:.4f}  fx={fx_full:.1f}  fy={fy_full:.1f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(cam_out / "diagnostics.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved {cam_out / 'diagnostics.png'}")

    # Save results as JSON
    results_json = {
        "camera": cam_id,
        "full_calibration": {
            "rms": float(rms_full),
            "fx": float(fx_full),
            "fy": float(fy_full),
            "num_frames": n,
        },
        "leave_one_out": loo_results,
        "problematic_frames": [r["frame"] for r in problematic],
    }
    with open(cam_out / "diagnostics.json", "w") as f:
        json.dump(results_json, f, indent=2)
    log.info(f"  Saved {cam_out / 'diagnostics.json'}")


@hydra.main(config_path="../configs", config_name="intrinsic_calibration", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Calibration Diagnostics\n{OmegaConf.to_yaml(cfg)}")

    cameras = OmegaConf.to_container(cfg.cameras, resolve=True)
    out_dir = Path(cfg.output.dir).parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter cameras if run_cameras is specified (comma-separated string)
    run_filter = cfg.get("run_cameras", None)
    if run_filter:
        if isinstance(run_filter, str):
            selected = [s.strip() for s in run_filter.split(",")]
        else:
            selected = list(run_filter)
        cameras = {k: v for k, v in cameras.items() if str(k) in selected}

    for cam_id, frame_names in cameras.items():
        cam_id = str(cam_id)
        log.info(f"\n{'='*50}  {cam_id}  {'='*50}")
        analyze_camera(cfg, cam_id, frame_names, out_dir)

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()