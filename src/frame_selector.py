"""
Intrinsic Calibration Frame Selector.

Iteratively builds a frame set that produces a calibration with fx
within a target range. Starts with a seed set, then greedily adds
frames that keep fx closest to the target.

Usage:
    python -m src.frame_selector
    python -m src.frame_selector +run_cameras=new
    python -m src.frame_selector +run_cameras=new +target_fx=1081 +fx_tolerance=100
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

DEFAULT_TARGET_FX = 1070.0
DEFAULT_FX_TOLERANCE = 20.0
MIN_SEED_FRAMES = 6


def detect_corners(img_path: Path, board_size: tuple):
    img = cv2.imread(str(img_path))
    if img is None:
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, board_size, None)
    if not ret:
        return None, None
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CORNER_CRITERIA)
    return corners, gray


def try_calibrate(objp, imgpoints_list, image_shape):
    """Calibrate and return (rms, K, dist) or None."""
    if len(imgpoints_list) < 4:
        return None
    objpoints = [objp] * len(imgpoints_list)
    try:
        rms, K, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints_list, image_shape, None, None)
        return rms, K, dist
    except cv2.error:
        return None


def select_frames(cfg, cam_id, frame_names, target_fx, fx_tol, out_dir):
    board_size = tuple(cfg.checkerboard.size)
    square_mm = cfg.checkerboard.square_size_mm
    cam_dir = Path(cfg.data.frames_dir) / cam_id

    objp = np.zeros((board_size[0] * board_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:board_size[0], 0:board_size[1]].T.reshape(-1, 2)
    objp *= square_mm

    # Detect corners in all frames
    candidates = []
    image_shape = None
    for fname in frame_names:
        corners, gray = detect_corners(cam_dir / fname, board_size)
        if corners is None:
            log.warning(f"  No corners in {fname}")
            continue
        if image_shape is None:
            image_shape = gray.shape[::-1]
        candidates.append((fname, corners))

    log.info(f"  {len(candidates)} frames with detected corners")

    if len(candidates) < MIN_SEED_FRAMES:
        log.error(f"  Need at least {MIN_SEED_FRAMES} frames")
        return

    fx_lo = target_fx - fx_tol
    fx_hi = target_fx + fx_tol

    # Score each frame individually: calibrate with small subsets to get
    # a rough per-frame fx estimate. We use each frame + a few neighbors.
    log.info(f"  Target fx: {target_fx:.1f} ± {fx_tol:.1f}  (range [{fx_lo:.1f}, {fx_hi:.1f}])")

    # Phase 1: Find a good seed set
    # Try all combinations of MIN_SEED_FRAMES consecutive frames, pick the
    # seed whose fx is closest to target.
    log.info(f"\n  Phase 1: Finding best seed set ({MIN_SEED_FRAMES} frames)...")

    best_seed_idx = None
    best_seed_fx_dist = float("inf")
    best_seed_fx = None

    # Instead of all combos, try sliding windows and random samples
    indices = list(range(len(candidates)))

    # Sliding windows
    for start in range(len(candidates) - MIN_SEED_FRAMES + 1):
        seed_idx = indices[start:start + MIN_SEED_FRAMES]
        imgpts = [candidates[i][1] for i in seed_idx]
        result = try_calibrate(objp, imgpts, image_shape)
        if result is None:
            continue
        rms, K, _ = result
        fx = K[0, 0]
        dist_to_target = abs(fx - target_fx)
        if dist_to_target < best_seed_fx_dist:
            best_seed_fx_dist = dist_to_target
            best_seed_idx = seed_idx
            best_seed_fx = fx

    # Also try evenly spaced samples
    for offset in range(3):
        step = max(1, len(candidates) // MIN_SEED_FRAMES)
        seed_idx = [min(i * step + offset, len(candidates) - 1) for i in range(MIN_SEED_FRAMES)]
        seed_idx = sorted(set(seed_idx))
        if len(seed_idx) < MIN_SEED_FRAMES:
            continue
        imgpts = [candidates[i][1] for i in seed_idx]
        result = try_calibrate(objp, imgpts, image_shape)
        if result is None:
            continue
        rms, K, _ = result
        fx = K[0, 0]
        dist_to_target = abs(fx - target_fx)
        if dist_to_target < best_seed_fx_dist:
            best_seed_fx_dist = dist_to_target
            best_seed_idx = seed_idx
            best_seed_fx = fx

    if best_seed_idx is None:
        log.error("  Could not find any valid seed set")
        return

    seed_names = [candidates[i][0] for i in best_seed_idx]
    log.info(f"  Best seed: {seed_names}  fx={best_seed_fx:.1f}")

    # Phase 2: Greedy frame addition
    log.info(f"\n  Phase 2: Greedy frame addition...")

    selected_idx = list(best_seed_idx)
    remaining_idx = [i for i in indices if i not in selected_idx]
    history = [{
        "step": 0,
        "action": "seed",
        "frame": "seed",
        "n_frames": len(selected_idx),
        "fx": float(best_seed_fx),
        "rms": None,
    }]

    # Get current calibration state
    imgpts_selected = [candidates[i][1] for i in selected_idx]
    result = try_calibrate(objp, imgpts_selected, image_shape)
    current_fx = result[1][0, 0] if result else best_seed_fx
    current_rms = result[0] if result else None
    history[0]["rms"] = float(current_rms) if current_rms else None

    step = 1
    rejected = []
    for _ in range(len(remaining_idx)):
        if not remaining_idx:
            break

        # Try adding each remaining frame, pick the one that keeps fx closest to target
        best_candidate = None
        best_fx_after = None
        best_rms_after = None
        best_fx_dist = float("inf")

        for idx in remaining_idx:
            test_imgpts = imgpts_selected + [candidates[idx][1]]
            result = try_calibrate(objp, test_imgpts, image_shape)
            if result is None:
                continue
            rms, K, _ = result
            fx = K[0, 0]

            # Must stay within tolerance
            if fx < fx_lo or fx > fx_hi:
                continue

            dist_to_target = abs(fx - target_fx)
            if dist_to_target < best_fx_dist:
                best_fx_dist = dist_to_target
                best_candidate = idx
                best_fx_after = fx
                best_rms_after = rms

        if best_candidate is None:
            # All remaining frames push fx out of range
            rejected.extend(remaining_idx)
            log.info(f"  Step {step}: No more frames can be added within fx tolerance. "
                     f"Rejected {len(remaining_idx)} remaining frames.")
            break

        # Add the best candidate
        selected_idx.append(best_candidate)
        imgpts_selected.append(candidates[best_candidate][1])
        remaining_idx.remove(best_candidate)
        current_fx = best_fx_after
        current_rms = best_rms_after

        history.append({
            "step": step,
            "action": "add",
            "frame": candidates[best_candidate][0],
            "n_frames": len(selected_idx),
            "fx": float(best_fx_after),
            "rms": float(best_rms_after),
        })

        log.info(f"  Step {step}: +{candidates[best_candidate][0]}  "
                 f"n={len(selected_idx)}  fx={best_fx_after:.1f}  rms={best_rms_after:.4f}")
        step += 1

    # Final calibration
    log.info(f"\n  Final calibration with {len(selected_idx)} frames...")
    result = try_calibrate(objp, imgpts_selected, image_shape)
    if result is None:
        log.error("  Final calibration failed")
        return

    rms_final, K_final, dist_final = result
    fx_final, fy_final = K_final[0, 0], K_final[1, 1]
    cx_final, cy_final = K_final[0, 2], K_final[1, 2]

    log.info(f"  RMS={rms_final:.4f}  fx={fx_final:.1f}  fy={fy_final:.1f}  "
             f"cx={cx_final:.1f}  cy={cy_final:.1f}")

    selected_names = [candidates[i][0] for i in selected_idx]
    rejected_names = [candidates[i][0] for i in rejected] + \
                     [candidates[i][0] for i in indices if i not in selected_idx and i not in rejected]

    log.info(f"\n  Selected {len(selected_names)} frames: {selected_names}")
    if rejected_names:
        log.info(f"  Rejected {len(rejected_names)} frames: {rejected_names}")

    # Save outputs
    cam_out = out_dir / cam_id
    cam_out.mkdir(parents=True, exist_ok=True)

    # Save selected intrinsics
    intrinsics = {
        "camera_name": cam_id,
        "resolution": list(image_shape),
        "fx": float(fx_final),
        "fy": float(fy_final),
        "cx": float(cx_final),
        "cy": float(cy_final),
        "k_matrix": K_final.tolist(),
        "dist_coeffs": dist_final.tolist(),
        "rms_error": float(rms_final),
        "num_frames_used": len(selected_idx),
    }
    with open(cam_out / "intrinsics.json", "w") as f:
        json.dump(intrinsics, f, indent=4)
    log.info(f"  Saved intrinsics -> {cam_out / 'intrinsics.json'}")

    # Save selection report
    report = {
        "camera": cam_id,
        "target_fx": target_fx,
        "fx_tolerance": fx_tol,
        "final_fx": float(fx_final),
        "final_fy": float(fy_final),
        "final_rms": float(rms_final),
        "selected_frames": selected_names,
        "rejected_frames": rejected_names,
        "history": history,
    }
    with open(cam_out / "frame_selection.json", "w") as f:
        json.dump(report, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # fx over steps
    ax = axes[0]
    steps = [h["step"] for h in history]
    fxs = [h["fx"] for h in history]
    ax.plot(steps, fxs, "b-o", markersize=5)
    ax.axhline(y=target_fx, color="green", linestyle="--", label=f"Target fx={target_fx:.0f}")
    ax.axhspan(fx_lo, fx_hi, alpha=0.1, color="green", label=f"±{fx_tol:.0f} tolerance")
    ax.set_xlabel("Step")
    ax.set_ylabel("fx")
    ax.set_title("fx During Frame Selection")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # RMS over steps
    ax = axes[1]
    rms_vals = [h["rms"] for h in history if h["rms"] is not None]
    rms_steps = [h["step"] for h in history if h["rms"] is not None]
    ax.plot(rms_steps, rms_vals, "r-o", markersize=5)
    ax.set_xlabel("Step")
    ax.set_ylabel("RMS (px)")
    ax.set_title("RMS During Frame Selection")
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"{cam_id} — Frame Selection\n"
        f"Selected {len(selected_names)}/{len(candidates)} frames  "
        f"fx={fx_final:.1f}  rms={rms_final:.4f}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(str(cam_out / "frame_selection.png"), dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved plot -> {cam_out / 'frame_selection.png'}")


@hydra.main(config_path="../configs", config_name="intrinsic_calibration", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Frame Selector\n{OmegaConf.to_yaml(cfg)}")

    target_fx = float(cfg.get("target_fx", DEFAULT_TARGET_FX))
    fx_tol = float(cfg.get("fx_tolerance", DEFAULT_FX_TOLERANCE))

    cameras = OmegaConf.to_container(cfg.cameras, resolve=True)
    out_dir = Path(cfg.output.dir).parent / "frame_selection"
    out_dir.mkdir(parents=True, exist_ok=True)

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
        select_frames(cfg, cam_id, frame_names, target_fx, fx_tol, out_dir)

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()