"""
AprilTag Ground Truth Detection for Recovery Experiments.

Detects AprilTag in each 'apriltag' segment and computes GT camera poses.
Since the tag is fixed in the scene, each detection gives T_cam_tag
(camera pose relative to tag). Relative poses between segments are:

    T_A_B = T_cam_tag_A @ inv(T_cam_tag_B)

This maps points from camera B's frame to camera A's frame.

The script processes all extracted segments under output/recovery/<env>/
and saves:
  - Per-segment: apriltag_gt.json with T_cam_tag and detection stats
  - Per-environment: gt_relative_poses.json with all pairwise GT transforms

Usage:
    python -m recovery.apriltag_gt
    python -m recovery.apriltag_gt environment=bathroom
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
from pupil_apriltags import Detector

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DETECTOR_KWARGS = dict(
    nthreads=4,
    quad_decimate=1.0,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0,
)


# ---------------------------------------------------------------------------
# Intrinsics loading
# ---------------------------------------------------------------------------

def load_intrinsics(intrinsics_dir: str, cam_id: str, fallback: dict) -> tuple[np.ndarray, list[float], np.ndarray | None]:
    """Load color intrinsics for the camera."""
    intrinsics_path = Path(intrinsics_dir) / cam_id / "intrinsics.json"
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            data = json.load(f)
        K = np.array(data["k_matrix"], dtype=np.float64)
        cam_params = [data["fx"], data["fy"], data["cx"], data["cy"]]
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64) if "dist_coeffs" in data else None
        log.info(f"  Loaded intrinsics from {intrinsics_path}")
        log.info(f"    fx={data['fx']:.2f} fy={data['fy']:.2f} cx={data['cx']:.2f} cy={data['cy']:.2f}")
        return K, cam_params, dist_coeffs

    log.warning(f"  Intrinsics not found at {intrinsics_path}, using fallback")
    K = np.array([
        [fallback.fx, 0.0, fallback.cx],
        [0.0, fallback.fy, fallback.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    cam_params = [fallback.fx, fallback.fy, fallback.cx, fallback.cy]
    return K, cam_params, None


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect_tag_in_frame(detector: Detector, frame_path: Path,
                        cam_params: list[float], tag_size: float,
                        tag_id: int, K: np.ndarray,
                        dist_coeffs: np.ndarray | None,
                        undistort: bool) -> dict | None:
    """Detect a specific tag in one frame. Returns T_cam_tag (4x4) or None."""
    img = cv2.imread(str(frame_path))
    if img is None:
        return None

    if undistort and dist_coeffs is not None:
        img = cv2.undistort(img, K, dist_coeffs)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=tag_size,
    )

    for det in detections:
        if det.tag_id == tag_id:
            T = np.eye(4)
            T[:3, :3] = det.pose_R
            T[:3, 3] = det.pose_t.flatten()
            return {
                "T_cam_tag": T,
                "corners": det.corners.tolist(),
                "center": det.center.tolist(),
                "decision_margin": float(det.decision_margin),
            }

    return None


def detect_tag_in_segment(detector: Detector, frames_dir: Path,
                          cam_params: list[float], tag_size: float,
                          tag_id: int, K: np.ndarray,
                          dist_coeffs: np.ndarray | None,
                          undistort: bool,
                          max_frames: int = 0) -> dict:
    """
    Detect tag across all frames in a segment directory.
    Returns averaged T_cam_tag and detection statistics.

    max_frames=0 means use all frames.
    """
    frame_files = sorted(frames_dir.glob("*.jpg"))
    if not frame_files:
        return {"error": "no frames found", "num_detections": 0}

    if max_frames > 0 and len(frame_files) > max_frames:
        # Sample evenly
        indices = np.linspace(0, len(frame_files) - 1, max_frames, dtype=int)
        frame_files = [frame_files[i] for i in indices]

    all_T = []
    all_margins = []

    for fp in frame_files:
        result = detect_tag_in_frame(
            detector, fp, cam_params, tag_size, tag_id,
            K, dist_coeffs, undistort)
        if result is not None:
            all_T.append(result["T_cam_tag"])
            all_margins.append(result["decision_margin"])

    if not all_T:
        return {"error": "tag not detected in any frame", "num_detections": 0}

    # Average rotation via SVD projection and average translation
    R_sum = np.zeros((3, 3))
    t_sum = np.zeros(3)
    for T in all_T:
        R_sum += T[:3, :3]
        t_sum += T[:3, 3]

    # Project averaged R back to SO(3)
    U, _, Vt = np.linalg.svd(R_sum)
    R_avg = U @ Vt
    if np.linalg.det(R_avg) < 0:
        U[:, -1] *= -1
        R_avg = U @ Vt

    t_avg = t_sum / len(all_T)

    T_avg = np.eye(4)
    T_avg[:3, :3] = R_avg
    T_avg[:3, 3] = t_avg

    # Compute spread (std of translation components)
    t_all = np.array([T[:3, 3] for T in all_T])
    t_std = t_all.std(axis=0)

    return {
        "T_cam_tag": T_avg,
        "num_frames": len(frame_files),
        "num_detections": len(all_T),
        "detection_rate": len(all_T) / len(frame_files),
        "avg_decision_margin": float(np.mean(all_margins)),
        "translation_std": t_std.tolist(),
    }


# ---------------------------------------------------------------------------
# Relative pose computation
# ---------------------------------------------------------------------------

def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def compute_relative_pose(T_cam_tag_ref: np.ndarray,
                          T_cam_tag_tgt: np.ndarray) -> np.ndarray:
    """
    Compute relative transform from tgt camera frame to ref camera frame.
    T_ref_tgt = T_cam_tag_ref @ inv(T_cam_tag_tgt)
    """
    return T_cam_tag_ref @ np.linalg.inv(T_cam_tag_tgt)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="apriltag_gt", version_base=None)
def main(cfg: DictConfig):
    log.info(f"AprilTag Ground Truth Detection\n{OmegaConf.to_yaml(cfg)}")

    recovery_dir = Path(cfg.recovery_dir)
    env = cfg.environment

    env_dir = recovery_dir / env
    if not env_dir.exists():
        log.error(f"Environment directory not found: {env_dir}")
        return

    # Load intrinsics
    K, cam_params, dist_coeffs = load_intrinsics(
        cfg.intrinsics.dir, cfg.camera_id, cfg.intrinsics.fallback)

    # Create detector
    detector = Detector(families=cfg.apriltag.family, **DETECTOR_KWARGS)

    tag_id = cfg.apriltag.tag_id
    tag_size = cfg.apriltag.tag_size_m
    undistort = cfg.apriltag.get("undistort", True)

    log.info(f"Tag: family={cfg.apriltag.family} id={tag_id} size={tag_size}m")
    log.info(f"Undistort: {undistort}")

    # Find all segment directories with apriltag subdirs
    gt_poses = {}

    for segment_dir in sorted(env_dir.iterdir()):
        if not segment_dir.is_dir():
            continue

        apriltag_dir = segment_dir / "apriltag"
        if not apriltag_dir.exists():
            continue

        frames_dir = apriltag_dir / "frames"
        if not frames_dir.exists():
            log.warning(f"  {segment_dir.name}/apriltag: no frames/ directory")
            continue

        log.info(f"\n{'='*50}")
        log.info(f"  {segment_dir.name}/apriltag")
        log.info(f"{'='*50}")

        result = detect_tag_in_segment(
            detector, frames_dir, cam_params, tag_size, tag_id,
            K, dist_coeffs, undistort,
            max_frames=cfg.get("max_frames_per_segment", 0),
        )

        if "error" in result:
            log.warning(f"  {result['error']}")
            # Save error result
            with open(apriltag_dir / "apriltag_gt.json", "w") as f:
                json.dump({"segment": segment_dir.name, **result}, f, indent=2)
            continue

        log.info(f"  Detections: {result['num_detections']}/{result['num_frames']} "
                 f"({result['detection_rate']:.0%})")
        log.info(f"  Avg margin: {result['avg_decision_margin']:.1f}")
        log.info(f"  Translation std: {result['translation_std']}")

        T = result["T_cam_tag"]
        t = T[:3, 3]
        log.info(f"  T_cam_tag translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

        # Save per-segment result
        save_result = {
            "segment": segment_dir.name,
            "T_cam_tag": T.tolist(),
            "num_frames": result["num_frames"],
            "num_detections": result["num_detections"],
            "detection_rate": result["detection_rate"],
            "avg_decision_margin": result["avg_decision_margin"],
            "translation_std": result["translation_std"],
        }
        with open(apriltag_dir / "apriltag_gt.json", "w") as f:
            json.dump(save_result, f, indent=2)
        log.info(f"  Saved -> {apriltag_dir / 'apriltag_gt.json'}")

        gt_poses[segment_dir.name] = T

    # Compute pairwise relative poses against orig
    if "orig" not in gt_poses:
        log.warning("No 'orig' segment found — cannot compute relative poses")
        log.info(f"Available: {list(gt_poses.keys())}")
    else:
        T_orig = gt_poses["orig"]
        relative_poses = {}

        log.info(f"\n{'='*60}")
        log.info(f"  GT RELATIVE POSES (vs orig)")
        log.info(f"{'='*60}")

        for name, T_tgt in sorted(gt_poses.items()):
            if name == "orig":
                continue

            # T_orig_tgt: maps tgt camera frame to orig camera frame
            T_rel = compute_relative_pose(T_orig, T_tgt)
            rot_deg = rotation_angle_deg(T_rel[:3, :3])
            trans_m = float(np.linalg.norm(T_rel[:3, 3]))

            log.info(f"  {name:<35} rot={rot_deg:6.2f}°  trans={trans_m:.4f}m")

            relative_poses[name] = {
                "T_orig_tgt": T_rel.tolist(),
                "rotation_deg": rot_deg,
                "translation_m": trans_m,
            }

        # Save relative poses
        out_path = env_dir / "gt_relative_poses.json"
        with open(out_path, "w") as f:
            json.dump(relative_poses, f, indent=2)
        log.info(f"\nSaved GT relative poses -> {out_path}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY")
    log.info(f"{'='*60}")
    log.info(f"  Environment: {env}")
    log.info(f"  Segments with GT: {len(gt_poses)}")
    for name in sorted(gt_poses.keys()):
        log.info(f"    {name}")


if __name__ == "__main__":
    main()