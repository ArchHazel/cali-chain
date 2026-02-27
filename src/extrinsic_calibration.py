"""
Extrinsic Calibration Pipeline.

1. Detect apriltags in each camera's selected frame (undistorted inline)
2. Seed known tags from the anchor
3. Walk the chain: for each camera, solve its pose using the CLOSEST known tag
   (by camera-frame depth), then register all newly seen tags
4. Validate against ground-truth validation tags (uses ALL detections)
5. Save extrinsics

Duplicate tag ID handling:
  - Chain (solving + registering): picks detection closest to known world pose
  - Validation: picks detection closest to ground-truth world pose

Usage:
    python -m src.extrinsic_calibration
    python -m src.extrinsic_calibration dataset=calib_4 intrinsics.session=intrinsic_1
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
# Intrinsics
# ---------------------------------------------------------------------------

def load_intrinsics(cfg: DictConfig, cam_id: str) -> tuple[np.ndarray, list[float], np.ndarray | None]:
    """
    Returns (K, cam_params, dist_coeffs).
    dist_coeffs is None if no calibration data available (fallback).
    """
    intrinsics_path = Path(cfg.intrinsics.dir) / cam_id / "intrinsics.json"
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            data = json.load(f)
        K = np.array(data["k_matrix"], dtype=np.float64)
        cam_params = [data["fx"], data["fy"], data["cx"], data["cy"]]
        dist_coeffs = np.array(data["dist_coeffs"], dtype=np.float64) if "dist_coeffs" in data else None
        log.info(f"[{cam_id}] Loaded intrinsics (dist={'yes' if dist_coeffs is not None else 'no'})")
        return K, cam_params, dist_coeffs

    fb = cfg.intrinsics.fallback
    log.warning(f"[{cam_id}] Intrinsics not found, using fallback (no undistortion)")
    K = np.array([
        [fb.fx, 0.0, fb.cx],
        [0.0, fb.fy, fb.cy],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    cam_params = [fb.fx, fb.fy, fb.cx, fb.cy]
    return K, cam_params, None


# ---------------------------------------------------------------------------
# AprilTag detection
# ---------------------------------------------------------------------------

def detect_tags(
    detector: Detector,
    img_path: Path,
    cam_params: list[float],
    tag_size: float,
    K: np.ndarray,
    dist_coeffs: np.ndarray | None = None,
) -> list[dict]:
    """
    Detect apriltags. Undistorts the image first if dist_coeffs is provided.
    Uses the undistorted image for detection but cam_params from the original K
    (since undistort with newCameraMatrix=K preserves the intrinsics).
    """
    img = cv2.imread(str(img_path))
    if img is None:
        log.warning(f"Could not read: {img_path}")
        return []

    # if dist_coeffs is not None:
    #     img = cv2.undistort(img, K, dist_coeffs, None, K)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=cam_params,
        tag_size=tag_size,
    )

    results = []
    for det in detections:
        T = np.eye(4)
        T[:3, :3] = det.pose_R
        T[:3, 3] = det.pose_t.flatten()
        results.append({
            "tag_id": det.tag_id,
            "corners": det.corners,
            "T_cam_tag": T,
        })
    return results


def filter_detections(detections: list[dict], allowed_tags: list[int]) -> list[dict]:
    return [d for d in detections if d["tag_id"] in allowed_tags]


# ---------------------------------------------------------------------------
# Grouping detections by tag ID
# ---------------------------------------------------------------------------

def group_detections_by_id(detections: list[dict]) -> dict[int, list[dict]]:
    groups: dict[int, list[dict]] = {}
    for det in detections:
        groups.setdefault(det["tag_id"], []).append(det)
    return groups


def pick_closest_detection(
    dets: list[dict],
    T_world_cam: np.ndarray,
    target_world_pos: np.ndarray,
) -> dict:
    if len(dets) == 1:
        return dets[0]

    best_det = None
    best_dist = float("inf")
    for det in dets:
        est_pos = (T_world_cam @ det["T_cam_tag"])[:3, 3]
        dist = np.linalg.norm(est_pos - target_world_pos)
        if dist < best_dist:
            best_dist = dist
            best_det = det
    return best_det


# ---------------------------------------------------------------------------
# Tag world pose from config
# ---------------------------------------------------------------------------

def compute_tag_world_pose(tag_cfg: dict) -> np.ndarray:
    half_w = tag_cfg["width_m"] / 2.0
    half_h = tag_cfg["height_m"] / 2.0

    pos_xyz = tag_cfg["position_xyz"]
    corner_pos = np.array([p if p is not None else 0.0 for p in pos_xyz])

    corner = tag_cfg["measured_corner"]
    offsets = {
        "top_left":     np.array([ half_w,  half_h, 0]),
        "top_right":    np.array([-half_w,  half_h, 0]),
        "bottom_right": np.array([-half_w, -half_h, 0]),
        "bottom_left":  np.array([ half_w, -half_h, 0]),
    }
    local_offset = offsets[corner]

    facing = tag_cfg["wall_facing"]
    rotations = {
        "neg_x": np.array([[ 0, 0,  1], [-1, 0, 0], [0, -1, 0]]),
        "pos_x": np.array([[ 0, 0, -1], [ 1, 0, 0], [0, -1, 0]]),
        "neg_y": np.array([[ 1, 0,  0], [ 0, 0, 1], [0, -1, 0]]),
        "pos_y": np.array([[-1, 0,  0], [ 0, 0,-1], [0, -1, 0]]),
    }
    R = rotations[facing]

    center = corner_pos + (R @ local_offset)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = center
    return T


# ---------------------------------------------------------------------------
# Tag corner 3D coordinates in tag-local frame
# ---------------------------------------------------------------------------

def tag_corners_local(tag_size: float) -> np.ndarray:
    h = tag_size / 2.0
    return np.array([
        [-h,  h, 0],
        [ h,  h, 0],
        [ h, -h, 0],
        [-h, -h, 0],
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Single-tag PnP solve (closest known tag)
# ---------------------------------------------------------------------------

def solve_camera_pose(
    known_tags: dict[int, np.ndarray],
    detections: list[dict],
    K: np.ndarray,
    tag_size: float,
) -> tuple[np.ndarray | None, int | None]:
    """
    Solve camera pose using the single closest known tag.
    "Closest" = smallest depth (Z) in camera frame among known-tag detections.
    """
    corners_local = tag_corners_local(tag_size)
    grouped = group_detections_by_id(detections)

    best_tag_id = None
    best_depth = float("inf")
    best_det = None

    for tag_id, dets in grouped.items():
        if tag_id not in known_tags:
            continue
        for det in dets:
            depth = det["T_cam_tag"][2, 3]
            if 0 < depth < best_depth:
                best_depth = depth
                best_tag_id = tag_id
                best_det = det

    if best_tag_id is None:
        return None, None

    T_world_tag = known_tags[best_tag_id]

    # Disambiguate duplicates of the chosen tag
    dets_for_tag = grouped[best_tag_id]
    if len(dets_for_tag) > 1:
        p3d_temp, p2d_temp = [], []
        for i in range(4):
            p_local = np.append(corners_local[i], 1.0)
            p3d_temp.append((T_world_tag @ p_local)[:3])
            p2d_temp.append(best_det["corners"][i])
        ok, rv, tv = cv2.solvePnP(
            np.array(p3d_temp), np.array(p2d_temp), K, None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if ok:
            R_temp, _ = cv2.Rodrigues(rv)
            T_temp = np.eye(4)
            T_temp[:3, :3] = R_temp
            T_temp[:3, 3] = tv.flatten()
            T_world_cam_temp = np.linalg.inv(T_temp)
            best_det = pick_closest_detection(dets_for_tag, T_world_cam_temp, T_world_tag[:3, 3])
        log.info(f"  Tag {best_tag_id}: {len(dets_for_tag)} detections, disambiguated by proximity")

    # Solve with the chosen tag's 4 corners
    pts_3d, pts_2d = [], []
    for i in range(4):
        p_local = np.append(corners_local[i], 1.0)
        p_world = (T_world_tag @ p_local)[:3]
        pts_3d.append(p_world)
        pts_2d.append(best_det["corners"][i])

    success, rvec, tvec = cv2.solvePnP(
        np.array(pts_3d, dtype=np.float64),
        np.array(pts_2d, dtype=np.float64),
        K, None, flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None, best_tag_id

    R_cam_world, _ = cv2.Rodrigues(rvec)
    T_cam_world = np.eye(4)
    T_cam_world[:3, :3] = R_cam_world
    T_cam_world[:3, 3] = tvec.flatten()

    T_world_cam = np.linalg.inv(T_cam_world)
    return T_world_cam, best_tag_id


# ---------------------------------------------------------------------------
# Chain propagation
# ---------------------------------------------------------------------------

def propagate_chain(
    cfg: DictConfig,
    known_tags: dict[int, np.ndarray],
    chain_detections: dict[str, list[dict]],
    all_K: dict[str, np.ndarray],
    tag_size: float,
) -> dict[str, np.ndarray]:
    chain = OmegaConf.to_container(cfg.chain, resolve=True)
    pose_dict: dict[str, np.ndarray] = {}

    for tag_id, T in known_tags.items():
        pose_dict[f"tag{tag_id}"] = T

    for cam_id in chain:
        cam_id = str(cam_id)

        if cam_id not in chain_detections:
            log.warning(f"[{cam_id}] No detections available. Skipping.")
            continue

        detections = chain_detections[cam_id]
        K = all_K[cam_id]

        T_world_cam, used_tag = solve_camera_pose(known_tags, detections, K, tag_size)

        if T_world_cam is None:
            log.warning(f"[{cam_id}] PnP failed. Skipping.")
            continue

        pose_dict[f"cam{cam_id}"] = T_world_cam
        pos = T_world_cam[:3, 3]
        log.info(
            f"[{cam_id}] Solved using tag {used_tag}  "
            f"pos=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]"
        )

        # Register new tags
        grouped = group_detections_by_id(detections)
        new_tags = 0
        for tag_id, dets in grouped.items():
            if tag_id in known_tags:
                continue

            if len(dets) > 1:
                best = None
                best_depth = float("inf")
                for det in dets:
                    depth = det["T_cam_tag"][2, 3]
                    if 0 < depth < best_depth:
                        best_depth = depth
                        best = det
                det = best if best is not None else dets[0]
                log.info(f"  Tag {tag_id}: {len(dets)} detections, picked nearest (depth={best_depth:.3f}m)")
            else:
                det = dets[0]

            T_world_tag = T_world_cam @ det["T_cam_tag"]
            known_tags[tag_id] = T_world_tag
            pose_dict[f"tag{tag_id}"] = T_world_tag

            tag_pos = T_world_tag[:3, 3]
            log.info(
                f"  Registered tag{tag_id}  "
                f"pos=[{tag_pos[0]:.3f}, {tag_pos[1]:.3f}, {tag_pos[2]:.3f}]"
            )
            new_tags += 1

        if new_tags == 0:
            log.info(f"  No new tags registered")

    return pose_dict


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(
    cfg: DictConfig,
    pose_dict: dict[str, np.ndarray],
    all_detections: dict[str, list[dict]],
):
    validation_tags = OmegaConf.to_container(cfg.validation, resolve=True)
    if not validation_tags:
        return

    log.info(f"\n{'='*60}")
    log.info("  VALIDATION")
    log.info(f"{'='*60}")

    axis_names = ["X", "Y", "Z"]
    all_errors = []

    for vtag in validation_tags:
        tag_id = vtag["tag_id"]
        gt_pos_raw = vtag["position_xyz"]

        gt_pose = compute_tag_world_pose(vtag)
        gt_center = gt_pose[:3, 3]

        valid_axes = [i for i, v in enumerate(gt_pos_raw) if v is not None]
        valid_names = [axis_names[i] for i in valid_axes]

        log.info(f"\n  Tag {tag_id} (comparing axes: {valid_names})")
        log.info(f"    GT center: [{gt_center[0]:.4f}, {gt_center[1]:.4f}, {gt_center[2]:.4f}]")

        tag_errors = []

        for cam_key, T_world_cam in pose_dict.items():
            if not cam_key.startswith("cam"):
                continue
            cam_id = cam_key[3:]

            if cam_id not in all_detections:
                continue

            grouped = group_detections_by_id(all_detections[cam_id])
            dets = grouped.get(tag_id)
            if not dets:
                continue

            det = pick_closest_detection(dets, T_world_cam, gt_center)

            T_world_tag_est = T_world_cam @ det["T_cam_tag"]
            est_center = T_world_tag_est[:3, 3]

            delta = est_center - gt_center
            delta_valid = np.array([delta[i] for i in valid_axes])
            error = np.linalg.norm(delta_valid)
            tag_errors.append(error)
            all_errors.append(error)

            delta_str = ", ".join(f"{axis_names[i]}:{delta[i]:+.4f}" for i in valid_axes)
            n_dets = len(dets)
            dup_note = f" (picked from {n_dets} detections)" if n_dets > 1 else ""
            log.info(
                f"    {cam_key:<12} est=[{est_center[0]:.4f}, {est_center[1]:.4f}, {est_center[2]:.4f}]  "
                f"Δ=[{delta_str}]  ||Δ||={error:.4f}m{dup_note}"
            )

        if tag_errors:
            log.info(
                f"    Tag {tag_id} summary: "
                f"mean={np.mean(tag_errors):.4f}m  "
                f"max={np.max(tag_errors):.4f}m  "
                f"min={np.min(tag_errors):.4f}m  "
                f"({len(tag_errors)} cameras)"
            )
        else:
            log.info(f"    Tag {tag_id}: not seen by any calibrated camera")

    if all_errors:
        log.info(f"\n  {'='*50}")
        log.info(
            f"  Overall validation ({len(all_errors)} observations): "
            f"mean={np.mean(all_errors):.4f}m  max={np.max(all_errors):.4f}m"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="extrinsic_calibration", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Extrinsic Calibration Pipeline\n{OmegaConf.to_yaml(cfg)}")

    tag_size = cfg.apriltag.tag_size_m

    # 1. Create detector
    detector = Detector(families=cfg.apriltag.family, **DETECTOR_KWARGS)

    # 2. Detect ALL tags in each camera's frame (undistorted inline)
    cameras = OmegaConf.to_container(cfg.cameras, resolve=True)
    all_detections: dict[str, list[dict]] = {}
    chain_detections: dict[str, list[dict]] = {}
    all_K: dict[str, np.ndarray] = {}

    for cam_id, cam_cfg in cameras.items():
        cam_id = str(cam_id)
        frame_name = cam_cfg["frame"]
        allowed_tags = [int(t) for t in cam_cfg["tags"]]
        frame_path = Path(cfg.data.frames_dir) / cam_id / frame_name

        if not frame_path.exists():
            log.warning(f"[{cam_id}] Frame not found: {frame_path}")
            continue

        K, cam_params, dist_coeffs = load_intrinsics(cfg, cam_id)
        all_K[cam_id] = K

        dets = detect_tags(detector, frame_path, cam_params, tag_size, K, dist_coeffs)
        all_detections[cam_id] = dets
        chain_detections[cam_id] = filter_detections(dets, allowed_tags)

        all_ids = [d["tag_id"] for d in dets]
        chain_ids = [d["tag_id"] for d in chain_detections[cam_id]]
        log.info(f"[{cam_id}] All detected: {all_ids}  Chain filtered: {chain_ids}")

    # 3. Seed known tags from anchor
    anchor_cfg = OmegaConf.to_container(cfg.anchor, resolve=True)
    anchor_pose = compute_tag_world_pose(anchor_cfg)
    known_tags: dict[int, np.ndarray] = {cfg.anchor.tag_id: anchor_pose}
    log.info(f"Anchor tag{cfg.anchor.tag_id} pos: {anchor_pose[:3, 3].tolist()}")

    # 4. Propagate chain (uses filtered detections, closest known tag only)
    pose_dict = propagate_chain(cfg, known_tags, chain_detections, all_K, tag_size)

    # 5. Validate (uses ALL detections)
    validate(cfg, pose_dict, all_detections)

    # 6. Save camera extrinsics
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_extrinsics = {k: pose_dict[k].tolist() for k in sorted(pose_dict) if k.startswith("cam")}
    ext_path = out_dir / "cam_extrinsics.json"
    with open(ext_path, "w") as f:
        json.dump(cam_extrinsics, f, indent=2)
    log.info(f"Saved {len(cam_extrinsics)} camera extrinsics -> {ext_path}")

    # 7. Save all poses for debugging
    all_poses = {k: pose_dict[k].tolist() for k in sorted(pose_dict)}
    with open(out_dir / "all_poses.json", "w") as f:
        json.dump(all_poses, f, indent=2)

    # 8. Summary
    log.info(f"\n{'='*50}")
    log.info("FINAL WORLD POSITIONS")
    log.info(f"{'='*50}")
    for key in sorted(pose_dict):
        pos = pose_dict[key][:3, 3]
        look = pose_dict[key][:3, 2]
        log.info(
            f"  {key:<12} pos=[{pos[0]:7.3f}, {pos[1]:7.3f}, {pos[2]:7.3f}]  "
            f"look=[{look[0]:6.3f}, {look[1]:6.3f}, {look[2]:6.3f}]"
        )


if __name__ == "__main__":
    main()