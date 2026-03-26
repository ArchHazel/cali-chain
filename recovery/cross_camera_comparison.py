"""
Cross-Camera Pose Comparison.

Solves the relative pose between two cameras in the SAME session using:
  1. Direct ICP on full point clouds
  2. Semantic plane matching + SVD solve

Compares both against ground truth from AprilTag extrinsics.

This tests the non-overlapping view scenario: two cameras on different
walls likely see different parts of the room, so ICP may struggle while
the plane approach (which only needs shared structural surfaces, not
shared points) should still work.

Prerequisites:
    - semantic_plane_fit.py run on the session
    - Extrinsic calibration run on the session (for ground truth)

Usage:
    python -m recovery.cross_camera_comparison
    python -m recovery.cross_camera_comparison session=calib_4 cam_ref=HAR2 cam_tgt=HAR3
"""

import json
import logging
from pathlib import Path
from itertools import permutations

import numpy as np
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080
STRUCTURAL_CLASSES = ["wall", "floor", "ceiling"]


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


def load_extrinsics(extrinsics_dir: str, filename: str) -> dict[str, np.ndarray]:
    path = Path(extrinsics_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


def load_planes(planes_dir: str, cam_id: str) -> dict:
    path = Path(planes_dir) / f"{cam_id}_planes.json"
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    for key, plane in data.items():
        plane["normal"] = np.array(plane["normal"])
        if "centroid" in plane:
            plane["centroid"] = np.array(plane["centroid"])
        if "axis_0" in plane:
            plane["axis_0"] = np.array(plane["axis_0"])
        if "axis_1" in plane:
            plane["axis_1"] = np.array(plane["axis_1"])
    return data


# ---------------------------------------------------------------------------
# Depth processing (Y-up for plane solve consistency)
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


def accumulate_points(depth_dir, cam_id, K_depth, chunk, frame_idx,
                      num_frames, max_depth, subsample) -> np.ndarray:
    frames = load_depth_frames(depth_dir, cam_id, chunk, frame_idx, num_frames)
    if not frames:
        return np.zeros((0, 3))
    all_pts = []
    for depth_frame in frames:
        pts, depths = backproject_depth(depth_frame, K_depth)
        pts = pts[depths < max_depth]
        if subsample > 1:
            pts = pts[::subsample]
        all_pts.append(pts)
    if not all_pts:
        return np.zeros((0, 3))
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------------------------------------
# ICP (point-to-plane, SciPy)
# ---------------------------------------------------------------------------

def estimate_normals(pts: np.ndarray, k: int = 30) -> np.ndarray:
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbors = pts[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normals[i] = Vt[2]
    return normals


def run_icp(source_pts, target_pts, max_correspondence_distance=0.5,
            init_transform=None, max_iterations=200, tolerance=1e-8):
    from scipy.spatial import cKDTree

    src = source_pts.astype(np.float64).copy()
    tgt = target_pts.astype(np.float64)

    log.info(f"      Estimating normals on {len(tgt)} target points...")
    tgt_normals = estimate_normals(tgt, k=30)
    tree = cKDTree(tgt)

    if init_transform is None:
        init_transform = np.eye(4)

    R_accum = init_transform[:3, :3].copy()
    t_accum = init_transform[:3, 3].copy()
    src_t = (R_accum @ src.T).T + t_accum

    prev_rmse = np.inf
    for iteration in range(max_iterations):
        dists, indices = tree.query(src_t)
        inlier_mask = dists < max_correspondence_distance
        n_inliers = inlier_mask.sum()
        if n_inliers < 6:
            log.warning(f"      ICP: only {n_inliers} inliers at iter {iteration}")
            break

        p = src_t[inlier_mask]
        q = tgt[indices[inlier_mask]]
        n = tgt_normals[indices[inlier_mask]]

        cross = np.cross(p, n)
        A = np.hstack([cross, n])
        b = np.sum(n * (q - p), axis=1)
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        alpha = x[:3]
        dt = x[3:]
        dR = np.eye(3) + np.array([
            [0, -alpha[2], alpha[1]],
            [alpha[2], 0, -alpha[0]],
            [-alpha[1], alpha[0], 0],
        ])
        U, _, Vt = np.linalg.svd(dR)
        dR = U @ Vt

        R_accum = dR @ R_accum
        t_accum = dR @ t_accum + dt
        src_t = (R_accum @ src.T).T + t_accum

        rmse = np.sqrt(np.mean(dists[inlier_mask]**2))
        if abs(prev_rmse - rmse) < tolerance:
            log.info(f"      ICP converged at iter {iteration} (RMSE={rmse:.6f}m)")
            break
        prev_rmse = rmse

    dists_final, _ = tree.query(src_t)
    inlier_final = dists_final < max_correspondence_distance
    fitness = inlier_final.sum() / len(src_t)
    inlier_rmse = np.sqrt(np.mean(dists_final[inlier_final]**2)) if inlier_final.any() else np.inf

    T = np.eye(4)
    T[:3, :3] = R_accum
    T[:3, 3] = t_accum
    return {"T": T, "fitness": float(fitness), "inlier_rmse": float(inlier_rmse)}


# ---------------------------------------------------------------------------
# Plane matching (adapted for cross-camera)
# ---------------------------------------------------------------------------

def normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def solve_rotation(matches):
    H = np.zeros((3, 3))
    for m in matches:
        H += np.outer(m["tgt_normal"], m["ref_normal"])
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    return R


def solve_translation_full(matches, R):
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t


def solve_translation_constrained(matches, R):
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])
    _, S, Vt = np.linalg.svd(A)
    unconstrained_dir = Vt[-1]
    t_full, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t = t_full - np.dot(t_full, unconstrained_dir) * unconstrained_dir
    log.info(f"    Unconstrained dir: [{unconstrained_dir[0]:.3f}, "
             f"{unconstrained_dir[1]:.3f}, {unconstrained_dir[2]:.3f}]")
    return t


def count_independent_directions(matches, parallel_thresh_deg=15.0):
    cos_thresh = np.cos(np.radians(parallel_thresh_deg))
    directions = []
    for m in matches:
        n = m["ref_normal"]
        found = False
        for d in directions:
            if abs(np.dot(n, d)) > cos_thresh:
                found = True
                break
        if not found:
            directions.append(n)
    return len(directions)


# ---------------------------------------------------------------------------
# Scoring: Bounded point-to-plane (primary)
# ---------------------------------------------------------------------------

def _project_points_to_plane_cells(pts, plane, plane_dist_thresh,
                                    extent_margin, cell_size):
    """
    Project points onto a plane's surface and return the set of occupied
    grid cell IDs as (i0, i1) tuples.
    """
    n = plane["normal"]
    centroid = plane["centroid"]
    axis_0 = plane["axis_0"]
    axis_1 = plane["axis_1"]
    extent_0 = plane["extent_0"]
    extent_1 = plane["extent_1"]
    rho = plane["rho"]

    perp_dist = np.abs(pts @ n - rho)
    close_mask = perp_dist < plane_dist_thresh
    if not close_mask.any():
        return set()

    offsets = pts[close_mask] - centroid
    d0_signed = offsets @ axis_0
    d1_signed = offsets @ axis_1
    bounded_mask = ((np.abs(d0_signed) < extent_0 * extent_margin) &
                    (np.abs(d1_signed) < extent_1 * extent_margin))

    if not bounded_mask.any():
        return set()

    d0_inlier = d0_signed[bounded_mask]
    d1_inlier = d1_signed[bounded_mask]

    width_0 = 2.0 * extent_0 * extent_margin
    width_1 = 2.0 * extent_1 * extent_margin
    n_cells_0 = max(1, int(width_0 / cell_size))
    n_cells_1 = max(1, int(width_1 / cell_size))

    i0 = ((d0_inlier + extent_0 * extent_margin) / cell_size).astype(np.int32)
    i1 = ((d1_inlier + extent_1 * extent_margin) / cell_size).astype(np.int32)
    i0 = np.clip(i0, 0, n_cells_0 - 1)
    i1 = np.clip(i1, 0, n_cells_1 - 1)

    return set(zip(i0.tolist(), i1.tolist()))


def score_assignment_bounded(candidate, ref_planes, tgt_planes,
                             ref_pts, tgt_pts,
                             plane_dist_thresh=0.05, extent_margin=1.2,
                             cell_size=0.1):
    """
    Score by spatial overlap on reference plane surfaces.
    For each reference plane, project both ref points and transformed tgt
    points onto the plane surface, count shared grid cells (intersection).
    Score = -sum(per-plane intersection cell counts).
    """
    n_dirs = count_independent_directions(candidate)
    if n_dirs < 2:
        return float("inf"), {}

    R = solve_rotation(candidate)
    if n_dirs >= 3:
        t = solve_translation_full(candidate, R)
    else:
        t = solve_translation_constrained(candidate, R)

    tgt_transformed = (R @ tgt_pts.T).T + t

    total_overlap = 0
    per_plane_overlap = {}

    for label, plane in ref_planes.items():
        ref_cells = _project_points_to_plane_cells(
            ref_pts, plane, plane_dist_thresh, extent_margin, cell_size)
        tgt_cells = _project_points_to_plane_cells(
            tgt_transformed, plane, plane_dist_thresh, extent_margin, cell_size)

        if not ref_cells or not tgt_cells:
            per_plane_overlap[label] = 0.0
            continue

        intersection = ref_cells & tgt_cells
        total_overlap += len(intersection)
        per_plane_overlap[label] = len(intersection) * cell_size * cell_size

    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    trans_m = float(np.linalg.norm(t))

    info = {
        "n_planes": len(candidate),
        "n_dirs": n_dirs,
        "rot_deg": rot_deg,
        "trans_m": trans_m,
        "overlap": int(total_overlap),
        "per_plane": per_plane_overlap,
        "R": R,
        "t": t,
    }

    return -total_overlap, info


# ---------------------------------------------------------------------------
# Scoring: KDTree point cloud fitness (fallback)
# ---------------------------------------------------------------------------

def score_assignment_fitness(candidate, ref_pts, tgt_pts, max_corr_dist=0.5):
    """
    Score by KDTree point cloud fitness (fallback for old plane JSONs
    without centroid/radius).
    """
    from scipy.spatial import cKDTree

    n_dirs = count_independent_directions(candidate)
    if n_dirs < 2:
        return float("inf"), {}

    R = solve_rotation(candidate)
    if n_dirs >= 3:
        t = solve_translation_full(candidate, R)
    else:
        t = solve_translation_constrained(candidate, R)

    tgt_transformed = (R @ tgt_pts.T).T + t
    tree = cKDTree(ref_pts)
    dists, _ = tree.query(tgt_transformed)
    inlier_mask = dists < max_corr_dist
    fitness = float(inlier_mask.sum() / len(tgt_transformed))
    inlier_rmse = float(np.sqrt(np.mean(dists[inlier_mask]**2))) if inlier_mask.any() else float('inf')

    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    trans_m = float(np.linalg.norm(t))

    info = {
        "n_planes": len(candidate),
        "n_dirs": n_dirs,
        "rot_deg": rot_deg,
        "trans_m": trans_m,
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
    }

    return -fitness, info


# ---------------------------------------------------------------------------
# Cross-camera plane matching
# ---------------------------------------------------------------------------

def match_planes_cross_camera(ref_planes, tgt_planes, ref_pts, tgt_pts,
                              max_corr_dist=0.5, T_gt=None):
    """
    Match planes between two different cameras in the same session.

    Floor and ceiling are matched by semantic label (same physical surface).
    Walls are brute-forced since wall_0 in one camera may correspond to
    any wall in the other camera. All pairs are considered — the assignment
    scorer determines which is correct.

    Uses bounded plane scoring if available, falls back to KDTree fitness.

    If T_gt (4×4 ground truth transform in Y-up frame) is provided,
    per-candidate rotation and translation errors are logged.
    """

    # --- Fixed matches by label (floor only; ceiling excluded as unreliable) ---
    fixed_matches = []
    for class_name in ["floor"]:
        if class_name not in ref_planes or class_name not in tgt_planes:
            continue
        ref_n = normalize(ref_planes[class_name]["normal"])
        tgt_n = normalize(tgt_planes[class_name]["normal"])
        raw_dot = np.dot(ref_n, tgt_n)
        if raw_dot < 0:
            a_tgt_n = -tgt_n
            a_tgt_rho = -tgt_planes[class_name]["rho"]
        else:
            a_tgt_n = tgt_n
            a_tgt_rho = tgt_planes[class_name]["rho"]
        dot = abs(raw_dot)
        fixed_matches.append({
            "label": f"{class_name}<->{class_name}",
            "ref_normal": ref_n,
            "tgt_normal": a_tgt_n,
            "ref_rho": ref_planes[class_name]["rho"],
            "tgt_rho": a_tgt_rho,
            "dot": float(dot),
        })
        log.info(f"    Fixed match {class_name}: dot={dot:.4f}  "
                 f"ref_ρ={ref_planes[class_name]['rho']:.3f}  "
                 f"tgt_ρ={a_tgt_rho:.3f}")

    # --- Brute-force wall assignments ---
    ref_walls = {k: v for k, v in ref_planes.items() if k.startswith("wall_")}
    tgt_walls = {k: v for k, v in tgt_planes.items() if k.startswith("wall_")}

    log.info(f"    Ref walls: {sorted(ref_walls.keys())}")
    log.info(f"    Tgt walls: {sorted(tgt_walls.keys())}")

    if not ref_walls or not tgt_walls:
        return fixed_matches

    ref_keys = sorted(ref_walls.keys())
    tgt_keys = sorted(tgt_walls.keys())

    # Pre-compute all wall pairs
    pair_info = {}
    for rk in ref_keys:
        ref_n = normalize(ref_walls[rk]["normal"])
        for tk in tgt_keys:
            tgt_n = normalize(tgt_walls[tk]["normal"])
            raw_dot = np.dot(ref_n, tgt_n)
            if raw_dot < 0:
                a_tgt_n = -tgt_n
                a_tgt_rho = -tgt_walls[tk]["rho"]
            else:
                a_tgt_n = tgt_n
                a_tgt_rho = tgt_walls[tk]["rho"]
            dot = abs(raw_dot)
            pair_info[(rk, tk)] = {
                "ref_normal": ref_n,
                "tgt_normal": a_tgt_n,
                "ref_rho": ref_walls[rk]["rho"],
                "tgt_rho": a_tgt_rho,
                "dot": float(dot),
            }

    # Check if bounded planes are available
    has_bounds = (all("centroid" in p and "axis_0" in p for p in ref_planes.values()) and
                  all("centroid" in p and "axis_0" in p for p in tgt_planes.values()))
    if has_bounds:
        log.info(f"    Scoring by bounded point-to-plane ({len(tgt_pts)} tgt points)")
    else:
        log.info(f"    Scoring by KDTree fitness ({len(ref_pts)} ref, {len(tgt_pts)} tgt points)")

    # Enumerate all possible assignments
    n_ref = len(ref_keys)
    tgt_options = tgt_keys + [None] * n_ref

    best_score = float("inf")
    best_walls = []
    top_candidates = []
    seen = set()

    for perm in permutations(tgt_options, n_ref):
        used_tgt = [t for t in perm if t is not None]
        if len(used_tgt) != len(set(used_tgt)):
            continue

        wall_matches = []
        valid_pairs = []
        for rk, tk in zip(ref_keys, perm):
            if tk is None:
                continue
            info = pair_info[(rk, tk)]
            valid_pairs.append((rk, tk))
            wall_matches.append({
                "label": f"{rk}<->{tk}",
                "ref_normal": info["ref_normal"],
                "tgt_normal": info["tgt_normal"],
                "ref_rho": info["ref_rho"],
                "tgt_rho": info["tgt_rho"],
                "dot": info["dot"],
            })

        key = tuple(sorted(valid_pairs))
        if key in seen:
            continue
        seen.add(key)

        # Score: fixed matches + this wall assignment
        candidate = fixed_matches + wall_matches
        if len(candidate) < 2:
            continue

        n_dirs = count_independent_directions(candidate)
        if n_dirs < 2:
            continue

        if has_bounds:
            score, info = score_assignment_bounded(
                candidate, ref_planes, tgt_planes, ref_pts, tgt_pts,
                plane_dist_thresh=max_corr_dist, extent_margin=1.2)
        else:
            score, info = score_assignment_fitness(
                candidate, ref_pts, tgt_pts, max_corr_dist=max_corr_dist)

        if score < float("inf"):
            info["walls"] = wall_matches
            top_candidates.append({"score": score, **info})

        if score < best_score:
            best_score = score
            best_walls = wall_matches

    # Log top 10 scored candidates with GT comparison
    top_candidates.sort(key=lambda x: x["score"])
    n_show = min(10, len(top_candidates))
    log.info(f"    --- Top {n_show} of {len(top_candidates)} scored candidates ---")

    # Compute GT info if available
    if T_gt is not None:
        gt_rot = rotation_angle_deg(T_gt[:3, :3])
        gt_t = T_gt[:3, 3]
        # Convert GT rotation to Euler angles (XYZ) for per-axis comparison
        from scipy.spatial.transform import Rotation as ScipyRot
        gt_euler = ScipyRot.from_matrix(T_gt[:3, :3]).as_euler('xyz', degrees=True)
        log.info(f"    GT: rot={gt_rot:.2f}°  euler=[{gt_euler[0]:.1f}, {gt_euler[1]:.1f}, {gt_euler[2]:.1f}]°  "
                 f"t=[{gt_t[0]:.3f}, {gt_t[1]:.3f}, {gt_t[2]:.3f}]m")

    for rank, c in enumerate(top_candidates[:n_show]):
        labels = [m["label"] for m in c["walls"]]
        pp = c.get("per_plane", {})
        total_area = sum(pp.values()) if pp else 0.0

        # Full 6DOF output + GT comparison
        gt_str = ""
        if T_gt is not None and "R" in c:
            from scipy.spatial.transform import Rotation as ScipyRot
            c_euler = ScipyRot.from_matrix(c["R"]).as_euler('xyz', degrees=True)
            c_t = c["t"]
            R_err = c["R"] @ T_gt[:3, :3].T
            rot_err = rotation_angle_deg(R_err)
            t_err = c_t - gt_t
            gt_str = (f"  err={rot_err:.1f}°/{np.linalg.norm(t_err):.2f}m  "
                      f"eul=[{c_euler[0]:+.1f},{c_euler[1]:+.1f},{c_euler[2]:+.1f}]  "
                      f"t=[{c_t[0]:+.2f},{c_t[1]:+.2f},{c_t[2]:+.2f}]")

        log.info(f"    #{rank+1:2d} {total_area:5.1f}m²  "
                 f"r={c['rot_deg']:5.1f}° t={c['trans_m']:.2f}m  "
                 f"p={c['n_planes']} d={c['n_dirs']}"
                 f"{gt_str}  {labels}")

    all_matches = fixed_matches + best_walls
    log.info(f"    Best assignment ({len(all_matches)} planes, score={best_score:.4f}):")
    for m in all_matches:
        log.info(f"      {m['label']}: dot={m['dot']:.4f}  "
                 f"ref_ρ={m['ref_rho']:.3f}  tgt_ρ={m['tgt_rho']:.3f}")

    return all_matches


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def evaluate_transform(source_pts, target_pts, T, max_corr_dist=0.5):
    from scipy.spatial import cKDTree
    R, t = T[:3, :3], T[:3, 3]
    src_transformed = (R @ source_pts.T).T + t
    tree = cKDTree(target_pts)
    dists, _ = tree.query(src_transformed)
    inlier_mask = dists < max_corr_dist
    fitness = float(inlier_mask.sum() / len(src_transformed))
    inlier_rmse = float(np.sqrt(np.mean(dists[inlier_mask]**2))) if inlier_mask.any() else float('inf')
    return {"fitness": fitness, "inlier_rmse": inlier_rmse}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="cross_camera_comparison", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Cross-Camera Pose Comparison\n{OmegaConf.to_yaml(cfg)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cam_ref = cfg.cam_ref
    cam_tgt = cfg.cam_tgt

    # Load ground truth extrinsics
    extrinsics = load_extrinsics(cfg.extrinsics_dir, cfg.extrinsics_filename)

    # Find extrinsics keys
    T_world_ref = None
    T_world_tgt = None
    for key in [cam_ref, f"cam{cam_ref}"]:
        if key in extrinsics:
            T_world_ref = extrinsics[key]
            break
    for key in [cam_tgt, f"cam{cam_tgt}"]:
        if key in extrinsics:
            T_world_tgt = extrinsics[key]
            break

    if T_world_ref is None or T_world_tgt is None:
        log.error(f"Missing extrinsics for {cam_ref} or {cam_tgt}")
        return

    # Ground truth relative pose: T_ref_tgt maps tgt camera frame to ref camera frame
    # In extrinsics convention (Y-down + X-negate)
    T_gt_world = np.linalg.inv(T_world_ref) @ T_world_tgt
    gt_rot = rotation_angle_deg(T_gt_world[:3, :3])
    gt_trans = float(np.linalg.norm(T_gt_world[:3, 3]))

    # Convert to Y-up frame for comparison with plane solve
    C = np.diag([-1.0, -1.0, 1.0, 1.0])
    T_gt_yup = C @ T_gt_world @ C

    log.info(f"\n{'='*60}")
    log.info(f"  Ground Truth: {cam_ref} -> {cam_tgt}")
    log.info(f"{'='*60}")
    log.info(f"  Rotation: {gt_rot:.3f}°")
    log.info(f"  Translation: {gt_trans:.4f}m")
    log.info(f"  Ref world pos: {T_world_ref[:3, 3].tolist()}")
    log.info(f"  Tgt world pos: {T_world_tgt[:3, 3].tolist()}")

    # Load depth for both cameras
    kinect_ref = load_kinect_config(cam_ref)
    kinect_tgt = load_kinect_config(cam_tgt)

    ref_pts = accumulate_points(
        cfg.depth_dir, cam_ref, kinect_ref["K_depth"],
        cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
        cfg.depth.max_depth, cfg.depth.subsample,
    )
    tgt_pts = accumulate_points(
        cfg.depth_dir, cam_tgt, kinect_tgt["K_depth"],
        cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
        cfg.depth.max_depth, cfg.depth.subsample,
    )
    log.info(f"\n  Ref points ({cam_ref}): {len(ref_pts)}")
    log.info(f"  Tgt points ({cam_tgt}): {len(tgt_pts)}")

    results = {}

    # =========================================================================
    # Method 1: Plane Solve (fast — run first)
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info(f"  Method 1: Plane Solve")
    log.info(f"{'='*60}")

    ref_planes = load_planes(cfg.planes_dir, cam_ref)
    tgt_planes = load_planes(cfg.planes_dir, cam_tgt)

    if not ref_planes or not tgt_planes:
        log.warning(f"  Missing planes (ref: {len(ref_planes)}, tgt: {len(tgt_planes)})")
        results["plane_solve"] = {"error": "missing planes"}
    else:
        matches = match_planes_cross_camera(ref_planes, tgt_planes, ref_pts, tgt_pts,
                                            max_corr_dist=cfg.icp.max_corr_dist,
                                            T_gt=T_gt_yup)

        if len(matches) < 2:
            log.warning(f"  Only {len(matches)} matches, need at least 2")
            results["plane_solve"] = {"error": f"only {len(matches)} matches"}
        else:
            n_dirs = count_independent_directions(matches)
            log.info(f"  Independent directions: {n_dirs}")

            R = solve_rotation(matches)
            if n_dirs >= 3:
                t = solve_translation_full(matches, R)
                solve_type = "full_6dof"
            else:
                t = solve_translation_constrained(matches, R)
                solve_type = "constrained_5dof"

            T_plane = np.eye(4)
            T_plane[:3, :3] = R
            T_plane[:3, 3] = t

            plane_rot = rotation_angle_deg(R)
            plane_trans = float(np.linalg.norm(t))

            log.info(f"  Solve type: {solve_type}")
            log.info(f"  Rotation: {plane_rot:.3f}°  (GT: {gt_rot:.3f}°)")
            log.info(f"  Translation: {plane_trans:.4f}m  (GT: {gt_trans:.4f}m)")

            # Point cloud evaluation
            pc_eval = evaluate_transform(tgt_pts, ref_pts, T_plane,
                                         max_corr_dist=cfg.icp.max_corr_dist)
            log.info(f"  Fitness: {pc_eval['fitness']:.4f}")
            log.info(f"  Inlier RMSE: {pc_eval['inlier_rmse']:.4f}m")

            # Error vs ground truth
            R_err = T_plane[:3, :3] @ T_gt_yup[:3, :3].T
            rot_err = rotation_angle_deg(R_err)
            t_err = float(np.linalg.norm(T_plane[:3, 3] - T_gt_yup[:3, 3]))
            log.info(f"  Rotation error vs GT: {rot_err:.3f}°")
            log.info(f"  Translation error vs GT: {t_err:.4f}m")

            results["plane_solve"] = {
                "solve_type": solve_type,
                "rotation_deg": plane_rot,
                "translation_m": plane_trans,
                "fitness": pc_eval["fitness"],
                "inlier_rmse": pc_eval["inlier_rmse"],
                "rot_error_vs_gt_deg": rot_err,
                "trans_error_vs_gt_m": t_err,
                "num_matches": len(matches),
                "num_independent_dirs": n_dirs,
                "matches": [
                    {"label": m["label"], "dot": m["dot"],
                     "ref_rho": m["ref_rho"], "tgt_rho": m["tgt_rho"]}
                    for m in matches
                ],
            }

    # =========================================================================
    # Method 2: Direct ICP (slow — run second)
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info(f"  Method 2: Direct ICP")
    log.info(f"{'='*60}")
    try:
        icp_result = run_icp(tgt_pts, ref_pts,
                             max_correspondence_distance=cfg.icp.max_corr_dist)
        T_icp = icp_result["T"]
        icp_rot = rotation_angle_deg(T_icp[:3, :3])
        icp_trans = float(np.linalg.norm(T_icp[:3, 3]))

        log.info(f"  Rotation: {icp_rot:.3f}°  (GT: {gt_rot:.3f}°)")
        log.info(f"  Translation: {icp_trans:.4f}m  (GT: {gt_trans:.4f}m)")
        log.info(f"  Fitness: {icp_result['fitness']:.4f}")
        log.info(f"  Inlier RMSE: {icp_result['inlier_rmse']:.4f}m")

        # Error vs ground truth
        R_err = T_icp[:3, :3] @ T_gt_yup[:3, :3].T
        rot_err = rotation_angle_deg(R_err)
        t_err = float(np.linalg.norm(T_icp[:3, 3] - T_gt_yup[:3, 3]))
        log.info(f"  Rotation error vs GT: {rot_err:.3f}°")
        log.info(f"  Translation error vs GT: {t_err:.4f}m")

        results["direct_icp"] = {
            "rotation_deg": icp_rot,
            "translation_m": icp_trans,
            "fitness": icp_result["fitness"],
            "inlier_rmse": icp_result["inlier_rmse"],
            "rot_error_vs_gt_deg": rot_err,
            "trans_error_vs_gt_m": t_err,
        }
    except Exception as e:
        log.error(f"  ICP failed: {e}")
        results["direct_icp"] = {"error": str(e)}

    # =========================================================================
    # Summary
    # =========================================================================
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY: {cam_ref} -> {cam_tgt}")
    log.info(f"{'='*60}")
    log.info(f"  Ground Truth:  rot={gt_rot:.2f}°  trans={gt_trans:.4f}m")
    log.info(f"  {'Method':<18} {'Rot(°)':<10} {'Trans(m)':<10} {'ΔRot(°)':<10} {'ΔTrans(m)':<10} {'Fitness':<10} {'RMSE(m)':<10}")
    log.info(f"  {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for method, name in [("plane_solve", "Plane Solve"), ("direct_icp", "Direct ICP")]:
        if method not in results or "error" in results[method]:
            err = results.get(method, {}).get("error", "skipped")
            log.info(f"  {name:<18} {err}")
            continue
        r = results[method]
        log.info(f"  {name:<18} {r['rotation_deg']:>10.2f} {r['translation_m']:>10.4f} "
                 f"{r['rot_error_vs_gt_deg']:>10.2f} {r['trans_error_vs_gt_m']:>10.4f} "
                 f"{r['fitness']:>10.4f} {r['inlier_rmse']:>10.4f}")

    # Save
    with open(out_dir / "cross_camera_comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)
    log.info(f"\nSaved -> {out_dir}")


if __name__ == "__main__":
    main()