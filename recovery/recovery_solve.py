"""
Recovery Plane Solve + GT Evaluation.

For each non-orig segment in the extracted recovery data:
  1. Load planes from orig/no_tag/planes.json (reference)
  2. Load planes from <segment>/no_tag/planes.json (target)
  3. Match planes (floor by label, walls brute-force)
  4. Solve R, t via SVD + linear system
  5. Score by bounded spatial intersection
  6. Compare against AprilTag GT from gt_relative_poses.json

Output:
  - output/recovery/<env>/solve_results.json  (all results)
  - Per-segment logging with GT comparison

Usage:
    python -m recovery.recovery_solve
    python -m recovery.recovery_solve environment=bathroom
    python -m recovery.recovery_solve environment=living_room segments='[rotation_g01]'
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


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_kinect_config(cam_id: str, configs_dir: str = "configs/kinect") -> dict:
    config_path = Path(configs_dir) / f"{cam_id}.yaml"
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
    return Vt[-1].reshape(3, 4)


def load_planes(planes_path: Path, max_walls: int = 0) -> dict:
    if not planes_path.exists():
        return {}
    with open(planes_path) as f:
        data = json.load(f)
    # Filter walls if max_walls > 0
    if max_walls > 0:
        data = {k: v for k, v in data.items()
                if not k.startswith("wall_") or int(k.split("_")[1]) < max_walls}
    for key, plane in data.items():
        plane["normal"] = np.array(plane["normal"])
        if "centroid" in plane:
            plane["centroid"] = np.array(plane["centroid"])
        if "axis_0" in plane:
            plane["axis_0"] = np.array(plane["axis_0"])
        if "axis_1" in plane:
            plane["axis_1"] = np.array(plane["axis_1"])
    return data


def load_gt_relative_poses(env_dir: Path) -> dict:
    gt_path = env_dir / "gt_relative_poses.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        data = json.load(f)

    # T_orig_tgt from apriltag_gt.py = T_cam_tag_ref @ inv(T_cam_tag_tgt)
    # This maps tgt_cam -> ref_cam, which is the same direction as the plane
    # solve (R maps tgt normals to ref normals, t maps tgt origin to ref frame).
    #
    # However, AprilTag operates in the horizontally-flipped RGB frame (X-right,
    # Y-down) while plane fitting uses Y-up depth frame. These are related by
    # C = diag(-1,-1,1,1) conjugation: T_yup = C @ T_flipped @ C.
    #
    # No inversion needed — direction already matches.
    C = np.diag([-1.0, -1.0, 1.0, 1.0])

    result = {}
    for name, entry in data.items():
        T_flipped = np.array(entry["T_orig_tgt"])
        T_compare = C @ T_flipped @ C
        result[name] = {
            "T": T_compare,
            "rotation_deg": entry["rotation_deg"],
            "translation_m": entry["translation_m"],
        }
    return result


# ---------------------------------------------------------------------------
# Depth processing (Y-up)
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H - 1, -1, -1)[:, np.newaxis]
    cam[:, :, 2] = 1.0
    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T
    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]
    valid = depth_m > 0
    return cam_flat[valid], depth_m[valid]


def accumulate_points(depth_dir: Path, K_depth: np.ndarray,
                      num_frames: int, max_depth: float,
                      subsample: int) -> np.ndarray:
    depth_path = depth_dir / "depth_1.npy"
    if not depth_path.exists():
        return np.array([])
    depth_chunk = np.load(depth_path)
    n = min(depth_chunk.shape[0], num_frames)
    start = max(0, depth_chunk.shape[0] // 2 - n // 2)

    all_pts = []
    for i in range(start, start + n):
        pts, depths = backproject_depth(depth_chunk[i], K_depth)
        mask = depths < max_depth
        pts = pts[mask]
        if subsample > 1:
            pts = pts[::subsample]
        all_pts.append(pts)

    return np.concatenate(all_pts, axis=0) if all_pts else np.array([])


# ---------------------------------------------------------------------------
# Plane matching utilities
# ---------------------------------------------------------------------------

def normalize(v):
    return v / np.linalg.norm(v)


def count_independent_directions(matches, parallel_thresh_deg=10.0):
    cos_thresh = np.cos(np.radians(parallel_thresh_deg))
    directions = []
    for m in matches:
        n = m["ref_normal"]
        if not any(abs(np.dot(n, d)) > cos_thresh for d in directions):
            directions.append(n)
    return len(directions)


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
    return t


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ---------------------------------------------------------------------------
# Scoring: bounded spatial intersection
# ---------------------------------------------------------------------------

def _project_points_to_plane_cells(pts, plane, plane_dist_thresh,
                                    extent_margin, cell_size):
    centroid = plane["centroid"]
    axis_0 = plane["axis_0"]
    axis_1 = plane["axis_1"]
    normal = plane["normal"]
    ext_0 = plane["extent_0"] * extent_margin
    ext_1 = plane["extent_1"] * extent_margin

    offsets = pts - centroid
    perp_dist = np.abs(offsets @ normal)
    close_mask = perp_dist < plane_dist_thresh

    if not close_mask.any():
        return set()

    close_offsets = offsets[close_mask]
    proj_0 = close_offsets @ axis_0
    proj_1 = close_offsets @ axis_1

    in_bounds = ((np.abs(proj_0) < ext_0) & (np.abs(proj_1) < ext_1))
    if not in_bounds.any():
        return set()

    p0 = proj_0[in_bounds]
    p1 = proj_1[in_bounds]
    i0 = (p0 / cell_size).astype(np.int32)
    i1 = (p1 / cell_size).astype(np.int32)

    return set(zip(i0.tolist(), i1.tolist()))


def score_assignment_bounded(candidate, ref_planes, tgt_planes,
                              ref_pts, tgt_pts,
                              plane_dist_thresh=0.1, extent_margin=1.2,
                              cell_size=0.1):
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

    rot_deg = rotation_angle_deg(R)
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
# Plane matching
# ---------------------------------------------------------------------------

def build_floor_match(ref_planes, tgt_planes):
    matches = []
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
        matches.append({
            "label": f"{class_name}<->{class_name}",
            "ref_normal": ref_n,
            "tgt_normal": a_tgt_n,
            "ref_rho": ref_planes[class_name]["rho"],
            "tgt_rho": a_tgt_rho,
            "dot": float(abs(raw_dot)),
        })
    return matches


def enumerate_wall_assignments(ref_planes, tgt_planes):
    ref_walls = sorted([k for k in ref_planes if k.startswith("wall_")])
    tgt_walls = sorted([k for k in tgt_planes if k.startswith("wall_")])

    if not ref_walls or not tgt_walls:
        return [[]]

    # Pre-compute pair info
    pair_info = {}
    for rk in ref_walls:
        ref_n = normalize(ref_planes[rk]["normal"])
        for tk in tgt_walls:
            tgt_n = normalize(tgt_planes[tk]["normal"])
            raw_dot = np.dot(ref_n, tgt_n)
            if raw_dot < 0:
                a_tgt_n = -tgt_n
                a_tgt_rho = -tgt_planes[tk]["rho"]
            else:
                a_tgt_n = tgt_n
                a_tgt_rho = tgt_planes[tk]["rho"]
            pair_info[(rk, tk)] = {
                "label": f"{rk}<->{tk}",
                "ref_normal": ref_n,
                "tgt_normal": a_tgt_n,
                "ref_rho": ref_planes[rk]["rho"],
                "tgt_rho": a_tgt_rho,
                "dot": float(abs(raw_dot)),
            }

    # Generate all possible assignments (each ref wall maps to at most one tgt wall)
    assignments = []
    n_ref = len(ref_walls)
    n_tgt = len(tgt_walls)

    # For each subset size of ref walls that can be matched
    from itertools import combinations
    for n_match in range(min(n_ref, n_tgt), 0, -1):
        for ref_subset in combinations(range(n_ref), n_match):
            for tgt_perm in permutations(range(n_tgt), n_match):
                wall_matches = []
                for ri, ti in zip(ref_subset, tgt_perm):
                    rk = ref_walls[ri]
                    tk = tgt_walls[int(ti)]
                    wall_matches.append(pair_info[(rk, tk)])
                assignments.append(wall_matches)

    # Also include empty assignment (floor only)
    assignments.append([])
    return assignments


def solve_segment(ref_planes, tgt_planes, ref_pts, tgt_pts,
                  scoring_cfg, T_gt=None):
    """
    Run the full plane matching + solve for one segment pair.
    Returns the best result dict.
    """
    floor_matches = build_floor_match(ref_planes, tgt_planes)
    if floor_matches:
        log.info(f"    Floor: dot={floor_matches[0]['dot']:.4f}  "
                 f"ref_ρ={floor_matches[0]['ref_rho']:.3f}  "
                 f"tgt_ρ={floor_matches[0]['tgt_rho']:.3f}")

    assignments = enumerate_wall_assignments(ref_planes, tgt_planes)
    log.info(f"    Evaluating {len(assignments)} wall assignments...")

    best_score = float("inf")
    best_info = None
    best_walls = []
    all_candidates = []

    for wall_matches in assignments:
        candidate = floor_matches + wall_matches
        if len(candidate) < 2:
            continue

        score, info = score_assignment_bounded(
            candidate, ref_planes, tgt_planes, ref_pts, tgt_pts,
            plane_dist_thresh=scoring_cfg["plane_dist_thresh"],
            extent_margin=scoring_cfg["extent_margin"],
            cell_size=scoring_cfg["cell_size"],
        )

        if not info:
            continue

        # Log GT comparison if available
        if T_gt is not None:
            R_est = info["R"]
            t_est = info["t"]
            R_gt = T_gt[:3, :3]
            t_gt = T_gt[:3, 3]
            R_err = rotation_angle_deg(R_est @ R_gt.T)
            t_err = float(np.linalg.norm(t_est - t_gt))
            info["gt_rot_err"] = R_err
            info["gt_trans_err"] = t_err

        all_candidates.append({
            "score": score,
            "info": info,
            "walls": [m["label"] for m in wall_matches],
        })

        if score < best_score:
            best_score = score
            best_info = info
            best_walls = wall_matches

    if best_info is None:
        return None

    # Log top candidates
    all_candidates.sort(key=lambda x: x["score"])
    log.info(f"    Top candidates:")
    for i, cand in enumerate(all_candidates[:10]):
        ci = cand["info"]
        gt_str = ""
        if "gt_rot_err" in ci:
            gt_str = f"  GT_err: R={ci['gt_rot_err']:.2f}° t={ci['gt_trans_err']:.4f}m"
        log.info(f"      [{i}] score={cand['score']:>8d}  "
                 f"R={ci['rot_deg']:6.2f}°  t={ci['trans_m']:.4f}m  "
                 f"dirs={ci['n_dirs']}  overlap={ci['overlap']}"
                 f"  walls={cand['walls']}{gt_str}")

    # Build result
    R = best_info["R"]
    t = best_info["t"]
    n_dirs = best_info["n_dirs"]

    result = {
        "solve_type": "6DOF" if n_dirs >= 3 else "5DOF",
        "n_dirs": n_dirs,
        "rotation_deg": best_info["rot_deg"],
        "translation_m": best_info["trans_m"],
        "overlap_score": int(best_info["overlap"]),
        "per_plane_overlap": {k: float(v) for k, v in best_info["per_plane"].items()},
        "matched_walls": [m["label"] for m in best_walls],
        "matched_floor": len(floor_matches) > 0,
        "T_ref_tgt": np.eye(4),  # will be filled
    }

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    result["T_ref_tgt"] = T

    if T_gt is not None:
        R_gt = T_gt[:3, :3]
        t_gt = T_gt[:3, 3]
        result["gt_rotation_deg"] = rotation_angle_deg(R_gt)
        result["gt_translation_m"] = float(np.linalg.norm(t_gt))
        result["rotation_error_deg"] = rotation_angle_deg(R @ R_gt.T)
        result["translation_error_m"] = float(np.linalg.norm(t - t_gt))

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="recovery_solve", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Recovery Plane Solve\n{OmegaConf.to_yaml(cfg)}")

    env_dir = Path(cfg.recovery_dir) / cfg.environment
    if not env_dir.exists():
        log.error(f"Environment not found: {env_dir}")
        return

    # Load kinect config + DLT
    kinect = load_kinect_config(cfg.camera_id)
    K_depth = kinect["K_depth"]
    corr_path = kinect["corr_path"]
    corr = np.load(corr_path)
    P_dlt = fit_dlt(corr["points_3d"], corr["points_2d"])

    # Load GT
    gt_poses = load_gt_relative_poses(env_dir)
    log.info(f"GT poses loaded: {len(gt_poses)} segments")

    max_walls = cfg.get("max_walls", 0)

    # Load reference (orig) planes and points
    ref_planes_path = env_dir / "orig" / "no_tag" / "planes.json"
    ref_planes = load_planes(ref_planes_path, max_walls=max_walls)
    if not ref_planes:
        log.error("No reference planes found")
        return
    log.info(f"Reference planes: {list(ref_planes.keys())}")

    ref_depth_dir = env_dir / "orig" / "no_tag" / "depth"
    ref_pts = accumulate_points(ref_depth_dir, K_depth,
                                cfg.depth.num_frames, cfg.depth.max_depth,
                                cfg.depth.subsample)
    log.info(f"Reference points: {len(ref_pts)}")

    # Determine segments to process
    segment_filter = list(cfg.get("segments", [])) or None

    scoring_cfg = OmegaConf.to_container(cfg.scoring, resolve=True)

    # Process each target segment
    results = {}
    for segment_dir in sorted(env_dir.iterdir()):
        if not segment_dir.is_dir():
            continue
        if segment_dir.name in ("orig",):
            continue

        no_tag_dir = segment_dir / "no_tag"
        planes_path = no_tag_dir / "planes.json"
        if not planes_path.exists():
            continue

        if segment_filter and segment_dir.name not in segment_filter:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  {segment_dir.name}")
        log.info(f"{'='*60}")

        # Load target planes
        tgt_planes = load_planes(planes_path, max_walls=max_walls)
        log.info(f"  Target planes: {list(tgt_planes.keys())}")

        # Load target points
        tgt_depth_dir = no_tag_dir / "depth"
        tgt_pts = accumulate_points(tgt_depth_dir, K_depth,
                                    cfg.depth.num_frames, cfg.depth.max_depth,
                                    cfg.depth.subsample)
        log.info(f"  Target points: {len(tgt_pts)}")

        # GT
        T_gt = gt_poses.get(segment_dir.name, {}).get("T") if gt_poses else None
        if T_gt is not None:
            gt_info = gt_poses[segment_dir.name]
            log.info(f"  GT: rot={gt_info['rotation_deg']:.2f}°  "
                     f"trans={gt_info['translation_m']:.4f}m")

        # Solve
        result = solve_segment(ref_planes, tgt_planes, ref_pts, tgt_pts,
                               scoring_cfg, T_gt)

        if result is None:
            log.warning(f"  Solve failed")
            results[segment_dir.name] = {"error": "solve failed"}
            continue

        # Log result
        log.info(f"  Result: {result['solve_type']}  "
                 f"R={result['rotation_deg']:.2f}°  t={result['translation_m']:.4f}m")
        if "rotation_error_deg" in result:
            log.info(f"  GT error: R={result['rotation_error_deg']:.2f}°  "
                     f"t={result['translation_error_m']:.4f}m")

        # Serialize for JSON
        result_json = {k: v for k, v in result.items()}
        result_json["T_ref_tgt"] = result["T_ref_tgt"].tolist()
        results[segment_dir.name] = result_json

    # Save all results
    out_path = env_dir / "solve_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved results -> {out_path}")

    # Summary table
    log.info(f"\n{'='*80}")
    log.info(f"  {'Segment':<30} {'Type':>5} {'R_est':>7} {'R_gt':>7} {'R_err':>7} "
             f"{'t_est':>8} {'t_gt':>8} {'t_err':>8}")
    log.info(f"  {'-'*30} {'-'*5} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

    for name, res in sorted(results.items()):
        if "error" in res:
            log.info(f"  {name:<30} FAILED")
            continue
        r_est = res["rotation_deg"]
        t_est = res["translation_m"]
        r_gt = res.get("gt_rotation_deg", None)
        t_gt = res.get("gt_translation_m", None)
        r_err = res.get("rotation_error_deg", None)
        t_err = res.get("translation_error_m", None)

        r_gt_s = f"{r_gt:7.2f}" if r_gt is not None else "    N/A"
        t_gt_s = f"{t_gt:8.4f}" if t_gt is not None else "     N/A"
        r_err_s = f"{r_err:7.2f}" if r_err is not None else "    N/A"
        t_err_s = f"{t_err:8.4f}" if t_err is not None else "     N/A"

        log.info(f"  {name:<30} {res['solve_type']:>5} {r_est:7.2f} {r_gt_s} {r_err_s} "
                 f"{t_est:8.4f} {t_gt_s} {t_err_s}")


if __name__ == "__main__":
    main()