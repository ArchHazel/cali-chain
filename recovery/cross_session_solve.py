"""
Cross-Session Pose Solve.

Matches semantically-labeled planes between two sessions for each camera
and solves for the rigid transform (R, t) that maps the old session's
camera frame to the new session's camera frame.

Pipeline:
  1. Load plane parameters (normal + rho + centroid + radius) from both sessions
  2. Match planes by semantic label and closest normal direction
  3. Solve rotation via SVD on matched normals
  4. Solve translation via linear system from matched rho values
  5. If only 2 non-parallel planes: zero the unconstrained translation DOF
  6. Combine with new session extrinsics to recover old session world poses

Scoring:
  Wall assignments are scored using bounded point-to-plane distance.
  Each candidate's R and t transform the target point cloud; points are
  counted as inliers only if they are both perpendicular-close to a
  reference plane AND within the spatial bounds (centroid + radius) of
  where that plane was actually observed. This disambiguates parallel
  walls: the wrong assignment puts points at correct perpendicular distance
  but far from the plane's observed spatial extent.

  Score is raw inlier count (not a ratio), so points outside any plane's
  bounds are simply ignored rather than penalized.

  Falls back to KDTree point cloud fitness if planes lack centroid/radius
  (backward compatibility with old plane JSONs).

Prerequisites:
    Run semantic_plane_fit.py on both sessions first.

Usage:
    python -m recovery.cross_session_solve
    python -m recovery.cross_session_solve reference.session=calib_5 target.session=calib_3
"""

import json
import logging
from pathlib import Path
from itertools import permutations

import numpy as np
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_planes(planes_dir: str, cam_id: str) -> dict:
    """Load plane parameters from JSON."""
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


def load_extrinsics(extrinsics_dir: str, filename: str) -> dict[str, np.ndarray]:
    path = Path(extrinsics_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


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
    return {"K_depth": K}


def load_depth_frames(depth_dir: str, cam_id: str, chunk: int,
                      start_frame: int, num_frames: int) -> list[np.ndarray]:
    depth_path = Path(depth_dir) / cam_id / "depth" / f"depth_{chunk}.npy"
    if not depth_path.exists():
        return []
    depth_chunk = np.load(depth_path)
    end_frame = min(start_frame + num_frames, depth_chunk.shape[0])
    return [depth_chunk[i] for i in range(start_frame, end_frame)]


def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project to 3D in Y-up Kinect camera space."""
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


def accumulate_points(depth_dir: str, cam_id: str, K_depth: np.ndarray,
                      chunk: int, frame_idx: int, num_frames: int,
                      max_depth: float, subsample: int) -> np.ndarray:
    """Accumulate depth points in Y-up camera frame. Returns Nx3."""
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
# Plane matching utilities
# ---------------------------------------------------------------------------

def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


def build_floor_match(ref_planes: dict, tgt_planes: dict,
                      angle_thresh_deg: float = 30.0) -> list[dict]:
    """Match floor between sessions by label. Returns list with 0 or 1 match."""
    cos_thresh = np.cos(np.radians(angle_thresh_deg))
    matches = []

    if "floor" in ref_planes and "floor" in tgt_planes:
        ref_n = normalize(ref_planes["floor"]["normal"])
        tgt_n = normalize(tgt_planes["floor"]["normal"])

        if np.dot(ref_n, tgt_n) < 0:
            tgt_n = -tgt_n
            tgt_rho = -tgt_planes["floor"]["rho"]
        else:
            tgt_rho = tgt_planes["floor"]["rho"]

        dot = abs(np.dot(ref_n, tgt_n))
        if dot > cos_thresh:
            matches.append({
                "label": "floor",
                "ref_normal": ref_n,
                "tgt_normal": tgt_n,
                "ref_rho": ref_planes["floor"]["rho"],
                "tgt_rho": tgt_rho,
                "dot": dot,
            })
    return matches


def enumerate_wall_assignments(ref_planes: dict, tgt_planes: dict,
                                angle_thresh_deg: float = 30.0) -> list[list[dict]]:
    """
    Enumerate all valid wall-to-wall assignments.

    Each assignment is a list of matched wall pairs. A ref wall can match
    at most one tgt wall and vice versa. Walls whose normals differ by
    more than angle_thresh_deg are not paired.
    """
    cos_thresh = np.cos(np.radians(angle_thresh_deg))

    ref_walls = {k: v for k, v in ref_planes.items() if k.startswith("wall_")}
    tgt_walls = {k: v for k, v in tgt_planes.items() if k.startswith("wall_")}

    if not ref_walls or not tgt_walls:
        return [[]]

    ref_keys = sorted(ref_walls.keys())
    tgt_keys = sorted(tgt_walls.keys())

    # Pre-compute aligned normals and rhos for all pairs
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
            valid = dot > cos_thresh
            pair_info[(rk, tk)] = {
                "ref_normal": ref_n,
                "tgt_normal": a_tgt_n,
                "ref_rho": ref_walls[rk]["rho"],
                "tgt_rho": a_tgt_rho,
                "dot": float(dot),
                "valid": valid,
            }

    # Generate all possible assignments
    assignments = []
    n_ref = len(ref_keys)
    tgt_options = tgt_keys + [None] * n_ref

    seen = set()
    for perm in permutations(tgt_options, n_ref):
        used_tgt = [t for t in perm if t is not None]
        if len(used_tgt) != len(set(used_tgt)):
            continue

        # Build assignment, filtering out invalid (angle too large) pairs
        assignment = []
        valid_pairs = []
        for rk, tk in zip(ref_keys, perm):
            if tk is None:
                continue
            info = pair_info[(rk, tk)]
            if not info["valid"]:
                continue
            valid_pairs.append((rk, tk))
            assignment.append({
                "label": f"{rk}<->{tk}",
                "ref_normal": info["ref_normal"],
                "tgt_normal": info["tgt_normal"],
                "ref_rho": info["ref_rho"],
                "tgt_rho": info["tgt_rho"],
                "dot": info["dot"],
            })

        # Deduplicate by the actual surviving pairs after validity filtering.
        # Different permutations can produce the same effective assignment
        # when unmatched (None) or invalid pairs differ.
        key = tuple(sorted(valid_pairs))
        if key in seen:
            continue
        seen.add(key)

        assignments.append(assignment)

    return assignments


# ---------------------------------------------------------------------------
# Check non-parallelism
# ---------------------------------------------------------------------------

def count_independent_directions(matches: list[dict],
                                 parallel_thresh_deg: float = 15.0) -> int:
    """Count how many independent normal directions exist among matched planes."""
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
# Pose solve
# ---------------------------------------------------------------------------

def solve_rotation(matches: list[dict]) -> np.ndarray:
    """
    Solve rotation from matched plane normals via SVD.
    R maps target (old) normals to reference (new) normals: R @ tgt_n ≈ ref_n
    """
    H = np.zeros((3, 3))
    for m in matches:
        H += np.outer(m["tgt_normal"], m["ref_normal"])

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def solve_translation_full(matches: list[dict], R: np.ndarray) -> np.ndarray:
    """
    Solve full 3DOF translation from matched rho values.
    For each match: ref_n · t = ref_rho - tgt_rho
    """
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])
    t, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return t


def solve_translation_constrained(matches: list[dict], R: np.ndarray,
                                   directions: list = None) -> np.ndarray:
    """
    Solve translation with only 2 independent plane directions.
    The unconstrained direction gets zero translation.
    """
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])

    _, S, Vt = np.linalg.svd(A)
    unconstrained_dir = Vt[-1]

    t_full, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    t = t_full - np.dot(t_full, unconstrained_dir) * unconstrained_dir

    return t


# ---------------------------------------------------------------------------
# Scoring: Bounded point-to-plane (primary)
# ---------------------------------------------------------------------------

def score_assignment_bounded(candidate: list[dict], ref_planes: dict,
                             tgt_pts: np.ndarray,
                             plane_dist_thresh: float = 0.05,
                             extent_margin: float = 1.2) -> tuple[float, dict]:
    """
    Score a wall assignment using bounded point-to-plane distance.

    For each reference plane, a transformed target point is an inlier only if:
      1. Perpendicular distance to the plane < plane_dist_thresh
      2. On-plane projection falls within the oriented bounding rectangle
         (centroid + axis_0/axis_1 + extent_0/extent_1) with margin

    Uses an oriented bounding rectangle instead of a circle, which better
    captures elongated shapes like walls and pillars.

    Returns: (score, info_dict) where score is negative inlier count (lower=better).
    """
    n_dirs = count_independent_directions(candidate)
    if n_dirs < 2:
        return float("inf"), {}

    R = solve_rotation(candidate)
    if n_dirs >= 3:
        t = solve_translation_full(candidate, R)
    else:
        t = solve_translation_constrained(candidate, R)

    # Transform target points into reference frame
    tgt_transformed = (R @ tgt_pts.T).T + t

    # Count inliers across all reference planes
    total_inliers = 0
    for label, plane in ref_planes.items():
        n = plane["normal"]
        centroid = plane["centroid"]
        axis_0 = plane["axis_0"]
        axis_1 = plane["axis_1"]
        extent_0 = plane["extent_0"]
        extent_1 = plane["extent_1"]
        rho = plane["rho"]

        # 1. Perpendicular distance to infinite plane
        perp_dist = np.abs(tgt_transformed @ n - rho)
        close_mask = perp_dist < plane_dist_thresh

        if not close_mask.any():
            continue

        # 2. Project onto plane axes, check within oriented bounding rect
        close_pts = tgt_transformed[close_mask]
        offsets = close_pts - centroid
        d0 = np.abs(offsets @ axis_0)
        d1 = np.abs(offsets @ axis_1)
        bounded_mask = (d0 < extent_0 * extent_margin) & (d1 < extent_1 * extent_margin)

        total_inliers += bounded_mask.sum()

    rot_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))
    trans_m = float(np.linalg.norm(t))

    info = {
        "n_planes": len(candidate),
        "n_dirs": n_dirs,
        "rot_deg": rot_deg,
        "trans_m": trans_m,
        "total_inliers": int(total_inliers),
    }

    score = -total_inliers  # more inliers = lower score = better
    return score, info


# ---------------------------------------------------------------------------
# Scoring: KDTree point cloud fitness (fallback)
# ---------------------------------------------------------------------------

def score_assignment_fitness(candidate: list[dict],
                             ref_pts: np.ndarray, tgt_pts: np.ndarray,
                             max_corr_dist: float = 0.1) -> tuple[float, dict]:
    """
    Score by KDTree point cloud fitness (fallback for old plane JSONs
    without centroid/radius).

    Returns: (score, info_dict) where score is negative fitness (lower=better).
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

    # Transform target points
    tgt_transformed = (R @ tgt_pts.T).T + t

    # Measure fitness: fraction of target points with a neighbor within max_corr_dist
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

    score = -fitness
    return score, info


# ---------------------------------------------------------------------------
# Plane matching (main entry point)
# ---------------------------------------------------------------------------

def match_planes(ref_planes: dict, tgt_planes: dict,
                 ref_pts: np.ndarray, tgt_pts: np.ndarray,
                 angle_thresh_deg: float = 30.0,
                 max_corr_dist: float = 0.1) -> list[dict]:
    """
    Match planes between reference (new session) and target (old session).

    Strategy:
      - floor<->floor: direct match by label
      - walls: brute-force all valid assignments, score by bounded plane fitness
      - ceiling excluded (unreliable detection in most cameras)

    Scoring: if planes have centroid/radius (bounded planes), uses fast
    bounded point-to-plane scoring. Otherwise falls back to KDTree fitness.

    Returns list of matched pairs with aligned normals and rho values.
    """
    # Floor match (fixed by label)
    floor_matches = build_floor_match(ref_planes, tgt_planes, angle_thresh_deg)
    if floor_matches:
        log.info(f"  Matched floor: dot={floor_matches[0]['dot']:.4f}  "
                 f"ref_ρ={floor_matches[0]['ref_rho']:.3f}  "
                 f"tgt_ρ={floor_matches[0]['tgt_rho']:.3f}")

    # Enumerate all valid wall assignments
    assignments = enumerate_wall_assignments(ref_planes, tgt_planes, angle_thresh_deg)
    log.info(f"  Evaluating {len(assignments)} wall assignments...")

    # Check if bounded planes are available
    has_bounds = all("centroid" in p and "axis_0" in p for p in ref_planes.values())
    if has_bounds:
        log.info(f"  Scoring by bounded point-to-plane ({len(tgt_pts)} tgt points)")
    else:
        log.info(f"  Scoring by KDTree fitness ({len(ref_pts)} ref, {len(tgt_pts)} tgt points)")

    # Score each assignment
    best_score = float("inf")
    best_walls = []
    all_scored = []

    for wall_matches in assignments:
        candidate = floor_matches + wall_matches
        if len(candidate) < 2:
            continue

        n_dirs = count_independent_directions(candidate)
        if n_dirs < 2:
            continue

        if has_bounds:
            score, info = score_assignment_bounded(
                candidate, ref_planes, tgt_pts,
                plane_dist_thresh=max_corr_dist, extent_margin=1.2)
        else:
            score, info = score_assignment_fitness(
                candidate, ref_pts, tgt_pts, max_corr_dist)

        if score < float("inf"):
            info["walls"] = wall_matches
            info["mean_drho"] = float(np.mean([abs(m["ref_rho"] - m["tgt_rho"]) for m in candidate]))
            all_scored.append({"score": score, **info})

        if score < best_score:
            best_score = score
            best_walls = wall_matches

    # Log top 5 candidates
    all_scored.sort(key=lambda x: x["score"])
    for rank, c in enumerate(all_scored[:5]):
        labels = [m["label"] for m in c["walls"]]
        if "total_inliers" in c:
            metric = f"inliers={c['total_inliers']}"
        else:
            metric = f"fitness={c['fitness']:.4f}"
        log.info(f"  #{rank+1} {metric}  "
                 f"planes={c['n_planes']}  dirs={c['n_dirs']}  "
                 f"rot={c['rot_deg']:.2f}°  trans={c['trans_m']:.4f}m  "
                 f"walls={labels}")

    # Log best wall matches
    for m in best_walls:
        rho_diff = abs(m["ref_rho"] - m["tgt_rho"])
        log.info(f"  Matched {m['label']}: dot={m['dot']:.4f}  "
                 f"Δρ={rho_diff:.3f}m  "
                 f"ref_ρ={m['ref_rho']:.3f}  tgt_ρ={m['tgt_rho']:.3f}")

    log.info(f"  Best assignment score: {best_score:.4f}")

    # Log unmatched walls
    ref_walls = {k for k in ref_planes if k.startswith("wall_")}
    tgt_walls = {k for k in tgt_planes if k.startswith("wall_")}
    matched_ref = {m["label"].split("<->")[0] for m in best_walls}
    matched_tgt = {m["label"].split("<->")[1] for m in best_walls}
    for rk in sorted(ref_walls - matched_ref):
        log.info(f"  {rk}: no match")
    for tk in sorted(tgt_walls - matched_tgt):
        log.info(f"  (target {tk}: unmatched)")

    return floor_matches + best_walls


# ---------------------------------------------------------------------------
# Plane selection (for the final solve)
# ---------------------------------------------------------------------------

def select_best_planes(matches: list[dict],
                       parallel_thresh_deg: float = 15.0,
                       min_dot: float = 0.95) -> list[dict]:
    """
    Select the best subset of planes for the solve.

    Strategy: pick one plane per independent normal direction, choosing
    the one with the highest normal dot product (best geometric agreement).
    Reject any plane with dot < min_dot (unreliable normal).
    """
    cos_thresh = np.cos(np.radians(parallel_thresh_deg))

    # Filter out unreliable planes
    reliable = [(i, m) for i, m in enumerate(matches) if m["dot"] >= min_dot]
    rejected = [(i, m) for i, m in enumerate(matches) if m["dot"] < min_dot]
    for idx, m in rejected:
        log.info(f"  Rejected {m['label']}: dot={m['dot']:.4f} < {min_dot}")

    # Sort by dot product (best normal agreement first)
    reliable.sort(key=lambda x: -x[1]["dot"])

    # Greedily pick one per independent direction
    selected = []
    selected_normals = []

    for idx, m in reliable:
        n = m["ref_normal"]
        is_parallel = False
        for sn in selected_normals:
            if abs(np.dot(n, sn)) > cos_thresh:
                is_parallel = True
                break
        if not is_parallel:
            selected.append(m)
            selected_normals.append(n)
            rho_diff = abs(m["ref_rho"] - m["tgt_rho"])
            log.info(f"  Selected {m['label']}: dot={m['dot']:.4f}  Δρ={rho_diff:.3f}m")

    return selected


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_residuals(matches: list[dict], R: np.ndarray, t: np.ndarray) -> dict:
    """Compute per-match residuals after applying the solved transform."""
    residuals = {}
    for m in matches:
        rotated_n = R @ m["tgt_normal"]
        dot = np.clip(np.dot(rotated_n, m["ref_normal"]), -1, 1)
        angle_err = np.degrees(np.arccos(abs(dot)))

        rho_predicted = np.dot(m["ref_normal"], t) + m["tgt_rho"]
        rho_err = abs(rho_predicted - m["ref_rho"])

        residuals[m["label"]] = {
            "normal_angle_err_deg": float(angle_err),
            "rho_err_m": float(rho_err),
        }
    return residuals


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="cross_session_solve", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Cross-Session Pose Solve\n{OmegaConf.to_yaml(cfg)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference (new session) extrinsics
    ref_extrinsics = load_extrinsics(cfg.reference.extrinsics_dir,
                                     cfg.reference.extrinsics_filename)

    cameras = list(cfg.cameras)
    results = {}

    for cam_id in cameras:
        log.info(f"\n{'='*60}")
        log.info(f"  {cam_id}")
        log.info(f"{'='*60}")

        # Load planes from both sessions
        ref_planes = load_planes(cfg.reference.planes_dir, cam_id)
        tgt_planes = load_planes(cfg.target.planes_dir, cam_id)

        if not ref_planes:
            log.warning(f"[{cam_id}] No reference planes, skipping")
            continue
        if not tgt_planes:
            log.warning(f"[{cam_id}] No target planes, skipping")
            continue

        log.info(f"  Reference planes: {list(ref_planes.keys())}")
        log.info(f"  Target planes:    {list(tgt_planes.keys())}")

        # Load depth points for scoring
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue

        K_depth = kinect["K_depth"]
        ref_pts = accumulate_points(
            cfg.reference.depth_dir, cam_id, K_depth,
            cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
            cfg.depth.max_depth, cfg.depth.subsample,
        )
        tgt_pts = accumulate_points(
            cfg.target.depth_dir, cam_id, K_depth,
            cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
            cfg.depth.max_depth, cfg.depth.subsample,
        )
        log.info(f"  Depth points: {len(ref_pts)} ref, {len(tgt_pts)} tgt")

        if len(ref_pts) < 1000 or len(tgt_pts) < 1000:
            log.warning(f"[{cam_id}] Too few depth points for scoring, skipping")
            continue

        # Match planes
        log.info(f"  --- Plane Matching ---")
        matches = match_planes(ref_planes, tgt_planes, ref_pts, tgt_pts,
                               angle_thresh_deg=cfg.matching.angle_thresh_deg,
                               max_corr_dist=cfg.matching.get("max_corr_dist", 0.1))

        if len(matches) < 2:
            log.warning(f"[{cam_id}] Only {len(matches)} matches, need at least 2. Skipping.")
            continue

        # Select best planes: one per independent direction, highest dot
        log.info(f"  --- Plane Selection ---")
        selected = select_best_planes(matches)
        n_dirs = len(selected)
        log.info(f"  Selected {n_dirs} planes from {len(matches)} matches")

        if n_dirs < 2:
            log.warning(f"[{cam_id}] Only {n_dirs} independent direction(s). Need at least 2. Skipping.")
            continue

        # Solve rotation from selected planes
        log.info(f"  --- Rotation Solve ---")
        R = solve_rotation(selected)

        # Verify rotation quality
        rot_angle = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        log.info(f"  Rotation angle: {rot_angle:.3f}°")

        # Solve translation
        log.info(f"  --- Translation Solve ---")
        if n_dirs >= 3:
            t = solve_translation_full(selected, R)
            solve_type = "full_6dof"
            log.info(f"  Full 6DOF solve ({n_dirs} planes)")
        else:
            t = solve_translation_constrained(selected, R, None)
            solve_type = "constrained_5dof"
            log.info(f"  Constrained 5DOF solve ({n_dirs} planes)")

        log.info(f"  Translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]m")
        log.info(f"  Translation magnitude: {np.linalg.norm(t):.4f}m")

        # Build transform: T maps target (old) camera frame to reference (new) camera frame
        T_ref_tgt = np.eye(4)
        T_ref_tgt[:3, :3] = R
        T_ref_tgt[:3, 3] = t

        # Compute residuals on selected planes
        log.info(f"  --- Residuals (selected) ---")
        residuals = compute_residuals(selected, R, t)
        for label, res in residuals.items():
            log.info(f"  {label}: normal_err={res['normal_angle_err_deg']:.3f}°  "
                     f"rho_err={res['rho_err_m']:.4f}m")

        # Recover old session world pose
        # T_ref_tgt is in Y-up Kinect camera frame (from semantic_plane_fit.py)
        # T_world_ref (extrinsics) is in Y-down + X-negate camera frame (from AprilTag calibration)
        # Convert T_ref_tgt to the extrinsics convention before multiplying.
        # The conversion between Y-up and Y-down+X-negate is C = diag(-1, -1, 1)
        # T_corrected = C @ T_ref_tgt @ C  (C is its own inverse)
        C = np.diag([-1.0, -1.0, 1.0, 1.0])
        T_ref_tgt_corrected = C @ T_ref_tgt @ C

        ext_key = None
        for key in [cam_id, f"cam{cam_id}"]:
            if key in ref_extrinsics:
                ext_key = key
                break

        if ext_key is not None:
            T_world_ref = ref_extrinsics[ext_key]
            T_world_tgt = T_world_ref @ T_ref_tgt_corrected
            log.info(f"  --- Recovered Old Session Pose ---")
            pos = T_world_tgt[:3, 3]
            log.info(f"  World position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

            ref_pos = T_world_ref[:3, 3]
            log.info(f"  (Reference pos: [{ref_pos[0]:.3f}, {ref_pos[1]:.3f}, {ref_pos[2]:.3f}])")
            log.info(f"  Position shift: {np.linalg.norm(pos - ref_pos):.4f}m")
        else:
            T_world_tgt = None
            log.warning(f"[{cam_id}] No extrinsics found for world pose recovery")

        # Store results
        results[cam_id] = {
            "solve_type": solve_type,
            "rotation_angle_deg": float(rot_angle),
            "translation": t.tolist(),
            "translation_magnitude_m": float(np.linalg.norm(t)),
            "T_ref_tgt": T_ref_tgt.tolist(),
            "T_world_tgt": T_world_tgt.tolist() if T_world_tgt is not None else None,
            "num_matches": len(matches),
            "num_selected": n_dirs,
            "residuals": residuals,
            "selected": [
                {
                    "label": m["label"],
                    "ref_normal": m["ref_normal"].tolist(),
                    "tgt_normal": m["tgt_normal"].tolist(),
                    "ref_rho": m["ref_rho"],
                    "tgt_rho": m["tgt_rho"],
                    "dot": m["dot"],
                }
                for m in selected
            ],
        }

    # Save all results
    with open(out_dir / "cross_session_solve.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved results -> {out_dir / 'cross_session_solve.json'}")

    # Save recovered extrinsics for old session
    old_extrinsics = {}
    for cam_id, res in results.items():
        if res["T_world_tgt"] is not None:
            old_extrinsics[f"cam{cam_id}"] = res["T_world_tgt"]

    if old_extrinsics:
        ext_path = out_dir / "recovered_extrinsics.json"
        with open(ext_path, "w") as f:
            json.dump(old_extrinsics, f, indent=2)
        log.info(f"Saved recovered extrinsics -> {ext_path}")

    # Summary
    log.info(f"\n{'='*60}")
    log.info(f"  SUMMARY")
    log.info(f"{'='*60}")
    for cam_id, res in results.items():
        log.info(f"  {cam_id}: {res['solve_type']}  "
                 f"rot={res['rotation_angle_deg']:.2f}°  "
                 f"trans={res['translation_magnitude_m']:.4f}m  "
                 f"({res['num_matches']} matched, {res['num_selected']} selected)")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()