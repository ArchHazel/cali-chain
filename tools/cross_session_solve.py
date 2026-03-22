"""
Cross-Session Pose Solve.

Matches semantically-labeled planes between two sessions for each camera
and solves for the rigid transform (R, t) that maps the old session's
camera frame to the new session's camera frame.

Pipeline:
  1. Load plane parameters (normal + rho) from both sessions
  2. Match planes by semantic label and closest normal direction
  3. Solve rotation via SVD on matched normals
  4. Solve translation via linear system from matched rho values
  5. If only 2 non-parallel planes: zero the unconstrained translation DOF
  6. Combine with new session extrinsics to recover old session world poses

Prerequisites:
    Run semantic_plane_fit.py on both sessions first.

Usage:
    python -m tools.cross_session_solve
    python -m tools.cross_session_solve reference.session=calib_5 target.session=calib_3
"""

import json
import logging
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


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
    # Convert normal lists back to numpy arrays
    for key, plane in data.items():
        plane["normal"] = np.array(plane["normal"])
    return data


def load_extrinsics(extrinsics_dir: str, filename: str) -> dict[str, np.ndarray]:
    path = Path(extrinsics_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


# ---------------------------------------------------------------------------
# Plane matching
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

    Returns list of possible assignments (each is a list of match dicts).
    """
    from itertools import permutations

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
    # For each ref wall, it can match any tgt wall or be unmatched (None)
    # We enumerate permutations of tgt walls assigned to ref walls
    assignments = []
    n_ref = len(ref_keys)
    n_tgt = len(tgt_keys)

    # Pad tgt_keys with None to allow unmatched ref walls
    tgt_options = tgt_keys + [None] * n_ref

    seen = set()
    for perm in permutations(tgt_options, n_ref):
        # Check no duplicate tgt walls (Nones can repeat)
        used_tgt = [t for t in perm if t is not None]
        if len(used_tgt) != len(set(used_tgt)):
            continue

        # Deduplicate
        key = tuple(perm)
        if key in seen:
            continue
        seen.add(key)

        # Build assignment
        assignment = []
        for rk, tk in zip(ref_keys, perm):
            if tk is None:
                continue
            info = pair_info[(rk, tk)]
            if not info["valid"]:
                continue
            assignment.append({
                "label": f"{rk}<->{tk}",
                "ref_normal": info["ref_normal"],
                "tgt_normal": info["tgt_normal"],
                "ref_rho": info["ref_rho"],
                "tgt_rho": info["tgt_rho"],
                "dot": info["dot"],
            })

        assignments.append(assignment)

    return assignments


def score_assignment(matches: list[dict]) -> float:
    """
    Score a set of matched planes by solving R and t, then computing
    total residual (sum of normal angle error + rho error).
    Lower is better.
    """
    if len(matches) < 2:
        return float("inf")

    # Check independent directions
    n_dirs, _ = count_independent_directions(matches)
    if n_dirs < 2:
        return float("inf")

    R = solve_rotation(matches)

    # Select best planes for translation
    selected = select_best_planes(matches)
    n_sel = len(selected)
    if n_sel < 2:
        return float("inf")

    if n_sel >= 3:
        t = solve_translation_full(selected, R)
    else:
        t = solve_translation_constrained(selected, R, None)

    # Compute total residual across ALL matches (not just selected)
    total_residual = 0.0
    for m in matches:
        rotated_n = R @ m["tgt_normal"]
        dot = np.clip(abs(np.dot(rotated_n, m["ref_normal"])), 0, 1)
        angle_err = np.degrees(np.arccos(dot))

        rho_predicted = np.dot(m["ref_normal"], t) + m["tgt_rho"]
        rho_err = abs(rho_predicted - m["ref_rho"])

        total_residual += angle_err + rho_err  # combined score

    return total_residual


def match_planes(ref_planes: dict, tgt_planes: dict,
                 angle_thresh_deg: float = 30.0) -> list[dict]:
    """
    Match planes between reference (new session) and target (old session).

    Strategy:
      - floor<->floor: direct match by label
      - walls: brute-force all valid assignments, solve each, pick lowest total residual
      - ceiling excluded (unreliable detection in most cameras)

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

    # Score each assignment (floor + walls)
    best_score = float("inf")
    best_walls = []

    for wall_matches in assignments:
        candidate = floor_matches + wall_matches
        if len(candidate) < 2:
            continue
        score = score_assignment(candidate)
        if score < best_score:
            best_score = score
            best_walls = wall_matches

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
# Check non-parallelism
# ---------------------------------------------------------------------------

def count_independent_directions(matches: list[dict],
                                 parallel_thresh_deg: float = 15.0) -> tuple[int, list[int]]:
    """
    Count how many independent normal directions exist among matched planes.
    Returns (count, indices of representative matches for each direction).
    """
    cos_thresh = np.cos(np.radians(parallel_thresh_deg))
    directions = []  # list of (representative_normal, [match_indices])

    for i, m in enumerate(matches):
        n = m["ref_normal"]
        found = False
        for d in directions:
            if abs(np.dot(n, d[0])) > cos_thresh:
                d[1].append(i)
                found = True
                break
        if not found:
            directions.append((n, [i]))

    return len(directions), directions


def select_best_planes(matches: list[dict],
                       parallel_thresh_deg: float = 15.0,
                       min_dot: float = 0.95) -> list[dict]:
    """
    Select the best subset of planes for the solve.

    Strategy: pick one plane per independent normal direction, choosing
    the one with the highest normal dot product (best geometric agreement).
    Reject any plane with dot < min_dot (unreliable normal).
    This gives at most 3 planes for 6DOF or 2 for 5DOF.
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
        # Check if this direction is already covered
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
# Pose solve
# ---------------------------------------------------------------------------

def solve_rotation(matches: list[dict]) -> np.ndarray:
    """
    Solve rotation from matched plane normals via SVD.
    R maps target (old) normals to reference (new) normals: R @ tgt_n ≈ ref_n
    """
    # Build correlation matrix H = sum(tgt_n @ ref_n^T)
    H = np.zeros((3, 3))
    for m in matches:
        H += np.outer(m["tgt_normal"], m["ref_normal"])

    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return R


def solve_translation_full(matches: list[dict], R: np.ndarray) -> np.ndarray:
    """
    Solve full 3DOF translation from matched rho values.
    For each match: ref_n · t = ref_rho - tgt_rho (after rotation applied)
    """
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])

    # Least squares solve
    t, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    return t


def solve_translation_constrained(matches: list[dict], R: np.ndarray,
                                   directions: list) -> np.ndarray:
    """
    Solve translation with only 2 independent plane directions.
    The unconstrained direction gets zero translation.
    """
    A = np.array([m["ref_normal"] for m in matches])
    b = np.array([m["ref_rho"] - m["tgt_rho"] for m in matches])

    # Find the unconstrained direction: orthogonal to all available normals
    # With 2 independent directions, the null space of A has dimension 1
    _, S, Vt = np.linalg.svd(A)
    # The last row of Vt corresponds to the smallest singular value
    unconstrained_dir = Vt[-1]

    # Solve in the constrained subspace: project out the unconstrained direction
    # t = A_pinv @ b, but zero out the component along unconstrained_dir
    t_full, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Remove the component along the unconstrained direction
    t = t_full - np.dot(t_full, unconstrained_dir) * unconstrained_dir

    log.info(f"  Unconstrained direction: [{unconstrained_dir[0]:.3f}, "
             f"{unconstrained_dir[1]:.3f}, {unconstrained_dir[2]:.3f}]")
    log.info(f"  Zeroed translation component: {np.dot(t_full, unconstrained_dir):.4f}m")

    return t


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def compute_residuals(matches: list[dict], R: np.ndarray, t: np.ndarray) -> dict:
    """Compute per-match residuals after applying the solved transform."""
    residuals = {}
    for m in matches:
        # Normal residual: angle between R @ tgt_n and ref_n
        rotated_n = R @ m["tgt_normal"]
        dot = np.clip(np.dot(rotated_n, m["ref_normal"]), -1, 1)
        angle_err = np.degrees(np.arccos(abs(dot)))

        # Rho residual: after applying R and t
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

        # Match planes
        log.info(f"  --- Plane Matching ---")
        matches = match_planes(ref_planes, tgt_planes,
                               angle_thresh_deg=cfg.matching.angle_thresh_deg)

        if len(matches) < 2:
            log.warning(f"[{cam_id}] Only {len(matches)} matches, need at least 2. Skipping.")
            continue

        # Select best planes: one per independent direction, lowest Δρ
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