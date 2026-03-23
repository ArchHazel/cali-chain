"""
ICP Baseline Comparison for Cross-Session Pose Recovery.

Compares two ICP baselines against the semantic plane solve:
  1. Direct ICP — all depth points, no filtering
  2. Structural ICP — only wall/floor/ceiling points (from semantic masks)

Both use point-to-plane ICP (SciPy KDTree + NumPy) in the Y-up Kinect
camera frame (same coordinate space as semantic_plane_fit.py) for fair
comparison.

The plane solve results are loaded from cross_session_solve.json for
side-by-side comparison.

Prerequisites:
    - semantic_plane_fit.py run on both sessions (for masks + plane results)
    - cross_session_solve.py run (for plane solve results)
    - scipy (already in semseg env)

Usage:
    python -m tools.icp_comparison
    python -m tools.icp_comparison reference.session=calib_4 target.session=traj_0
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080
STRUCTURAL_CLASSES = ["wall", "floor", "ceiling"]


# ---------------------------------------------------------------------------
# Loading (reused from semantic_plane_fit.py)
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


# ---------------------------------------------------------------------------
# DLT (for semantic label lookup)
# ---------------------------------------------------------------------------

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


def project_to_color_pixels(pts_3d: np.ndarray, P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts_h = np.hstack([pts_3d, np.ones((len(pts_3d), 1))])
    proj = (P @ pts_h.T).T
    w = proj[:, 2]
    px = (proj[:, 0] / w).astype(np.int32)
    py = (proj[:, 1] / w).astype(np.int32)
    px = COLOR_W - 1 - px  # flip to match pre-flipped RGB
    valid = (w > 0) & (px >= 0) & (px < COLOR_W) & (py >= 0) & (py < COLOR_H)
    return np.stack([px, py], axis=1), valid


# ---------------------------------------------------------------------------
# Depth processing (Y-up, matching semantic_plane_fit.py)
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project to 3D in Y-up Kinect camera space (for DLT compatibility)."""
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


# ---------------------------------------------------------------------------
# Point cloud accumulation
# ---------------------------------------------------------------------------

def accumulate_points(
    depth_dir: str,
    cam_id: str,
    K_depth: np.ndarray,
    chunk: int,
    frame_idx: int,
    num_frames: int,
    max_depth: float,
    subsample: int,
) -> np.ndarray:
    """Accumulate depth points in Y-up camera frame. Returns Nx3."""
    frames = load_depth_frames(depth_dir, cam_id, chunk, frame_idx, num_frames)
    if not frames:
        return np.zeros((0, 3))

    all_pts = []
    for depth_frame in frames:
        pts, depths = backproject_depth(depth_frame, K_depth)
        mask = depths < max_depth
        pts = pts[mask]
        if subsample > 1:
            pts = pts[::subsample]
        all_pts.append(pts)

    if not all_pts:
        return np.zeros((0, 3))
    return np.concatenate(all_pts, axis=0)


def accumulate_points_with_labels(
    depth_dir: str,
    cam_id: str,
    K_depth: np.ndarray,
    P_dlt: np.ndarray,
    masks: dict[str, np.ndarray],
    chunk: int,
    frame_idx: int,
    num_frames: int,
    max_depth: float,
    subsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Accumulate depth points with semantic labels. Returns (Nx3 pts, N labels)."""
    frames = load_depth_frames(depth_dir, cam_id, chunk, frame_idx, num_frames)
    if not frames:
        return np.zeros((0, 3)), np.array([])

    all_pts = []
    all_labels = []

    for depth_frame in frames:
        pts, depths = backproject_depth(depth_frame, K_depth)
        mask = depths < max_depth
        pts = pts[mask]

        if len(pts) == 0:
            continue

        # Project to color pixels for semantic lookup
        pixels, valid = project_to_color_pixels(pts, P_dlt)
        pts = pts[valid]
        pixels = pixels[valid]

        # Label each point
        labels = np.full(len(pts), "other", dtype=object)
        for class_name, sem_mask in masks.items():
            px, py = pixels[:, 0], pixels[:, 1]
            on_mask = sem_mask[py, px].astype(bool)
            labels[on_mask] = class_name

        if subsample > 1:
            pts = pts[::subsample]
            labels = labels[::subsample]

        all_pts.append(pts)
        all_labels.append(labels)

    if not all_pts:
        return np.zeros((0, 3)), np.array([])
    return np.concatenate(all_pts, axis=0), np.concatenate(all_labels, axis=0)


# ---------------------------------------------------------------------------
# ICP
# ---------------------------------------------------------------------------

def estimate_normals(pts: np.ndarray, k: int = 30) -> np.ndarray:
    """Estimate surface normals via PCA on k nearest neighbors."""
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbors = pts[idx[i]]
        centered = neighbors - neighbors.mean(axis=0)
        _, _, Vt = np.linalg.svd(centered, full_matrices=False)
        normals[i] = Vt[2]  # smallest singular value = normal direction
    return normals


def run_icp(source_pts: np.ndarray, target_pts: np.ndarray,
            max_correspondence_distance: float = 0.1,
            init_transform: np.ndarray | None = None,
            max_iterations: int = 200,
            tolerance: float = 1e-8) -> dict:
    """
    Point-to-plane ICP using SciPy KDTree + NumPy.

    source = target session (old), target = reference session (new).
    Returns dict with T (4x4), fitness, inlier_rmse.
    """
    from scipy.spatial import cKDTree

    src = source_pts.astype(np.float64).copy()
    tgt = target_pts.astype(np.float64)

    # Estimate normals on target (reference) cloud
    log.info(f"      Estimating normals on {len(tgt)} target points...")
    tgt_normals = estimate_normals(tgt, k=30)

    # Build KD-tree on target
    tree = cKDTree(tgt)

    if init_transform is None:
        init_transform = np.eye(4)

    T_accum = init_transform.copy()
    R_accum = T_accum[:3, :3].copy()
    t_accum = T_accum[:3, 3].copy()

    # Apply initial transform to source
    src_t = (R_accum @ src.T).T + t_accum

    prev_rmse = np.inf

    for iteration in range(max_iterations):
        # Find closest points in target
        dists, indices = tree.query(src_t)

        # Filter by max correspondence distance
        inlier_mask = dists < max_correspondence_distance
        n_inliers = inlier_mask.sum()
        if n_inliers < 6:
            log.warning(f"      ICP: only {n_inliers} inliers at iter {iteration}")
            break

        p = src_t[inlier_mask]           # source points (transformed)
        q = tgt[indices[inlier_mask]]     # closest target points
        n = tgt_normals[indices[inlier_mask]]  # target normals

        # Point-to-plane linearized solve
        # Minimize sum of (n_i . (R*p_i + t - q_i))^2
        # Linearize R ≈ I + [α]_× for small rotation
        # Each row: [n×p, n] . [α; t] = n.(q - p)
        cross = np.cross(p, n)  # Nx3
        A = np.hstack([cross, n])  # Nx6
        b = np.sum(n * (q - p), axis=1)  # N

        # Solve least squares
        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        # Extract incremental rotation and translation
        alpha = x[:3]
        dt = x[3:]

        # Small-angle rotation matrix
        dR = np.eye(3) + np.array([
            [0, -alpha[2], alpha[1]],
            [alpha[2], 0, -alpha[0]],
            [-alpha[1], alpha[0], 0],
        ])
        # Re-orthogonalize via SVD
        U, _, Vt = np.linalg.svd(dR)
        dR = U @ Vt

        # Update accumulated transform
        R_accum = dR @ R_accum
        t_accum = dR @ t_accum + dt

        # Apply to source
        src_t = (R_accum @ src.T).T + t_accum

        # Check convergence
        rmse = np.sqrt(np.mean(dists[inlier_mask]**2))
        if abs(prev_rmse - rmse) < tolerance:
            log.info(f"      ICP converged at iter {iteration} (RMSE={rmse:.6f}m)")
            break
        prev_rmse = rmse

    # Final metrics
    dists_final, _ = tree.query(src_t)
    inlier_final = dists_final < max_correspondence_distance
    fitness = inlier_final.sum() / len(src_t)
    inlier_rmse = np.sqrt(np.mean(dists_final[inlier_final]**2)) if inlier_final.any() else np.inf

    T = np.eye(4)
    T[:3, :3] = R_accum
    T[:3, 3] = t_accum

    return {
        "T": T,
        "fitness": float(fitness),
        "inlier_rmse": float(inlier_rmse),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def rotation_angle_deg(R: np.ndarray) -> float:
    """Angle of rotation matrix in degrees."""
    trace = np.clip((np.trace(R) - 1) / 2, -1, 1)
    return float(np.degrees(np.arccos(trace)))


def translation_magnitude(t: np.ndarray) -> float:
    return float(np.linalg.norm(t))


def evaluate_transform(source_pts: np.ndarray, target_pts: np.ndarray,
                       T: np.ndarray, max_corr_dist: float = 0.1) -> dict:
    """
    Evaluate a transform by applying it to source points and measuring
    nearest-neighbor fitness and RMSE against target points.

    Same metric used by ICP, so results are directly comparable.
    """
    from scipy.spatial import cKDTree

    # Apply transform to source
    R, t = T[:3, :3], T[:3, 3]
    src_transformed = (R @ source_pts.T).T + t

    # Find nearest neighbors in target
    tree = cKDTree(target_pts)
    dists, _ = tree.query(src_transformed)

    inlier_mask = dists < max_corr_dist
    fitness = float(inlier_mask.sum() / len(src_transformed))
    if inlier_mask.any():
        inlier_rmse = float(np.sqrt(np.mean(dists[inlier_mask]**2)))
    else:
        inlier_rmse = float('inf')

    return {
        "fitness": fitness,
        "inlier_rmse": inlier_rmse,
    }


def compare_transforms(T_est: np.ndarray, T_gt: np.ndarray) -> dict:
    """
    Compare estimated transform against ground truth.
    Returns rotation error (degrees) and translation error (meters).
    """
    R_est, t_est = T_est[:3, :3], T_est[:3, 3]
    R_gt, t_gt = T_gt[:3, :3], T_gt[:3, 3]

    # Rotation error: angle of R_est @ R_gt^T
    R_err = R_est @ R_gt.T
    rot_err = rotation_angle_deg(R_err)

    # Translation error: Euclidean distance
    t_err = float(np.linalg.norm(t_est - t_gt))

    return {
        "rotation_error_deg": rot_err,
        "translation_error_m": t_err,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="icp_comparison", version_base=None)
def main(cfg: DictConfig):
    log.info(f"ICP Comparison\n{OmegaConf.to_yaml(cfg)}")

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load reference extrinsics (AprilTag-based)
    ref_extrinsics = load_extrinsics(cfg.reference.extrinsics_dir,
                                     cfg.reference.extrinsics_filename)

    # Load plane solve results for comparison
    plane_solve_path = Path(cfg.plane_solve.results_dir) / "cross_session_solve.json"
    plane_solve_results = {}
    if plane_solve_path.exists():
        with open(plane_solve_path) as f:
            plane_solve_results = json.load(f)
        log.info(f"Loaded plane solve results from {plane_solve_path}")
    else:
        log.warning(f"No plane solve results at {plane_solve_path}")

    # Convention conversion matrix (Y-up ↔ Y-down + X-negate)
    C = np.diag([-1.0, -1.0, 1.0, 1.0])

    cameras = list(cfg.cameras)
    results = {}

    for cam_id in cameras:
        log.info(f"\n{'='*60}")
        log.info(f"  {cam_id}")
        log.info(f"{'='*60}")

        # --- Load kinect config ---
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue

        K_depth = kinect["K_depth"]

        # --- Accumulate reference (new session) points ---
        ref_pts = accumulate_points(
            cfg.reference.depth_dir, cam_id, K_depth,
            cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
            cfg.depth.max_depth, cfg.depth.subsample,
        )
        if len(ref_pts) == 0:
            log.warning(f"[{cam_id}] No reference points, skipping")
            continue

        # --- Accumulate target (old session) points ---
        tgt_pts = accumulate_points(
            cfg.target.depth_dir, cam_id, K_depth,
            cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
            cfg.depth.max_depth, cfg.depth.subsample,
        )
        if len(tgt_pts) == 0:
            log.warning(f"[{cam_id}] No target points, skipping")
            continue

        log.info(f"  Reference points: {len(ref_pts)}")
        log.info(f"  Target points:    {len(tgt_pts)}")

        cam_results = {}

        # =====================================================================
        # Method 1: Direct ICP (all points)
        # =====================================================================
        log.info(f"  --- Method 1: Direct ICP ---")
        try:
            icp_all = run_icp(tgt_pts, ref_pts,
                              max_correspondence_distance=cfg.icp.max_corr_dist)
            T_icp_all = icp_all["T"]
            rot_all = rotation_angle_deg(T_icp_all[:3, :3])
            trans_all = translation_magnitude(T_icp_all[:3, 3])
            log.info(f"    Fitness: {icp_all['fitness']:.4f}")
            log.info(f"    Inlier RMSE: {icp_all['inlier_rmse']:.4f}m")
            log.info(f"    Rotation: {rot_all:.3f}°")
            log.info(f"    Translation: {trans_all:.4f}m")

            # Recover world pose
            T_corrected = C @ T_icp_all @ C
            ext_key = None
            for key in [cam_id, f"cam{cam_id}"]:
                if key in ref_extrinsics:
                    ext_key = key
                    break

            T_world_tgt = None
            if ext_key:
                T_world_tgt = ref_extrinsics[ext_key] @ T_corrected
                pos = T_world_tgt[:3, 3]
                ref_pos = ref_extrinsics[ext_key][:3, 3]
                log.info(f"    World pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                log.info(f"    Shift from ref: {np.linalg.norm(pos - ref_pos):.4f}m")

            cam_results["direct_icp"] = {
                "T_ref_tgt": T_icp_all.tolist(),
                "T_world_tgt": T_world_tgt.tolist() if T_world_tgt is not None else None,
                "rotation_deg": rot_all,
                "translation_m": trans_all,
                "fitness": icp_all["fitness"],
                "inlier_rmse": icp_all["inlier_rmse"],
                "num_source_pts": len(tgt_pts),
                "num_target_pts": len(ref_pts),
            }
        except Exception as e:
            log.error(f"    Direct ICP failed: {e}")
            cam_results["direct_icp"] = {"error": str(e)}

        # =====================================================================
        # Method 2: Structural ICP (wall/floor/ceiling only)
        # =====================================================================
        log.info(f"  --- Method 2: Structural ICP ---")

        # Need DLT + semantic masks for filtering
        corr_path = kinect.get("corr_path")
        if corr_path is None:
            corr_path = f"data/{cfg.reference.session}/videos/{cam_id}/depth3d_to_color2d_correspondences.npz"
        if not Path(corr_path).exists():
            log.warning(f"    No DLT correspondences, skipping structural ICP")
            cam_results["structural_icp"] = {"error": "no DLT correspondences"}
        else:
            try:
                corr = np.load(corr_path)
                P_dlt = fit_dlt(corr["points_3d"], corr["points_2d"])

                # Load semantic masks for both sessions
                ref_masks = {}
                for cls in STRUCTURAL_CLASSES:
                    mask_path = Path(cfg.reference.semantic_dir) / f"{cam_id}_{cls}_mask.npy"
                    if mask_path.exists():
                        ref_masks[cls] = np.load(mask_path)

                tgt_masks = {}
                for cls in STRUCTURAL_CLASSES:
                    mask_path = Path(cfg.target.semantic_dir) / f"{cam_id}_{cls}_mask.npy"
                    if mask_path.exists():
                        tgt_masks[cls] = np.load(mask_path)

                if not ref_masks or not tgt_masks:
                    log.warning(f"    Missing semantic masks (ref: {len(ref_masks)}, tgt: {len(tgt_masks)})")
                    cam_results["structural_icp"] = {"error": "missing semantic masks"}
                else:
                    # Accumulate with labels
                    ref_pts_l, ref_labels = accumulate_points_with_labels(
                        cfg.reference.depth_dir, cam_id, K_depth, P_dlt, ref_masks,
                        cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
                        cfg.depth.max_depth, cfg.depth.subsample,
                    )
                    tgt_pts_l, tgt_labels = accumulate_points_with_labels(
                        cfg.target.depth_dir, cam_id, K_depth, P_dlt, tgt_masks,
                        cfg.depth.chunk, cfg.depth.frame_idx, cfg.depth.num_frames,
                        cfg.depth.max_depth, cfg.depth.subsample,
                    )

                    # Filter to structural only
                    ref_struct = ref_pts_l[np.isin(ref_labels, STRUCTURAL_CLASSES)]
                    tgt_struct = tgt_pts_l[np.isin(tgt_labels, STRUCTURAL_CLASSES)]

                    log.info(f"    Ref structural: {len(ref_struct)} / {len(ref_pts_l)} total")
                    log.info(f"    Tgt structural: {len(tgt_struct)} / {len(tgt_pts_l)} total")

                    if len(ref_struct) < 500 or len(tgt_struct) < 500:
                        log.warning(f"    Too few structural points")
                        cam_results["structural_icp"] = {"error": "too few structural points"}
                    else:
                        icp_struct = run_icp(tgt_struct, ref_struct,
                                             max_correspondence_distance=cfg.icp.max_corr_dist)
                        T_icp_struct = icp_struct["T"]
                        rot_struct = rotation_angle_deg(T_icp_struct[:3, :3])
                        trans_struct = translation_magnitude(T_icp_struct[:3, 3])
                        log.info(f"    Fitness: {icp_struct['fitness']:.4f}")
                        log.info(f"    Inlier RMSE: {icp_struct['inlier_rmse']:.4f}m")
                        log.info(f"    Rotation: {rot_struct:.3f}°")
                        log.info(f"    Translation: {trans_struct:.4f}m")

                        # Recover world pose
                        T_corrected = C @ T_icp_struct @ C
                        T_world_tgt = None
                        if ext_key:
                            T_world_tgt = ref_extrinsics[ext_key] @ T_corrected
                            pos = T_world_tgt[:3, 3]
                            ref_pos = ref_extrinsics[ext_key][:3, 3]
                            log.info(f"    World pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
                            log.info(f"    Shift from ref: {np.linalg.norm(pos - ref_pos):.4f}m")

                        cam_results["structural_icp"] = {
                            "T_ref_tgt": T_icp_struct.tolist(),
                            "T_world_tgt": T_world_tgt.tolist() if T_world_tgt is not None else None,
                            "rotation_deg": rot_struct,
                            "translation_m": trans_struct,
                            "fitness": icp_struct["fitness"],
                            "inlier_rmse": icp_struct["inlier_rmse"],
                            "num_source_pts": len(tgt_struct),
                            "num_target_pts": len(ref_struct),
                        }

            except Exception as e:
                log.error(f"    Structural ICP failed: {e}")
                cam_results["structural_icp"] = {"error": str(e)}

        # =====================================================================
        # Method 3: Plane solve (loaded from existing results)
        # =====================================================================
        log.info(f"  --- Method 3: Plane Solve ---")
        if cam_id in plane_solve_results:
            ps = plane_solve_results[cam_id]
            T_plane = np.array(ps["T_ref_tgt"])
            rot_plane = ps["rotation_angle_deg"]
            trans_plane = ps["translation_magnitude_m"]
            log.info(f"    Rotation: {rot_plane:.3f}°")
            log.info(f"    Translation: {trans_plane:.4f}m")
            log.info(f"    Solve type: {ps['solve_type']} ({ps['num_selected']} planes)")

            # Evaluate on point cloud (same metric as ICP)
            log.info(f"    Evaluating on point cloud...")
            pc_eval = evaluate_transform(tgt_pts, ref_pts, T_plane,
                                         max_corr_dist=cfg.icp.max_corr_dist)
            log.info(f"    Fitness: {pc_eval['fitness']:.4f}")
            log.info(f"    Inlier RMSE: {pc_eval['inlier_rmse']:.4f}m")

            cam_results["plane_solve"] = {
                "solve_type": ps["solve_type"],
                "rotation_deg": rot_plane,
                "translation_m": trans_plane,
                "num_selected": ps["num_selected"],
                "residuals": ps["residuals"],
                "fitness": pc_eval["fitness"],
                "inlier_rmse": pc_eval["inlier_rmse"],
            }
        else:
            log.info(f"    Not available")
            cam_results["plane_solve"] = {"error": "not available"}

        results[cam_id] = cam_results

    # =========================================================================
    # Save results
    # =========================================================================
    with open(out_dir / "icp_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved results -> {out_dir / 'icp_comparison.json'}")

    # Save recovered extrinsics for each ICP method
    for method in ["direct_icp", "structural_icp"]:
        ext = {}
        for cam_id, cam_res in results.items():
            if method in cam_res and "T_world_tgt" in cam_res[method]:
                T = cam_res[method].get("T_world_tgt")
                if T is not None:
                    ext[f"cam{cam_id}"] = T
        if ext:
            ext_path = out_dir / f"recovered_extrinsics_{method}.json"
            with open(ext_path, "w") as f:
                json.dump(ext, f, indent=2)
            log.info(f"Saved {method} extrinsics -> {ext_path}")

    # =========================================================================
    # Summary table
    # =========================================================================
    log.info(f"\n{'='*80}")
    log.info(f"  COMPARISON SUMMARY")
    log.info(f"{'='*80}")
    log.info(f"  {'Camera':<8} {'Method':<18} {'Rot(°)':<10} {'Trans(m)':<10} {'Fitness':<10} {'RMSE(m)':<10}")
    log.info(f"  {'-'*8} {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for cam_id in cameras:
        if cam_id not in results:
            continue
        cam_res = results[cam_id]
        for method_name, display_name in [
            ("direct_icp", "Direct ICP"),
            ("structural_icp", "Structural ICP"),
            ("plane_solve", "Plane Solve"),
        ]:
            if method_name not in cam_res or "error" in cam_res[method_name]:
                err = cam_res.get(method_name, {}).get("error", "skipped")
                log.info(f"  {cam_id:<8} {display_name:<18} {'—':>10} {'—':>10} {'—':>10} {err}")
                continue

            r = cam_res[method_name]
            rot = f"{r['rotation_deg']:.2f}"
            trans = f"{r['translation_m']:.4f}"
            fit = f"{r.get('fitness', 'n/a')}"
            if isinstance(fit, float):
                fit = f"{fit:.4f}"
            rmse = r.get("inlier_rmse", "n/a")
            if isinstance(rmse, float):
                rmse = f"{rmse:.4f}"
            log.info(f"  {cam_id:<8} {display_name:<18} {rot:>10} {trans:>10} {fit:>10} {rmse:>10}")
        log.info(f"  {'-'*8} {'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()