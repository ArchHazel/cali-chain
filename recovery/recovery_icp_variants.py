"""
ICP variant comparison for recovery calibration.

Implements three ICP variants (all identity init):
  1. Point-to-Point ICP
  2. Point-to-Plane ICP
  3. Symmetric (Plane-to-Plane) ICP

Uses same depth data / GT as recovery_solve.py.
Run: python -m recovery.icp_variants max_walls=3
"""
import json
import logging
import time
from pathlib import Path
import yaml

import hydra
import numpy as np
from omegaconf import DictConfig
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config helpers (shared with recovery_solve)
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



def load_gt_relative_poses(env_dir: Path) -> dict:
    gt_path = env_dir / "gt_relative_poses.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        data = json.load(f)

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
# Depth -> point cloud (Y-up, same as recovery_solve)
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame, K):
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


def accumulate_points(depth_dir, K_depth, num_frames, max_depth, subsample):
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
# Utilities
# ---------------------------------------------------------------------------

def voxel_downsample(pts, voxel_size):
    if len(pts) == 0:
        return pts
    keys = np.floor(pts / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx]


def estimate_normals(pts, k=30):
    tree = cKDTree(pts)
    _, idx = tree.query(pts, k=min(k, len(pts)))
    normals = np.zeros_like(pts)
    for i in range(len(pts)):
        neighbors = pts[idx[i]]
        cov = np.cov(neighbors.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normals[i] = eigvecs[:, 0]
    dots = np.sum(normals * pts, axis=1)
    normals[dots > 0] *= -1
    return normals


def rotation_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def _skew(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def _reorthogonalize(R):
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def _final_rmse(src, R, t, tgt_tree):
    src_t = (R @ src.T).T + t
    dists, _ = tgt_tree.query(src_t, k=1)
    return float(np.sqrt(np.mean(dists ** 2)))


# ---------------------------------------------------------------------------
# 1. Point-to-Point ICP
# ---------------------------------------------------------------------------

def icp_point_to_point(src, tgt, max_iter=80, tolerance=1e-6, max_dist=0.3):
    """
    Classic point-to-point ICP using SVD closed-form for each step.
    Minimizes sum ||R @ p_i + t - q_i||^2.
    """
    R_cur = np.eye(3)
    t_cur = np.zeros(3)
    tree = cKDTree(tgt)
    prev_error = float("inf")

    for iteration in range(max_iter):
        src_t = (R_cur @ src.T).T + t_cur
        dists, indices = tree.query(src_t, k=1)

        mask = dists < max_dist
        if mask.sum() < 6:
            break

        p = src_t[mask]
        q = tgt[indices[mask]]

        # Centroids
        p_mean = p.mean(axis=0)
        q_mean = q.mean(axis=0)
        p_c = p - p_mean
        q_c = q - q_mean

        # SVD
        H = p_c.T @ q_c
        U, S, Vt = np.linalg.svd(H)
        R_inc = Vt.T @ U.T
        if np.linalg.det(R_inc) < 0:
            Vt[-1, :] *= -1
            R_inc = Vt.T @ U.T
        t_inc = q_mean - R_inc @ p_mean

        R_cur = R_inc @ R_cur
        t_cur = R_inc @ t_cur + t_inc

        error = float(np.mean(dists[mask]))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    T = np.eye(4)
    T[:3, :3] = R_cur
    T[:3, 3] = t_cur
    rmse = _final_rmse(src, R_cur, t_cur, tree)
    return T, rmse


# ---------------------------------------------------------------------------
# 2. Point-to-Plane ICP
# ---------------------------------------------------------------------------

def icp_point_to_plane(src, tgt, tgt_normals, max_iter=80,
                       tolerance=1e-6, max_dist=0.3):
    """
    Point-to-plane ICP. Minimizes sum (n_i . (R p_i + t - q_i))^2.
    Linearized solve per iteration.
    """
    R_cur = np.eye(3)
    t_cur = np.zeros(3)
    tree = cKDTree(tgt)
    prev_error = float("inf")

    for iteration in range(max_iter):
        src_t = (R_cur @ src.T).T + t_cur
        dists, indices = tree.query(src_t, k=1)

        mask = dists < max_dist
        if mask.sum() < 6:
            break

        p = src_t[mask]
        q = tgt[indices[mask]]
        n = tgt_normals[indices[mask]]

        # Linearized: [p x n | n] [alpha; t_delta] = n . (q - p)
        c = np.cross(p, n)
        A = np.hstack([c, n])
        b = np.sum(n * (q - p), axis=1)

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        alpha = x[:3]
        t_delta = x[3:]

        R_inc = _reorthogonalize(np.eye(3) + _skew(alpha))
        R_cur = R_inc @ R_cur
        t_cur = R_inc @ t_cur + t_delta

        error = float(np.mean(np.abs(b)))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    T = np.eye(4)
    T[:3, :3] = R_cur
    T[:3, 3] = t_cur
    rmse = _final_rmse(src, R_cur, t_cur, tree)
    return T, rmse


# ---------------------------------------------------------------------------
# 3. Symmetric (Plane-to-Plane) ICP
# ---------------------------------------------------------------------------

def icp_symmetric(src, src_normals, tgt, tgt_normals,
                  max_iter=80, tolerance=1e-6, max_dist=0.3):
    """
    Symmetric ICP (Rusinkiewicz 2019).
    Uses averaged normals from both source and target at each correspondence.
    Minimizes sum ((n_p + n_q) . (R p + t - q))^2.
    More robust to local minima than point-to-plane.
    """
    R_cur = np.eye(3)
    t_cur = np.zeros(3)
    tgt_tree = cKDTree(tgt)
    prev_error = float("inf")

    for iteration in range(max_iter):
        src_t = (R_cur @ src.T).T + t_cur
        # Also rotate source normals
        src_n_t = (R_cur @ src_normals.T).T

        dists, indices = tgt_tree.query(src_t, k=1)

        mask = dists < max_dist
        if mask.sum() < 6:
            break

        p = src_t[mask]
        q = tgt[indices[mask]]
        np_ = src_n_t[mask]      # source normals (rotated)
        nq = tgt_normals[indices[mask]]  # target normals

        # Symmetric normal: average of source and target normals
        n_sym = np_ + nq
        norms = np.linalg.norm(n_sym, axis=1, keepdims=True)
        # Skip degenerate pairs where normals cancel
        valid = norms.squeeze() > 0.1
        if valid.sum() < 6:
            break
        p = p[valid]
        q = q[valid]
        n_sym = n_sym[valid]
        n_sym = n_sym / norms[valid]

        # Same linearized system as point-to-plane but with symmetric normal
        c = np.cross(p, n_sym)
        A = np.hstack([c, n_sym])
        b = np.sum(n_sym * (q - p), axis=1)

        x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        alpha = x[:3]
        t_delta = x[3:]

        R_inc = _reorthogonalize(np.eye(3) + _skew(alpha))
        R_cur = R_inc @ R_cur
        t_cur = R_inc @ t_cur + t_delta

        error = float(np.mean(np.abs(b)))
        if abs(prev_error - error) < tolerance:
            break
        prev_error = error

    T = np.eye(4)
    T[:3, :3] = R_cur
    T[:3, 3] = t_cur
    rmse = _final_rmse(src, R_cur, t_cur, tgt_tree)
    return T, rmse


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="recovery_solve", version_base=None)
def main(cfg: DictConfig):
    log.info("ICP Variant Comparison")

    env_dir = Path(cfg.recovery_dir) / cfg.environment
    if not env_dir.exists():
        log.error(f"Environment not found: {env_dir}")
        return

    kinect = load_kinect_config(cfg.camera_id)
    K_depth = kinect["K_depth"]

    gt_poses = load_gt_relative_poses(env_dir)
    log.info(f"GT poses loaded: {len(gt_poses)} segments")

    # Parameters
    icp_max_dist = 0.5   # 50cm max correspondence (bigger convergence basin)
    icp_max_iter = 200   # more iterations to converge
    normal_k = 30        # more neighbors for smoother normals

    # Load reference (same points as plane solve: subsample only, no voxel)
    ref_depth_dir = env_dir / "orig" / "no_tag" / "depth"
    ref_pts = accumulate_points(ref_depth_dir, K_depth,
                                cfg.depth.num_frames, cfg.depth.max_depth,
                                cfg.depth.subsample)
    log.info(f"Reference points: {len(ref_pts)}")

    log.info("Estimating reference normals...")
    ref_normals = estimate_normals(ref_pts, k=normal_k)
    log.info("Done")

    segment_filter = list(cfg.get("segments", [])) or None

    METHODS = ["pt2pt", "pt2pl", "symmetric"]
    results = {}

    for segment_dir in sorted(env_dir.iterdir()):
        if not segment_dir.is_dir():
            continue
        if segment_dir.name in ("orig",):
            continue

        no_tag_dir = segment_dir / "no_tag"
        depth_dir = no_tag_dir / "depth"
        if not (depth_dir / "depth_1.npy").exists():
            continue

        if segment_filter and segment_dir.name not in segment_filter:
            continue

        log.info(f"\n{'='*60}")
        log.info(f"  {segment_dir.name}")
        log.info(f"{'='*60}")

        T_gt = gt_poses.get(segment_dir.name, {}).get("T")
        if T_gt is not None:
            gt_info = gt_poses[segment_dir.name]
            log.info(f"  GT: rot={gt_info['rotation_deg']:.2f}°  "
                     f"trans={gt_info['translation_m']:.4f}m")

        # Load target
        tgt_pts = accumulate_points(depth_dir, K_depth,
                                    cfg.depth.num_frames, cfg.depth.max_depth,
                                    cfg.depth.subsample)
        log.info(f"  Target points: {len(tgt_pts)}")

        # Target normals (needed for pt2pl and symmetric)
        log.info(f"  Estimating target normals...")
        tgt_normals = estimate_normals(tgt_pts, k=normal_k)

        seg_results = {}

        # --- Point-to-Point ---
        log.info(f"  Running Point-to-Point ICP...")
        t0 = time.time()
        T_p2p, rmse_p2p = icp_point_to_point(
            tgt_pts, ref_pts,
            max_iter=icp_max_iter, max_dist=icp_max_dist)
        dt_p2p = time.time() - t0
        seg_results["pt2pt"] = _make_result(T_p2p, rmse_p2p, dt_p2p, T_gt, gt_poses, segment_dir.name)
        _log_result(log, "Pt2Pt", seg_results["pt2pt"])

        # --- Point-to-Plane ---
        log.info(f"  Running Point-to-Plane ICP...")
        t0 = time.time()
        T_p2l, rmse_p2l = icp_point_to_plane(
            tgt_pts, ref_pts, ref_normals,
            max_iter=icp_max_iter, max_dist=icp_max_dist)
        dt_p2l = time.time() - t0
        seg_results["pt2pl"] = _make_result(T_p2l, rmse_p2l, dt_p2l, T_gt, gt_poses, segment_dir.name)
        _log_result(log, "Pt2Pl", seg_results["pt2pl"])

        # --- Symmetric ---
        log.info(f"  Running Symmetric ICP...")
        t0 = time.time()
        T_sym, rmse_sym = icp_symmetric(
            tgt_pts, tgt_normals, ref_pts, ref_normals,
            max_iter=icp_max_iter, max_dist=icp_max_dist)
        dt_sym = time.time() - t0
        seg_results["symmetric"] = _make_result(T_sym, rmse_sym, dt_sym, T_gt, gt_poses, segment_dir.name)
        _log_result(log, "Symm ", seg_results["symmetric"])

        results[segment_dir.name] = seg_results

    # Save
    out_path = env_dir / "icp_variants_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved results -> {out_path}")

    # Summary table
    log.info(f"\n{'='*110}")
    log.info(f"  {'Segment':<28} {'GT_R':>5} | {'Point-to-Point':^18} | {'Point-to-Plane':^18} | {'Symmetric':^18}")
    log.info(f"  {'':28} {'':>5} | {'R_err':>7} {'t_err':>8} | {'R_err':>7} {'t_err':>8} | {'R_err':>7} {'t_err':>8}")
    log.info(f"  {'-'*28} {'-'*5}-+-{'-'*18}-+-{'-'*18}-+-{'-'*18}")

    for name in sorted(results.keys()):
        seg = results[name]
        gt_r = seg.get("pt2pl", {}).get("gt_rotation_deg")
        gt_r_s = f"{gt_r:5.1f}" if gt_r else "  N/A"

        parts = [f"  {name:<28} {gt_r_s}"]
        for method in METHODS:
            r = seg.get(method, {})
            r_err = r.get("rotation_error_deg")
            t_err = r.get("translation_error_m")
            if r_err is not None:
                parts.append(f"{r_err:7.2f} {t_err:8.4f}")
            else:
                parts.append(f"{'N/A':>7} {'N/A':>8}")

        log.info(" | ".join(parts))


def _make_result(T_icp, rmse, dt, T_gt, gt_poses, seg_name):
    R = T_icp[:3, :3]
    t = T_icp[:3, 3]
    result = {
        "rotation_deg": rotation_angle_deg(R),
        "translation_m": float(np.linalg.norm(t)),
        "rmse": rmse,
        "time_s": dt,
    }
    if T_gt is not None:
        R_gt = np.array(T_gt)[:3, :3]
        t_gt = np.array(T_gt)[:3, 3]
        result["gt_rotation_deg"] = float(gt_poses[seg_name]["rotation_deg"])
        result["gt_translation_m"] = float(gt_poses[seg_name]["translation_m"])
        result["rotation_error_deg"] = rotation_angle_deg(R @ R_gt.T)
        result["translation_error_m"] = float(np.linalg.norm(t - t_gt))
    return result


def _log_result(logger, label, result):
    r_est = result["rotation_deg"]
    t_est = result["translation_m"]
    rmse = result["rmse"]
    dt = result["time_s"]
    s = f"    {label}: R={r_est:.2f}°  t={t_est:.4f}m  rmse={rmse:.4f}  time={dt:.1f}s"
    if "rotation_error_deg" in result:
        s += f"  | GT err: R={result['rotation_error_deg']:.2f}°  t={result['translation_error_m']:.4f}m"
    logger.info(s)


if __name__ == "__main__":
    main()