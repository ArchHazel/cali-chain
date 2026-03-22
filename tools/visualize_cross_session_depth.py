"""
Cross-session depth visualization with RGB colors.

For each session, back-projects depth points into world frame and colors
them using the actual RGB image (via DLT depth-to-color projection).
Renders separate bird's-eye, front, and side views for ref and target.

Usage:
    python -m tools.visualize_cross_session_depth_rgb
    python -m tools.visualize_cross_session_depth_rgb reference.session=calib_5 target.session=calib_3
"""

import json
import logging
from pathlib import Path

import cv2
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    return {
        "K_depth": K,
        "corr_path": cfg.get("depth_to_color_correspondences", None),
    }


def load_extrinsics(extrinsics_dir: str, filename: str) -> dict[str, np.ndarray]:
    ext_path = Path(extrinsics_dir) / filename
    if not ext_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {ext_path}")
    with open(ext_path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


def load_depth_frames(depth_dir: str, cam_id: str, chunk: int,
                      start_frame: int, num_frames: int) -> list[np.ndarray]:
    depth_path = Path(depth_dir) / cam_id / "depth" / f"depth_{chunk}.npy"
    if not depth_path.exists():
        log.warning(f"Depth file not found: {depth_path}")
        return []
    depth_chunk = np.load(depth_path)
    end_frame = min(start_frame + num_frames, depth_chunk.shape[0])
    return [depth_chunk[i] for i in range(start_frame, end_frame)]


def load_rgb_frame(frames_dir: str, cam_id: str, frame_name: str = "000000.jpg") -> np.ndarray | None:
    """Load an RGB frame (already pre-flipped during preprocessing)."""
    path = Path(frames_dir) / cam_id / frame_name
    if not path.exists():
        log.warning(f"RGB frame not found: {path}")
        return None
    img = cv2.imread(str(path))
    if img is None:
        log.warning(f"Could not read RGB frame: {path}")
        return None
    if img.shape[:2] != (COLOR_H, COLOR_W):
        img = cv2.resize(img, (COLOR_W, COLOR_H))
    return img


# ---------------------------------------------------------------------------
# Hardcoded HAR camera positions (for target session extrinsics)
# A = position, B = look-at point
# ---------------------------------------------------------------------------

HAR_CAMERAS = {
    "HAR1": {"A": [5.03, 8.45, 0.90], "B": [5.73, 8.18, 0.90]},
    "HAR2": {"A": [0.73, 5.71, 0.90], "B": [1.47, 5.80, 0.90]},
    "HAR3": {"A": [1.42, 7.75, 0.90], "B": [2.18, 7.79, 0.90]},
    "HAR4": {"A": [3.72, 0.32, 0.90], "B": [4.27, 0.86, 0.90]},
    "HAR6": {"A": [4.23, 4.36, 0.90], "B": [3.50, 4.03, 0.90]},
    "HAR8": {"A": [5.00, 4.00, 0.90], "B": [5.28, 3.52, 0.90]},
}


def build_extrinsic_from_lookat(position: list, lookat: list) -> np.ndarray:
    """
    Build a 4x4 T_world_cam matrix from camera position and look-at point.

    OpenCV camera convention: +Z = forward (into scene), +X = right, +Y = down.
    World convention: Z = up.
    """
    pos = np.array(position, dtype=np.float64)
    target = np.array(lookat, dtype=np.float64)

    # Camera forward = +Z axis in camera frame
    forward = target - pos
    forward = forward / np.linalg.norm(forward)

    # World up
    world_up = np.array([0.0, 0.0, 1.0])

    # Camera right = +X axis
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        # forward is parallel to world_up, pick arbitrary right
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)

    # Camera down = +Y axis (OpenCV convention: Y points down)
    down = np.cross(forward, right)
    down = down / np.linalg.norm(down)

    # T_world_cam: columns are camera axes expressed in world frame
    # col0 = right (cam X in world), col1 = down (cam Y in world), col2 = forward (cam Z in world)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = down
    T[:3, 2] = forward
    T[:3, 3] = pos

    return T


def build_har_extrinsics() -> dict[str, np.ndarray]:
    """Build T_world_cam for all HAR cameras from position + look-at."""
    extrinsics = {}
    for cam_id, params in HAR_CAMERAS.items():
        extrinsics[cam_id] = build_extrinsic_from_lookat(params["A"], params["B"])
        log.info(f"[{cam_id}] Built extrinsic from look-at: pos={params['A']}, lookat={params['B']}")
    return extrinsics


# ---------------------------------------------------------------------------
# DLT
# ---------------------------------------------------------------------------

def fit_dlt(points_3d, points_2d):
    n = len(points_3d)
    A = np.zeros((2 * n, 12))
    for i in range(n):
        X, Y, Z = points_3d[i]
        u, v = points_2d[i]
        A[2*i]   = [X, Y, Z, 1, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u]
        A[2*i+1] = [0, 0, 0, 0, X, Y, Z, 1, -v*X, -v*Y, -v*Z, -v]
    _, _, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape(3, 4)
    return P


def load_dlt_matrix(cam_id: str, corr_path: str | None, session: str) -> np.ndarray | None:
    """Load correspondences and fit DLT for a camera."""
    if corr_path is not None and Path(corr_path).exists():
        npz_path = corr_path
    else:
        # Fallback to session data
        npz_path = f"data/{session}/videos/{cam_id}/depth3d_to_color2d_correspondences.npz"
    if not Path(npz_path).exists():
        log.warning(f"[{cam_id}] No correspondences found at {npz_path}")
        return None
    corr = np.load(npz_path)
    P = fit_dlt(corr['points_3d'], corr['points_2d'])
    return P


# ---------------------------------------------------------------------------
# Depth processing with RGB color sampling
# ---------------------------------------------------------------------------

def backproject_depth_yup(depth_frame: np.ndarray, K: np.ndarray):
    """
    Back-project depth to 3D in depth camera space with Y-up convention
    (matching the Kinect SDK CameraSpacePoint convention used for DLT fitting).
    Returns all pixels (including invalid) so indices stay aligned.
    """
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H - 1, -1, -1)[:, np.newaxis]  # Y-up
    cam[:, :, 2] = 1.0

    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]

    return cam_flat, depth_m


def backproject_depth_ydown(depth_frame: np.ndarray, K: np.ndarray):
    """
    Back-project depth to 3D in depth camera space with Y-down convention
    (standard image coords, used for world-frame transform with extrinsics).
    """
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H)[:, np.newaxis]  # Y-down
    cam[:, :, 2] = 1.0

    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]

    return cam_flat, depth_m


def sample_rgb_via_dlt(cam_space_yup: np.ndarray, depth_m: np.ndarray,
                       P: np.ndarray, rgb_image: np.ndarray) -> np.ndarray:
    """
    Project depth-camera-space points (Y-up convention) through DLT into
    the color image and sample RGB values.

    Returns Nx3 float32 array of RGB colors in [0, 1] for ALL input points.
    Points that don't project validly get color (0, 0, 0).
    """
    n = len(cam_space_yup)
    colors = np.zeros((n, 3), dtype=np.float32)

    valid = depth_m > 0
    pts = cam_space_yup[valid]
    if len(pts) == 0:
        return colors

    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    proj = (P @ pts_h.T).T

    w = proj[:, 2]
    ok = w > 0
    px = np.full(len(pts), -1, dtype=np.int32)
    py = np.full(len(pts), -1, dtype=np.int32)
    px[ok] = (proj[ok, 0] / w[ok]).astype(np.int32)
    py[ok] = (proj[ok, 1] / w[ok]).astype(np.int32)

    # Flip to match pre-flipped RGB frames
    px[ok] = COLOR_W - 1 - px[ok]

    in_bounds = ok & (px >= 0) & (px < COLOR_W) & (py >= 0) & (py < COLOR_H)

    # Sample colors from RGB image (BGR -> RGB, normalized)
    sampled = rgb_image[py[in_bounds], px[in_bounds]]  # Nx3 uint8 BGR
    sampled_rgb = sampled[:, ::-1].astype(np.float32) / 255.0  # -> RGB [0,1]

    # Place back into full array
    valid_indices = np.where(valid)[0]
    colors[valid_indices[in_bounds]] = sampled_rgb

    return colors


def to_world(pts_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return (R @ pts_cam.T).T + t


def accumulate_world_points_rgb(
    depth_frames: list[np.ndarray],
    K_depth: np.ndarray,
    T_world_cam: np.ndarray,
    P_dlt: np.ndarray | None,
    rgb_image: np.ndarray | None,
    max_depth: float = 6.0,
    subsample: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Back-project depth frames to world frame and sample RGB colors.

    Returns:
        pts_world: Nx3 world points
        colors: Nx3 RGB colors in [0, 1]
    """
    all_pts = []
    all_colors = []

    for depth_frame in depth_frames:
        # Y-down for world-frame transform (matching extrinsics convention)
        cam_ydown, depth_m = backproject_depth_ydown(depth_frame, K_depth)

        # Y-up for DLT color sampling
        if P_dlt is not None and rgb_image is not None:
            cam_yup, _ = backproject_depth_yup(depth_frame, K_depth)
            colors = sample_rgb_via_dlt(cam_yup, depth_m, P_dlt, rgb_image)
        else:
            colors = np.full((len(cam_ydown), 3), 0.5, dtype=np.float32)  # gray fallback

        # Filter valid depth
        valid = depth_m > 0
        pts = cam_ydown[valid]
        colors = colors[valid]
        depths = depth_m[valid]

        # Filter by max depth
        mask = depths < max_depth
        pts = pts[mask]
        colors = colors[mask]

        # Negate X to match horizontally flipped RGB frames
        pts[:, 0] = -pts[:, 0]

        # Subsample
        if subsample > 1:
            pts = pts[::subsample]
            colors = colors[::subsample]

        # Transform to world
        pts_world = to_world(pts, T_world_cam)
        all_pts.append(pts_world)
        all_colors.append(colors)

    if not all_pts:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.concatenate(all_pts, axis=0), np.concatenate(all_colors, axis=0)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def compute_global_limits(clouds_a: dict, clouds_b: dict, margin: float = 0.5) -> dict:
    all_pts = []
    for d in [clouds_a, clouds_b]:
        for pts, _ in d.values():
            if len(pts) > 0:
                all_pts.append(pts)
    if not all_pts:
        return {"xlim": (-5, 5), "ylim": (-5, 5), "zlim": (-5, 5)}
    all_pts = np.concatenate(all_pts, axis=0)
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)
    return {"xlim": xlim, "ylim": ylim, "zlim": zlim}


def render_rgb_view(ax, pts: np.ndarray, colors: np.ndarray,
                    axis0: int, axis1: int, xlim, ylim, resolution: int = 800):
    """
    Render a 2D view of the point cloud using actual RGB colors.
    Bins points into a grid and averages colors per bin.
    """
    if len(pts) == 0:
        img = np.zeros((resolution, resolution, 3), dtype=np.float32)
        ax.imshow(img, extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
                  aspect='equal', origin='lower', zorder=0)
        return

    x = pts[:, axis0]
    y = pts[:, axis1]

    # Bin indices
    bx = ((x - xlim[0]) / (xlim[1] - xlim[0]) * resolution).astype(np.int32)
    by = ((y - ylim[0]) / (ylim[1] - ylim[0]) * resolution).astype(np.int32)
    bx = np.clip(bx, 0, resolution - 1)
    by = np.clip(by, 0, resolution - 1)

    # Accumulate color and count per bin
    color_sum = np.zeros((resolution, resolution, 3), dtype=np.float64)
    count = np.zeros((resolution, resolution), dtype=np.int32)

    np.add.at(color_sum, (by, bx), colors)
    np.add.at(count, (by, bx), 1)

    # Average
    mask = count > 0
    img = np.zeros((resolution, resolution, 3), dtype=np.float32)
    img[mask] = (color_sum[mask] / count[mask, np.newaxis]).astype(np.float32)

    ax.imshow(np.clip(img, 0, 1),
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              aspect='equal', origin='lower', zorder=0)


def plot_session_views(pts: np.ndarray, colors: np.ndarray,
                       session_name: str,
                       extrinsics: dict[str, np.ndarray],
                       limits: dict,
                       out_path: Path,
                       resolution: int = 800,
                       draw_cameras: bool = True):
    """Render bird's-eye, front, and side views for one session with RGB colors."""
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    fig.patch.set_facecolor('black')

    view_configs = [
        ("Bird's Eye (XY)", 0, 1, "xlim", "ylim", "X (m)", "Y (m)"),
        ("Front View (XZ)",  0, 2, "xlim", "zlim", "X (m)", "Z (m)"),
        ("Side View (YZ)",   1, 2, "ylim", "zlim", "Y (m)", "Z (m)"),
    ]

    for ax, (title, a0, a1, xkey, ykey, xlabel, ylabel) in zip(axes, view_configs):
        ax.set_facecolor('black')
        render_rgb_view(ax, pts, colors, a0, a1, limits[xkey], limits[ykey], resolution)

        if draw_cameras:
            for cam_id, T in extrinsics.items():
                pos = T[:3, 3]
                coords = [pos[a0], pos[a1]]
                ax.plot(coords[0], coords[1], 'w^', markersize=8,
                        markeredgecolor='black', markeredgewidth=0.5, zorder=5)
                ax.annotate(cam_id, (coords[0], coords[1]), fontsize=7,
                            ha='center', va='bottom', color='white',
                            fontweight='bold', zorder=5)

        ax.set_xlim(limits[xkey])
        ax.set_ylim(limits[ykey])
        ax.set_xlabel(xlabel, color='white')
        ax.set_ylabel(ylabel, color='white')
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)
        ax.set_title(title, fontsize=14, color='white')
        ax.tick_params(colors='white')

    fig.suptitle(f"RGB Point Cloud: {session_name}", fontsize=16, fontweight='bold', color='white')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', facecolor='black')
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def plot_per_camera_rgb(ref_clouds: dict[str, tuple[np.ndarray, np.ndarray]],
                        tgt_clouds: dict[str, tuple[np.ndarray, np.ndarray]],
                        ref_session: str, tgt_session: str,
                        extrinsics: dict[str, np.ndarray],
                        limits: dict,
                        out_dir: Path,
                        resolution: int = 800):
    """Per-camera 3-panel views for ref and target separately."""
    cam_dir = out_dir / "per_camera_rgb"
    cam_dir.mkdir(parents=True, exist_ok=True)

    all_cam_ids = sorted(set(list(ref_clouds.keys()) + list(tgt_clouds.keys())))

    for cam_id in all_cam_ids:
        # Reference
        if cam_id in ref_clouds:
            pts, cols = ref_clouds[cam_id]
            if len(pts) > 0:
                plot_session_views(pts, cols,
                                   f"{ref_session} / {cam_id}",
                                   extrinsics, limits,
                                   cam_dir / f"{cam_id}_ref.png",
                                   resolution,
                                   draw_cameras=False)

        # Target
        if cam_id in tgt_clouds:
            pts, cols = tgt_clouds[cam_id]
            if len(pts) > 0:
                plot_session_views(pts, cols,
                                   f"{tgt_session} / {cam_id}",
                                   extrinsics, limits,
                                   cam_dir / f"{cam_id}_tgt.png",
                                   resolution,
                                   draw_cameras=False)

    log.info(f"Per-camera RGB views -> {cam_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="visualize_cross_session_depth", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Cross-Session Depth RGB Visualization\n{OmegaConf.to_yaml(cfg)}")

    # Load reference extrinsics (from AprilTag calibration)
    ref_extrinsics = load_extrinsics(cfg.reference.extrinsics_dir,
                                     cfg.reference.extrinsics_filename)
    log.info(f"Loaded ref extrinsics for: {list(ref_extrinsics.keys())}")

    if cfg.target.get("extrinsics_dir") and cfg.target.get("extrinsics_filename"):
        tgt_extrinsics = load_extrinsics(cfg.target.extrinsics_dir, cfg.target.extrinsics_filename)
        log.info(f"Loaded target extrinsics from file: {list(tgt_extrinsics.keys())}")
    else:
        tgt_extrinsics = build_har_extrinsics()
        log.info(f"Using hardcoded HAR extrinsics: {list(tgt_extrinsics.keys())}")

    # Derive frames directories from session names
    ref_frames_dir = f"data/{cfg.reference.session}/frames"
    tgt_frames_dir = f"data/{cfg.target.session}/frames"

    cameras = list(cfg.cameras)
    ref_clouds = {}  # cam_id -> (pts, colors)
    tgt_clouds = {}

    for cam_id in cameras:
        # Find reference extrinsics key
        T_ref = None
        for key in [cam_id, f"cam{cam_id}"]:
            if key in ref_extrinsics:
                T_ref = ref_extrinsics[key]
                break
        
        T_tgt = None
        if cfg.target.get("extrinsics_dir") and cfg.target.get("extrinsics_filename"):
            for key in [cam_id, f"cam{cam_id}"]:
                if key in tgt_extrinsics:
                    T_tgt = tgt_extrinsics[key]
                    break
        else:
            T_tgt = tgt_extrinsics.get(cam_id)
        
        if T_ref is None and T_tgt is None:
            log.warning(f"[{cam_id}] No extrinsics found for either session, skipping")
            continue

        # Load kinect config (with corr_path)
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue
        K_depth = kinect["K_depth"]

        # Load DLT matrix
        P_dlt = load_dlt_matrix(cam_id, kinect.get("corr_path"), cfg.reference.session)
        if P_dlt is None:
            log.warning(f"[{cam_id}] No DLT matrix, will use gray fallback")

        # Load RGB frames for color sampling (use frame 0 as representative)
        ref_rgb = load_rgb_frame(ref_frames_dir, cam_id)
        tgt_rgb = load_rgb_frame(tgt_frames_dir, cam_id)

        # Reference session (uses AprilTag extrinsics)
        if T_ref is not None:
            ref_depth = load_depth_frames(
                cfg.reference.depth_dir, cam_id, cfg.depth.chunk,
                cfg.depth.frame_idx, cfg.depth.num_frames
            )
            if ref_depth:
                pts, cols = accumulate_world_points_rgb(ref_depth, K_depth, T_ref, P_dlt, ref_rgb)
                ref_clouds[cam_id] = (pts, cols)
                log.info(f"[{cam_id}] Reference: {len(pts)} world points")
            else:
                log.warning(f"[{cam_id}] No reference depth frames")

        # Target session (uses HAR look-at extrinsics)
        if T_tgt is not None:
            tgt_depth = load_depth_frames(
                cfg.target.depth_dir, cam_id, cfg.depth.chunk,
                cfg.depth.frame_idx, cfg.depth.num_frames
            )
            if tgt_depth:
                pts, cols = accumulate_world_points_rgb(tgt_depth, K_depth, T_tgt, P_dlt, tgt_rgb)
                tgt_clouds[cam_id] = (pts, cols)
                log.info(f"[{cam_id}] Target: {len(pts)} world points")
            else:
                log.warning(f"[{cam_id}] No target depth frames")

    # Visualize
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use ref extrinsics for camera markers on overview plots
    all_extrinsics = {**ref_extrinsics}
    # Also add any HAR-only cameras
    for cam_id, T in tgt_extrinsics.items():
        if cam_id not in all_extrinsics:
            all_extrinsics[cam_id] = T

    if ref_clouds or tgt_clouds:
        limits = compute_global_limits(ref_clouds, tgt_clouds)

        # Combine all points per session
        if ref_clouds:
            ref_pts_all = np.concatenate([p for p, _ in ref_clouds.values() if len(p) > 0], axis=0)
            ref_cols_all = np.concatenate([c for _, c in ref_clouds.values() if len(c) > 0], axis=0)
            plot_session_views(ref_pts_all, ref_cols_all,
                               cfg.reference.session, all_extrinsics, limits,
                               out_dir / "rgb_ref.png",
                               draw_cameras=False)

        if tgt_clouds:
            tgt_pts_all = np.concatenate([p for p, _ in tgt_clouds.values() if len(p) > 0], axis=0)
            tgt_cols_all = np.concatenate([c for _, c in tgt_clouds.values() if len(c) > 0], axis=0)
            plot_session_views(tgt_pts_all, tgt_cols_all,
                               cfg.target.session, all_extrinsics, limits,
                               out_dir / "rgb_tgt.png",
                               draw_cameras=False)

        # Per-camera views
        plot_per_camera_rgb(ref_clouds, tgt_clouds,
                            cfg.reference.session, cfg.target.session,
                            all_extrinsics, limits, out_dir)
    else:
        log.warning("Not enough data for visualization")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()