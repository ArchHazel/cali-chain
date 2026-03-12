"""
Cross-session depth alignment visualization.

Takes depth from two sessions, projects both into world frame using the
reference session's extrinsics, and renders bird's-eye and side views
to visualize camera drift between sessions.

Usage:
    python -m src.cross_session_depth
    python -m src.cross_session_depth reference.session=calib_5 target.session=calib_3
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
    return {"K_depth": K}


def load_extrinsics(extrinsics_dir: str, filename: str) -> dict[str, np.ndarray]:
    ext_path = Path(extrinsics_dir) / filename
    if not ext_path.exists():
        raise FileNotFoundError(f"Extrinsics not found: {ext_path}")
    with open(ext_path) as f:
        data = json.load(f)
    return {cam: np.array(mat) for cam, mat in data.items()}


def load_depth_frames(depth_dir: str, cam_id: str, chunk: int,
                      start_frame: int, num_frames: int) -> list[np.ndarray]:
    """Load multiple depth frames from a chunk."""
    depth_path = Path(depth_dir) / cam_id / "depth" / f"depth_{chunk}.npy"
    if not depth_path.exists():
        log.warning(f"Depth file not found: {depth_path}")
        return []
    depth_chunk = np.load(depth_path)
    end_frame = min(start_frame + num_frames, depth_chunk.shape[0])
    return [depth_chunk[i] for i in range(start_frame, end_frame)]


# ---------------------------------------------------------------------------
# Depth processing
# ---------------------------------------------------------------------------

def backproject_depth(depth_frame: np.ndarray, K: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Back-project depth pixels to 3D in depth camera space (Y-down, OpenCV convention)."""
    H, W = depth_frame.shape
    cam = np.zeros((H, W, 3), dtype=np.float32)
    cam[:, :, 0] = np.arange(W)
    cam[:, :, 1] = np.arange(H)[:, np.newaxis]  # standard image coords (Y-down)
    cam[:, :, 2] = 1.0

    cam_flat = cam.reshape(-1, 3)
    cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

    depth_m = depth_frame.flatten().astype(np.float32) * 0.001
    cam_flat *= depth_m[:, np.newaxis]

    valid = depth_m > 0
    return cam_flat[valid], depth_m[valid]


def to_world(pts_cam: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """Transform points from camera frame to world frame."""
    R = T_world_cam[:3, :3]
    t = T_world_cam[:3, 3]
    return (R @ pts_cam.T).T + t


def accumulate_world_points(depth_frames: list[np.ndarray], K_depth: np.ndarray,
                            T_world_cam: np.ndarray,
                            max_depth: float = 6.0,
                            subsample: int = 4) -> np.ndarray:
    """
    Back-project multiple depth frames, transform to world frame,
    and accumulate into a single point cloud.
    """
    all_pts = []
    for depth_frame in depth_frames:
        pts, depths = backproject_depth(depth_frame, K_depth)

        # Filter by max depth
        mask = depths < max_depth
        pts = pts[mask]

        # Negate X to match the horizontally flipped RGB frames
        # that the extrinsics were computed from
        pts[:, 0] = -pts[:, 0]

        # Subsample for performance
        if subsample > 1:
            pts = pts[::subsample]

        # Transform to world
        pts_world = to_world(pts, T_world_cam)
        all_pts.append(pts_world)

    if not all_pts:
        return np.zeros((0, 3))
    return np.concatenate(all_pts, axis=0)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def compute_global_limits(ref_clouds: dict[str, np.ndarray],
                          tgt_clouds: dict[str, np.ndarray],
                          margin: float = 0.5) -> dict:
    """Compute shared axis limits from all point clouds."""
    all_pts = []
    for pts in list(ref_clouds.values()) + list(tgt_clouds.values()):
        if len(pts) > 0:
            all_pts.append(pts)
    if not all_pts:
        return {"xlim": (-5, 5), "ylim": (-5, 5), "zlim": (-5, 5)}
    all_pts = np.concatenate(all_pts, axis=0)

    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    return {"xlim": xlim, "ylim": ylim, "zlim": zlim}


def render_density_bg(ax, ref_pts, tgt_pts, axis0, axis1, xlim, ylim, resolution=800):
    """
    Render density-blended background image on an axes.
    Blue = reference, Red = target, Purple = overlap.
    """
    ref_density = np.zeros((resolution, resolution))
    tgt_density = np.zeros((resolution, resolution))

    if len(ref_pts) > 0:
        h, _, _ = np.histogram2d(
            ref_pts[:, axis0], ref_pts[:, axis1],
            bins=resolution, range=[list(xlim), list(ylim)]
        )
        ref_density = np.log1p(h.T)

    if len(tgt_pts) > 0:
        h, _, _ = np.histogram2d(
            tgt_pts[:, axis0], tgt_pts[:, axis1],
            bins=resolution, range=[list(xlim), list(ylim)]
        )
        tgt_density = np.log1p(h.T)

    # Normalize each independently
    if ref_density.max() > 0:
        ref_density /= ref_density.max()
    if tgt_density.max() > 0:
        tgt_density /= tgt_density.max()

    # Blend into RGB: red=target, blue=reference, overlap=purple
    rgb = np.zeros((resolution, resolution, 3), dtype=np.float32)
    rgb[:, :, 0] = tgt_density        # red channel
    rgb[:, :, 2] = ref_density         # blue channel
    rgb[:, :, 1] = np.minimum(ref_density, tgt_density) * 0.3  # slight green in overlap

    ax.imshow(np.clip(rgb, 0, 1),
              extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
              aspect='equal', origin='lower', zorder=0)


def plot_birds_eye(ref_clouds: dict[str, np.ndarray],
                   tgt_clouds: dict[str, np.ndarray],
                   ref_session: str, tgt_session: str,
                   extrinsics: dict[str, np.ndarray],
                   limits: dict,
                   out_path: Path,
                   ref_all: np.ndarray = None,
                   tgt_all: np.ndarray = None):
    """
    Bird's-eye (XY), front (XZ), and side (YZ) views.
    Reference in blue, target in red, overlap in purple.
    """
    if ref_all is None:
        ref_all = np.concatenate([p for p in ref_clouds.values() if len(p) > 0], axis=0) \
            if any(len(p) > 0 for p in ref_clouds.values()) else np.zeros((0, 3))
    if tgt_all is None:
        tgt_all = np.concatenate([p for p in tgt_clouds.values() if len(p) > 0], axis=0) \
            if any(len(p) > 0 for p in tgt_clouds.values()) else np.zeros((0, 3))

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # XY view (bird's eye)
    ax = axes[0]
    ax.set_title("Bird's Eye View (XY)", fontsize=14)
    render_density_bg(ax, ref_all, tgt_all, 0, 1, limits["xlim"], limits["ylim"])

    for cam_id, T in extrinsics.items():
        pos = T[:3, 3]
        ax.plot(pos[0], pos[1], 'w^', markersize=8,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.annotate(cam_id, (pos[0], pos[1]), fontsize=7,
                    ha='center', va='bottom', color='white',
                    fontweight='bold', zorder=5)

    ax.set_xlim(limits["xlim"])
    ax.set_ylim(limits["ylim"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)
    ax.legend(
        handles=[
            plt.Line2D([0], [0], marker='s', color='blue', linestyle='', markersize=8,
                       label=f'{ref_session} (reference)'),
            plt.Line2D([0], [0], marker='s', color='red', linestyle='', markersize=8,
                       label=f'{tgt_session} (target)'),
            plt.Line2D([0], [0], marker='s', color='purple', linestyle='', markersize=8,
                       label='overlap'),
        ],
        loc='upper right', facecolor='black', edgecolor='white',
        labelcolor='white', fontsize=9
    )

    # XZ view (front)
    ax = axes[1]
    ax.set_title("Front View (XZ)", fontsize=14)
    render_density_bg(ax, ref_all, tgt_all, 0, 2, limits["xlim"], limits["zlim"])

    for cam_id, T in extrinsics.items():
        pos = T[:3, 3]
        ax.plot(pos[0], pos[2], 'w^', markersize=8,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.annotate(cam_id, (pos[0], pos[2]), fontsize=7,
                    ha='center', va='bottom', color='white',
                    fontweight='bold', zorder=5)

    ax.set_xlim(limits["xlim"])
    ax.set_ylim(limits["zlim"])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)

    # YZ view (side)
    ax = axes[2]
    ax.set_title("Side View (YZ)", fontsize=14)
    render_density_bg(ax, ref_all, tgt_all, 1, 2, limits["ylim"], limits["zlim"])

    for cam_id, T in extrinsics.items():
        pos = T[:3, 3]
        ax.plot(pos[1], pos[2], 'w^', markersize=8,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        ax.annotate(cam_id, (pos[1], pos[2]), fontsize=7,
                    ha='center', va='bottom', color='white',
                    fontweight='bold', zorder=5)

    ax.set_xlim(limits["ylim"])
    ax.set_ylim(limits["zlim"])
    ax.set_xlabel("Y (m)")
    ax.set_ylabel("Z (m)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)

    # Title
    if ref_session and tgt_session:
        title = f"Cross-Session Depth Alignment: {ref_session} (blue) vs {tgt_session} (red)"
    elif ref_session:
        title = f"Depth Point Cloud: {ref_session} (reference)"
    else:
        title = f"Depth Point Cloud: {tgt_session} (target)"
    fig.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def render_three_views(ax_xy, ax_xz, ax_yz, pts_a, pts_b, limits, resolution=800):
    """Render 3-panel density view for one or two point clouds."""
    render_density_bg(ax_xy, pts_a, pts_b, 0, 1, limits["xlim"], limits["ylim"], resolution)
    ax_xy.set_xlim(limits["xlim"])
    ax_xy.set_ylim(limits["ylim"])
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_aspect("equal")
    ax_xy.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)
    ax_xy.set_title("Bird's Eye (XY)")

    render_density_bg(ax_xz, pts_a, pts_b, 0, 2, limits["xlim"], limits["zlim"], resolution)
    ax_xz.set_xlim(limits["xlim"])
    ax_xz.set_ylim(limits["zlim"])
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_aspect("equal")
    ax_xz.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)
    ax_xz.set_title("Front View (XZ)")

    render_density_bg(ax_yz, pts_a, pts_b, 1, 2, limits["ylim"], limits["zlim"], resolution)
    ax_yz.set_xlim(limits["ylim"])
    ax_yz.set_ylim(limits["zlim"])
    ax_yz.set_xlabel("Y (m)")
    ax_yz.set_ylabel("Z (m)")
    ax_yz.set_aspect("equal")
    ax_yz.grid(True, alpha=0.3, color='white', linewidth=0.5, zorder=1)
    ax_yz.set_title("Side View (YZ)")


def plot_per_camera(ref_clouds: dict[str, np.ndarray],
                    tgt_clouds: dict[str, np.ndarray],
                    ref_session: str, tgt_session: str,
                    limits: dict,
                    out_dir: Path):
    """Per-camera views: individual sessions + combined overlay."""
    cam_ids = sorted(set(ref_clouds.keys()) & set(tgt_clouds.keys()))
    empty = np.zeros((0, 3))

    for cam_id in cam_ids:
        ref_pts = ref_clouds[cam_id]
        tgt_pts = tgt_clouds[cam_id]

        if len(ref_pts) == 0 and len(tgt_pts) == 0:
            continue

        # Reference only
        if len(ref_pts) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            render_three_views(axes[0], axes[1], axes[2], ref_pts, empty, limits)
            fig.suptitle(f"{cam_id}: {ref_session} (reference)",
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(out_dir / f"{cam_id}_ref.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved: {out_dir / f'{cam_id}_ref.png'}")

        # Target only
        if len(tgt_pts) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            render_three_views(axes[0], axes[1], axes[2], empty, tgt_pts, limits)
            fig.suptitle(f"{cam_id}: {tgt_session} (target)",
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(out_dir / f"{cam_id}_tgt.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved: {out_dir / f'{cam_id}_tgt.png'}")

        # Combined
        if len(ref_pts) > 0 and len(tgt_pts) > 0:
            fig, axes = plt.subplots(1, 3, figsize=(24, 7))
            render_three_views(axes[0], axes[1], axes[2], ref_pts, tgt_pts, limits)
            fig.suptitle(f"{cam_id}: {ref_session} (blue) vs {tgt_session} (red)",
                         fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(str(out_dir / f"{cam_id}_combined.png"), dpi=150, bbox_inches='tight')
            plt.close(fig)
            log.info(f"Saved: {out_dir / f'{cam_id}_combined.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="visualize_cross_session_depth", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Cross-Session Depth Alignment\n{OmegaConf.to_yaml(cfg)}")

    # Load reference extrinsics
    extrinsics = load_extrinsics(cfg.reference.extrinsics_dir,
                                 cfg.reference.extrinsics_filename)
    log.info(f"Loaded extrinsics for: {list(extrinsics.keys())}")

    cameras = list(cfg.cameras)
    ref_clouds = {}
    tgt_clouds = {}

    for cam_id in cameras:
        # Find extrinsics key (might be "camHAR1" or "HAR1")
        T = None
        for key in [cam_id, f"cam{cam_id}"]:
            if key in extrinsics:
                T = extrinsics[key]
                break
        if T is None:
            log.warning(f"[{cam_id}] No extrinsics found, skipping")
            continue

        # Load kinect config
        try:
            kinect = load_kinect_config(cam_id)
        except FileNotFoundError:
            log.warning(f"[{cam_id}] No kinect config, skipping")
            continue
        K_depth = kinect["K_depth"]

        # Load reference depth frames
        ref_frames = load_depth_frames(
            cfg.reference.depth_dir, cam_id, cfg.depth.chunk,
            cfg.depth.frame_idx, cfg.depth.num_frames
        )
        if ref_frames:
            ref_clouds[cam_id] = accumulate_world_points(ref_frames, K_depth, T)
            log.info(f"[{cam_id}] Reference: {len(ref_clouds[cam_id])} world points "
                     f"from {len(ref_frames)} frames")
        else:
            log.warning(f"[{cam_id}] No reference depth frames")

        # Load target depth frames
        tgt_frames = load_depth_frames(
            cfg.target.depth_dir, cam_id, cfg.depth.chunk,
            cfg.depth.frame_idx, cfg.depth.num_frames
        )
        if tgt_frames:
            tgt_clouds[cam_id] = accumulate_world_points(tgt_frames, K_depth, T)
            log.info(f"[{cam_id}] Target: {len(tgt_clouds[cam_id])} world points "
                     f"from {len(tgt_frames)} frames")
        else:
            log.warning(f"[{cam_id}] No target depth frames")

    # Visualize
    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if ref_clouds and tgt_clouds:
        limits = compute_global_limits(ref_clouds, tgt_clouds)

        # Combine all points per session
        ref_all = np.concatenate([p for p in ref_clouds.values() if len(p) > 0], axis=0) \
            if any(len(p) > 0 for p in ref_clouds.values()) else np.zeros((0, 3))
        tgt_all = np.concatenate([p for p in tgt_clouds.values() if len(p) > 0], axis=0) \
            if any(len(p) > 0 for p in tgt_clouds.values()) else np.zeros((0, 3))
        empty = np.zeros((0, 3))

        # Reference only overview
        plot_birds_eye({}, {}, cfg.reference.session, "",
                       extrinsics, limits, out_dir / "overview_ref.png",
                       ref_all=ref_all, tgt_all=empty)

        # Target only overview
        plot_birds_eye({}, {}, "", cfg.target.session,
                       extrinsics, limits, out_dir / "overview_tgt.png",
                       ref_all=empty, tgt_all=tgt_all)

        # Combined overview
        plot_birds_eye(ref_clouds, tgt_clouds,
                       cfg.reference.session, cfg.target.session,
                       extrinsics, limits, out_dir / "overview_combined.png")

        plot_per_camera(ref_clouds, tgt_clouds,
                        cfg.reference.session, cfg.target.session,
                        limits, out_dir)
    else:
        log.warning("Not enough data for visualization")

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()