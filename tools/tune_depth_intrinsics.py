"""
Interactive GUI for tuning Kinect V2 depth intrinsics.

Displays depth-on-RGB overlay with trackbars to adjust fx, fy, cx, cy.
Updates in real-time so you can visually align depth edges with RGB edges.

Usage:
    python tune_depth_intrinsics.py <session> <cam_id> --rgb_frame <path>
"""

import numpy as np
import cv2
import argparse
from pathlib import Path


DEPTH_W, DEPTH_H = 512, 424
COLOR_W, COLOR_H = 1920, 1080

# Display resolution
DISP_W, DISP_H = 1440, 810


class IntrinsicsTuner:
    def __init__(self, depth_frame, rgb, P_dlt, K_depth_init):
        self.depth_frame = depth_frame
        self.rgb = rgb
        self.P_dlt = P_dlt
        self.alpha = 100  # percent

        # Store initial values (will be adjusted via trackbars)
        # Trackbars use integers, so we scale by 10 for 0.1 precision
        self.fx_10 = int(K_depth_init[0, 0] * 10)
        self.fy_10 = int(K_depth_init[1, 1] * 10)
        self.cx_10 = int(K_depth_init[0, 2] * 10)
        self.cy_10 = int(K_depth_init[1, 2] * 10)

        self.setup_gui()
        self.update(0)

    def setup_gui(self):
        cv2.namedWindow("Depth Overlay", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Depth Overlay", DISP_W, DISP_H)

        # fx: 300.0 - 430.0 (3000 - 4300 in trackbar units)
        cv2.createTrackbar("fx*10", "Depth Overlay", self.fx_10, 4300, self.update)
        cv2.setTrackbarMin("fx*10", "Depth Overlay", 3000)

        # fy: 300.0 - 430.0
        cv2.createTrackbar("fy*10", "Depth Overlay", self.fy_10, 4300, self.update)
        cv2.setTrackbarMin("fy*10", "Depth Overlay", 3000)

        # cx: 200.0 - 320.0 (2000 - 3200)
        cv2.createTrackbar("cx*10", "Depth Overlay", self.cx_10, 3200, self.update)
        cv2.setTrackbarMin("cx*10", "Depth Overlay", 2000)

        # cy: 160.0 - 260.0 (1600 - 2600)
        cv2.createTrackbar("cy*10", "Depth Overlay", self.cy_10, 2600, self.update)
        cv2.setTrackbarMin("cy*10", "Depth Overlay", 1600)

        # Alpha: 0 - 100
        cv2.createTrackbar("alpha%", "Depth Overlay", self.alpha, 100, self.update)

    def backproject(self, fx, fy, cx, cy):
        """Back-project depth to 3D with current intrinsics."""
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        H, W = self.depth_frame.shape

        cam = np.zeros((H, W, 3), dtype=np.float32)
        cam[:, :, 0] = np.arange(W)
        cam[:, :, 1] = np.arange(H - 1, -1, -1)[:, np.newaxis]
        cam[:, :, 2] = 1.0

        cam_flat = cam.reshape(-1, 3)
        cam_flat = (np.linalg.inv(K) @ cam_flat.T).T

        depth_m = self.depth_frame.flatten().astype(np.float32) * 0.001
        cam_flat *= depth_m[:, np.newaxis]

        return cam_flat, depth_m

    def project_dlt(self, cam_space, depth_m):
        """Project 3D points to color image via DLT."""
        valid = depth_m > 0
        pts = cam_space[valid]
        depths = depth_m[valid]

        pts_h = np.hstack([pts, np.ones((len(pts), 1))])
        proj = (self.P_dlt @ pts_h.T).T

        w = proj[:, 2]
        mask = w > 0
        px = (proj[:, 0] / w).astype(np.int32)
        py = (proj[:, 1] / w).astype(np.int32)

        in_bounds = mask & (px >= 0) & (px < COLOR_W) & (py >= 0) & (py < COLOR_H)
        px, py, depths = px[in_bounds], py[in_bounds], depths[in_bounds]

        depth_img = np.full((COLOR_H, COLOR_W), np.inf, dtype=np.float32)
        np.minimum.at(depth_img, (py, px), depths)
        depth_img[depth_img == np.inf] = 0

        return depth_img

    def colorize(self, depth_img):
        valid = depth_img > 0
        norm = np.zeros_like(depth_img)
        norm[valid] = 1.0 - np.clip((depth_img[valid] - 0.5) / 5.5, 0, 1)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    def update(self, _):
        # Read trackbar values
        fx = cv2.getTrackbarPos("fx*10", "Depth Overlay") / 10.0
        fy = cv2.getTrackbarPos("fy*10", "Depth Overlay") / 10.0
        cx = cv2.getTrackbarPos("cx*10", "Depth Overlay") / 10.0
        cy = cv2.getTrackbarPos("cy*10", "Depth Overlay") / 10.0
        alpha = cv2.getTrackbarPos("alpha%", "Depth Overlay") / 100.0

        # Back-project with current intrinsics and project via DLT
        cam_space, depth_m = self.backproject(fx, fy, cx, cy)
        depth_in_color = self.project_dlt(cam_space, depth_m)
        colored = self.colorize(depth_in_color)

        # Blend
        depth_valid = depth_in_color > 0
        overlay = self.rgb.copy()
        overlay[depth_valid] = cv2.addWeighted(
            self.rgb[depth_valid], 1.0 - alpha,
            colored[depth_valid], alpha, 0
        )

        # Add text showing current values
        text = f"fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
        cv2.putText(overlay, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (255, 255, 255), 2, cv2.LINE_AA)

        # Resize for display
        display = cv2.resize(overlay, (DISP_W, DISP_H))
        cv2.imshow("Depth Overlay", display)

    def run(self):
        print("Adjust sliders to align depth with RGB.")
        print("Press 's' to save current values, 'q' to quit.")

        while True:
            key = cv2.waitKey(50) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                fx = cv2.getTrackbarPos("fx*10", "Depth Overlay") / 10.0
                fy = cv2.getTrackbarPos("fy*10", "Depth Overlay") / 10.0
                cx = cv2.getTrackbarPos("cx*10", "Depth Overlay") / 10.0
                cy = cv2.getTrackbarPos("cy*10", "Depth Overlay") / 10.0
                print(f"\nCurrent intrinsics:")
                print(f"  fx = {fx:.1f}")
                print(f"  fy = {fy:.1f}")
                print(f"  cx = {cx:.1f}")
                print(f"  cy = {cy:.1f}")
                print(f"K_DEPTH = np.array([")
                print(f"    [{fx:.3f}, 0.0, {cx:.3f}],")
                print(f"    [0.0, {fy:.3f}, {cy:.3f}],")
                print(f"    [0.0, 0.0, 1.0]")
                print(f"])")

        cv2.destroyAllWindows()


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

    pts_h = np.hstack([points_3d, np.ones((n, 1))])
    proj = (P @ pts_h.T).T
    proj = proj[:, :2] / proj[:, 2:3]
    err = np.linalg.norm(proj - points_2d, axis=1)
    print(f"DLT fit: mean={err.mean():.2f}px, max={err.max():.2f}px")
    return P


def load_kinect_config(cam_id, configs_dir="configs/kinect"):
    """Load depth intrinsics from the Hydra kinect config for this camera."""
    config_path = Path(configs_dir) / f"{cam_id}.yaml"
    if not config_path.exists():
        print(f"Warning: {config_path} not found, using defaults")
        return None, None

    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    di = cfg.get("depth_intrinsics", {})
    K = np.array([
        [di["fx"], 0.0, di["cx"]],
        [0.0, di["fy"], di["cy"]],
        [0.0, 0.0, 1.0]
    ])

    corr_path = cfg.get("depth_to_color_correspondences", None)
    return K, corr_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("session")
    parser.add_argument("cam_id")
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=1)
    parser.add_argument("--data_root", default="data")
    parser.add_argument("--rgb_frame", required=True)
    args = parser.parse_args()

    base = Path(args.data_root) / args.session / "videos" / args.cam_id

    # Load depth intrinsics from kinect config
    K_init, corr_cfg_path = load_kinect_config(args.cam_id)
    if K_init is None:
        print("No kinect config found, using fallback values")
        K_init = np.array([
            [365.939, 0.0, 256.0],
            [0.0, 365.939, 212.0],
            [0.0, 0.0, 1.0]
        ])

    # Load correspondences - prefer config path, fall back to session data
    if corr_cfg_path and Path(corr_cfg_path).exists():
        npz_path = Path(corr_cfg_path)
    else:
        npz_path = base / "depth3d_to_color2d_correspondences.npz"
    print(f"Loading correspondences from {npz_path}")
    corr = np.load(npz_path)
    P = fit_dlt(corr['points_3d'], corr['points_2d'])

    # Load depth
    depth_path = base / "depth" / f"depth_{args.chunk}.npy"
    print(f"Loading {depth_path}")
    depth_frame = np.load(depth_path)[args.frame_idx]
    print(f"Depth: {depth_frame.shape}, non-zero: {(depth_frame > 0).sum()}")

    # Load RGB (undo preprocessing flip)
    rgb = cv2.imread(args.rgb_frame)
    rgb = np.fliplr(rgb).copy()
    if rgb.shape[:2] != (COLOR_H, COLOR_W):
        rgb = cv2.resize(rgb, (COLOR_W, COLOR_H))

    print(f"Initial depth intrinsics: fx={K_init[0,0]:.1f} fy={K_init[1,1]:.1f} "
          f"cx={K_init[0,2]:.1f} cy={K_init[1,2]:.1f}")

    tuner = IntrinsicsTuner(depth_frame, rgb, P, K_init)
    tuner.run()


if __name__ == "__main__":
    main()