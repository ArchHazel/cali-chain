from pathlib import Path
import json
import numpy as np

DEFAULT_K = np.array([
    [1.110516063691994532e03, 0.0, 9.560036883813438635e02],
    [0.0, 1.119431526830967186e03, 4.795592441694046784e02],
    [0.0, 0.0, 1.0],
])

def load_cam_intrinsics(cam_name, videos_dir=Path("data/videos")):
    """Load intrinsics for a camera from its folder, falling back to default."""
    intrinsics_path = videos_dir / cam_name / "kinect_rgb_intrinsics.json"
    if not intrinsics_path.exists() and cam_name.startswith("cam"):
        intrinsics_path = videos_dir / cam_name[3:] / "kinect_rgb_intrinsics.json"
    if intrinsics_path.exists():
        with open(intrinsics_path) as f:
            data = json.load(f)
        K = np.array(data["k_matrix"], dtype=np.float64)
        cam_params = [data["fx"], data["fy"], data["cx"], data["cy"]]
        return K, cam_params
    
    print("[WARNING] " + cam_name + " intrinsics not found, falling back to default")
    # fallback to hardcoded defaults
    K = DEFAULT_K.copy()
    cam_params = [K[0,0], K[1,1], K[0,2], K[1,2]]
    return K, cam_params