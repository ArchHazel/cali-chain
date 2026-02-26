from pathlib import Path
import json
import numpy as np

# DEFAULT_K = np.array([
#     [1.110516063691994532e03, 0.0, 9.560036883813438635e02],
#     [0.0, 1.119431526830967186e03, 4.795592441694046784e02],
#     [0.0, 0.0, 1.0],
# ])

# DEFAULT_K = np.array([
#     [1.0327407219495085e+03, 0.0, 9.5685930301206076e+02],
#     [0.0, 1.0323427647512485e+03, 5.3914133979587950e+02],
#     [0.0, 0.0, 1.0],
# ])

DEFAULT_K = np.array([
    [1081.37, 0.0,     959.5],
    [0.0,     1081.37, 539.5],
    [0.0,     0.0,     1.0],
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

def get_tag_pose_in_world(tag_cfg):
    """
    Converts config (measured corner + wall facing) -> tag center + rotation matrix.

    AprilTag local frame (defined from the viewer looking AT the tag):
        +X = viewer's right
        +Y = down
        +Z = X × Y = into the tag / into the wall (away from viewer)

    The outward normal (toward the camera) is therefore -Z in tag frame.

    "wall_facing" in config = direction of the outward normal = tag's -Z in world.
    So tag +Z in world = opposite of wall_facing.

    All tags are upright on walls, so tag +Y = world -Z (down) always.

    To determine tag +X: stand in front of the tag, facing it. Your right hand
    direction in world coordinates is tag +X. When facing direction D:
        facing +X => right is -Y
        facing -X => right is +Y
        facing +Y => right is +X
        facing -Y => right is -X

    World frame: +X = right, +Y = forward (into room from door), +Z = up.
    """
    width_m = tag_cfg.width_m
    height_m = tag_cfg.height_m

    half_w = width_m / 2.0
    half_h = height_m / 2.0
    corner_pos_world = np.array(tag_cfg.position_xyz)

    # Offset from measured corner to tag center in tag-local frame
    if tag_cfg.measured_corner == "top_left":
        local_offset = np.array([half_w, half_h, 0])
    elif tag_cfg.measured_corner == "top_right":
        local_offset = np.array([-half_w, half_h, 0])
    elif tag_cfg.measured_corner == "bottom_right":
        local_offset = np.array([-half_w, -half_h, 0])
    elif tag_cfg.measured_corner == "bottom_left":
        local_offset = np.array([half_w, -half_h, 0])
    else:
        raise ValueError(f"Invalid corner: {tag_cfg.measured_corner}")

    facing = tag_cfg.wall_facing

    # Rotation matrix R: world_vec = R @ tag_vec
    # Columns are [tag_+X_in_world, tag_+Y_in_world, tag_+Z_in_world]

    if facing == "neg_x":
        # Outward normal = -X. Viewer faces +X. Right = -Y.
        # tag +X = -Y,  tag +Y = -Z,  tag +Z = (-Y)×(-Z) = +X
        r_matrix = np.array([
            [ 0,  0,  1],
            [-1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "pos_x":
        # Outward normal = +X. Viewer faces -X. Right = +Y.
        # tag +X = +Y,  tag +Y = -Z,  tag +Z = (+Y)×(-Z) = -X
        r_matrix = np.array([
            [ 0,  0, -1],
            [ 1,  0,  0],
            [ 0, -1,  0]
        ])
    elif facing == "neg_y":
        # Outward normal = -Y. Viewer faces +Y. Right = +X.
        # tag +X = +X,  tag +Y = -Z,  tag +Z = (+X)×(-Z) = +Y
        r_matrix = np.array([
            [ 1,  0,  0],
            [ 0,  0,  1],
            [ 0, -1,  0]
        ])
    elif facing == "pos_y":
        # Outward normal = +Y. Viewer faces -Y. Right = -X.
        # tag +X = -X,  tag +Y = -Z,  tag +Z = (-X)×(-Z) = -Y
        r_matrix = np.array([
            [-1,  0,  0],
            [ 0,  0, -1],
            [ 0, -1,  0]
        ])
    else:
        raise ValueError(f"Unknown facing: {facing}")

    center_pos_world = corner_pos_world + (r_matrix @ local_offset)
    return center_pos_world, r_matrix, width_m, height_m
