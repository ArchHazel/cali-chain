import json
import numpy as np
import trimesh
from pathlib import Path
from types import SimpleNamespace
from scipy.spatial.transform import Rotation as R

from utils import get_tag_pose_in_world

TAG1_CONFIG = SimpleNamespace(
    width_m=0.12,
    height_m=0.12,
    wall_facing="pos_x",
    measured_corner="bottom_left",
    position_xyz=[0, 0.1413, 1.2383],
)

# Manual measurements: A = camera position, B = look-at point
HAR_CAMERAS = {
    "HAR1": {"A": [5.03, 8.45, 0.90], "B": [5.73, 8.18, 0.90]},
    "HAR2": {"A": [0.73, 5.71, 0.90], "B": [1.47, 5.80, 0.90]},
    "HAR3": {"A": [1.42, 7.75, 0.90], "B": [2.18, 7.79, 0.90]},
    "HAR4": {"A": [3.72, 0.32, 0.90], "B": [4.27, 0.86, 0.90]},
    "HAR6": {"A": [4.23, 4.36, 0.90], "B": [3.50, 4.03, 0.90]},
    "HAR8": {"A": [5.00, 4.00, 0.90], "B": [5.28, 3.52, 0.90]},
}


def get_translation(matrix):
    """Extracts the X, Y, Z translation vector from a 4x4 matrix."""
    return matrix[:3, 3]


def get_euler_angles(matrix, degrees=True):
    """Extracts Euler angles (XYZ) from the rotation part of a 4x4 matrix."""
    rot = R.from_matrix(matrix[:3, :3])
    return rot.as_euler('xyz', degrees=degrees)


def get_look_direction(matrix):
    """Camera +Z axis in world = third column of rotation matrix."""
    return matrix[:3, 2]


def camera_frustum_mesh(length=0.15, back_size=0.04, front_size=0.10, edge_radius=0.002):
    """
    Create a camera frustum mesh in camera frame: origin at camera, looking along +Z.
    Small rectangle at back (z=0), larger at front (z=length).
    Black faces for the body, blue edges (thin cylinders).
    """
    s = back_size / 2
    b = front_size / 2
    f = length
    vertices = np.array([
        [-s, -s, 0], [s, -s, 0], [s, s, 0], [-s, s, 0],   # back (z=0)
        [-b, -b, f], [b, -b, f], [b, b, f], [-b, b, f],   # front (z=f)
    ], dtype=np.float64)
    faces = np.array([
        [0, 2, 1], [0, 3, 2],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [1, 2, 6], [1, 6, 5],
        [2, 3, 7], [2, 7, 6],
        [3, 0, 4], [3, 4, 7],
    ], dtype=np.int32)
    frustum = trimesh.Trimesh(vertices=vertices, faces=faces)
    n_frustum_faces = len(frustum.faces)
    frustum.visual.face_colors = np.tile([0, 0, 0, 255], (n_frustum_faces, 1)).astype(np.uint8)

    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    edge_meshes = []
    for i, j in edges:
        seg = np.array([vertices[i], vertices[j]], dtype=np.float64)
        cyl = trimesh.creation.cylinder(radius=edge_radius, segment=seg)
        cyl.visual.face_colors = np.tile([0, 0, 255, 255], (len(cyl.faces), 1)).astype(np.uint8)
        edge_meshes.append(cyl)

    combined = trimesh.util.concatenate([frustum] + edge_meshes)
    n_combined = len(combined.faces)
    face_colors = np.zeros((n_combined, 4), dtype=np.uint8)
    face_colors[:n_frustum_faces] = [0, 0, 0, 255]
    face_colors[n_frustum_faces:] = [0, 0, 255, 255]
    combined.visual.face_colors = face_colors
    return combined


if __name__ == "__main__":

    out_dir = Path("output")
    try:
        with open("output/apriltag_results.json", "r") as f:
            apriltag_results = json.load(f)
    except FileNotFoundError:
        print("Error: 'output/apriltag_results.json' not found.")
        exit(1)

    # Dependency Chain
    chain_list = [
        ['tag1', 'camHAR6'],
        ['camHAR6', 'tag7'],
        ['tag7', 'camnew'],
        ['camnew', 'tag10'],
        ['camnew', 'tag2'],
        ['tag2', 'camHAR2'],
        ['camHAR2', 'tag9'],
        ['camHAR2', 'tag8'],
        ['tag9', 'camHAR1'],
        ['tag8', 'camHAR3'],
        ['tag10', 'camHAR8'],
    ]

    # --- INITIALIZATION ---
    center, rotation, _, _ = get_tag_pose_in_world(TAG1_CONFIG)
    T_world_tag1 = np.eye(4)
    T_world_tag1[:3, :3] = rotation
    T_world_tag1[:3, 3] = center
    
    pose_dict = {
        'tag1': T_world_tag1
    }

    # --- PROCESS CHAIN ---
    for chain in chain_list:
        source = chain[0]
        target = chain[-1]
        
        if 'tag' in source:
            tag_id_str = source[3:]
            cam_id_str = target[3:]
            inverse_needed = True 
        else:
            cam_id_str = source[3:]
            tag_id_str = target[3:]
            inverse_needed = False

        trans_mat = None
        if cam_id_str in apriltag_results:
            for detected_tag in apriltag_results[cam_id_str]:
                if detected_tag['tag_id'] == int(tag_id_str):
                    trans_mat = np.array(detected_tag['trans_mat'])
                    break
        
        if trans_mat is None:
            print(f"⚠️  MISSING DATA: Could not find Tag {tag_id_str} in Camera {cam_id_str}")
            continue

        if source not in pose_dict:
             print(f"❌ ERROR: Source '{source}' pose not calculated yet. Check chain order.")
             continue
        T_world_source = pose_dict[source]

        if inverse_needed:
            T_source_target = np.linalg.inv(trans_mat)
        else:
            T_source_target = trans_mat

        T_world_target = T_world_source @ T_source_target
        pose_dict[target] = T_world_target
        
        pos = get_translation(T_world_target)
        print(f"🔗 Linked {source} -> {target}")
        print(f"   New World Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # --- FINAL OUTPUT ---
    print("\n" + "="*50)
    print("FINAL WORLD POSITIONS")
    print("="*50)
    
    cam_extrinsics = {k: pose_dict[k].tolist() for k in pose_dict if "cam" in k}
    with open(out_dir / "cam_extrinsics.json", "w") as f:
        json.dump(cam_extrinsics, f, indent=2)

    sorted_keys = sorted(pose_dict.keys())
    
    indoor_scene = []
    for k in sorted_keys:
        pos = get_translation(pose_dict[k])
        euler = get_euler_angles(pose_dict[k])
        look = get_look_direction(pose_dict[k])
        print(f"{k:<12}: pos=[{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]  "
              f"look=[{look[0]:6.3f}, {look[1]:6.3f}, {look[2]:6.3f}]  "
              f"euler=[{euler[0]:7.2f}°, {euler[1]:7.2f}°, {euler[2]:7.2f}°]")

        if 'tag' in k:
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.1])
            mesh.apply_transform(pose_dict[k])
            indoor_scene.append(mesh)
        else:
            mesh = camera_frustum_mesh(length=0.15, back_size=0.04, front_size=0.10)
            mesh.apply_transform(pose_dict[k])
            indoor_scene.append(mesh)
    
    indoor_scene = trimesh.util.concatenate(indoor_scene)
    indoor_scene.export(out_dir / "indoor_scene.ply")

    # --- COMPARISON WITH MANUAL MEASUREMENTS ---
    print("\n" + "="*70)
    print("  CALIBRATED vs MANUAL MEASUREMENT COMPARISON")
    print("="*70)

    diffs_pos = []
    diffs_angle = []

    for cam_key, ref in sorted(HAR_CAMERAS.items()):
        pose_key = f"cam{cam_key}"
        if pose_key not in pose_dict:
            print(f"\n  {cam_key}: -- not calibrated --")
            continue

        T = pose_dict[pose_key]
        cal_pos = get_translation(T)
        cal_look = get_look_direction(T)
        cal_look_norm = cal_look / np.linalg.norm(cal_look)

        ref_A = np.array(ref["A"])
        ref_B = np.array(ref["B"])
        ref_look = ref_B - ref_A
        ref_look_norm = ref_look / np.linalg.norm(ref_look)

        pos_diff = np.linalg.norm(cal_pos - ref_A)
        pos_delta = cal_pos - ref_A
        dot = np.clip(np.dot(cal_look_norm, ref_look_norm), -1, 1)
        angle_diff = np.degrees(np.arccos(dot))

        diffs_pos.append(pos_diff)
        diffs_angle.append(angle_diff)

        print(f"\n  {cam_key}:")
        print(f"    Position:   cal=[{cal_pos[0]:6.3f}, {cal_pos[1]:6.3f}, {cal_pos[2]:6.3f}]  "
              f"ref=[{ref_A[0]:6.3f}, {ref_A[1]:6.3f}, {ref_A[2]:6.3f}]  "
              f"Δ=[{pos_delta[0]:+.3f}, {pos_delta[1]:+.3f}, {pos_delta[2]:+.3f}]  "
              f"||Δ||={pos_diff:.3f}m")
        print(f"    Look dir:   cal=[{cal_look_norm[0]:6.3f}, {cal_look_norm[1]:6.3f}, {cal_look_norm[2]:6.3f}]  "
              f"ref=[{ref_look_norm[0]:6.3f}, {ref_look_norm[1]:6.3f}, {ref_look_norm[2]:6.3f}]  "
              f"angle_diff={angle_diff:.1f}°")

    if diffs_pos:
        print(f"\n  {'='*50}")
        print(f"  Summary ({len(diffs_pos)} cameras compared):")
        print(f"    Position error:  mean={np.mean(diffs_pos):.3f}m  "
              f"max={np.max(diffs_pos):.3f}m  min={np.min(diffs_pos):.3f}m")
        print(f"    Angle error:     mean={np.mean(diffs_angle):.1f}°  "
              f"max={np.max(diffs_angle):.1f}°  min={np.min(diffs_angle):.1f}°")