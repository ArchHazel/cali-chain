import json
import numpy as np
import trimesh
from pathlib import Path


def get_translation(matrix):
    """Extracts the X, Y, Z translation vector from a 4x4 matrix."""
    return matrix[:3, 3]


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
    # Triangles: back (2), front (2), 4 sides (2 each) = 12 faces
    faces = np.array([
        [0, 2, 1], [0, 3, 2],           # back
        [4, 5, 6], [4, 6, 7],           # front
        [0, 1, 5], [0, 5, 4],           # bottom
        [1, 2, 6], [1, 6, 5],           # right
        [2, 3, 7], [2, 7, 6],           # top
        [3, 0, 4], [3, 4, 7],           # left
    ], dtype=np.int32)
    frustum = trimesh.Trimesh(vertices=vertices, faces=faces)
    n_frustum_faces = len(frustum.faces)
    frustum.visual.face_colors = np.tile([0, 0, 0, 255], (n_frustum_faces, 1)).astype(np.uint8)

    # 12 edges: back 4, front 4, connecting 4
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
    edge_meshes = []
    for i, j in edges:
        seg = np.array([vertices[i], vertices[j]], dtype=np.float64)
        cyl = trimesh.creation.cylinder(radius=edge_radius, segment=seg)
        cyl.visual.face_colors = np.tile([0, 0, 255, 255], (len(cyl.faces), 1)).astype(np.uint8)
        edge_meshes.append(cyl)

    combined = trimesh.util.concatenate([frustum] + edge_meshes)
    # Re-apply face colors (concatenate can drop per-mesh visuals)
    n_combined = len(combined.faces)
    face_colors = np.zeros((n_combined, 4), dtype=np.uint8)
    face_colors[:n_frustum_faces] = [0, 0, 0, 255]
    face_colors[n_frustum_faces:] = [0, 0, 255, 255]
    combined.visual.face_colors = face_colors
    return combined


if __name__ == "__main__":

    out_dir = Path("output")
    # Load the results
    try:
        with open("output/apriltag_results.json", "r") as f:
            apriltag_results = json.load(f)
    except FileNotFoundError:
        print("Error: 'output/apriltag_results.json' not found.")
        exit(1)

    # Dependency Chain
    chain_list = [
        ['tag1', 'cam8'],
        ['cam8', 'tag7'],
        ['tag7', 'camnew'],
        ['camnew', 'tag10'],
        ['camnew', 'tag2'],
        ['tag2', 'cam4'],
        ['cam4', 'tag9'],
        ['cam4', 'tag8'],
        ['tag9', 'cam12'],
        ['tag8', 'cam3'],
        ['tag10', 'cam4_8'],
    ]

    # --- INITIALIZATION ---
    # Initialize dictionary to store 4x4 World Pose Matrices
    # We assume Tag 1 is at [0, 1.685, 0.228] with Identity rotation (aligned with World)
    
    T_world_tag1 = np.eye(4)
    # T_world_tag1[:3, 3] = np.array([0, 0.2013, 1.2983])
    T_world_tag1[:3, 3] = np.array([0.2013, -1.2983, 0])
    
    pose_dict = {
        'tag1': T_world_tag1
    }

    # --- PROCESS CHAIN ---
    for chain in chain_list:
        source = chain[0]
        
        target = chain[-1]
        
        # Determine which ID is the Camera and which is the Tag
        # And determine the direction of the transform
        if 'tag' in source:
            # Case: Moving from Tag -> Camera
            tag_id_str = source[3:]
            cam_id_str = target[3:]
            
            # We are at Source (Tag). We want Target (Cam).
            # Detection gives T_cam_tag. We need T_tag_cam = inv(T_cam_tag)
            inverse_needed = True 
        else:
            # Case: Moving from Camera -> Tag
            cam_id_str = source[3:]
            tag_id_str = target[3:]
            
            # We are at Source (Cam). We want Target (Tag).
            # Detection gives T_cam_tag. We can use it directly.
            inverse_needed = False

        # Find the detection matrix in the JSON data
        trans_mat = None
        
        # Check if the camera exists in the results
        if cam_id_str in apriltag_results:
            for detected_tag in apriltag_results[cam_id_str]:
                if detected_tag['tag_id'] == int(tag_id_str):
                    trans_mat = np.array(detected_tag['trans_mat'])
                    break
        
        if trans_mat is None:
            print(f"⚠️  MISSING DATA: Could not find Tag {tag_id_str} in Camera {cam_id_str}")
            continue # Skip this link

        # --- THE MATH ---
        
        # 1. Get the World Pose of the Source (Where we are now)
        # Result: T_world_source
        if source not in pose_dict:
             print(f"❌ ERROR: Source '{source}' pose not calculated yet. Check chain order.")
             continue
        T_world_source = pose_dict[source]

        # 2. Calculate the Relative Pose (Source -> Target)
        # Detection Matrix (trans_mat) is always T_cam_tag (Tag in Cam frame)
        if inverse_needed:
            # Source=Tag, Target=Cam. 
            # We need T_tag_cam.
            T_source_target = np.linalg.inv(trans_mat)
        else:
            # Source=Cam, Target=Tag. 
            # We need T_cam_tag.
            T_source_target = trans_mat

        # 3. Chain the transformations
        # T_world_target = T_world_source * T_source_target
        T_world_target = T_world_source @ T_source_target
        # T_world_target = T_source_target @ T_world_source

        # Store the result
        pose_dict[target] = T_world_target
        
        # Print update
        pos = get_translation(T_world_target)
        print(f"🔗 Linked {source} -> {target}")
        print(f"   New World Pos: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    # --- FINAL OUTPUT ---
    print("\n" + "="*30)
    print("FINAL WORLD POSITIONS")
    print("="*30)
    
    # Sort keys for cleaner output
    sorted_keys = sorted(pose_dict.keys())
    
    indoor_scene = []
    for k in sorted_keys:
        # Get translation vector
        pos = get_translation(pose_dict[k])
        # Format: Name: [x, y, z]
        print(f"{k:<10}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")

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