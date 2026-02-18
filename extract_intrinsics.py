import json
from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime

def main():
    # 1. Initialize Kinect
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color)
    
    # --- Step 0: Get Actual Resolution from Hardware ---
    # We access the underlying sensor -> ColorSource -> FrameDescription
    frame_desc = kinect._sensor.ColorFrameSource.FrameDescription
    width = frame_desc.Width
    height = frame_desc.Height

    print(f"🔎 Detected Resolution: {width} x {height}")

    # Access the mapper
    mapper = kinect._mapper

    # --- Step 1: Find Principal Point (cx, cy) ---
    pt_center = PyKinectV2._CameraSpacePoint()
    pt_center.x = 0.0
    pt_center.y = 0.0
    pt_center.z = 1.0

    color_pt_center = mapper.MapCameraPointToColorSpace(pt_center)
    cx = color_pt_center.x
    cy = color_pt_center.y

    # --- Step 2: Find Focal Length X (fx) ---
    pt_right = PyKinectV2._CameraSpacePoint()
    pt_right.x = 0.1
    pt_right.y = 0.0
    pt_right.z = 1.0

    color_pt_right = mapper.MapCameraPointToColorSpace(pt_right)
    fx = abs((color_pt_right.x - cx) * (1.0 / 0.1))

    # --- Step 3: Find Focal Length Y (fy) ---
    pt_up = PyKinectV2._CameraSpacePoint()
    pt_up.x = 0.0
    pt_up.y = 0.1
    pt_up.z = 1.0

    color_pt_up = mapper.MapCameraPointToColorSpace(pt_up)
    fy = abs((color_pt_up.y - cy) * (1.0 / 0.1))

    # --- Construct Data ---
    intrinsics_data = {
        "camera_name": "kinect_v2_rgb",
        "resolution": [width, height],
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
        "k_matrix": [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ]
    }

    # --- Save to File ---
    filename = "kinect_rgb_intrinsics.json"
    with open(filename, "w") as f:
        json.dump(intrinsics_data, f, indent=4)

    print(f"✅ Saved to '{filename}' with resolution {width}x{height}")
    
    kinect.close()

if __name__ == "__main__":
    main()