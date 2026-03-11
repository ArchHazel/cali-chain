import numpy as np
from pykinect2 import PyKinectV2, PyKinectRuntime

def extract_point_pairs():
    print("Initializing Kinect...")
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Depth)
    mapper = kinect._mapper

    points_3d = []
    points_2d = []

    print("Generating 3D grid and mapping to Color Space...")
    
    # Create a dense grid from -1m to 1m wide/tall, and 1m to 4m deep
    for x in np.linspace(-1.0, 1.0, 10):
        for y in np.linspace(-1.0, 1.0, 10):
            for z in np.linspace(1.0, 4.0, 10):
                pt_3d = [x, y, z]
                
                # Format for PyKinect2
                csp = PyKinectV2._CameraSpacePoint()
                csp.x, csp.y, csp.z = pt_3d
                
                # Query the SDK's internal calibration
                color_pt = mapper.MapCameraPointToColorSpace(csp)
                
                # Filter out points that fall outside the camera's view
                if not np.isinf(color_pt.x) and not np.isinf(color_pt.y):
                    points_3d.append(pt_3d)
                    points_2d.append([color_pt.x, color_pt.y])

    # Convert to numpy arrays
    points_3d = np.array(points_3d, dtype=np.float32)
    points_2d = np.array(points_2d, dtype=np.float32)

    # Save the matched point pairs
    filename = "depth3d_to_color2d_correspondences.npz"
    np.savez(filename, points_3d=points_3d, points_2d=points_2d)
    
    print(f"✅ Saved {len(points_3d)} coordinate pairs to '{filename}'")
    kinect.close()

if __name__ == "__main__":
    extract_point_pairs()