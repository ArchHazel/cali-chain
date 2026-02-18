from pykinect2 import PyKinectV2
from pykinect2 import PyKinectRuntime
import ctypes

def main():
    # 1. Initialize the Kinect V2
    # We only need the Depth frame source to access the coordinate mapper
    kinect = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth)

    # 2. Access the Coordinate Mapper
    # The mapper holds the calibration data for the device
    mapper = kinect._mapper

    # 3. Get the Depth Camera Intrinsics
    # This returns a struct with FocalLengthX, FocalLengthY, PrincipalPointX, PrincipalPointY
    intrinsics = mapper.GetDepthCameraIntrinsics()

    # 4. Extract values
    fx = intrinsics.FocalLengthX
    fy = intrinsics.FocalLengthY
    cx = intrinsics.PrincipalPointX
    cy = intrinsics.PrincipalPointY

    # 5. Print in Matrix Format
    print("-" * 30)
    print("KINECT V2 DEPTH INTRINSICS")
    print("-" * 30)
    print(f"Focal Length X (fx): {fx:.4f}")
    print(f"Focal Length Y (fy): {fy:.4f}")
    print(f"Principal Pt X (cx): {cx:.4f}")
    print(f"Principal Pt Y (cy): {cy:.4f}")
    print("-" * 30)
    print("Matrix Form (K):")
    print(f"[{fx:.4f},    0,    {cx:.4f}]")
    print(f"[   0,    {fy:.4f}, {cy:.4f}]")
    print(f"[   0,       0,       1    ]")
    print("-" * 30)

    # Close the kinect sensor (optional explicitly, but good practice)
    kinect.close()

if __name__ == "__main__":
    main()