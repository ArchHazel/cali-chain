import cv2
import numpy as np
import json
import os

# --- Configuration ---
JSON_PATH = 'data/intrinsics.json'
IMG_DIR = 'data/imgs/HAR2/'
OUT_DIR_UNDISTORTED = 'output/HAR2_undistorted/'
OUT_DIR_CORNERS = 'output/HAR2_corners/' 
OUT_DIR_UNDISTORTED_CORNERS = 'output/HAR2_undistorted_corners/'

# Number of inner corners on the checkerboard (width, height)
CHECKERBOARD_SIZE = (7, 10) 

# Size of a single square in mm
SQUARE_SIZE = 15.0  

def calibrate_visualize_and_undistort():
    # 1. Load the target image frames
    try:
        with open(JSON_PATH, 'r') as f:
            data = json.load(f)
        frames = data.get("frames", [])
    except FileNotFoundError:
        print(f"Error: JSON file not found at {JSON_PATH}")
        return

    if not frames:
        print("No frames found in the JSON file.")
        return

    # Create all output directories
    os.makedirs(OUT_DIR_UNDISTORTED, exist_ok=True)
    os.makedirs(OUT_DIR_CORNERS, exist_ok=True)
    os.makedirs(OUT_DIR_UNDISTORTED_CORNERS, exist_ok=True)

    # 2. Prepare 3D object points
    objp = np.zeros((CHECKERBOARD_SIZE[0] * CHECKERBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD_SIZE[0], 0:CHECKERBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    objpoints = [] 
    imgpoints = [] 

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    image_shape = None

    # 3. Detect corners on original distorted images
    print(f"Detecting corners and saving original visualizations to: {OUT_DIR_CORNERS}")
    for frame in frames:
        img_path = os.path.join(IMG_DIR, frame)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Warning: Could not read {img_path}. Skipping.")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        if image_shape is None:
            image_shape = gray.shape[::-1]

        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD_SIZE, None)

        if ret:
            objpoints.append(objp)
            refined_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(refined_corners)
            
            # Draw the corners and the connecting lines
            vis_img = img.copy()
            cv2.drawChessboardCorners(vis_img, CHECKERBOARD_SIZE, refined_corners, ret)
            
            # Save the visualization
            vis_path = os.path.join(OUT_DIR_CORNERS, frame)
            cv2.imwrite(vis_path, vis_img)
        else:
            print(f"Warning: Checkerboard corners not found in {frame}.")

    if not objpoints:
        print("Error: Could not detect the checkerboard in any of the provided images.")
        return

    # 4. Perform the camera calibration
    print("\nRunning camera calibration...")
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_shape, None, None)

    print("\n--- Calibration Results ---")
    print(f"RMS Re-projection Error: {ret:.4f} pixels")
    print("\nCamera Matrix:\n", mtx)
    print("\nDistortion Coefficients:\n", dist)

    # 5. Undistort, detect corners again, and save images
    print(f"\nUndistorting images and saving to: {OUT_DIR_UNDISTORTED}")
    print(f"Saving undistorted corner visualizations to: {OUT_DIR_UNDISTORTED_CORNERS}")
    
    new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, image_shape, 1, image_shape)
    x, y, w, h = roi

    for frame in frames:
        img_path = os.path.join(IMG_DIR, frame)
        img = cv2.imread(img_path)
        
        if img is None:
            continue
            
        # Apply undistortion
        dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)
        dst_cropped = dst[y:y+h, x:x+w]
        
        # Save the clean undistorted image
        out_path = os.path.join(OUT_DIR_UNDISTORTED, frame)
        cv2.imwrite(out_path, dst_cropped)

        # Detect corners on the newly undistorted and cropped image
        gray_undist = cv2.cvtColor(dst_cropped, cv2.COLOR_BGR2GRAY)
        ret_undist, corners_undist = cv2.findChessboardCorners(gray_undist, CHECKERBOARD_SIZE, None)

        if ret_undist:
            refined_corners_undist = cv2.cornerSubPix(gray_undist, corners_undist, (11, 11), (-1, -1), criteria)
            
            vis_img_undist = dst_cropped.copy()
            cv2.drawChessboardCorners(vis_img_undist, CHECKERBOARD_SIZE, refined_corners_undist, ret_undist)
            
            vis_path_undist = os.path.join(OUT_DIR_UNDISTORTED_CORNERS, frame)
            cv2.imwrite(vis_path_undist, vis_img_undist)
        else:
            print(f"Warning: Could not detect checkerboard in the undistorted version of {frame}.")
        
    print("\nProcessing complete.")

if __name__ == "__main__":
    calibrate_visualize_and_undistort()