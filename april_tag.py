from pupil_apriltags import Detector
from pathlib import Path
import json
import cv2
import numpy as np

from utils import load_cam_intrinsics

if __name__ == "__main__":

    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)


    apriltag_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )

    with open("data/cam_frame_ids.json", "r") as f:
        cam_frame_ids = json.load(f)
    
    apriltag_results = {}
    for cam_name, frame_id in cam_frame_ids.items():
        img_path = Path(f"data/imgs/{cam_name}/{frame_id}.jpg")
        if not img_path.exists():
            print(f"Image {img_path} does not exist")
            continue

        _, cam_params = load_cam_intrinsics(cam_name)

        img = cv2.imread(str(img_path))
        # flip the image horizontally
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # flip the image horizontally
        detections = apriltag_detector.detect(
            img_gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=0.12,
        )

        apriltag_cam = []
        if detections:
            print(f"Detected {len(detections)} tags in {cam_name}")

            # save the transformation matrix of the tag
            for detection in detections:
                trans_mat = np.eye(4)
                trans_mat[:3, :3] = detection.pose_R
                trans_mat[:3, 3] = detection.pose_t.flatten()

                apriltag_cam.append({
                    "tag_id": detection.tag_id,
                    "trans_mat": trans_mat.tolist()
                })

                # visualize the contour of the tag
                # print(detection.corners)
                
                for corner_id in range(4):
                    l0 = detection.corners[corner_id].astype(int)
                    l1 = detection.corners[(corner_id + 1) % 4].astype(int)
                    cv2.line(img, (l0[0], l0[1]), (l1[0], l1[1]), (0, 255, 0), 2)

                # Draw tag ID at center
                cx, cy = detection.center.astype(int)
                cv2.putText(
                    img,
                    str(detection.tag_id),
                    (cx - 20, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
            

            cv2.imwrite(str(out_dir / f"{cam_name}.jpg"), img)
            cv2.imshow("img", img)
            cv2.waitKey(0)
            
            apriltag_results[cam_name] = apriltag_cam
        else:
            print(f"No tags detected in {cam_name}")

    with open(out_dir / "apriltag_results.json", "w") as f:
        json.dump(apriltag_results, f, indent=4)
