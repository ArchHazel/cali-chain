"""
Live AprilTag detection demo using the laptop camera.
Same detector config and camera params as april_tag.py.
Press 'q' to quit.
"""

from pupil_apriltags import Detector
import cv2
import numpy as np

# Same as april_tag.py (calibrated camera). For accurate pose on your laptop, calibrate it.
K = np.array([
    [1.110516063691994532e+03, 0.000000000000000000e+00, 9.560036883813438635e+02],
    [0.000000000000000000e+00, 1.119431526830967186e+03, 4.795592441694046784e+02],
    [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00],
])

cam_params = [
    1.110516063691994532e+03,
    1.119431526830967186e+03,
    9.560036883813438635e+02,
    4.795592441694046784e+02,
]
TAG_SIZE = 0.4318


def main():
    apriltag_detector = Detector(
        families="tag36h11",
        nthreads=4,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open laptop camera.")
        return

    print("AprilTag live demo. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # frame = cv2.flip(frame, 1)
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = apriltag_detector.detect(
            img_gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=TAG_SIZE,
        )

        if len(detections) > 0:
            print(f"Detected {len(detections)} tags")

        for detection in detections:
            print(detection.pose_R, detection.pose_t)
            # Draw tag contour (green)
            pts = detection.corners.astype(np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Draw tag ID at center
            cx, cy = detection.center.astype(int)
            cv2.putText(
                frame,
                str(detection.tag_id),
                (cx - 20, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )

            # Optional: draw pose translation (e.g. distance) if you want
            t = detection.pose_t.flatten()
            cv2.putText(
                frame,
                f"t: ({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})",
                (pts[0][0], pts[0][1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
            )

        # Status line
        cv2.putText(
            frame,
            f"Tags: {len(detections)} | Press 'q' to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow("AprilTag Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
