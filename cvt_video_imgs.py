import cv2
from pathlib import Path


if __name__ == "__main__":

    video_list = Path("data/videos").glob("*/rgb.avi")
    out_dir = Path("data/imgs")
    for video_path in video_list:
        video_name = video_path.parent.name
        print("processing video: ", video_name)
        out_img_dir = out_dir / video_name
        out_img_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(str(video_path))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            cv2.imwrite(str(out_img_dir / f"{frame_count:06d}.jpg"), frame)
            frame_count += 1
        cap.release()
        print(f"Processed {frame_count} frames from {video_name}")