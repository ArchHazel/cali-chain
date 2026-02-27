"""
Preprocessing Pipeline: extract all frames from camera videos with horizontal flip.

Usage:
    python -m src.preprocessing
    python -m src.preprocessing dataset=intrinsic_1
"""

import logging
from pathlib import Path

import cv2
import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def discover_cameras(cfg: DictConfig) -> list[str]:
    video_dir = Path(cfg.data.raw_video_dir)
    if not video_dir.exists():
        raise FileNotFoundError(f"Video directory not found: {video_dir}")
    cam_ids = sorted(
        d.name for d in video_dir.iterdir()
        if d.is_dir() and (d / cfg.data.video_filename).exists()
    )
    log.info(f"Discovered {len(cam_ids)} cameras: {cam_ids}")
    return cam_ids


def extract_frames(cfg: DictConfig, cam_id: str) -> int:
    video_path = Path(cfg.data.raw_video_dir) / cam_id / cfg.data.video_filename
    out_dir = Path(cfg.data.frames_dir) / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"[{cam_id}] Could not open: {video_path}")
        return 0

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        fname = cfg.data.frame_format.format(frame_id=count)
        cv2.imwrite(str(out_dir / fname), frame)
        count += 1

    cap.release()
    log.info(f"[{cam_id}] {count} frames -> {out_dir}")
    return count


@hydra.main(config_path="../configs", config_name="preprocessing", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Preprocessing Pipeline\n{OmegaConf.to_yaml(cfg)}")

    cam_ids = discover_cameras(cfg)
    for cam_id in cam_ids:
        extract_frames(cfg, cam_id)

    log.info("Done.")


if __name__ == "__main__":
    main()