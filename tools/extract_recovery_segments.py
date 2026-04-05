"""
Extract labeled segments from recovery recordings into pipeline-ready format.

Reads segments.json manifests from each recording directory, extracts
RGB frames (horizontally flipped) and depth arrays for each segment,
and organizes them into the output structure expected by the pipeline.

Output structure:
  output/recovery/<environment>/
    orig/
      apriltag/
        frames/000000.jpg, 000001.jpg, ...
        depth/depth_1.npy
        segment_info.json
      no_tag/
        ...
    new_static/
      apriltag/
      no_tag/
    rotation_g01/
      apriltag/
      no_tag/
    translation_g05/
      ...
    translation_rotation_g03/
      ...

Each leaf directory contains:
  - frames/  : horizontally flipped RGB frames as {frame_id:06d}.jpg
  - depth/   : depth_1.npy with shape (N, 424, 512), uint16
  - segment_info.json : traceability metadata

Usage:
    python extract_segments.py data/recovery --environment living_room
    python extract_segments.py data/recovery --environment bathroom
    python extract_segments.py data/recovery  # all environments
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

DEPTH_CHUNK_SIZE = 1440  # frames per depth_x.npy file


# ---------------------------------------------------------------------------
# Depth loading
# ---------------------------------------------------------------------------

def load_depth_frame_range(depth_dir: Path, start: int, end: int) -> np.ndarray:
    """
    Load depth frames [start, end) from chunked depth files.

    Global frame N lives in depth_{N // CHUNK_SIZE + 1}.npy at index N % CHUNK_SIZE.
    Returns array of shape (end - start, 424, 512), uint16.
    """
    frames = []
    for global_idx in range(start, end):
        chunk_num = global_idx // DEPTH_CHUNK_SIZE + 1
        local_idx = global_idx % DEPTH_CHUNK_SIZE

        chunk_path = depth_dir / f"depth_{chunk_num}.npy"
        if not chunk_path.exists():
            log.warning(f"  Depth chunk not found: {chunk_path}")
            continue

        chunk = np.load(chunk_path, mmap_mode="r")
        if local_idx >= chunk.shape[0]:
            log.warning(f"  Frame index {local_idx} out of range for {chunk_path} "
                        f"(shape {chunk.shape})")
            continue

        frames.append(chunk[local_idx].copy())

    if not frames:
        return np.array([], dtype=np.uint16)

    return np.stack(frames, axis=0)


# ---------------------------------------------------------------------------
# RGB extraction
# ---------------------------------------------------------------------------

def extract_rgb_frames(video_path: Path, start: int, end: int,
                       out_dir: Path):
    """
    Extract frames [start, end) from video, horizontally flip, save as jpg.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        log.error(f"  Could not open {video_path}")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)

    count = 0
    for frame_idx in range(start, end):
        ret, frame = cap.read()
        if not ret:
            log.warning(f"  Could not read frame {frame_idx}")
            break
        frame = cv2.flip(frame, 1)  # horizontal flip
        fname = f"{count:06d}.jpg"
        cv2.imwrite(str(out_dir / fname), frame)
        count += 1

    cap.release()
    return count


# ---------------------------------------------------------------------------
# Directory naming
# ---------------------------------------------------------------------------

def infer_recording_type(recording_name: str) -> str:
    """
    Infer the recording type from its directory name.
    Returns: 'orig', 'new_static', 'rotation', 'translation',
             'translation_rotation', or 'unknown'
    """
    name = recording_name.lower()

    if "orig" in name:
        return "orig"

    # Check for movement types (order matters: check combined first)
    if "translation" in name and "rotation" in name:
        return "translation_rotation"
    if "rotation" in name:
        return "rotation"
    if "translation" in name:
        return "translation"

    # New static (furniture moved, camera same pose)
    # e.g., living_room_new, living_room_new_tag
    if "new" in name:
        return "new_static"

    return "unknown"


def segment_output_dir(base_dir: Path, recording_name: str,
                       segment: dict) -> Path:
    """
    Determine the output directory for a segment.

    Returns path like:
      base_dir/orig/apriltag
      base_dir/new_static/no_tag
      base_dir/rotation_g01/apriltag
    """
    rec_type = infer_recording_type(recording_name)
    label = segment["label"]
    group = segment.get("group")

    if rec_type in ("orig", "new_static"):
        # No group needed
        return base_dir / rec_type / label
    elif group is not None:
        return base_dir / f"{rec_type}_g{group:02d}" / label
    else:
        # Ungrouped segment in a movement recording — use frame range as ID
        start = segment["rgb_frame_start"]
        return base_dir / f"{rec_type}_f{start:05d}" / label


# ---------------------------------------------------------------------------
# Main extraction
# ---------------------------------------------------------------------------

def extract_segment(rec_dir: Path, segment: dict, out_dir: Path) -> dict:
    """Extract one segment's RGB frames and depth data."""
    start = segment["rgb_frame_start"]
    end = segment["rgb_frame_end"]
    label = segment["label"]
    group = segment.get("group")

    log.info(f"  Extracting {label} frames {start}-{end} "
             f"({end - start} frames)"
             f"{f'  group={group}' if group else ''}")

    # RGB
    video_path = rec_dir / "rgb.avi"
    frames_dir = out_dir / "frames"
    n_rgb = extract_rgb_frames(video_path, start, end, frames_dir)
    log.info(f"    RGB: {n_rgb} frames -> {frames_dir}")

    # Depth
    depth_dir = rec_dir / "depth"
    depth_out = out_dir / "depth"
    depth_out.mkdir(parents=True, exist_ok=True)

    depth_frames = load_depth_frame_range(depth_dir, start, end)
    if depth_frames.size > 0:
        depth_path = depth_out / "depth_1.npy"
        np.save(depth_path, depth_frames)
        log.info(f"    Depth: {depth_frames.shape} -> {depth_path}")
    else:
        log.warning(f"    Depth: no frames extracted")

    # Segment info (traceability)
    info = {
        "source_recording": rec_dir.name,
        "source_dir": str(rec_dir.resolve()),
        "label": label,
        "rgb_frame_start": start,
        "rgb_frame_end": end,
        "num_frames": end - start,
        "num_rgb_extracted": n_rgb,
        "num_depth_extracted": len(depth_frames) if depth_frames.size > 0 else 0,
    }
    if group is not None:
        info["group"] = group

    with open(out_dir / "segment_info.json", "w") as f:
        json.dump(info, f, indent=2)

    return info


def process_recording(rec_dir: Path, env_out_dir: Path) -> list[dict]:
    """Process all segments in one recording."""
    manifest_path = rec_dir / "segments.json"
    if not manifest_path.exists():
        return []

    with open(manifest_path) as f:
        manifest = json.load(f)

    recording_name = manifest["recording_name"]
    segments = manifest["segments"]

    if not segments:
        log.info(f"  {recording_name}: no segments, skipping")
        return []

    log.info(f"\n{'='*60}")
    log.info(f"  {recording_name} ({len(segments)} segments)")
    log.info(f"{'='*60}")

    results = []
    for seg in segments:
        out_dir = segment_output_dir(env_out_dir, recording_name, seg)

        if out_dir.exists():
            log.info(f"  Skipping {out_dir.name} (already exists)")
            results.append({"skipped": True, "dir": str(out_dir)})
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        info = extract_segment(rec_dir, seg, out_dir)
        info["output_dir"] = str(out_dir)
        results.append(info)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Extract labeled segments into pipeline-ready format")
    parser.add_argument("recovery_dir", type=str,
                        help="Path to recovery data directory (contains recording subdirs)")
    parser.add_argument("--environment", type=str, default=None,
                        help="Only process this environment (living_room, bathroom)")
    parser.add_argument("--output", type=str, default="output/recovery",
                        help="Output base directory (default: output/recovery)")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing extracted segments")
    args = parser.parse_args()

    recovery_dir = Path(args.recovery_dir)
    if not recovery_dir.exists():
        print(f"Error: {recovery_dir} does not exist")
        sys.exit(1)

    output_base = Path(args.output)

    # Find all recording directories with manifests
    rec_dirs = sorted([
        d for d in recovery_dir.iterdir()
        if d.is_dir() and (d / "segments.json").exists()
    ])

    if not rec_dirs:
        print(f"No recordings with segments.json found in {recovery_dir}")
        sys.exit(1)

    log.info(f"Found {len(rec_dirs)} recordings with manifests")

    # Filter by environment if specified
    if args.environment:
        rec_dirs = [d for d in rec_dirs
                    if args.environment in d.name]
        log.info(f"Filtered to {len(rec_dirs)} recordings for {args.environment}")

    # Group by environment
    env_recordings = {}
    for rec_dir in rec_dirs:
        manifest_path = rec_dir / "segments.json"
        with open(manifest_path) as f:
            manifest = json.load(f)
        env = manifest.get("environment", "unknown")
        env_recordings.setdefault(env, []).append(rec_dir)

    # Process each environment
    all_results = {}
    for env, recordings in sorted(env_recordings.items()):
        env_out = output_base / env
        log.info(f"\n{'#'*60}")
        log.info(f"  Environment: {env} ({len(recordings)} recordings)")
        log.info(f"  Output: {env_out}")
        log.info(f"{'#'*60}")

        env_results = []
        for rec_dir in recordings:
            if args.force:
                # Remove existing output for this recording's segments
                pass  # handled per-segment below

            results = process_recording(rec_dir, env_out)
            env_results.extend(results)

        all_results[env] = env_results

    # Summary
    log.info(f"\n\n{'='*60}")
    log.info(f"  EXTRACTION SUMMARY")
    log.info(f"{'='*60}")
    for env, results in all_results.items():
        extracted = [r for r in results if not r.get("skipped")]
        skipped = [r for r in results if r.get("skipped")]
        log.info(f"  {env}: {len(extracted)} extracted, {len(skipped)} skipped")

    # List output structure
    log.info(f"\nOutput structure:")
    for env in sorted(all_results.keys()):
        env_dir = output_base / env
        if env_dir.exists():
            for d in sorted(env_dir.iterdir()):
                if d.is_dir():
                    subdirs = sorted([s.name for s in d.iterdir() if s.is_dir()])
                    log.info(f"  {env}/{d.name}/  ->  {', '.join(subdirs)}")


if __name__ == "__main__":
    main()