"""
Segment Labeling Tool for Recovery Experiments.

Opens an RGB video with a frame scrubber and lets you mark segment
boundaries with keyboard shortcuts. Saves a JSON manifest describing
each labeled segment.

Experimental structure:
  - Each recording dir is one "recording"
  - A recording belongs to an environment (bathroom, living_room)
  - A recording has a role: "original" (reference pose) or "new" (moved pose)
  - Within each recording, you label segments:
      "apriltag"  — static scene with AprilTag visible (for GT)
      "no_tag"    — static scene without AprilTag (for plane solve)
      "movement"  — camera moving (rotation/translation/both)
      "other"     — anything else (transition, junk, etc.)

Keyboard controls:
  Scrubbing:
    LEFT/RIGHT arrow  — step 1 frame
    , / .             — step 10 frames
    - / =             — step 100 frames
    HOME / END        — jump to start / end

  Segment marking:
    [ — set current frame as segment START
    ] — set current frame as segment END and save segment
    1 — label = "apriltag"
    2 — label = "no_tag"
    3 — label = "movement"
    4 — label = "other"

  Groups (optional, for pairing apriltag/no_tag at the same pose):
    g — start new group (auto-increments)
    G — clear group (segments saved without group)

  Workflow: select label (1-4), optionally press g for a new group,
            navigate to start, press [, navigate to end, press ].
            Segment is saved. For paired segments, use the same group
            for both the apriltag and no_tag segments.

  Management:
    d — delete last segment
    p — print all segments so far
    s — save manifest to JSON (also auto-saves on quit)
    q / ESC — quit and save

  Display:
    SPACE — toggle play/pause (plays at video FPS)

Usage:
    python label_segments.py /path/to/recording_dir
    python label_segments.py /path/to/data/recovery/bathroom_new_rotation

    # Resume labeling (loads existing manifest):
    python label_segments.py /path/to/recording_dir  (auto-loads if manifest exists)

    # Override metadata:
    python label_segments.py /path/to/recording_dir --environment bathroom --role new
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------

def default_manifest(rec_dir: Path, args) -> dict:
    """Create a new manifest with metadata inferred from directory name."""
    name = rec_dir.name

    # Infer environment
    if args.environment:
        env = args.environment
    elif "bathroom" in name:
        env = "bathroom"
    elif "living_room" in name:
        env = "living_room"
    else:
        env = "unknown"

    # Infer role
    if args.role:
        role = args.role
    elif "orig" in name:
        role = "original"
    else:
        role = "new"

    return {
        "recording_dir": str(rec_dir.resolve()),
        "recording_name": name,
        "environment": env,
        "role": role,
        "segments": [],
    }


def load_manifest(manifest_path: Path) -> dict | None:
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return None


def save_manifest(manifest: dict, manifest_path: Path):
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Saved manifest -> {manifest_path}  ({len(manifest['segments'])} segments)")


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

LABEL_COLORS = {
    "apriltag":  (0, 255, 0),    # green
    "no_tag":    (255, 200, 0),  # cyan-ish
    "movement":  (0, 0, 255),    # red
    "other":     (128, 128, 128),
}

LABEL_KEYS = {
    ord("1"): "apriltag",
    ord("2"): "no_tag",
    ord("3"): "movement",
    ord("4"): "other",
}

WINDOW_NAME = "Segment Labeler"
DISPLAY_W = 960  # display width (scaled)


def run_labeler(rec_dir: Path, args):
    video_path = rec_dir / "rgb.avi"
    if not video_path.exists():
        print(f"Error: {video_path} not found")
        sys.exit(1)

    manifest_path = rec_dir / "segments.json"

    # Load or create manifest
    manifest = load_manifest(manifest_path)
    if manifest is not None:
        print(f"  Loaded existing manifest with {len(manifest['segments'])} segments")
        # Update metadata from args if provided
        if args.environment:
            manifest["environment"] = args.environment
        if args.role:
            manifest["role"] = args.role
    else:
        manifest = default_manifest(rec_dir, args)
        print(f"  New manifest: env={manifest['environment']} "
              f"role={manifest['role']}")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Could not open {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = DISPLAY_W / vid_w
    disp_h = int(vid_h * scale)

    print(f"  Video: {vid_w}x{vid_h} @ {fps:.1f} FPS, {total_frames} frames")
    print(f"  Display: {DISPLAY_W}x{disp_h}")

    # State
    current_frame = 0
    current_label = "apriltag"
    current_group = None  # optional group ID for pairing apriltag/no_tag segments
    seg_start = None  # frame number of segment start, or None
    playing = False

    def read_frame(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        return frame if ret else None

    def draw_overlay(frame, frame_idx):
        """Draw HUD overlay on the frame."""
        disp = cv2.resize(frame, (DISPLAY_W, disp_h))

        # --- Top bar: frame info ---
        bar_h = 30
        cv2.rectangle(disp, (0, 0), (DISPLAY_W, bar_h), (0, 0, 0), -1)

        time_s = frame_idx / fps
        group_str = f"  Group: {current_group}" if current_group is not None else ""
        text = (f"Frame {frame_idx}/{total_frames-1}  "
                f"Time {time_s:.2f}s  "
                f"Label: {current_label.upper()}{group_str}")
        cv2.putText(disp, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, LABEL_COLORS.get(current_label, (255, 255, 255)), 1)

        # Show segment start marker
        if seg_start is not None:
            cv2.putText(disp, f"  [START: {seg_start}]", (650, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # --- Bottom bar: existing segments timeline ---
        timeline_y = disp_h - 20
        cv2.rectangle(disp, (0, timeline_y - 5), (DISPLAY_W, disp_h), (0, 0, 0), -1)

        for seg in manifest["segments"]:
            x0 = int(seg["rgb_frame_start"] / total_frames * DISPLAY_W)
            x1 = int(seg["rgb_frame_end"] / total_frames * DISPLAY_W)
            color = LABEL_COLORS.get(seg["label"], (128, 128, 128))
            cv2.rectangle(disp, (x0, timeline_y - 3), (max(x1, x0 + 1), timeline_y + 10),
                          color, -1)

        # Current position marker
        cx = int(frame_idx / total_frames * DISPLAY_W)
        cv2.line(disp, (cx, timeline_y - 5), (cx, disp_h), (255, 255, 255), 2)

        # --- Help text (bottom-left) ---
        help_y = disp_h - 30
        cv2.putText(disp, "1-4:label  g:group  [:start  ]:end  d:del  p:print  s:save  q:quit",
                    (10, help_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        return disp

    # --- Trackbar callback ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    def on_trackbar(val):
        nonlocal current_frame
        current_frame = val

    cv2.createTrackbar("Frame", WINDOW_NAME, 0, max(total_frames - 1, 1), on_trackbar)

    print("\n  Controls:")
    print("    1=apriltag  2=no_tag  3=movement  4=other")
    print("    g=new group  G=clear group  [=set start  ]=set end & save")
    print("    LEFT/RIGHT=±1  ,/.=±10  -/==±100")
    print("    SPACE=play/pause  d=delete last  p=print  s=save  q=quit")

    frame = read_frame(0)
    if frame is None:
        print("Error: Could not read first frame")
        cap.release()
        return

    while True:
        # Read and display current frame
        frame = read_frame(current_frame)
        if frame is not None:
            disp = draw_overlay(frame, current_frame)
            cv2.imshow(WINDOW_NAME, disp)
            cv2.setTrackbarPos("Frame", WINDOW_NAME, current_frame)

        # Wait for key
        wait_ms = int(1000 / fps) if playing else 0
        key = cv2.waitKey(max(wait_ms, 1)) & 0xFFFF

        if key == 0xFFFF or key == -1:
            # No key (timeout during play)
            if playing:
                current_frame = min(current_frame + 1, total_frames - 1)
                if current_frame >= total_frames - 1:
                    playing = False
            continue

        # --- Key handling ---
        # Note: arrow keys with modifiers produce different key codes
        # on different platforms. We handle the common cases.

        if key == ord("q") or key == 27:  # q or ESC
            save_manifest(manifest, manifest_path)
            break

        elif key == ord(" "):  # space — play/pause
            playing = not playing

        elif key == ord("s"):
            save_manifest(manifest, manifest_path)

        elif key == ord("p"):
            print(f"\n  Segments ({len(manifest['segments'])}):")
            for i, seg in enumerate(manifest["segments"]):
                dur = (seg["rgb_frame_end"] - seg["rgb_frame_start"]) / fps
                group_str = f"  group={seg['group']}" if "group" in seg else ""
                print(f"    [{i}] {seg['label']:<10} frames {seg['rgb_frame_start']}-{seg['rgb_frame_end']} "
                      f"({dur:.1f}s){group_str}")

        elif key == ord("d"):
            if manifest["segments"]:
                removed = manifest["segments"].pop()
                print(f"  Deleted: {removed['label']} "
                      f"frames {removed['rgb_frame_start']}-{removed['rgb_frame_end']}")
            else:
                print("  No segments to delete")

        elif key in LABEL_KEYS:
            current_label = LABEL_KEYS[key]
            print(f"  Label set to: {current_label}")

        elif key == ord("g"):
            # Increment group: find max existing group and add 1
            existing = [seg.get("group") for seg in manifest["segments"]
                        if seg.get("group") is not None]
            current_group = max(existing, default=0) + 1
            print(f"  Group set to: {current_group}")

        elif key == ord("G"):
            current_group = None
            print(f"  Group cleared (no group)")

        elif key == ord("["):
            seg_start = current_frame
            print(f"  Segment START set: frame {seg_start}")

        elif key == ord("]"):
            if seg_start is None:
                print("  No start set! Press [ first.")
            elif current_frame <= seg_start:
                print(f"  End ({current_frame}) must be after start ({seg_start})")
            else:
                seg = {
                    "label": current_label,
                    "rgb_frame_start": seg_start,
                    "rgb_frame_end": current_frame,
                }
                if current_group is not None:
                    seg["group"] = current_group
                manifest["segments"].append(seg)
                dur = (current_frame - seg_start) / fps
                group_str = f"  group={current_group}" if current_group is not None else ""
                print(f"  Saved segment: {current_label} "
                      f"frames {seg_start}-{current_frame} ({dur:.1f}s){group_str}")
                seg_start = None

        # Arrow keys — platform-dependent codes
        # OpenCV on Linux: LEFT=65361, RIGHT=65363, UP=65362, DOWN=65364
        # OpenCV on Mac/Windows: LEFT=2, RIGHT=3 (sometimes 63234/63235)
        # We also check for 0-255 range

        elif key in (65363, 3, 83):  # RIGHT arrow
            current_frame = min(current_frame + 1, total_frames - 1)
            playing = False
        elif key in (65361, 2, 81):  # LEFT arrow
            current_frame = max(current_frame - 1, 0)
            playing = False

        # Shift+arrow: some platforms send different codes
        # Fallback: use , and . for ±10
        elif key == ord(",") or key == ord("<"):
            current_frame = max(current_frame - 10, 0)
            playing = False
        elif key == ord(".") or key == ord(">"):
            current_frame = min(current_frame + 10, total_frames - 1)
            playing = False

        # Ctrl+arrow fallback: use - and = for ±100
        elif key == ord("-") or key == ord("_"):
            current_frame = max(current_frame - 100, 0)
            playing = False
        elif key == ord("=") or key == ord("+"):
            current_frame = min(current_frame + 100, total_frames - 1)
            playing = False

        elif key == 65360:  # HOME
            current_frame = 0
            playing = False
        elif key == 65367:  # END
            current_frame = total_frames - 1
            playing = False

        else:
            # Debug: uncomment to see key codes on your platform
            # print(f"  Unknown key: {key}")
            pass

    cap.release()
    cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Label experimental segments in recovery recordings")
    parser.add_argument("recording_dir", type=str,
                        help="Path to recording directory (contains rgb.avi)")
    parser.add_argument("--environment", type=str, default=None,
                        help="Override environment: bathroom, living_room")
    parser.add_argument("--role", type=str, default=None,
                        help="Override role: original, new")
    args = parser.parse_args()

    rec_dir = Path(args.recording_dir)
    if not rec_dir.exists():
        print(f"Error: {rec_dir} does not exist")
        sys.exit(1)

    print(f"\n  Segment Labeling Tool")
    print(f"  Recording: {rec_dir.name}")

    run_labeler(rec_dir, args)


if __name__ == "__main__":
    main()