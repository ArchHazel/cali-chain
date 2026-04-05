#!/usr/bin/env python
"""
Linux-refactored Kinect v2 data capture script.

Original: Windows-only using pykinect2, pygame, w32tm, admin elevation.
Refactored: Linux using pylibfreenect2 (freenect2), OpenCV display,
            ntpdate/chronyc for time sync.

Dependencies:
    pip install numpy opencv-python pylibfreenect2 arrow

Hardware prerequisites:
    - libfreenect2 installed: https://github.com/OpenKinect/libfreenect2
    - Kinect v2 connected via USB 3.0
"""

from __future__ import print_function, division

import argparse
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import arrow
import cv2
import numpy as np

# Configure logging
logging.basicConfig(
    filename="sync_time_log.log",
    filemode="a",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)


# ---------------------------------------------------------------------------
# Graceful exit handler
# ---------------------------------------------------------------------------
class GracefulExiter:
    def __init__(self):
        self.state = False
        signal.signal(signal.SIGINT, self.change_state)

    def change_state(self, signum, frame):
        print("\nExit flag set to True (preparing to exit)")
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        self.state = True

    def exit(self):
        return self.state


# ---------------------------------------------------------------------------
# Kinect v2 capture using pylibfreenect2
# ---------------------------------------------------------------------------
class KinectCapture:
    """Wraps pylibfreenect2 to provide color and depth frames.

    NOTE: pylibfreenect2 does NOT provide skeleton/body tracking.
    To add pose estimation on Linux, integrate MediaPipe Pose or OpenPose
    and feed the color frames through it.
    """

    def __init__(self):
        try:
            import pylibfreenect2 as freenect2
            self._fn2 = freenect2
        except ImportError:
            raise ImportError(
                "pylibfreenect2 is required. Install libfreenect2 first, then:\n"
                "  pip install pylibfreenect2"
            )

        self._fnect = freenect2.Freenect2()
        num_devices = self._fnect.enumerateDevices()
        if num_devices == 0:
            raise RuntimeError("No Kinect v2 devices found.")

        serial_number = self._fnect.getDeviceSerialNumber(0)

        # Select packet pipeline: prefer OpenGL, fall back to CPU
        try:
            self._pipeline = freenect2.OpenGLPacketPipeline()
        except AttributeError:
            self._pipeline = freenect2.CpuPacketPipeline()

        # Pass pipeline to openDevice so it is actually used
        self._device = self._fnect.openDevice(
            serial_number, pipeline=self._pipeline
        )

        # Use FrameType (not Frame) for the listener bitmask
        self._listener = freenect2.SyncMultiFrameListener(
            freenect2.FrameType.Color | freenect2.FrameType.Depth
        )
        self._device.setColorFrameListener(self._listener)
        self._device.setIrAndDepthFrameListener(self._listener)
        self._device.start()

        self._registration = freenect2.Registration(
            self._device.getIrCameraParams(),
            self._device.getColorCameraParams(),
        )

        logging.info("Kinect v2 opened (serial: %s)", serial_number)

    def get_frames(self):
        """Return (color_bgr, depth) or (None, None) if timeout."""
        frames = self._listener.waitForNewFrame()
        if frames is None:
            return None, None

        # Use FrameType for dictionary keys
        color = frames[self._fn2.FrameType.Color]
        depth = frames[self._fn2.FrameType.Depth]

        # color frame: BGRX uint8, 1920x1080
        color_array = np.copy(color.asarray(np.uint8))[:, :, :3]  # drop alpha
        # depth frame: float32, 512x424
        depth_array = np.copy(depth.asarray(np.float32)).astype(np.uint16)

        self._listener.release(frames)
        return color_array, depth_array

    def close(self):
        self._device.stop()
        self._device.close()
        logging.info("Kinect v2 closed.")


# ---------------------------------------------------------------------------
# Main recording runtime
# ---------------------------------------------------------------------------
class KinectRecorder:
    """Captures color video and depth frames from a Kinect v2 on Linux."""

    def __init__(self, flag, pth_dat):
        self.flag = flag
        self._done = False
        self._init_paths(pth_dat)

    def _init_paths(self, pth_dat):
        self.pth_dat = Path(pth_dat)
        self.pth_dat.mkdir(parents=True, exist_ok=True)

        self.pth_har_ske = self.pth_dat / "ske_har.txt"
        self.pth_har_ts = self.pth_dat / "ske_ts.txt"
        self.pth_rgb_video = self.pth_dat / "rgb.avi"
        self.pth_rgb_ts = self.pth_dat / "rgb_ts.txt"
        self.pth_depth = self.pth_dat / "depth"
        self.pth_depth.mkdir(exist_ok=True)

    def run(
        self,
        display=True,
        save_depth=True,
        target_fps=15,
        depth_save_interval=1440,
    ):
        print("Starting Kinect capture")
        start_time = time.time()

        kinect = KinectCapture()

        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(
            str(self.pth_rgb_video), fourcc, target_fps, (1920, 1080)
        )

        depth_buffer = []
        frame_count = 0

        shutdown_flag_path = Path.home() / "shutdown.flag"
        frame_interval = 1.0 / target_fps

        try:
            while not self._done:
                loop_start = time.time()

                # Check for external shutdown flag
                if shutdown_flag_path.exists():
                    print("Shutdown flag found — finishing capture.")
                    try:
                        shutdown_flag_path.unlink()
                    except OSError as e:
                        print("Error deleting shutdown flag: {}".format(e))
                    self._done = True
                    break

                # Check graceful exit (Ctrl-C)
                if self.flag.exit():
                    self._done = True
                    break

                # Grab frames
                color_frame, depth_frame = kinect.get_frames()

                if color_frame is not None:
                    out.write(color_frame)

                    cur_ts = time.time()
                    with open(str(self.pth_rgb_ts), "a") as f:
                        f.write("{:.6f}\n".format(cur_ts))

                    if display:
                        resized = cv2.resize(color_frame, (1200, 600))
                        cv2.imshow("Recording Kinect Video Stream", resized)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            self._done = True
                            break

                    # Depth capture
                    if save_depth and depth_frame is not None:
                        depth_buffer.append(depth_frame)
                        frame_count += 1

                        if frame_count % depth_save_interval == 0:
                            batch_idx = frame_count // depth_save_interval
                            np.save(
                                str(self.pth_depth / "depth_{}".format(batch_idx)),
                                np.array(depth_buffer),
                            )
                            depth_buffer = []

                # Rate-limit
                elapsed = time.time() - loop_start
                sleep_time = frame_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

        finally:
            # Save remaining depth buffer
            if depth_buffer:
                batch_idx = frame_count // depth_save_interval + 1
                np.save(
                    str(self.pth_depth / "depth_{}".format(batch_idx)),
                    np.array(depth_buffer),
                )

            out.release()
            cv2.destroyAllWindows()
            kinect.close()

            end_time = time.time()
            duration = end_time - start_time
            hours = int(duration // 3600)
            minutes = int((duration % 3600) // 60)
            seconds = duration % 60
            print(
                "Recording duration: {} hr {} min {:.2f} sec".format(
                    hours, minutes, seconds
                )
            )


def kinect_start_wrapper(flag, dat_path, display, save_depth):
    recorder = KinectRecorder(flag, pth_dat=dat_path)
    recorder.run(display=display, save_depth=save_depth)


# ---------------------------------------------------------------------------
# Time synchronization (Linux: chronyc or ntpdate)
# ---------------------------------------------------------------------------

def _run_command(command, timeout=10):
    try:
        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
            timeout=timeout,
        )
        logging.info("Command OK: %s", command)
        return result.stdout
    except subprocess.TimeoutExpired:
        logging.error("Command timed out: %s", command)
        return None
    except subprocess.CalledProcessError as e:
        logging.error("Command failed (%s): %s", command, e.stderr)
        return None


def sync_time():
    """Synchronize system clock via NTP. Requires root / sudo."""
    if os.geteuid() != 0:
        logging.warning("Time sync requires root. Skipping.")
        print("Warning: time sync requires root (run with sudo). Skipping.")
        return False

    # Try chronyc first (common on modern distros)
    out = _run_command("chronyc makestep", timeout=10)
    if out is not None:
        logging.info("Time synced via chronyc.")
        print("Time synced (chronyc).")
        return True

    # Fall back to ntpdate
    ntp_servers = "0.debian.pool.ntp.org 1.debian.pool.ntp.org 2.debian.pool.ntp.org"
    out = _run_command("ntpdate {}".format(ntp_servers), timeout=15)
    if out is not None:
        logging.info("Time synced via ntpdate.")
        print("Time synced (ntpdate).")
        return True

    logging.error("Time sync failed with all methods.")
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Kinect v2 capture (Linux)")
    parser.add_argument(
        "--initials",
        type=str,
        default="Guest",
        help="Subject initials (default: Guest)",
    )
    parser.add_argument(
        "--dir-name",
        type=str,
        default="Data_Collection",
        help="Root data directory name (default: Data_Collection)",
    )
    parser.add_argument(
        "--naming-file",
        type=str,
        default=None,
        help="Path to a text file containing subject initials (one line). "
        "Overrides --initials if provided.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable the live preview window.",
    )
    parser.add_argument(
        "--no-depth",
        action="store_true",
        help="Disable depth frame capture.",
    )
    parser.add_argument(
        "--no-time-sync",
        action="store_true",
        help="Skip NTP time synchronization.",
    )
    args = parser.parse_args()

    # ---- Time sync (best-effort) ----
    if not args.no_time_sync:
        sync_attempts = 0
        max_retries = 3
        while sync_attempts < max_retries:
            sync_attempts += 1
            if sync_time():
                break
            logging.info("Retry time sync %d/%d", sync_attempts, max_retries)
        else:
            logging.error(
                "Time sync failed after %d retries. Continuing.", max_retries
            )

    logging.info("Starting capture...")

    # ---- Subject initials ----
    initials = args.initials
    if args.naming_file and Path(args.naming_file).exists():
        initials = (
            Path(args.naming_file).read_text().strip().splitlines()[0].strip()
        )
        print("Subject initials from file: {}".format(initials))

    # ---- Data directory ----
    dir_root = Path.cwd().parent
    ts = arrow.now().format("YYYY_MM_DD_HH_mm_ss")
    usr_name = "{}_{}".format(ts, initials)

    dat_path = dir_root / args.dir_name / usr_name
    dat_path.mkdir(parents=True, exist_ok=True)
    print("Data path: {}".format(dat_path))

    # ---- Launch capture ----
    flag = GracefulExiter()

    kinect_thread = threading.Thread(
        target=kinect_start_wrapper,
        args=(flag, str(dat_path), not args.no_display, not args.no_depth),
    )
    kinect_thread.start()
    kinect_thread.join()

    print("Capture finished.")


if __name__ == "__main__":
    main()