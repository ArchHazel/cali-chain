"""
Download a session from Google Drive into data/<session>/videos/.

Setup:
    pip install gdown

    Share your Drive folder: right-click → Share → "Anyone with the link" → Viewer

Usage:
    python download_from_drive.py --folder-id FOLDER_ID --session calib_4
    python download_from_drive.py --folder-id FOLDER_ID --session intrinsic_1
"""

import argparse
import sys
from pathlib import Path

try:
    import gdown
except ImportError:
    print("gdown not installed. Run:  pip install gdown")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Download session data from Google Drive")
    parser.add_argument("--folder-id", required=True, help="Drive folder ID of the session")
    parser.add_argument("--session", required=True, help="Session name (used for local directory)")
    args = parser.parse_args()

    dest = Path("data") / args.session / "videos"
    dest.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{args.folder_id}"
    print(f"Downloading {url} -> {dest}")

    gdown.download_folder(url=url, output=str(dest), quiet=False, use_cookies=False)

    print(f"\nDone. Files in: {dest.resolve()}")


if __name__ == "__main__":
    main()