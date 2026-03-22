"""
Semantic Segmentation for Structural Planes (Wall, Floor, Ceiling).

Uses Mask2Former (Swin-L backbone) trained on ADE20K to label pixels
in RGB frames as wall, floor, ceiling, or other. Outputs per-camera
visualization overlays and binary masks.

Setup:
    conda env create -f environment_semseg.yaml
    conda activate semseg

Usage:
    python -m tools.semantic_planes
    python -m tools.semantic_planes session=calib_5
    python -m tools.semantic_planes session=calib_5 cameras='[HAR1,HAR6]'
"""

import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import hydra
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ADE20K class IDs (1-indexed as in the dataset)
# Mask2Former with HuggingFace uses 0-indexed, so wall=0, floor=3, ceiling=5
# ---------------------------------------------------------------------------

STRUCTURAL_CLASSES = {
    "wall": 0,
    "floor": 3,
    "ceiling": 5,
}

STRUCTURAL_COLORS = {
    "wall": (0.2, 0.6, 1.0),      # blue
    "floor": (0.2, 0.8, 0.2),     # green
    "ceiling": (1.0, 0.4, 0.4),   # red
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str, device: str):
    """Load Mask2Former model and processor from HuggingFace."""
    from transformers import Mask2FormerForUniversalSegmentation, AutoImageProcessor

    log.info(f"Loading model: {model_name}")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name)
    model = model.to(device)
    model.eval()
    log.info(f"Model loaded on {device}")
    return model, processor


def predict_semantic(model, processor, image: Image.Image, device: str) -> np.ndarray:
    """
    Run semantic segmentation on a PIL image.
    Returns a (H, W) array of class indices (0-indexed ADE20K 150 classes).
    """
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    # Post-process to get semantic segmentation map
    result = processor.post_process_semantic_segmentation(
        outputs, target_sizes=[image.size[::-1]]
    )[0]

    return result.cpu().numpy()


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_segmentation(image: np.ndarray, seg_map: np.ndarray,
                           cam_id: str, out_path: Path):
    """
    Side-by-side: original image | structural overlay | structural mask only.
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title("RGB Image", fontsize=12)
    axes[0].axis("off")

    # Overlay: structural classes on top of RGB
    overlay = image.copy().astype(np.float32) / 255.0
    alpha = 0.5

    for class_name, class_id in STRUCTURAL_CLASSES.items():
        mask = seg_map == class_id
        color = STRUCTURAL_COLORS[class_name]
        for c in range(3):
            overlay[:, :, c] = np.where(
                mask,
                overlay[:, :, c] * (1 - alpha) + color[c] * alpha,
                overlay[:, :, c]
            )

    axes[1].imshow(np.clip(overlay, 0, 1))
    axes[1].set_title("Structural Overlay", fontsize=12)
    axes[1].axis("off")

    # Mask only: structural classes on black background
    mask_vis = np.zeros((*seg_map.shape, 3), dtype=np.float32)
    for class_name, class_id in STRUCTURAL_CLASSES.items():
        mask = seg_map == class_id
        color = STRUCTURAL_COLORS[class_name]
        for c in range(3):
            mask_vis[:, :, c] = np.where(mask, color[c], mask_vis[:, :, c])

    axes[2].imshow(mask_vis)
    axes[2].set_title("Structural Mask", fontsize=12)
    axes[2].axis("off")

    # Legend
    legend_elements = [
        Patch(facecolor=STRUCTURAL_COLORS[name], label=f"{name} (id={cid})")
        for name, cid in STRUCTURAL_CLASSES.items()
    ]
    axes[2].legend(handles=legend_elements, loc="upper right", fontsize=9)

    fig.suptitle(f"{cam_id} — Semantic Segmentation", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Saved: {out_path}")


def save_masks(seg_map: np.ndarray, cam_id: str, out_dir: Path):
    """Save individual binary masks as .npy files."""
    for class_name, class_id in STRUCTURAL_CLASSES.items():
        mask = (seg_map == class_id).astype(np.uint8)
        pixel_count = mask.sum()
        ratio = pixel_count / mask.size
        out_path = out_dir / f"{cam_id}_{class_name}_mask.npy"
        np.save(out_path, mask)
        log.info(f"  {class_name}: {pixel_count} pixels ({ratio:.1%}) -> {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(config_path="../configs", config_name="semantic_segmentation", version_base=None)
def main(cfg: DictConfig):
    log.info(f"Semantic Plane Segmentation\n{OmegaConf.to_yaml(cfg)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")

    # Load model
    model, processor = load_model(cfg.model.name, device)

    out_dir = Path(cfg.output.dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras = list(cfg.cameras)

    for cam_id in cameras:
        log.info(f"\n{'='*50}")
        log.info(f"  {cam_id}")
        log.info(f"{'='*50}")

        # Find RGB frame
        frame_path = Path(cfg.frames_dir) / cam_id / cfg.frame_name
        if not frame_path.exists():
            log.warning(f"[{cam_id}] Frame not found: {frame_path}")
            continue

        # Load and predict
        image = Image.open(frame_path).convert("RGB")
        log.info(f"[{cam_id}] Loaded {frame_path} ({image.size[0]}x{image.size[1]})")

        seg_map = predict_semantic(model, processor, image, device)
        log.info(f"[{cam_id}] Segmentation complete, {len(np.unique(seg_map))} classes detected")

        # Log structural class stats
        for class_name, class_id in STRUCTURAL_CLASSES.items():
            count = (seg_map == class_id).sum()
            ratio = count / seg_map.size
            log.info(f"  {class_name}: {count} pixels ({ratio:.1%})")

        # Save visualization
        image_np = np.array(image)
        visualize_segmentation(image_np, seg_map, cam_id, out_dir / f"{cam_id}_semantic.png")

        # Save masks
        save_masks(seg_map, cam_id, out_dir)

        # Save full segmentation map
        np.save(out_dir / f"{cam_id}_segmap.npy", seg_map.astype(np.int16))

    log.info(f"\nAll outputs -> {out_dir}")


if __name__ == "__main__":
    main()