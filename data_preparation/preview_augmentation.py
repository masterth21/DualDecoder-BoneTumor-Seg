"""
Preview tool for Medical X-ray Augmentation.
Generates a side-by-side comparison grid of ~16 samples for each preset:
[Original Image | Augmented Image | Overlay Mask | Overlay Boundary GT]
Covers benign, malignant, and non-tumor radiographs.
"""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

from data_generators.augmentations import XrayAugmenter, AUGMENTATION_PRESETS
from utils.boundary_utils import extract_boundary_tf
from utils.images_utils import prepare_image, prepare_mask


def find_diverse_samples(dataset_dirs, target_count=16):
    """
    Search for benign, malignant, and empty mask samples.
    """
    candidates = {"benign": [], "malignant": [], "empty": []}

    for d in dataset_dirs:
        img_dir = os.path.join(d, "images")
        mask_dir = os.path.join(d, "masks")
        if not os.path.isdir(mask_dir):
            mask_dir = os.path.join(d, "mask")
        if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
            continue

        for fname in os.listdir(img_dir):
            stem, _ = os.path.splitext(fname)
            m_path = os.path.join(mask_dir, stem + ".png")
            if not os.path.exists(m_path):
                m_path = os.path.join(mask_dir, stem + ".jpeg")
                if not os.path.exists(m_path):
                    continue

            i_path = os.path.join(img_dir, fname)
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue

            u = np.unique(m)
            if np.all(m == 0):
                if len(candidates["empty"]) < 6:
                    candidates["empty"].append((i_path, m_path))
            elif np.any(m > 192) or (2 in u and np.max(u) <= 2):
                if len(candidates["malignant"]) < 6:
                    candidates["malignant"].append((i_path, m_path))
            else:
                if len(candidates["benign"]) < 6:
                    candidates["benign"].append((i_path, m_path))

            if (len(candidates["benign"]) >= 6 and
                len(candidates["malignant"]) >= 6 and
                len(candidates["empty"]) >= 4):
                break

    selected = []
    # Balance across categories
    selected.extend(candidates["benign"][:6])
    selected.extend(candidates["malignant"][:6])
    selected.extend(candidates["empty"][:4])

    # If still not enough, generate synthetic fallbacks
    if len(selected) < target_count:
        needed = target_count - len(selected)
        for idx in range(needed):
            selected.append(("synth", f"synth_{idx}"))

    return selected[:target_count]


def create_synthetic_sample(sample_id: int, size=(384, 384)):
    """
    Generate synthetic X-ray bone and tumor patch for testing/preview when data is absent.
    """
    h, w = size
    img = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.int32)

    # Simulated bone shaft gradient
    y, x = np.ogrid[:h, :w]
    bone = np.exp(-((x - w / 2) ** 2) / (2 * (w / 6) ** 2)) * 0.7
    img += bone[:, :, np.newaxis]

    # Type of tumor: 0=benign, 1=malignant, 2=empty
    case = sample_id % 3
    if case == 0:  # Benign: regular round ellipse
        cy, cx = int(h * 0.45), int(w * 0.5)
        r = 35
        dist = ((y - cy) ** 2) + ((x - cx) ** 2)
        tumor = dist <= (r ** 2)
        mask[tumor] = 1
        img[tumor] += 0.2
    elif case == 1:  # Malignant: irregular shape
        cy, cx = int(h * 0.55), int(w * 0.52)
        dist = (((y - cy) * 1.3) ** 2) + (((x - cx) * 0.8) ** 2)
        tumor = dist <= (45 ** 2)
        mask[tumor] = 2
        img[tumor] += 0.25

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img, mask


def overlay_mask(image, mask, alpha=0.45):
    """
    Overlay 3-class mask on RGB image:
    Class 1 (Benign): Green
    Class 2 (Malignant): Red
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = np.repeat(vis[:, :, np.newaxis], 3, axis=-1)

    overlay = vis.copy()
    overlay[mask == 1] = [0.0, 0.9, 0.2]  # Green
    overlay[mask == 2] = [0.95, 0.1, 0.1] # Red

    has_mask = (mask > 0)
    vis[has_mask] = (1.0 - alpha) * vis[has_mask] + alpha * overlay[has_mask]
    return np.clip(vis, 0.0, 1.0)


def overlay_boundary(image, boundary_mask, alpha=0.8):
    """
    Overlay boundary on image in Yellow.
    """
    vis = image.copy()
    if vis.ndim == 2:
        vis = np.repeat(vis[:, :, np.newaxis], 3, axis=-1)

    bnd = (boundary_mask > 0.5)
    vis[bnd] = (1.0 - alpha) * vis[bnd] + alpha * np.array([1.0, 0.9, 0.0], dtype=np.float32)
    return np.clip(vis, 0.0, 1.0)


def generate_preview_grid(preset_name: str, samples, output_path: str, seed: int = 42):
    """
    Create a 16 x 4 visualization grid for the given preset.
    Columns: Original Image | Augmented Image | Overlay Mask | Overlay Boundary GT
    """
    augmenter = XrayAugmenter({"ENABLE": preset_name != "none", "PRESET": preset_name}, seed=seed)

    n_samples = len(samples)
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3.2 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, 0)

    col_titles = [
        "1. Original Image",
        f"2. Augmented ({preset_name})",
        "3. Overlay Mask",
        "4. Overlay Boundary GT"
    ]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    resize_cfg = OmegaConf.create({"VALUE": True, "HEIGHT": 384, "WIDTH": 384})
    norm_mask_cfg = OmegaConf.create({"VALUE": True, "NORMALIZE_VALUE": 255})

    for i, (img_item, mask_item) in enumerate(samples):
        if img_item == "synth":
            orig_img, orig_mask = create_synthetic_sample(i, size=(384, 384))
        else:
            orig_img = prepare_image(img_item, resize_cfg, "normalize")
            orig_mask = prepare_mask(mask_item, resize_cfg, norm_mask_cfg)

        # Apply augmentation
        aug_img, aug_mask = augmenter(orig_img, orig_mask)

        # Compute Boundary GT from augmented mask
        # Expand to one-hot for boundary extractor
        mask_clamped = np.clip(aug_mask, 0, 2).astype(np.int32)
        one_hot_mask = np.eye(3, dtype=np.float32)[mask_clamped]
        boundary_tensor = extract_boundary_tf(np.expand_dims(one_hot_mask, 0), kernel_size=5)
        # Combine tumor boundaries (classes 1 and 2)
        boundary_2d = np.squeeze(boundary_tensor[0, :, :, 1:].numpy().max(axis=-1))

        # Render visualizations
        mask_vis = overlay_mask(aug_img, aug_mask)
        bnd_vis = overlay_boundary(aug_img, boundary_2d)

        axes[i, 0].imshow(orig_img, cmap="gray" if orig_img.ndim == 2 else None)
        axes[i, 0].set_ylabel(f"Sample {i+1}", fontsize=10, rotation=0, labelpad=25)

        axes[i, 1].imshow(aug_img, cmap="gray" if aug_img.ndim == 2 else None)
        axes[i, 2].imshow(mask_vis)
        axes[i, 3].imshow(bnd_vis)

        for col in range(4):
            axes[i, col].set_xticks([])
            axes[i, col].set_yticks([])

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Preview saved to: {output_path}")


def main():
    candidate_dirs = [
        "D:/TumorBone/data/BTXRD_NEW/Split_full/train",
        "D:/TumorBone/data/BTXRD_NEW/Split/train",
        "D:/TumorBone/data/BTXRD_split/Split_full/train",
        "D:/TumorBone/data/BTXRD_split/Split/train",
        "./BTXRD/Split_full/train",
        "./BTXRD/Split/train"
    ]

    samples = find_diverse_samples(candidate_dirs, target_count=16)
    print(f"[INFO] Selected {len(samples)} diverse samples for augmentation preview.")

    out_dir = "reports"
    os.makedirs(out_dir, exist_ok=True)

    for preset in ["none", "light", "medium"]:
        out_path = os.path.join(out_dir, f"preview_{preset}.png")
        generate_preview_grid(preset, samples, out_path, seed=42)


if __name__ == "__main__":
    main()
