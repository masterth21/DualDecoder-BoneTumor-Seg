"""
Medical X-ray Data Augmentation Pipeline for DualDecoder-BoneTumor-Seg.
Provides XrayAugmenter with configurable presets: 'none', 'light', 'medium',
joint geometric transformations, tumor area loss protection (MAX_TUMOR_LOSS),
and intensity transformations tailored for bone tumor radiographs.
"""

import copy
import logging
import cv2
import numpy as np
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

# Standard preset configurations
AUGMENTATION_PRESETS = {
    "none": {
        "ENABLE": False,
        "PRESET": "none",
        "MAX_TUMOR_LOSS": 0.10,
        "HORIZONTAL_FLIP": {"PROB": 0.0},
        "ROTATION": {"PROB": 0.0, "ANGLE_RANGE": [-10, 10]},
        "ZOOM": {"PROB": 0.0, "RANGE": [0.9, 1.1]},
        "SHIFT": {"PROB": 0.0, "RANGE": [-0.05, 0.05]},
        "ELASTIC": {"PROB": 0.0, "ALPHA": 15.0, "SIGMA": 4.0},
        "CLAHE": {"PROB": 0.0, "CLIP_LIMIT": 2.0, "TILE_GRID_SIZE": 8},
        "BRIGHTNESS": {"PROB": 0.0, "RANGE": [-0.08, 0.08]},
        "CONTRAST": {"PROB": 0.0, "RANGE": [0.9, 1.1]},
        "GAMMA": {"PROB": 0.0, "RANGE": [0.85, 1.15]},
        "GAUSSIAN_NOISE": {"PROB": 0.0, "STDDEV": 0.01},
        "GAUSSIAN_BLUR": {"PROB": 0.0, "SIGMA_RANGE": [0.5, 1.0]},
    },
    "light": {
        "ENABLE": True,
        "PRESET": "light",
        "MAX_TUMOR_LOSS": 0.10,
        "HORIZONTAL_FLIP": {"PROB": 0.5},
        "ROTATION": {"PROB": 0.3, "ANGLE_RANGE": [-5, 5]},
        "ZOOM": {"PROB": 0.3, "RANGE": [0.95, 1.05]},
        "SHIFT": {"PROB": 0.0, "RANGE": [-0.05, 0.05]},
        "ELASTIC": {"PROB": 0.0, "ALPHA": 15.0, "SIGMA": 4.0},
        "CLAHE": {"PROB": 0.0, "CLIP_LIMIT": 2.0, "TILE_GRID_SIZE": 8},
        "BRIGHTNESS": {"PROB": 0.5, "RANGE": [-0.05, 0.05]},
        "CONTRAST": {"PROB": 0.5, "RANGE": [0.95, 1.05]},
        "GAMMA": {"PROB": 0.0, "RANGE": [0.85, 1.15]},
        "GAUSSIAN_NOISE": {"PROB": 0.0, "STDDEV": 0.01},
        "GAUSSIAN_BLUR": {"PROB": 0.0, "SIGMA_RANGE": [0.5, 1.0]},
    },
    "medium": {
        "ENABLE": True,
        "PRESET": "medium",
        "MAX_TUMOR_LOSS": 0.10,
        "HORIZONTAL_FLIP": {"PROB": 0.5},
        "ROTATION": {"PROB": 0.5, "ANGLE_RANGE": [-10, 10]},
        "ZOOM": {"PROB": 0.5, "RANGE": [0.9, 1.1]},
        "SHIFT": {"PROB": 0.3, "RANGE": [-0.05, 0.05]},
        "ELASTIC": {"PROB": 0.2, "ALPHA": 15.0, "SIGMA": 4.0},
        "CLAHE": {"PROB": 0.3, "CLIP_LIMIT": 2.0, "TILE_GRID_SIZE": 8},
        "BRIGHTNESS": {"PROB": 0.5, "RANGE": [-0.08, 0.08]},
        "CONTRAST": {"PROB": 0.5, "RANGE": [0.9, 1.1]},
        "GAMMA": {"PROB": 0.5, "RANGE": [0.85, 1.15]},
        "GAUSSIAN_NOISE": {"PROB": 0.5, "STDDEV": 0.01},
        "GAUSSIAN_BLUR": {"PROB": 0.5, "SIGMA_RANGE": [0.5, 1.0]},
    },
}

ALLOWED_TOP_KEYS = {
    "ENABLE", "PRESET", "MAX_TUMOR_LOSS",
    "HORIZONTAL_FLIP", "ROTATION", "ZOOM", "SHIFT", "ELASTIC",
    "CLAHE", "BRIGHTNESS", "CONTRAST", "GAMMA", "GAUSSIAN_NOISE", "GAUSSIAN_BLUR"
}

ALLOWED_SUBKEYS = {
    "HORIZONTAL_FLIP": {"PROB"},
    "ROTATION": {"PROB", "ANGLE_RANGE"},
    "ZOOM": {"PROB", "RANGE"},
    "SHIFT": {"PROB", "RANGE"},
    "ELASTIC": {"PROB", "ALPHA", "SIGMA"},
    "CLAHE": {"PROB", "CLIP_LIMIT", "TILE_GRID_SIZE"},
    "BRIGHTNESS": {"PROB", "RANGE"},
    "CONTRAST": {"PROB", "RANGE"},
    "GAMMA": {"PROB", "RANGE"},
    "GAUSSIAN_NOISE": {"PROB", "STDDEV"},
    "GAUSSIAN_BLUR": {"PROB", "SIGMA_RANGE"},
}


def build_augmentation_config(cfg_or_dict=None) -> dict:
    """
    Build and validate effective augmentation configuration from preset and overrides.
    Raises ValueError on any unrecognized keys.
    """
    if cfg_or_dict is None:
        user_dict = {}
    elif isinstance(cfg_or_dict, DictConfig):
        user_dict = OmegaConf.to_container(cfg_or_dict, resolve=True)
    elif isinstance(cfg_or_dict, dict):
        user_dict = copy.deepcopy(cfg_or_dict)
    else:
        raise TypeError(f"Expected DictConfig or dict, got {type(cfg_or_dict)}")

    # Check for unrecognized top-level keys
    unknown_keys = set(user_dict.keys()) - ALLOWED_TOP_KEYS
    if unknown_keys:
        raise ValueError(
            f"Unrecognized augmentation config key(s): {sorted(list(unknown_keys))}. "
            f"Allowed keys are: {sorted(list(ALLOWED_TOP_KEYS))}"
        )

    # Determine preset
    preset_name = str(user_dict.get("PRESET", "medium")).lower()
    if preset_name not in AUGMENTATION_PRESETS:
        raise ValueError(
            f"Unknown augmentation preset '{preset_name}'. "
            f"Allowed presets: {list(AUGMENTATION_PRESETS.keys())}"
        )

    # Start with base preset
    effective = copy.deepcopy(AUGMENTATION_PRESETS[preset_name])

    # Apply overrides
    for k, v in user_dict.items():
        if k in ("ENABLE", "PRESET", "MAX_TUMOR_LOSS"):
            if k == "ENABLE":
                effective["ENABLE"] = bool(v)
            elif k == "PRESET":
                effective["PRESET"] = str(v).lower()
            elif k == "MAX_TUMOR_LOSS":
                effective["MAX_TUMOR_LOSS"] = float(v)
        elif isinstance(v, dict):
            # Check subkeys
            allowed_sub = ALLOWED_SUBKEYS.get(k, set())
            unknown_sub = set(v.keys()) - allowed_sub
            if unknown_sub:
                raise ValueError(
                    f"Unrecognized parameter(s) under '{k}': {sorted(list(unknown_sub))}. "
                    f"Allowed parameters: {sorted(list(allowed_sub))}"
                )
            for sub_k, sub_v in v.items():
                if sub_k == "PROB":
                    effective[k]["PROB"] = float(sub_v)
                elif sub_k in ("ALPHA", "SIGMA", "STDDEV", "CLIP_LIMIT"):
                    effective[k][sub_k] = float(sub_v)
                elif sub_k == "TILE_GRID_SIZE":
                    effective[k][sub_k] = int(sub_v)
                elif sub_k in ("ANGLE_RANGE", "RANGE", "SIGMA_RANGE"):
                    effective[k][sub_k] = [float(x) for x in sub_v]
                else:
                    effective[k][sub_k] = sub_v
        else:
            raise ValueError(f"Value for section '{k}' must be a dictionary, got {type(v)}")

    # Handle ENABLE flag: explicit override in user_dict takes precedence,
    # otherwise preset "none" defaults to ENABLE=False
    if "ENABLE" in user_dict:
        effective["ENABLE"] = bool(user_dict["ENABLE"])
    elif effective["PRESET"] == "none":
        effective["ENABLE"] = False

    return effective


class XrayAugmenter:
    """
    Data Augmenter tailored for Bone Tumor Radiographs.
    Applies joint geometric transformations (image + mask) and photometric transformations (image only).
    Protects tumor coverage via MAX_TUMOR_LOSS constraint.
    """
    def __init__(self, config=None, seed: int = 42):
        self.effective_config = build_augmentation_config(config)
        self.enabled = bool(self.effective_config["ENABLE"])
        self.max_tumor_loss = float(self.effective_config.get("MAX_TUMOR_LOSS", 0.10))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def reseed(self, seed: int):
        """Update RNG seed for reproducibility"""
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def get_effective_config(self) -> dict:
        """Return the dictionary of active augmentation settings"""
        return copy.deepcopy(self.effective_config)

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        return self.augment(image, mask)

    def augment(self, image: np.ndarray, mask: np.ndarray = None):
        """
        Augment image and optional mask.
        Args:
            image: float32 ndarray in [0.0, 1.0], shape (H, W, C) or (H, W)
            mask: int32 ndarray with class indices {0, 1, 2}, shape (H, W) or (H, W, 1)
        Returns:
            augmented_image, augmented_mask (if mask provided) or augmented_image
        """
        if not self.enabled:
            if mask is not None:
                return image, mask
            return image

        orig_image = image.copy()
        orig_mask = mask.copy() if mask is not None else None

        img_h, img_w = image.shape[:2]
        is_mask_squeezed = False
        if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask[:, :, 0]
            is_mask_squeezed = True

        # Calculate initial tumor area for tumor loss protection
        a_orig = 0
        if mask is not None:
            a_orig = int(np.count_nonzero(mask > 0))

        # =========================================================================
        # 1. JOINT GEOMETRIC TRANSFORMS (image + mask)
        # =========================================================================
        zoom_scale = 1.0
        geom_applied = False

        # --- A. Horizontal Flip ---
        p_flip = self.effective_config["HORIZONTAL_FLIP"]["PROB"]
        if p_flip > 0 and self.rng.random() < p_flip:
            image = np.ascontiguousarray(np.fliplr(image))
            if mask is not None:
                mask = np.ascontiguousarray(np.fliplr(mask))
            geom_applied = True

        # --- B. Combined Affine Transform: Rotation + Zoom + Shift ---
        p_rot = self.effective_config["ROTATION"]["PROB"]
        p_zoom = self.effective_config["ZOOM"]["PROB"]
        p_shift = self.effective_config["SHIFT"]["PROB"]

        rot_deg = 0.0
        if p_rot > 0 and self.rng.random() < p_rot:
            ang_range = self.effective_config["ROTATION"]["ANGLE_RANGE"]
            rot_deg = float(self.rng.uniform(ang_range[0], ang_range[1]))

        if p_zoom > 0 and self.rng.random() < p_zoom:
            z_range = self.effective_config["ZOOM"]["RANGE"]
            zoom_scale = float(self.rng.uniform(z_range[0], z_range[1]))

        shift_x, shift_y = 0.0, 0.0
        if p_shift > 0 and self.rng.random() < p_shift:
            s_range = self.effective_config["SHIFT"]["RANGE"]
            shift_x = float(self.rng.uniform(s_range[0], s_range[1])) * img_w
            shift_y = float(self.rng.uniform(s_range[0], s_range[1])) * img_h

        if abs(rot_deg) > 1e-5 or abs(zoom_scale - 1.0) > 1e-5 or abs(shift_x) > 1e-3 or abs(shift_y) > 1e-3:
            center = (img_w / 2.0, img_h / 2.0)
            M = cv2.getRotationMatrix2D(center, rot_deg, zoom_scale)
            M[0, 2] += shift_x
            M[1, 2] += shift_y

            image = cv2.warpAffine(
                image, M, (img_w, img_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0
            )
            if mask is not None:
                mask = cv2.warpAffine(
                    mask.astype(np.float32), M, (img_w, img_h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                ).astype(mask.dtype)
            geom_applied = True

        # --- C. Elastic Deformation ---
        p_elastic = self.effective_config["ELASTIC"]["PROB"]
        if p_elastic > 0 and self.rng.random() < p_elastic:
            alpha = float(self.effective_config["ELASTIC"]["ALPHA"])
            sigma = float(self.effective_config["ELASTIC"]["SIGMA"])
            image, mask = self._apply_elastic(image, mask, alpha, sigma, img_h, img_w)
            geom_applied = True

        # --- D. MAX_TUMOR_LOSS Protection ---
        # Cancel geometric transform if tumor lost > 10% expected area
        if geom_applied and mask is not None and a_orig > 0:
            a_expected = a_orig * (zoom_scale ** 2)
            a_after = int(np.count_nonzero(mask > 0))
            if a_expected > 1e-6:
                tumor_loss = max(0.0, 1.0 - (a_after / a_expected))
            else:
                tumor_loss = 0.0

            if tumor_loss > self.max_tumor_loss:
                # Revert geometric transformations
                image = orig_image.copy()
                mask = orig_mask.copy()
                if mask is not None and mask.ndim == 3 and mask.shape[-1] == 1:
                    mask = mask[:, :, 0]

        # =========================================================================
        # 2. INTENSITY TRANSFORMS (image only)
        # =========================================================================
        # --- A. CLAHE (Contrast Limited Adaptive Histogram Equalization) ---
        p_clahe = self.effective_config["CLAHE"]["PROB"]
        if p_clahe > 0 and self.rng.random() < p_clahe:
            clip_limit = float(self.effective_config["CLAHE"]["CLIP_LIMIT"])
            tile_grid = int(self.effective_config["CLAHE"]["TILE_GRID_SIZE"])
            image = self._apply_clahe(image, clip_limit, tile_grid)

        # --- B. Brightness ---
        p_bright = self.effective_config["BRIGHTNESS"]["PROB"]
        if p_bright > 0 and self.rng.random() < p_bright:
            b_range = self.effective_config["BRIGHTNESS"]["RANGE"]
            delta = float(self.rng.uniform(b_range[0], b_range[1]))
            image = image + delta

        # --- C. Contrast ---
        p_contrast = self.effective_config["CONTRAST"]["PROB"]
        if p_contrast > 0 and self.rng.random() < p_contrast:
            c_range = self.effective_config["CONTRAST"]["RANGE"]
            factor = float(self.rng.uniform(c_range[0], c_range[1]))
            mean_val = float(np.mean(image))
            image = (image - mean_val) * factor + mean_val

        # --- D. Gamma Correction ---
        p_gamma = self.effective_config["GAMMA"]["PROB"]
        if p_gamma > 0 and self.rng.random() < p_gamma:
            g_range = self.effective_config["GAMMA"]["RANGE"]
            gamma = float(self.rng.uniform(g_range[0], g_range[1]))
            image = np.power(np.maximum(image, 0.0), gamma)

        # --- E. Gaussian Noise ---
        p_noise = self.effective_config["GAUSSIAN_NOISE"]["PROB"]
        if p_noise > 0 and self.rng.random() < p_noise:
            stddev = float(self.effective_config["GAUSSIAN_NOISE"]["STDDEV"])
            noise_2d = self.rng.normal(0.0, stddev, size=(img_h, img_w)).astype(np.float32)
            if image.ndim == 3:
                # Apply identical noise across all channels for grayscale radiograph consistency
                noise = np.repeat(noise_2d[:, :, np.newaxis], image.shape[-1], axis=-1)
            else:
                noise = noise_2d
            image = image + noise

        # --- F. Gaussian Blur ---
        p_blur = self.effective_config["GAUSSIAN_BLUR"]["PROB"]
        if p_blur > 0 and self.rng.random() < p_blur:
            b_range = self.effective_config["GAUSSIAN_BLUR"]["SIGMA_RANGE"]
            sigma = float(self.rng.uniform(b_range[0], b_range[1]))
            ksize = int(np.ceil(sigma * 3.0)) * 2 + 1
            image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

        # Final clip and dtype enforcement
        image = np.clip(image, 0.0, 1.0).astype(np.float32)

        if mask is not None:
            if is_mask_squeezed:
                mask = mask[:, :, np.newaxis]
            mask = mask.astype(np.int32)
            return image, mask

        return image

    def _apply_elastic(self, image: np.ndarray, mask: np.ndarray, alpha: float, sigma: float, h: int, w: int):
        """
        Fast coarse-grid elastic displacement with bilinear upsampling.
        Ensures NEAREST interpolation and constant 0 border on mask.
        """
        down = 4
        ch, cw = max(4, h // down), max(4, w // down)

        dx_small = self.rng.uniform(-1.0, 1.0, size=(ch, cw)).astype(np.float32)
        dy_small = self.rng.uniform(-1.0, 1.0, size=(ch, cw)).astype(np.float32)

        sigma_coarse = max(1.0, sigma / down)
        dx_smooth = cv2.GaussianBlur(dx_small, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse) * alpha
        dy_smooth = cv2.GaussianBlur(dy_small, (0, 0), sigmaX=sigma_coarse, sigmaY=sigma_coarse) * alpha

        dx = cv2.resize(dx_smooth, (w, h), interpolation=cv2.INTER_LINEAR)
        dy = cv2.resize(dy_smooth, (w, h), interpolation=cv2.INTER_LINEAR)

        grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        map_x = np.clip(grid_x + dx, 0, w - 1).astype(np.float32)
        map_y = np.clip(grid_y + dy, 0, h - 1).astype(np.float32)

        image_warped = cv2.remap(
            image, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0
        )

        mask_warped = None
        if mask is not None:
            mask_warped = cv2.remap(
                mask.astype(np.float32), map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ).astype(mask.dtype)

        return image_warped, mask_warped

    @staticmethod
    def _apply_clahe(image: np.ndarray, clip_limit: float, tile_grid: int) -> np.ndarray:
        """
        CLAHE operation on uint8 [0, 255] and converted back to float32 [0.0, 1.0].
        Applies equally across channels to preserve grayscale integrity.
        """
        u8 = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))

        if u8.ndim == 2:
            res = clahe.apply(u8)
            return (res.astype(np.float32) / 255.0)

        c = u8.shape[-1]
        is_replicated = (c == 3 and np.array_equal(u8[:, :, 0], u8[:, :, 1]) and np.array_equal(u8[:, :, 0], u8[:, :, 2]))
        if is_replicated or c == 1:
            ch0 = clahe.apply(u8[:, :, 0])
            res = np.repeat(ch0[:, :, np.newaxis], c, axis=-1)
        else:
            lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            res = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

        return np.clip(res.astype(np.float32) / 255.0, 0.0, 1.0)
