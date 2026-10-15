"""
Tensorflow data generator class with medical X-ray augmentation and optional malignant oversampling.
"""
import os
import tensorflow as tf
import numpy as np
import cv2
from omegaconf import DictConfig
from utils.general_utils import get_data_paths
from utils.images_utils import prepare_image, prepare_mask
from .augmentations import XrayAugmenter, build_augmentation_config


class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg: DictConfig, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        np.random.seed(cfg.SEED)

        self.mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
            cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

        data_paths = get_data_paths(cfg, mode, self.mask_available)

        self.images_paths = list(data_paths[0])
        if self.mask_available:
            self.mask_paths = list(data_paths[1])

        # Optional Malignant Oversampling for TRAIN
        oversample_factor = float(getattr(getattr(cfg, "DATA", None), "OVERSAMPLE_MALIGNANT", 1.0))
        if self.mode == "TRAIN" and oversample_factor > 1.0 and self.mask_available:
            self._apply_oversampling(oversample_factor)

        # Configure Augmentation: ONLY TRAIN mode is augmented
        aug_cfg_raw = getattr(cfg, "DATA_AUGMENTATION", None)
        if self.mode == "TRAIN" and aug_cfg_raw is not None:
            effective_aug_cfg = build_augmentation_config(aug_cfg_raw)
            if effective_aug_cfg.get("ENABLE", False):
                self.augmenter = XrayAugmenter(effective_aug_cfg, seed=cfg.SEED)
                print(f"[INFO] DataGenerator ({mode}): Augmentation ENABLED (Preset: '{effective_aug_cfg['PRESET']}')")
            else:
                self.augmenter = None
                print(f"[INFO] DataGenerator ({mode}): Augmentation DISABLED (ENABLE=False or preset=none)")
        else:
            self.augmenter = None
            if self.mode != "TRAIN":
                print(f"[INFO] DataGenerator ({mode}): Augmentation strictly DISABLED for {mode}")

        self.on_epoch_end()

    def _apply_oversampling(self, factor: float):
        """
        Oversample malignant cases (class 2) in TRAIN by replicating paths.
        """
        malignant_indices = []
        for idx, m_path in enumerate(self.mask_paths):
            if not os.path.exists(m_path):
                continue
            m = cv2.imread(m_path, cv2.IMREAD_GRAYSCALE)
            if m is not None:
                # Raw mask: malignant > 192; pre-mapped: malignant == 2
                if np.any(m > 192) or (np.any(m == 2) and not np.any(m > 2)):
                    malignant_indices.append(idx)

        if malignant_indices:
            mult = factor - 1.0
            num_full = int(np.floor(mult))
            fraction = mult - num_full
            extra_indices = []
            for _ in range(num_full):
                extra_indices.extend(malignant_indices)
            if fraction > 0:
                n_sample = int(round(len(malignant_indices) * fraction))
                if n_sample > 0:
                    rng = np.random.default_rng(self.cfg.SEED)
                    sampled = rng.choice(malignant_indices, size=n_sample, replace=False)
                    extra_indices.extend(sampled.tolist())

            for idx in extra_indices:
                self.images_paths.append(self.images_paths[idx])
                self.mask_paths.append(self.mask_paths[idx])

            print(f"[INFO] Malignant oversampling (factor={factor}): added {len(extra_indices)} samples. "
                  f"Total train images: {len(self.images_paths)}")

    def __len__(self):
        return int(np.floor(len(self.images_paths) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images_paths))
        if self.cfg.PREPROCESS_DATA.SHUFFLE[self.mode].VALUE:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        return self.__data_generation(indexes)

    def __data_generation(self, indexes):
        batch_images = np.zeros(
            (self.batch_size, self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.INPUT.CHANNELS),
            dtype=np.float32
        )

        if self.mask_available:
            batch_masks = np.zeros(
                (self.batch_size, self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.OUTPUT.CLASSES),
                dtype=np.float32
            )

        for i, index in enumerate(indexes):
            img_path = self.images_paths[int(index)]
            image = prepare_image(
                img_path,
                self.cfg.PREPROCESS_DATA.RESIZE,
                self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE
            )

            mask = None
            if self.mask_available:
                mask_path = self.mask_paths[int(index)]
                mask = prepare_mask(
                    mask_path,
                    self.cfg.PREPROCESS_DATA.RESIZE,
                    self.cfg.PREPROCESS_DATA.NORMALIZE_MASK
                )

            # ================= DATA AUGMENTATION (CHỈ DÀNH CHO TẬP TRAIN) =================
            # Augmentation runs on numpy arrays BEFORE one-hot encoding and BEFORE boundary extraction
            if self.mode == "TRAIN" and self.augmenter is not None:
                if self.mask_available:
                    image, mask = self.augmenter(image, mask)
                else:
                    image = self.augmenter(image, None)
            # ==============================================================================

            # Ensure image is in [0.0, 1.0] float32
            image = np.clip(image, 0.0, 1.0).astype(np.float32)
            batch_images[i] = image

            # Convert mask to one-hot encoding
            if self.mask_available:
                if self.cfg.OUTPUT.CLASSES == 1:
                    mask_out = np.expand_dims(mask, axis=-1).astype(np.float32)
                else:
                    # One-hot encode integer classes {0, 1, 2}
                    mask_clamped = np.clip(mask, 0, self.cfg.OUTPUT.CLASSES - 1).astype(np.int32)
                    mask_out = np.eye(self.cfg.OUTPUT.CLASSES, dtype=np.float32)[mask_clamped]

                batch_masks[i] = mask_out

        if self.mask_available:
            return batch_images, batch_masks
        else:
            return batch_images,

    @staticmethod
    def tf_func(*args):
        return args