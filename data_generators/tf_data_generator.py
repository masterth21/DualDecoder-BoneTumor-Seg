"""
Tensorflow data generator class.
"""
import tensorflow as tf
import numpy as np
from omegaconf import DictConfig
from utils.general_utils import get_data_paths
from utils.images_utils import prepare_image, prepare_mask

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, cfg: DictConfig, mode: str):
        self.cfg = cfg
        self.mode = mode
        self.batch_size = self.cfg.HYPER_PARAMETERS.BATCH_SIZE
        np.random.seed(cfg.SEED)

        self.mask_available = False if cfg.DATASET[mode].MASK_PATH is None or str(
            cfg.DATASET[mode].MASK_PATH).lower() == "none" else True

        data_paths = get_data_paths(cfg, mode, self.mask_available)

        self.images_paths = data_paths[0]
        if self.mask_available:
            self.mask_paths = data_paths[1]

        self.on_epoch_end()

    def __len__(self):
        self.on_epoch_end()
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
            (self.cfg.HYPER_PARAMETERS.BATCH_SIZE, self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.INPUT.CHANNELS)
        ).astype(np.float32)

        if self.mask_available:
            batch_masks = np.zeros(
                (self.cfg.HYPER_PARAMETERS.BATCH_SIZE, self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.OUTPUT.CLASSES)
            ).astype(np.float32)

        for i, index in enumerate(indexes):
            img_path = self.images_paths[int(index)]
            if self.mask_available:
                mask_path = self.mask_paths[int(index)]

            image = prepare_image(img_path, self.cfg.PREPROCESS_DATA.RESIZE, self.cfg.PREPROCESS_DATA.IMAGE_PREPROCESSING_TYPE)

            if self.mask_available:
                mask = prepare_mask(mask_path, self.cfg.PREPROCESS_DATA.RESIZE, self.cfg.PREPROCESS_DATA.NORMALIZE_MASK)

            if self.mask_available:
                image, mask = tf.numpy_function(self.tf_func, [image, mask], [tf.float32, tf.int32])
            else:
                image = tf.numpy_function(self.tf_func, [image, ], [tf.float32, ])

            image.set_shape([self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.INPUT.CHANNELS])

            if self.mask_available:
                if self.cfg.OUTPUT.CLASSES == 1:
                    mask = tf.expand_dims(mask, axis=-1)
                else:
                    mask = tf.one_hot(mask, self.cfg.OUTPUT.CLASSES, dtype=tf.int32)
                
                mask = tf.cast(mask, tf.float32) # Cast sang float để chuẩn bị Augmentation
                mask.set_shape([self.cfg.INPUT.HEIGHT, self.cfg.INPUT.WIDTH, self.cfg.OUTPUT.CLASSES])

            # ================= DATA AUGMENTATION (CHỈ DÀNH CHO TẬP TRAIN) =================
            if self.mode == "TRAIN":
                # Lật ngang 50%
                if np.random.rand() > 0.5:
                    image = tf.image.flip_left_right(image)
                    if self.mask_available:
                        mask = tf.image.flip_left_right(mask)
                
                # Lật dọc 50%
                if np.random.rand() > 0.5:
                    image = tf.image.flip_up_down(image)
                    if self.mask_available:
                        mask = tf.image.flip_up_down(mask)
                
                # Biến thiên độ sáng nhẹ
                image = tf.image.random_brightness(image, max_delta=0.05)
            # ==============================================================================

            batch_images[i] = image
            if self.mask_available:
                batch_masks[i] = mask

        if self.mask_available:
            return batch_images, batch_masks
        else:
            return batch_images,

    @staticmethod
    def tf_func(*args):
        return args