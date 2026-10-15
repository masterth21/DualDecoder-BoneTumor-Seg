"""
Data generator
"""
import os
import tensorflow as tf
from omegaconf import DictConfig

from utils.general_utils import join_paths, get_gpus_count
from utils.boundary_utils import extract_boundary_tf
from .tf_data_generator import DataGenerator as tf_data_generator

class DualDecoderWrapper(tf.keras.utils.Sequence):
    """
    Wrapper biến đổi nhãn đơn (1 output region mask) thành 3 outputs:
    1. region_output: Ground truth region mask
    2. boundary_output: Ground truth boundary mask (tính tự động)
    3. refined_output: Ground truth region mask (tinh chỉnh)
    """
    def __init__(self, base_generator, output_names=None, kernel_size=3):
        self.base_generator = base_generator
        self.output_names = output_names
        self.kernel_size = kernel_size

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, i):
        item = self.base_generator[i]
        if isinstance(item, tuple) and len(item) == 2:
            x, y_region = item
            y_boundary = extract_boundary_tf(y_region, kernel_size=self.kernel_size)
            targets = {
                'region_output': y_region,
                'boundary_output': y_boundary,
                'refined_output': y_region
            }
            if self.output_names:
                targets = {k: v for k, v in targets.items() if k in self.output_names}
                if len(self.output_names) == 1:
                    return x, targets[self.output_names[0]]
            return x, targets
        return item

    def on_epoch_end(self):
        if hasattr(self.base_generator, 'on_epoch_end'):
            self.base_generator.on_epoch_end()


class MultiOutputWrapper(tf.keras.utils.Sequence):
    """
    Wrapper biến đổi nhãn đơn (1 output) thành nhãn đa tầng (6 outputs) cho Deep Supervision.
    """
    def __init__(self, base_generator):
        self.base_generator = base_generator

    def __len__(self):
        return len(self.base_generator)

    def __getitem__(self, i):
        item = self.base_generator[i]
        if isinstance(item, tuple) and len(item) == 2:
            x, y = item
            # Tạo nhãn Classification CGM (Batch_size, output_channels) bằng Global Max Pooling
            y_cls = tf.reduce_max(y, axis=[1, 2])
            # Trả về bộ 6 nhãn khớp với 6 đầu ra của mô hình (đều ở độ phân giải gốc 384x384 do model upsample)
            return x, (y, y, y, y, y, y_cls)
        return item

    def on_epoch_end(self):
        if hasattr(self.base_generator, 'on_epoch_end'):
            self.base_generator.on_epoch_end()



try:
    from .dali_data_generator import data_generator as dali_data_generator
except ModuleNotFoundError:
    print("NVIDIA DALI not installed, please install it."
          "\nNote: DALI is only available on Linux platform. For Window "
          "you can use TensorFlow generator for training.")


def get_data_generator(cfg: DictConfig,
                       mode: str,
                       strategy: tf.distribute.Strategy = None):
    """
    Creates and return data generator object based on given type.
    """
    # Check DALI compatibility with medical X-ray augmentation
    aug_cfg_raw = getattr(cfg, "DATA_AUGMENTATION", None)
    aug_enabled = False
    if aug_cfg_raw is not None:
        aug_enabled = bool(getattr(aug_cfg_raw, "ENABLE", True)) and str(getattr(aug_cfg_raw, "PRESET", "none")).lower() != "none"

    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR":
        print(f"Using TensorFlow generator for {mode} data")
        generator = tf_data_generator(cfg, mode)
    elif cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR":
        if aug_enabled:
            raise ValueError(
                "DALI pipeline does not support medical X-ray augmentation "
                "(elastic deformation, CLAHE, tumor loss protection). "
                "Please set DATA_GENERATOR_TYPE='TF_GENERATOR' or set DATA_AUGMENTATION.ENABLE=False."
            )
        print(f"Using NVIDIA DALI generator for {mode} data")
        if cfg.USE_MULTI_GPUS.VALUE:
            generator = dali_data_generator(cfg, mode, strategy)
        else:
            generator = dali_data_generator(cfg, mode)
    else:
        raise ValueError(
            "Wrong generator type passed."
            "\nPossible options are TF_GENERATOR and DALI_GENERATOR"
        )
    return generator


def update_batch_size(cfg: DictConfig):
    """
    Scale up batch size to multi gpus in case of TensorFlow generator.
    """
    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR" and cfg.USE_MULTI_GPUS.VALUE:
        # change batch size according to available gpus
        cfg.HYPER_PARAMETERS.BATCH_SIZE = \
            cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()


def get_batch_size(cfg: DictConfig):
    """
    Return batch size.
    In case of DALI generator scale up batch size to multi gpus.
    """
    if cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR" and cfg.USE_MULTI_GPUS.VALUE:
        # change batch size according to available gpus
        return cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count()
    else:
        return cfg.HYPER_PARAMETERS.BATCH_SIZE


def get_iterations(cfg: DictConfig, mode: str):
    """
    Return steps per epoch
    """
    images_length = len(
        os.listdir(
            join_paths(
                cfg.WORK_DIR,
                cfg.DATASET[mode].IMAGES_PATH
            )
        )
    )

    if cfg.DATA_GENERATOR_TYPE == "TF_GENERATOR":
        training_steps = images_length // cfg.HYPER_PARAMETERS.BATCH_SIZE
    elif cfg.DATA_GENERATOR_TYPE == "DALI_GENERATOR":
        if cfg.USE_MULTI_GPUS.VALUE:
            training_steps = images_length // (
                    cfg.HYPER_PARAMETERS.BATCH_SIZE * get_gpus_count())
        else:
            training_steps = images_length // cfg.HYPER_PARAMETERS.BATCH_SIZE
    else:
        raise ValueError("Wrong generator type passed.")

    return training_steps
