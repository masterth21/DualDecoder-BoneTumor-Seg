import os
import tensorflow as tf
from omegaconf import OmegaConf
import gc
from train_dual_decoder import train_dual_decoder

def train_test(dual_decoder, gate):
    tf.keras.backend.clear_session()
    gc.collect()
    
    cfg = OmegaConf.create({
        "WORK_DIR": ".",
        "INPUT": {"HEIGHT": 384, "WIDTH": 384, "CHANNELS": 3},
        "OUTPUT": {"CLASSES": 3},
        "MODEL": {
            "TYPE": "dual_decoder_resnet",
            "DUAL_DECODER": dual_decoder,
            "BACKBONE": {"TYPE": "resnet34"},
            "DECODER": {
                "TYPE": "unet3plus",
                "CAT_CHANNELS_REGION": 64,
                "CAT_CHANNELS_BOUNDARY": 32,
            },
            "SKIP_ATTENTION": {
                "REGION": {"TYPE": gate, "STAGES": [3, 4]},
                "BOUNDARY": {"TYPE": "none", "STAGES": [3, 4]},
            }
        },
        "OPTIMIZATION": {"AMP": False, "XLA": False},
        "HYPER_PARAMETERS": {
            "EPOCHS": 2,
            "BATCH_SIZE": 8,
            "LEARNING_RATE": 1e-4,
            "GRADIENT_CLIP": 1.0,
            "WEIGHT_DECAY": 1e-5
        },
        "DATA_GENERATOR_TYPE": "TF_GENERATOR",
        "USE_MULTI_GPUS": {"VALUE": False},
        "DATA_PREPARATION": {},
        "DATASET": {
            "TRAIN": {"IMAGES_PATH": "./BTXRD/Split/train/images", "MASK_PATH": "./BTXRD/Split/train/mask"},
            "VAL": {"IMAGES_PATH": "./BTXRD/Split/val/images", "MASK_PATH": "./BTXRD/Split/val/mask"}
        },
        "PREPROCESS_DATA": {
            "RESIZE": {"VALUE": True, "HEIGHT": 384, "WIDTH": 384},
            "IMAGE_PREPROCESSING_TYPE": "normalize",
            "NORMALIZE_MASK": {"VALUE": True, "NORMALIZE_VALUE": 255},
            "SHUFFLE": {"TRAIN": {"VALUE": True}, "VAL": {"VALUE": False}},
            "AUGMENTATION": {
                "ROTATION": {"VALUE": False}, "FLIP": {"VALUE": False},
                "ELASTIC_DEFORMATION": {"VALUE": False}, "ZOOM": {"VALUE": False},
                "BRIGHTNESS": {"VALUE": False}, "CONTRAST": {"VALUE": False},
                "CLAHE": {"VALUE": False}, "GAUSSIAN_NOISE": {"VALUE": False},
                "GAUSSIAN_BLUR": {"VALUE": False}
            }
        },
        "LOSS_CONFIG": {"TYPE": "weighted_dice_loss", "WEIGHTS": [0.1, 0.9]},
        "CALLBACKS": {
            "TENSORBOARD": {"PATH": "checkpoint/tb_logs"},
            "EARLY_STOPPING": {"PATIENCE": 20, "MIN_DELTA": 1e-4},
            "MODEL_CHECKPOINT": {"PATH": "checkpoint", "SAVE_WEIGHTS_ONLY": True, "SAVE_BEST_ONLY": True},
            "CSV_LOGGER": {"PATH": "checkpoint", "APPEND_LOGS": False},
            "REDUCE_LR_ON_PLATEAU": {"PATIENCE": 8}
        },
        "VERBOSE": 1,
        "DATALOADER_WORKERS": 4,
        "SEED": 42
    })
    
    print(f"\n=======================================================")
    print(f"TRAIN TEST: DUAL_DECODER={dual_decoder}, REGION_GATE={gate}")
    print(f"=======================================================")
    
    tf.config.experimental.reset_memory_stats('GPU:0')
    try:
        train_dual_decoder(cfg)
        mem_info = tf.config.experimental.get_memory_info('GPU:0')
        print(f"PEAK VRAM: {mem_info['peak'] / (1024**2):.2f} MB")
    except Exception as e:
        print(f"TRAIN FAILED: {e}")

if __name__ == "__main__":
    train_test(True, "nalaformer")
    train_test(False, "none")
