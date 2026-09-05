"""
Training script for Dual-Decoder (Region + Boundary + Refinement) ResNet Architecture.
Handles boundary ground truth generation, multi-output compilation, mixed precision, and callbacks.
"""


import os
os.environ["TF_ENABLE_GPU_GARBAGE_COLLECTION"] = "false"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
from datetime import datetime, timedelta
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
import hydra
from omegaconf import DictConfig
import tensorflow as tf
try:
    for _gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau
)

from data_generators import data_generator
from data_generators.data_generator import DualDecoderWrapper
from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths, set_gpus, suppress_warnings
from models.model import prepare_model
from losses.loss import MacroDiceMetric, DiceCoefficient
from losses.dual_decoder_loss import get_dual_decoder_losses
from callbacks.timing_callback import TimingCallback
from callbacks.comprehensive_metrics_callback import ComprehensiveMetricsCallback
from callbacks.progress_callback import CompactProgressCallback



def create_training_folders(cfg: DictConfig):
    """Create directory structure for logs and checkpoints"""
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH))
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_dual_decoder(cfg: DictConfig):
    """Main training function for Dual-Decoder ResNet model"""
    # Check and configure GPU devices
    physical_gpus = tf.config.list_physical_devices('GPU')
    if physical_gpus:
        print(f"[INFO] Detected {len(physical_gpus)} GPU device(s):")
        for gpu in physical_gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"   -> {gpu.name} (Memory Growth: ON)")
            except Exception:
                pass
    else:
        print("[WARNING] No GPU detected! Running on CPU.")

    # Force model type to dual_decoder_resnet if not already set
    cfg.MODEL.TYPE = "dual_decoder_resnet"

    print("\n" + "=" * 80)
    print("[START] DUAL-DECODER (REGION + BOUNDARY + REFINEMENT) RESNET PIPELINE")
    print("=" * 80)
    print(f"Model Type: {cfg.MODEL.TYPE}")
    print(f"Backbone: {getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')}")
    print(f"Input Shape: {cfg.INPUT.HEIGHT}x{cfg.INPUT.WIDTH}x{cfg.INPUT.CHANNELS}")
    print(f"Batch Size: {cfg.HYPER_PARAMETERS.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.HYPER_PARAMETERS.LEARNING_RATE:.2e}")
    print(f"Epochs: {cfg.HYPER_PARAMETERS.EPOCHS}")
    print("=" * 80 + "\n")

    print("[INFO] Verifying data paths...")
    verify_data(cfg)

    if cfg.USE_MULTI_GPUS.VALUE:
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        data_generator.update_batch_size(cfg)

    create_training_folders(cfg)

    if cfg.OPTIMIZATION.AMP:
        print("[INFO] Enabling Automatic Mixed Precision (AMP)")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        print("[INFO] Enabling Accelerated Linear Algebra (XLA)")
        tf.config.optimizer.set_jit(True)

    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print(f'[INFO] Multi-GPU Strategy enabled across {strategy.num_replicas_in_sync} GPUs\n')
        with strategy.scope():
            clipnorm_val = float(getattr(cfg.HYPER_PARAMETERS, 'GRADIENT_CLIP', 1.0))
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE,
                clipnorm=clipnorm_val
            )
            if cfg.OPTIMIZATION.AMP:
                optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
            dice_coef_refined = MacroDiceMetric(classes=cfg.OUTPUT.CLASSES, name="dice_coef")
            dice_coef_region = MacroDiceMetric(classes=cfg.OUTPUT.CLASSES, name="dice_coef")
            model = prepare_model(cfg, training=True)
    else:
        clipnorm_val = float(getattr(cfg.HYPER_PARAMETERS, 'GRADIENT_CLIP', 1.0))
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE,
            clipnorm=clipnorm_val
        )
        if cfg.OPTIMIZATION.AMP:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        dice_coef_refined = MacroDiceMetric(classes=cfg.OUTPUT.CLASSES, name="dice_coef")
        dice_coef_region = MacroDiceMetric(classes=cfg.OUTPUT.CLASSES, name="dice_coef")
        model = prepare_model(cfg, training=True)

    # Get multi-output loss dictionary and loss weights
    losses_dict, loss_weights = get_dual_decoder_losses()

    model.compile(
        optimizer=optimizer,
        loss=losses_dict,
        loss_weights=loss_weights,
        metrics={'refined_output': [dice_coef_refined], 'region_output': [dice_coef_region]}
    )

    print("\n[INFO] Model Summary:")
    model.summary()

    # Data Generators wrapped with DualDecoderWrapper (Ground Truth Boundary Generator)
    base_train_gen = data_generator.get_data_generator(cfg, "TRAIN", strategy)
    base_val_gen = data_generator.get_data_generator(cfg, "VAL", strategy)

    train_generator = DualDecoderWrapper(base_train_gen)
    val_generator = DualDecoderWrapper(base_val_gen)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH, f"dual_decoder_{run_timestamp}")
    ckpt_ext = ".weights.h5" if cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY else ".keras"
    weights_name = getattr(cfg.MODEL, "WEIGHTS_FILE_NAME", "model_dual_decoder_resnet")
    checkpoint_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH, f"{weights_name}_{run_timestamp}{ckpt_ext}")
    print(f"[INFO] Model checkpoint will be saved to: {checkpoint_path}")

    csv_log_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.CSV_LOGGER.PATH, f"training_logs_dual_decoder_{run_timestamp}.csv")

    evaluation_metric = "val_refined_output_dice_coef"

    timing_callback = TimingCallback()
    reduce_lr_patience = getattr(cfg.CALLBACKS.REDUCE_LR_ON_PLATEAU, "PATIENCE", 15)
    reduce_lr = ReduceLROnPlateau(
        monitor=evaluation_metric, factor=0.5, patience=reduce_lr_patience, min_lr=1e-7, mode='max', verbose=cfg.VERBOSE
    )
    tensorboard_callback = TensorBoard(
        log_dir=tb_log_dir, write_graph=False, profile_batch=0, update_freq='epoch'
    )
    early_stopping = EarlyStopping(
        patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE, verbose=cfg.VERBOSE, monitor=evaluation_metric, mode='max', restore_best_weights=True
    )
    model_checkpoint = ModelCheckpoint(
        checkpoint_path, verbose=cfg.VERBOSE, save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY, save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY, monitor=evaluation_metric, mode="max"
    )
    csv_logger = CSVLogger(csv_log_path, append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS)

    detailed_metrics = ComprehensiveMetricsCallback(
        val_generator=val_generator,
        cfg=cfg,
        log_dir=join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH),
        print_table=False
    )

    training_steps = data_generator.get_iterations(cfg, mode="TRAIN")
    validation_steps = data_generator.get_iterations(cfg, mode="VAL")

    compact_progress = CompactProgressCallback(
        total_steps=training_steps,
        epochs=cfg.HYPER_PARAMETERS.EPOCHS
    )

    callbacks = [
        compact_progress,
        early_stopping,
        model_checkpoint,
        csv_logger,
        timing_callback,
        reduce_lr,
        detailed_metrics,
        tensorboard_callback
    ]

    print(f"Training Steps per Epoch: {training_steps} | Validation Steps: {validation_steps}\n")
    print("[INFO] Starting training pipeline...\n")

    history = model.fit(
        x=train_generator,
        steps_per_epoch=training_steps,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=cfg.HYPER_PARAMETERS.EPOCHS,
        callbacks=callbacks,
        workers=cfg.DATALOADER_WORKERS,
        max_queue_size=10,
        use_multiprocessing=False,
        verbose=0,
    )

    print("\n" + "=" * 80)
    print("[SUCCESS] TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)



if __name__ == "__main__":
    train_dual_decoder()
