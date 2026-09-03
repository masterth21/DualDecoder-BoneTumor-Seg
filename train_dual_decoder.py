"""
Training script for Dual-Decoder (Region + Boundary + Refinement) ResNet Architecture.
Handles boundary ground truth generation, multi-output compilation, mixed precision, and callbacks.
"""

from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig
import tensorflow as tf
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
from losses.loss import DiceCoefficient
from losses.dual_decoder_loss import get_dual_decoder_losses
from callbacks.timing_callback import TimingCallback


def create_training_folders(cfg: DictConfig):
    """Create directory structure for logs and checkpoints"""
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH))
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH))


@hydra.main(version_base=None, config_path="configs", config_name="config")
def train_dual_decoder(cfg: DictConfig):
    """Main training function for Dual-Decoder ResNet model"""
    suppress_warnings()

    # Force model type to dual_decoder_resnet if not already set
    cfg.MODEL.TYPE = "dual_decoder_resnet"

    print("\n" + "=" * 80)
    print("🎯 DUAL-DECODER (REGION + BOUNDARY + REFINEMENT) RESNET PIPELINE")
    print("=" * 80)
    print(f"Model Type: {cfg.MODEL.TYPE}")
    print(f"Backbone: {getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')}")
    print(f"Input Shape: {cfg.INPUT.HEIGHT}x{cfg.INPUT.WIDTH}x{cfg.INPUT.CHANNELS}")
    print(f"Batch Size: {cfg.HYPER_PARAMETERS.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.HYPER_PARAMETERS.LEARNING_RATE:.2e}")
    print(f"Epochs: {cfg.HYPER_PARAMETERS.EPOCHS}")
    print("=" * 80 + "\n")

    print("✓ Verifying data paths...")
    verify_data(cfg)

    if cfg.USE_MULTI_GPUS.VALUE:
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        data_generator.update_batch_size(cfg)

    create_training_folders(cfg)

    if cfg.OPTIMIZATION.AMP:
        print("✓ Enabling Automatic Mixed Precision (AMP)")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        print("✓ Enabling Accelerated Linear Algebra (XLA)")
        tf.config.optimizer.set_jit(True)

    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print(f'✓ Multi-GPU Strategy enabled across {strategy.num_replicas_in_sync} GPUs\n')
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
            if cfg.OPTIMIZATION.AMP:
                optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
            dice_coef_refined = tf.keras.metrics.MeanMetricWrapper(
                name="dice_coef",
                fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
            )
            dice_coef_region = tf.keras.metrics.MeanMetricWrapper(
                name="dice_coef",
                fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
            )
            model = prepare_model(cfg, training=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
        if cfg.OPTIMIZATION.AMP:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        dice_coef_refined = tf.keras.metrics.MeanMetricWrapper(
            name="dice_coef",
            fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
        )
        dice_coef_region = tf.keras.metrics.MeanMetricWrapper(
            name="dice_coef",
            fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
        )
        model = prepare_model(cfg, training=True)

    # Get multi-output loss dictionary and loss weights
    losses_dict, loss_weights = get_dual_decoder_losses()

    model.compile(
        optimizer=optimizer,
        loss=losses_dict,
        loss_weights=loss_weights,
        metrics={'refined_output': [dice_coef_refined], 'region_output': [dice_coef_region]}
    )

    print("\n📐 Model Summary:")
    model.summary()

    # Data Generators wrapped with DualDecoderWrapper (Ground Truth Boundary Generator)
    base_train_gen = data_generator.get_data_generator(cfg, "TRAIN", strategy)
    base_val_gen = data_generator.get_data_generator(cfg, "VAL", strategy)

    train_generator = DualDecoderWrapper(base_train_gen)
    val_generator = DualDecoderWrapper(base_val_gen)

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tb_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH, f"dual_decoder_{run_timestamp}")
    checkpoint_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH, "model_dual_decoder_resnet.hdf5")
    csv_log_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.CSV_LOGGER.PATH, f"training_logs_dual_decoder_{run_timestamp}.csv")

    evaluation_metric = "val_refined_output_dice_coef"

    timing_callback = TimingCallback()
    reduce_lr = ReduceLROnPlateau(
        monitor=evaluation_metric, factor=0.5, patience=10, min_lr=1e-7, mode='max', verbose=cfg.VERBOSE
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

    callbacks = [tensorboard_callback, early_stopping, model_checkpoint, csv_logger, timing_callback, reduce_lr]

    training_steps = data_generator.get_iterations(cfg, mode="TRAIN")
    validation_steps = data_generator.get_iterations(cfg, mode="VAL")

    print(f"Training Steps per Epoch: {training_steps} | Validation Steps: {validation_steps}\n")
    print("🚀 Starting training pipeline...\n")

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
    )

    print("\n" + "=" * 80)
    print("✅ TRAINING COMPLETED SUCCESSFULLY")
    print("=" * 80)


if __name__ == "__main__":
    train_dual_decoder()
