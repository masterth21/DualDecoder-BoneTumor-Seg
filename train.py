"""
Training script
"""
import numpy as np
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
    ReduceLROnPlateau # <-- Đã thêm Scheduler tự động hãm phanh
)

from data_generators import data_generator
from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths, set_gpus, suppress_warnings
from models.model import prepare_model
from losses.loss import DiceCoefficient
from losses.unet_loss import unet3p_hybrid_loss, weighted_dice_loss # <-- Import hàm loss mới
from callbacks.timing_callback import TimingCallback

def create_training_folders(cfg: DictConfig):
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH))
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH))

def train(cfg: DictConfig):
    suppress_warnings()
    print("Verifying data ...")
    verify_data(cfg)

    if cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        raise ValueError("Model exist but training script is not supported for this variant")

    if cfg.USE_MULTI_GPUS.VALUE:
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        data_generator.update_batch_size(cfg)

    create_training_folders(cfg)

    if cfg.OPTIMIZATION.AMP:
        print("Enabling Automatic Mixed Precision(AMP) training")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        print("Enabling Accelerated Linear Algebra(XLA) training")
        tf.config.optimizer.set_jit(True)

    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
        print('Number of visible gpu devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
            if cfg.OPTIMIZATION.AMP:
                optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
            dice_coef = DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
            dice_coef = tf.keras.metrics.MeanMetricWrapper(name="dice_coef", fn=dice_coef)
            model = prepare_model(cfg, training=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
        if cfg.OPTIMIZATION.AMP:
            optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        dice_coef = DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES)
        dice_coef = tf.keras.metrics.MeanMetricWrapper(name="dice_coef", fn=dice_coef)
        model = prepare_model(cfg, training=True)

    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        from losses.dual_decoder_loss import get_dual_decoder_losses
        losses_dict, loss_weights = get_dual_decoder_losses()
        model.compile(
            optimizer=optimizer,
            loss=losses_dict,
            loss_weights=loss_weights,
            metrics={'refined_output': [dice_coef], 'region_output': [dice_coef]}
        )
    else:
        model.compile(
            optimizer=optimizer,
            loss=weighted_dice_loss, # <--- ĐÃ ĐỔI SANG DÙNG WEIGHTED LOSS
            metrics=[dice_coef],
        )
    model.summary()

    train_generator = data_generator.get_data_generator(cfg, "TRAIN", strategy)
    val_generator = data_generator.get_data_generator(cfg, "VAL", strategy)

    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        from data_generators.data_generator import DualDecoderWrapper
        train_generator = DualDecoderWrapper(train_generator)
        val_generator = DualDecoderWrapper(val_generator)


    tb_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH, "{}".format(datetime.now().strftime("%Y.%m.%d.%H.%M.%S")))
    print("TensorBoard directory\n" + tb_log_dir)

    checkpoint_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH, f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5")
    print("Weights path\n" + checkpoint_path)

    csv_log_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.CSV_LOGGER.PATH, f"training_logs_{cfg.MODEL.TYPE}.csv")
    print("Logs path\n" + csv_log_path)

    evaluation_metric = "val_dice_coef"
    if len(model.outputs) > 1:
        evaluation_metric = f"val_{model.output_names[0]}_dice_coef"

    timing_callback = TimingCallback()
    
    # <-- KHAI BÁO CƠ CHẾ TỰ ĐỘNG GIẢM LEARNING RATE
    reduce_lr = ReduceLROnPlateau(
        monitor=evaluation_metric, 
        factor=0.5,      # Giảm một nửa learning rate
        patience=6,      # Nếu 6 epoch val_dice không tăng thì giảm LR
        min_lr=1e-6,
        mode='max',
        verbose=cfg.VERBOSE
    )

    callbacks = [
        TensorBoard(log_dir=tb_log_dir, write_graph=False, profile_batch=0),
        EarlyStopping(
            patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE,
            verbose=cfg.VERBOSE,
            monitor=evaluation_metric,
            mode='max'
        ),
        ModelCheckpoint(
            checkpoint_path,
            verbose=cfg.VERBOSE,
            save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY,
            save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY,
            monitor=evaluation_metric,
            mode="max"
        ),
        CSVLogger(csv_log_path, append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS),
        timing_callback,
        reduce_lr # <--- THÊM VÀO PIPELINE
    ]

    training_steps = data_generator.get_iterations(cfg, mode="TRAIN")
    validation_steps = data_generator.get_iterations(cfg, mode="VAL")

    model.fit(
        x=train_generator,
        steps_per_epoch=training_steps,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=cfg.HYPER_PARAMETERS.EPOCHS,
        callbacks=callbacks,
        workers=cfg.DATALOADER_WORKERS,
    )

    training_time = timing_callback.train_end_time - timing_callback.train_start_time
    training_time = timedelta(seconds=training_time)
    print(f"Total training time {training_time}")

    mean_time = np.mean(timing_callback.batch_time)
    throughput = data_generator.get_batch_size(cfg) / mean_time
    print(f"Training latency: {round(mean_time * 1e3, 2)} msec")
    print(f"Training throughput/FPS: {round(throughput, 2)} samples/sec")

@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    main()