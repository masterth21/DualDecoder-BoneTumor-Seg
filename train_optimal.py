"""
Script huấn luyện tối ưu cho U-Net++ và UNet 3+ DeepSup CGM
Hỗ trợ Multi-Output Loss và tự động Scale Nhãn
"""

import numpy as np
from datetime import datetime, timedelta
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import mixed_precision
from losses.unet_loss import hybrid_boundary_loss, hybrid_abedice_loss, paper2_cedice_loss, paper3_point_dice_loss, classification_loss
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
    CSVLogger,
    ReduceLROnPlateau
)

from data_generators import data_generator
from data_generators.data_generator import MultiOutputWrapper
from data_preparation.verify_data import verify_data
from utils.general_utils import create_directory, join_paths, set_gpus, suppress_warnings
from models.model import prepare_model
from losses.loss import DiceCoefficient,weighted_dice_loss,bmt_boundary_aware_loss
# from losses.unet_loss import weighted_dice_loss
from callbacks.timing_callback import TimingCallback


def create_training_folders(cfg: DictConfig):
    """Tạo các thư mục cần thiết cho huấn luyện"""
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH))
    create_directory(join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH))


class WarmupCallback(tf.keras.callbacks.Callback):
    """Callback khởi động learning rate"""
    def __init__(self, initial_lr, target_lr, warmup_epochs):
        super().__init__()
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr + (self.target_lr - self.initial_lr) * (epoch / self.warmup_epochs)
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
            print(f"🔄 Epoch Khởi động {epoch + 1}/{self.warmup_epochs}, LR: {lr:.2e}")


# MultiOutputWrapper is now imported from data_generators.data_generator


def train_optimal(cfg: DictConfig):
    """Hàm huấn luyện được tối ưu hóa"""
    suppress_warnings()
    
    print("\n" + "="*80)
    print("🎯 PIPELINE HUẤN LUYỆN TỐI ƯU")
    print("="*80)
    print(f"Loại Mô Hình: {cfg.MODEL.TYPE}")
    print(f"Backbone: {cfg.MODEL.BACKBONE.TYPE}")
    print(f"Kích thước Đầu vào: {cfg.INPUT.HEIGHT}x{cfg.INPUT.WIDTH}")
    print(f"Batch Size: {cfg.HYPER_PARAMETERS.BATCH_SIZE}")
    print(f"Learning Rate: {cfg.HYPER_PARAMETERS.LEARNING_RATE:.2e}")
    print(f"Số Epoch: {cfg.HYPER_PARAMETERS.EPOCHS}")
    print("="*80 + "\n")
    
    print("✓ Đang xác minh dữ liệu ...")
    verify_data(cfg)
    
    if cfg.USE_MULTI_GPUS.VALUE:
        set_gpus(cfg.USE_MULTI_GPUS.GPU_IDS)
        data_generator.update_batch_size(cfg)
    
    create_training_folders(cfg)
    
    if cfg.OPTIMIZATION.AMP:
        print("✓ Bật Automatic Mixed Precision (AMP) training")
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
    
    if cfg.OPTIMIZATION.XLA:
        print("✓ Bật Accelerated Linear Algebra (XLA) training")
        tf.config.optimizer.set_jit(True)
    
    strategy = None
    if cfg.USE_MULTI_GPUS.VALUE:
        strategy = tf.distribute.MirroredStrategy(
            cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
        )
        print(f'✓ Số GPU khả dụng: {strategy.num_replicas_in_sync}\n')
        with strategy.scope():
            optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
            if cfg.OPTIMIZATION.AMP: optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
            dice_coef = tf.keras.metrics.MeanMetricWrapper(name="dice_coef", fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES))
            model = prepare_model(cfg, training=True)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate=cfg.HYPER_PARAMETERS.LEARNING_RATE)
        if cfg.OPTIMIZATION.AMP: optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
        dice_coef = tf.keras.metrics.MeanMetricWrapper(name="dice_coef", fn=DiceCoefficient(post_processed=True, classes=cfg.OUTPUT.CLASSES))
        model = prepare_model(cfg, training=True)
    
    # === THIẾT LẬP LOSS MỚI (BOUNDARY DOU LOSS) ===
    if cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        print("✓ Kích hoạt hệ thống Loss đa tầng cho Deep Supervision CGM")
        losses = [
            paper2_cedice_loss,  # d1
            paper2_cedice_loss,  # d2
            paper2_cedice_loss,  # d3
            paper2_cedice_loss,  # d4
            paper2_cedice_loss,  # e5
            classification_loss  # cls
        ]
        loss_weights = [1.0, 0.5, 0.5, 0.5, 0.5, 0.1]
        model.compile(
            optimizer=optimizer,
            loss=losses,
            loss_weights=loss_weights,
            metrics={model.output_names[0]: [dice_coef]}
        )
    else:
        print("✓ Kích hoạt hàm CE-pp2 Loss từ bài báo khoa học")
        model.compile(
            optimizer=optimizer, 
            loss=paper2_cedice_loss, 
            metrics=[dice_coef]
        )
    
    print("\n📐 Kiến trúc Mô Hình:")
    model.summary()
    
    # === TẠO VÀ WRAP DATA GENERATOR ===
    train_generator = data_generator.get_data_generator(cfg, "TRAIN", strategy)
    val_generator = data_generator.get_data_generator(cfg, "VAL", strategy)
    
    if cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        train_generator = MultiOutputWrapper(train_generator)
        val_generator = MultiOutputWrapper(val_generator)
    
    # 1. Lấy mốc thời gian thực ngay lúc bấm Train (Định dạng: NămThángNgày_GiờPhútGiây)
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    tb_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.TENSORBOARD.PATH, f"{run_timestamp}")
    checkpoint_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH, f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5")
    
    # 2. TẠO TÊN FILE CSV ĐỘC NHẤT (VD: training_logs_unet_plus_plus_20260716_105845.csv)
    csv_log_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.CSV_LOGGER.PATH, f"training_logs_{cfg.MODEL.TYPE}_{run_timestamp}.csv")
    
    evaluation_metric = "val_dice_coef"
    if len(model.outputs) > 1:
        evaluation_metric = f"val_{model.output_names[0]}_dice_coef"
    
    timing_callback = TimingCallback()
    warmup_callback = WarmupCallback(initial_lr=cfg.HYPER_PARAMETERS.LEARNING_RATE / 10, target_lr=cfg.HYPER_PARAMETERS.LEARNING_RATE, warmup_epochs=5)
    reduce_lr = ReduceLROnPlateau(monitor=evaluation_metric, factor=0.5, patience=10, min_lr=1e-7, mode='max', verbose=cfg.VERBOSE)
    tensorboard_callback = TensorBoard(log_dir=tb_log_dir, write_graph=False, profile_batch=0, update_freq='epoch')
    early_stopping = EarlyStopping(patience=cfg.CALLBACKS.EARLY_STOPPING.PATIENCE, verbose=cfg.VERBOSE, monitor=evaluation_metric, mode='max', restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(checkpoint_path, verbose=cfg.VERBOSE, save_weights_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_WEIGHTS_ONLY, save_best_only=cfg.CALLBACKS.MODEL_CHECKPOINT.SAVE_BEST_ONLY, monitor=evaluation_metric, mode="max")
    csv_logger = CSVLogger(csv_log_path, append=cfg.CALLBACKS.CSV_LOGGER.APPEND_LOGS)
    
    callbacks = [tensorboard_callback, early_stopping, model_checkpoint, csv_logger, timing_callback, warmup_callback, reduce_lr]
    
    training_steps = data_generator.get_iterations(cfg, mode="TRAIN")
    validation_steps = data_generator.get_iterations(cfg, mode="VAL")
    
    print(f"Bước Huấn Luyện: {training_steps} | Bước Xác Thực: {validation_steps}\n")
    print("🚀 Bắt đầu huấn luyện...\n")
    
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
    
    training_time = timedelta(seconds=timing_callback.train_end_time - timing_callback.train_start_time)
    print("\n" + "="*80)
    print("✅ HUẤN LUYỆN HOÀN TẤT")
    print("="*80)
    print(f"⏱️  Tổng thời gian huấn luyện: {training_time}")
    
    mean_time = np.mean(timing_callback.batch_time)
    throughput = data_generator.get_batch_size(cfg) / mean_time
    print(f"⚡ Độ trễ huấn luyện: {round(mean_time * 1e3, 2)} msec")
    print(f"📈 Thông lượng huấn luyện: {round(throughput, 2)} samples/sec")
    
    if hasattr(history, 'history'):
        best_val_dice = np.max(history.history.get(evaluation_metric, [0]))
        print(f"\n🏆 Dice Score Validation Tốt Nhất: {best_val_dice:.4f}")
    
    print("="*80 + "\n")


@hydra.main(version_base=None, config_path="configs", config_name="optimal_config")
def main(cfg: DictConfig):
    train_optimal(cfg)

if __name__ == "__main__":
    main()