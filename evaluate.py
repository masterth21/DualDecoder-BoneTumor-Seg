"""
Evaluation script used to calculate comprehensive accuracy of trained model:
Dice (Overall, Benign, Malignant), IoU, Precision, Recall, Loss.
Prints formatted summary table to terminal and logs results to a separate CSV.
"""
import os
import csv
from datetime import datetime
import numpy as np
import hydra
from omegaconf import DictConfig
import tensorflow as tf
from tensorflow.keras import mixed_precision

from data_generators import data_generator
from utils.general_utils import join_paths, set_gpus, suppress_warnings
from models.model import prepare_model


@hydra.main(version_base=None, config_path="configs", config_name="config")
def evaluate(cfg: DictConfig):
    suppress_warnings()

    print("\n" + "=" * 88)
    print("🔍 COMPREHENSIVE MODEL EVALUATION")
    print("=" * 88)
    print(f"Model Type: {cfg.MODEL.TYPE}")
    print(f"Backbone: {getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')}")
    print(f"Input Shape: {cfg.INPUT.HEIGHT}x{cfg.INPUT.WIDTH}x{cfg.INPUT.CHANNELS}")
    print(f"Classes: {cfg.OUTPUT.CLASSES} (0: Background, 1: Benign, 2: Malignant)")
    print("=" * 88 + "\n")

    if cfg.OPTIMIZATION.AMP:
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if cfg.OPTIMIZATION.XLA:
        tf.config.optimizer.set_jit(True)

    # 1. Build model
    model = prepare_model(cfg, training=False)

    # 2. Checkpoint path
    checkpoint_path = join_paths(
        cfg.WORK_DIR,
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH,
        f"{cfg.MODEL.WEIGHTS_FILE_NAME}.hdf5"
    )
    if not os.path.exists(checkpoint_path):
        # Fallback to standard dual decoder name if weights file name differs
        alt_path = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH, "model_dual_decoder_resnet.hdf5")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path

    print(f"✓ Loading model weights from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint does not exist at:\n{checkpoint_path}"
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)

    # 3. Data Generator
    val_generator = data_generator.get_data_generator(cfg, "VAL", strategy=None)
    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        from data_generators.data_generator import DualDecoderWrapper
        val_generator = DualDecoderWrapper(val_generator)
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        from data_generators.data_generator import MultiOutputWrapper
        val_generator = MultiOutputWrapper(val_generator)

    validation_steps = len(val_generator)
    print(f"✓ Total validation batches to evaluate: {validation_steps}\n")

    # 4. Accumulate Confusion Matrix (TP, FP, FN) for Class 1 (Benign) and Class 2 (Malignant)
    tp = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}

    print("⏳ Running inference on validation dataset...")
    for i in range(validation_steps):
        batch = val_generator[i]
        x_val, y_targets = batch[0], batch[1]

        if isinstance(y_targets, dict):
            y_true = y_targets.get('refined_output', y_targets.get('region_output'))
        elif isinstance(y_targets, (list, tuple)):
            y_true = y_targets[-1]
        else:
            y_true = y_targets

        preds = model.predict(x_val, verbose=0)
        if isinstance(preds, (list, tuple)):
            y_pred = preds[-1]
        else:
            y_pred = preds

        if y_true.shape[-1] > 1:
            y_true_cls = np.argmax(y_true, axis=-1)
        else:
            y_true_cls = np.squeeze(y_true, axis=-1).astype(int)

        y_pred_cls = np.argmax(y_pred, axis=-1)

        for c in [1, 2]:
            true_c = (y_true_cls == c)
            pred_c = (y_pred_cls == c)
            tp[c] += np.sum(true_c & pred_c)
            fp[c] += np.sum((~true_c) & pred_c)
            fn[c] += np.sum(true_c & (~pred_c))

    # 5. Compute metrics
    eps = 1e-7
    def calc_metrics(c):
        dice = (2.0 * tp[c] + eps) / (2.0 * tp[c] + fp[c] + fn[c] + eps)
        iou = (tp[c] + eps) / (tp[c] + fp[c] + fn[c] + eps)
        prec = (tp[c] + eps) / (tp[c] + fp[c] + eps)
        rec = (tp[c] + eps) / (tp[c] + fn[c] + eps)
        return dice, iou, prec, rec

    dice_b, iou_b, prec_b, rec_b = calc_metrics(1)  # Benign
    dice_m, iou_m, prec_m, rec_m = calc_metrics(2)  # Malignant

    dice_avg = (dice_b + dice_m) / 2.0
    iou_avg = (iou_b + iou_m) / 2.0
    prec_avg = (prec_b + prec_m) / 2.0
    rec_avg = (rec_b + rec_m) / 2.0

    eval_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 6. Print Report to Terminal
    print("\n" + "=" * 88)
    print(f"🎯 [EVALUATION REPORT / KET QUA DANH GIA] - {eval_timestamp}")
    print(f"   Model: {cfg.MODEL.TYPE} | Backbone: {getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')}")
    print(f"   Weights: {checkpoint_path}")
    print(f"   Data: {cfg.DATASET.VAL.IMAGES_PATH}")
    print("-" * 88)
    print(f"   {'Class / Metric':<20} | {'Dice Score':<14} | {'IoU':<12} | {'Precision':<12} | {'Recall':<12}")
    print("-" * 88)
    print(f"   {'U Lanh (Benign)':<20} | {dice_b:<14.4f} | {iou_b:<12.4f} | {prec_b:<12.4f} | {rec_b:<12.4f}")
    print(f"   {'U Ac (Malignant)':<20} | {dice_m:<14.4f} | {iou_m:<12.4f} | {prec_m:<12.4f} | {rec_m:<12.4f}")
    print("-" * 88)
    print(f"   {'TONG HOP (OVERALL)':<20} | {dice_avg:<14.4f} | {iou_avg:<12.4f} | {prec_avg:<12.4f} | {rec_avg:<12.4f}")
    print("=" * 88 + "\n")

    # 7. Write to separate evaluation log file
    eval_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH)
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_csv_path = join_paths(eval_log_dir, "evaluation_detailed_logs.csv")
    file_exists = os.path.exists(eval_csv_path)

    with open(eval_csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model_Type", "Backbone", "Weights_Path", "Val_Data_Path",
                "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                "Recall_Overall", "Recall_Benign", "Recall_Malignant"
            ])
        writer.writerow([
            eval_timestamp, cfg.MODEL.TYPE, getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34'),
            checkpoint_path, cfg.DATASET.VAL.IMAGES_PATH,
            f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
            f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
            f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
            f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}"
        ])

    print(f"✓ Saved evaluation logs to: {eval_csv_path}\n")


if __name__ == "__main__":
    evaluate()
