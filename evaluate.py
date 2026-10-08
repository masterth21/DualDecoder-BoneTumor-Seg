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
from utils.seg_metrics import hd95, boundary_f1, contour, boundary_dice, summarize_hd95


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

    # 2. Checkpoint path (auto-detect newest .weights.h5, .keras, or .hdf5)
    checkpoint_path = getattr(cfg, "CHECKPOINT_PATH", None)
    ckpt_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        import glob
        pattern = os.path.join(ckpt_dir, "*model*")
        found = [f for f in glob.glob(pattern) if f.endswith(('.weights.h5', '.keras', '.hdf5', '.h5'))]
        if found:
            found.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = found[0]
        else:
            checkpoint_path = join_paths(ckpt_dir, f"{cfg.MODEL.WEIGHTS_FILE_NAME}.weights.h5")

    print(f"✓ Loading model weights from: {checkpoint_path}")
    assert os.path.exists(checkpoint_path), f"Checkpoint does not exist at:\n{checkpoint_path}\nPlease train a model first!"

    # Keras 3 warm-up pass: Chạy 1 batch tensor rỗng để toàn bộ sub-layers (như Dense trong Attention) được build biến
    try:
        dummy_shape = (1, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS)
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    except Exception as e:
        print(f"[WARNING] Warm-up forward pass: {e}")

    try:
        if str(checkpoint_path).endswith(('.weights.h5', '.keras')):
            model.load_weights(checkpoint_path)
        else:
            model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    except (TypeError, ValueError):
        model.load_weights(checkpoint_path)

    # 3. Data Generator
    val_generator = data_generator.get_data_generator(cfg, "VAL", strategy=None)
    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        from data_generators.data_generator import DualDecoderWrapper
        val_generator = DualDecoderWrapper(val_generator, output_names=model.output_names)
    elif cfg.MODEL.TYPE == "unet3plus_deepsup_cgm":
        from data_generators.data_generator import MultiOutputWrapper
        val_generator = MultiOutputWrapper(val_generator)

    validation_steps = len(val_generator)
    print(f"✓ Total validation batches to evaluate: {validation_steps}\n")

    # 4. Accumulate Confusion Matrix (TP, FP, FN) for Class 1 (Benign) and Class 2 (Malignant)
    tp = {1: 0, 2: 0, 'tumor': 0}
    fp = {1: 0, 2: 0, 'tumor': 0}
    fn = {1: 0, 2: 0, 'tumor': 0}
    
    hd95_vals = {1: [], 2: [], 'tumor': []}
    bf_vals = []

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
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        out = dict(zip(model.output_names, preds))
        y_pred = out["refined_output"]

        if y_true.shape[-1] > 1:
            y_true_cls = np.argmax(y_true, axis=-1)
        else:
            y_true_cls = np.squeeze(y_true, axis=-1).astype(int)

        y_pred_cls = np.argmax(y_pred, axis=-1)

        
        y_pred_boundary = None
        if "boundary_output" in out:
            bp = out["boundary_output"]
            if bp.shape[-1] > 1:
                bp_merged = np.max(bp[..., 1:], axis=-1)
            else:
                bp_merged = bp[..., 0]
            y_pred_boundary = bp_merged > 0.5
            
        for b_idx in range(x_val.shape[0]):
            yt_cls = y_true_cls[b_idx]
            yp_cls = y_pred_cls[b_idx]
            
            for c in [1, 2]:
                true_c = (yt_cls == c)
                pred_c = (yp_cls == c)
                tp[c] += np.sum(true_c & pred_c)
                fp[c] += np.sum((~true_c) & pred_c)
                fn[c] += np.sum(true_c & (~pred_c))
                
                v, st = hd95(pred_c, true_c)
                hd95_vals[c].append((v, st))
                
            true_u = (yt_cls > 0)
            pred_u = (yp_cls > 0)
            tp['tumor'] += np.sum(true_u & pred_u)
            fp['tumor'] += np.sum((~true_u) & pred_u)
            fn['tumor'] += np.sum(true_u & (~pred_u))
            
            v, st = hd95(pred_u, true_u)
            hd95_vals['tumor'].append((v, st))
            
            if y_pred_boundary is not None:
                yp_b = y_pred_boundary[b_idx]
                gt_b = contour(true_u)
                bf = boundary_f1(yp_b, gt_b, tol=getattr(cfg.BOUNDARY, "EVAL_TOL", 2))
                if bf is not None:
                    bf_vals.append(bf)

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
    sum_b = summarize_hd95(hd95_vals[1])
    sum_m = summarize_hd95(hd95_vals[2])
    sum_u = summarize_hd95(hd95_vals['tumor'])
    
    print(f"   {'Class / Metric':<20} | {'Dice':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'HD95 mean':<12}")
    print("-" * 88)
    print(f"   {'U Lanh (Benign)':<20} | {dice_b:<10.4f} | {iou_b:<10.4f} | {prec_b:<10.4f} | {rec_b:<10.4f} | {sum_b['mean']:<12.1f}")
    print(f"   {'U Ac (Malignant)':<20} | {dice_m:<10.4f} | {iou_m:<10.4f} | {prec_m:<10.4f} | {rec_m:<10.4f} | {sum_m['mean']:<12.1f}")
    print("-" * 88)
    print(f"   {'TONG HOP (OVERALL)':<20} | {dice_avg:<10.4f} | {iou_avg:<10.4f} | {prec_avg:<10.4f} | {rec_avg:<10.4f} | {sum_u['mean']:<12.1f}")
    if bf_vals:
        bf_m = np.mean([v['f1'] for v in bf_vals])
        print(f"   Boundary F1: {bf_m:.4f}")
    print("=" * 88 + "\n")

    # 7. Write to separate evaluation log file
    skip_attn_type = getattr(cfg.MODEL, "SKIP_ATTENTION_TYPE", "None")
    skip_attn_stages = getattr(cfg.MODEL, "SKIP_ATTENTION_STAGES", [])
    if isinstance(skip_attn_stages, (list, tuple)):
        skip_stages_str = f"Stages {list(skip_attn_stages)}"
    else:
        skip_stages_str = str(skip_attn_stages)
    bottleneck_attn = getattr(cfg.MODEL, "BOTTLENECK_ATTENTION", "None")

    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        attention_summary = f"Skip: {skip_attn_type} ({skip_stages_str}) | Bottleneck: {bottleneck_attn}"
    elif cfg.MODEL.TYPE == "hybrid_transunet":
        attention_summary = "Bottleneck: MultiHeadAttention (Transformer 8 heads)"
    else:
        attention_summary = "None (Standard CNN)"

    eval_log_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH)
    os.makedirs(eval_log_dir, exist_ok=True)
    eval_csv_path = join_paths(eval_log_dir, "evaluation_detailed_logs.csv")
    file_exists = os.path.exists(eval_csv_path)

    with open(eval_csv_path, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "Model_Type", "Backbone", "Attention_Summary",
                "Skip_Attention_Type", "Skip_Attention_Stages", "Bottleneck_Attention",
                "Weights_Path", "Val_Data_Path",
                "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                "Recall_Overall", "Recall_Benign", "Recall_Malignant",
                "HD95_Benign", "HD95_Malignant", "HD95_Tumor", "Boundary_F1"
            ])
        writer.writerow([
            eval_timestamp, cfg.MODEL.TYPE, getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34'),
            attention_summary, skip_attn_type, skip_stages_str, bottleneck_attn,
            checkpoint_path, cfg.DATASET.VAL.IMAGES_PATH,
            f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
            f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
            f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
            f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}",
            f"{sum_b['mean']:.1f}", f"{sum_m['mean']:.1f}", f"{sum_u['mean']:.1f}",
            f"{np.mean([v['f1'] for v in bf_vals]):.4f}" if bf_vals else "N/A"
        ])

    print(f"✓ Saved evaluation logs to: {eval_csv_path}\n")


if __name__ == "__main__":
    evaluate()
