import os
import csv
from datetime import datetime
import numpy as np
import tensorflow as tf


class ComprehensiveMetricsCallback(tf.keras.callbacks.Callback):
    """
    Callback tuy bien:
    1. Tinh toan chi tiet sau moi epoch: Loss, Dice (Tong, Lanh, Ac), IoU, Precision, Recall.
    2. In bang chi so dep mat, truc quan ra Terminal.
    3. Ghi nhat ky day du ra file CSV kem: Thoi gian, Ten mo hinh, Backbone, Duong dan du lieu...
    """

    def __init__(self, val_generator, cfg, log_dir="checkpoint", print_table=False):
        super().__init__()
        self.val_generator = val_generator
        self.cfg = cfg
        self.log_dir = log_dir
        self.print_table = print_table
        os.makedirs(log_dir, exist_ok=True)
        
        # File log rieng cho qua trinh train
        self.csv_file = os.path.join(log_dir, f"training_detailed_logs_{cfg.MODEL.TYPE}.csv")
        self._init_csv()

    def _init_csv(self):
        file_exists = os.path.exists(self.csv_file)
        with open(self.csv_file, mode='a' if file_exists else 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp", "Epoch", "Model_Type", "Backbone",
                    "Train_Data_Path", "Val_Data_Path", "Learning_Rate", "Batch_Size",
                    "Train_Loss", "Val_Loss",
                    "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                    "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                    "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                    "Recall_Overall", "Recall_Benign", "Recall_Malignant"
                ])

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        train_loss = logs.get('loss', 0.0)
        val_loss = logs.get('val_loss', 0.0)
        
        # Khoi tao ma tran tich luy TP, FP, FN cho tung lop
        # Class 1: U Lanh (Benign), Class 2: U Ac (Malignant)
        tp = {1: 0, 2: 0}
        fp = {1: 0, 2: 0}
        fn = {1: 0, 2: 0}

        # Quet qua toan bo tap validation
        for i in range(len(self.val_generator)):
            batch = self.val_generator[i]
            x_val, y_targets = batch[0], batch[1]
            
            # Lay Ground Truth Mask (tu refined_output hoac mask truc tiep)
            if isinstance(y_targets, dict):
                y_true = y_targets.get('refined_output', y_targets.get('region_output'))
            elif isinstance(y_targets, (list, tuple)):
                y_true = y_targets[-1]
            else:
                y_true = y_targets

            # Du doan tu mo hinh
            preds = self.model.predict(x_val, verbose=0)
            if isinstance(preds, (list, tuple)):
                # Lay dau ra refined_output (dau ra thu 3 hoac cuoi cung)
                y_pred = preds[-1]
            else:
                y_pred = preds

            # Chuyen ve nhan lop (0: Nen, 1: U lanh, 2: U ac)
            if y_true.shape[-1] > 1:
                y_true_cls = np.argmax(y_true, axis=-1)
            else:
                y_true_cls = np.squeeze(y_true, axis=-1).astype(int)

            y_pred_cls = np.argmax(y_pred, axis=-1)

            # Tinh TP, FP, FN cho tung lop U
            for c in [1, 2]:
                true_c = (y_true_cls == c)
                pred_c = (y_pred_cls == c)
                tp[c] += np.sum(true_c & pred_c)
                fp[c] += np.sum((~true_c) & pred_c)
                fn[c] += np.sum(true_c & (~pred_c))

        # Tinh toan cac chi so chi tiet (kem epsilon chong chia cho 0)
        eps = 1e-7
        def calc_metrics(c):
            dice = (2.0 * tp[c] + eps) / (2.0 * tp[c] + fp[c] + fn[c] + eps)
            iou = (tp[c] + eps) / (tp[c] + fp[c] + fn[c] + eps)
            prec = (tp[c] + eps) / (tp[c] + fp[c] + eps)
            rec = (tp[c] + eps) / (tp[c] + fn[c] + eps)
            return dice, iou, prec, rec

        dice_b, iou_b, prec_b, rec_b = calc_metrics(1)  # U lanh
        dice_m, iou_m, prec_m, rec_m = calc_metrics(2)  # U ac

        # Trung binh cong 2 lop khoi u (Macro Average tren cac khoi u)
        dice_avg = (dice_b + dice_m) / 2.0
        iou_avg = (iou_b + iou_m) / 2.0
        prec_avg = (prec_b + prec_m) / 2.0
        rec_avg = (rec_b + rec_m) / 2.0

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))

        # -------------------------------------------------------------
        # 1. IN BANG CHI SO RA TERMINAL (KHI DUOC YEU CAU)
        # -------------------------------------------------------------
        if self.print_table:
            print("\n" + "=" * 88)
            print(f"[EPOCH {epoch + 1:03d} SUMMARY] - {current_time}")
            print(f"   Model: {self.cfg.MODEL.TYPE} ({getattr(self.cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')}) | LR: {lr:.2e}")
            print(f"   Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print("-" * 88)
            print(f"   {'Class / Metric':<20} | {'Dice Score':<14} | {'IoU':<12} | {'Precision':<12} | {'Recall':<12}")
            print("-" * 88)
            print(f"   {'U Lanh (Benign)':<20} | {dice_b:<14.4f} | {iou_b:<12.4f} | {prec_b:<12.4f} | {rec_b:<12.4f}")
            print(f"   {'U Ac (Malignant)':<20} | {dice_m:<14.4f} | {iou_m:<12.4f} | {prec_m:<12.4f} | {rec_m:<12.4f}")
            print("-" * 88)
            print(f"   {'TONG HOP (OVERALL)':<20} | {dice_avg:<14.4f} | {iou_avg:<12.4f} | {prec_avg:<12.4f} | {rec_avg:<12.4f}")
            print("=" * 88 + "\n")

        # -------------------------------------------------------------
        # 2. GHI DONG LOG VAO FILE CSV HUAN LUYEN
        # -------------------------------------------------------------
        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_time, epoch + 1, self.cfg.MODEL.TYPE, getattr(self.cfg.MODEL.BACKBONE, 'TYPE', 'resnet34'),
                self.cfg.DATASET.TRAIN.IMAGES_PATH, self.cfg.DATASET.VAL.IMAGES_PATH,
                f"{lr:.2e}", self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                f"{train_loss:.4f}", f"{val_loss:.4f}",
                f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
                f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
                f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
                f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}"
            ])
