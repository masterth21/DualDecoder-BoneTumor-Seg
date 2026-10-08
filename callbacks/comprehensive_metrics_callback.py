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
        # Giữ on_epoch_end gọn gàng: để CompactProgressCallback in đúng 1 dòng tóm tắt.
        # Không in bảng cồng kềnh mỗi epoch để tránh rối mắt terminal.
        pass

    def on_train_end(self, logs=None):
        """
        Chỉ in bảng tổng kết toàn diện 1 LẦN DUY NHẤT khi toàn bộ quá trình train kết thúc!
        """
        print("\n" + "=" * 88)
        print("🏁 [TỔNG KẾT TOÀN DIỆN KHI KẾT THÚC HUẤN LUYỆN]")
        print(f"   Model: {self.cfg.MODEL.TYPE} ({getattr(self.cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')})")
        print("   Đang đánh giá chi tiết trên tập Validation...")
        print("=" * 88)

        tp = {1: 0, 2: 0, 'tumor': 0}
        fp = {1: 0, 2: 0, 'tumor': 0}
        fn = {1: 0, 2: 0, 'tumor': 0}
        
        hd95_vals = {1: [], 2: [], 'tumor': []}
        bf_vals = []
        
        import sys
        sys.path.append('.')
        from utils.seg_metrics import hd95, boundary_f1, contour, summarize_hd95
        
        tol = getattr(self.cfg.BOUNDARY, "EVAL_TOL", 2)

        for i in range(len(self.val_generator)):
            batch = self.val_generator[i]
            x_val, y_targets = batch[0], batch[1]

            if isinstance(y_targets, dict):
                y_true = y_targets.get('refined_output', y_targets.get('region_output'))
            elif isinstance(y_targets, (list, tuple)):
                y_true = y_targets[-1]
            else:
                y_true = y_targets

            preds = self.model(x_val, training=False)
            if not isinstance(preds, (list, tuple)):
                preds = [preds]
            out = dict(zip(self.model.output_names, preds))
            y_pred = out["refined_output"].numpy()
            
            y_pred_boundary = None
            if "boundary_output" in out:
                bp = out["boundary_output"].numpy()
                if bp.shape[-1] > 1:
                    bp_merged = np.max(bp[..., 1:], axis=-1)
                else:
                    bp_merged = bp[..., 0]
                y_pred_boundary = bp_merged > 0.5
            
            # shape (B, H, W, C)
            for b_idx in range(x_val.shape[0]):
                yt = y_true[b_idx]
                yp = y_pred[b_idx]
                
                if yt.shape[-1] > 1:
                    yt_cls = np.argmax(yt, axis=-1)
                else:
                    yt_cls = np.squeeze(yt, axis=-1).astype(int)
                
                yp_cls = np.argmax(yp, axis=-1)
                
                for c in [1, 2]:
                    true_c = (yt_cls == c)
                    pred_c = (yp_cls == c)
                    tp[c] += np.sum(true_c & pred_c)
                    fp[c] += np.sum((~true_c) & pred_c)
                    fn[c] += np.sum(true_c & (~pred_c))
                    
                    val, st = hd95(pred_c, true_c)
                    hd95_vals[c].append((val, st))
                    
                true_u = (yt_cls > 0)
                pred_u = (yp_cls > 0)
                tp['tumor'] += np.sum(true_u & pred_u)
                fp['tumor'] += np.sum((~true_u) & pred_u)
                fn['tumor'] += np.sum(true_u & (~pred_u))
                
                val, st = hd95(pred_u, true_u)
                hd95_vals['tumor'].append((val, st))
                
                if y_pred_boundary is not None:
                    yp_b = y_pred_boundary[b_idx]
                    gt_b = contour(true_u)
                    bf = boundary_f1(yp_b, gt_b, tol=tol)
                    if bf is not None:
                        bf_vals.append(bf)
                        
        eps = 1e-7
        def calc_metrics(c):
            dice = (2.0 * tp[c] + eps) / (2.0 * tp[c] + fp[c] + fn[c] + eps)
            iou = (tp[c] + eps) / (tp[c] + fp[c] + fn[c] + eps)
            prec = (tp[c] + eps) / (tp[c] + fp[c] + eps)
            rec = (tp[c] + eps) / (tp[c] + fn[c] + eps)
            return dice, iou, prec, rec

        dice_b, iou_b, prec_b, rec_b = calc_metrics(1)
        dice_m, iou_m, prec_m, rec_m = calc_metrics(2)
        dice_u, iou_u, prec_u, rec_u = calc_metrics('tumor')
        
        dice_avg = (dice_b + dice_m) / 2.0
        iou_avg = (iou_b + iou_m) / 2.0
        prec_avg = (prec_b + prec_m) / 2.0
        rec_avg = (rec_b + rec_m) / 2.0

        sum_b = summarize_hd95(hd95_vals[1])
        sum_m = summarize_hd95(hd95_vals[2])
        sum_u = summarize_hd95(hd95_vals['tumor'])

        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        print("\n" + "=" * 105)
        print(f"📊 [BẢNG TỔNG KẾT CHỈ SỐ CUỐI CÙNG] - {current_time}")
        print("-" * 105)
        print(f"   {'Class / Metric':<20} | {'Dice':<10} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10} | {'HD95 mean':<12} | {'HD95 med':<12}")
        print("-" * 105)
        print(f"   {'U Lanh (Benign)':<20} | {dice_b:<10.4f} | {iou_b:<10.4f} | {prec_b:<10.4f} | {rec_b:<10.4f} | {sum_b['mean']:<12.1f} | {sum_b['median']:<12.1f}")
        print(f"   {'U Ac (Malignant)':<20} | {dice_m:<10.4f} | {iou_m:<10.4f} | {prec_m:<10.4f} | {rec_m:<10.4f} | {sum_m['mean']:<12.1f} | {sum_m['median']:<12.1f}")
        print(f"   {'U (nhi phan)':<20} | {dice_u:<10.4f} | {iou_u:<10.4f} | {prec_u:<10.4f} | {rec_u:<10.4f} | {sum_u['mean']:<12.1f} | {sum_u['median']:<12.1f}")
        print("-" * 105)
        if y_pred_boundary is not None:
            bf1_mean = np.mean([v['f1'] for v in bf_vals]) if bf_vals else 0.0
            bp_mean = np.mean([v['precision'] for v in bf_vals]) if bf_vals else 0.0
            br_mean = np.mean([v['recall'] for v in bf_vals]) if bf_vals else 0.0
            print(f"   Boundary F1@{tol}px: {bf1_mean*100:.1f}% (P {bp_mean*100:.1f}% | R {br_mean*100:.1f}%)   |   HD95 bỏ qua: missed {sum_u['missed']}, false_pos {sum_u['false_pos']}")
        else:
            print(f"   Boundary F1: N/A   |   HD95 bỏ qua: missed {sum_u['missed']}, false_pos {sum_u['false_pos']}")
        print("=" * 105 + "\n")

        with open(self.csv_file, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                current_time, "FINAL", self.cfg.MODEL.TYPE, getattr(self.cfg.MODEL.BACKBONE, 'TYPE', 'resnet34'),
                self.cfg.DATASET.TRAIN.IMAGES_PATH, self.cfg.DATASET.VAL.IMAGES_PATH,
                "N/A", self.cfg.HYPER_PARAMETERS.BATCH_SIZE,
                "N/A", "N/A",
                f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
                f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
                f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
                f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}"
            ])
