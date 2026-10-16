import re

with open("evaluate.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Import
import_str = "from models.model import prepare_model"
new_import_str = "from models.model import prepare_model\nfrom utils.seg_metrics import hd95, boundary_f1, contour, boundary_dice, summarize_hd95"
code = code.replace(import_str, new_import_str)

# 2. Add vars
var_str = """    tp = {1: 0, 2: 0}
    fp = {1: 0, 2: 0}
    fn = {1: 0, 2: 0}"""
new_var_str = """    tp = {1: 0, 2: 0, 'tumor': 0}
    fp = {1: 0, 2: 0, 'tumor': 0}
    fn = {1: 0, 2: 0, 'tumor': 0}
    
    hd95_vals = {1: [], 2: [], 'tumor': []}
    bf_vals = []"""
code = code.replace(var_str, new_var_str)

# 3. Inside loop
loop_calc_str = """        for c in [1, 2]:
            true_c = (y_true_cls == c)
            pred_c = (y_pred_cls == c)
            tp[c] += np.sum(true_c & pred_c)
            fp[c] += np.sum((~true_c) & pred_c)
            fn[c] += np.sum(true_c & (~pred_c))"""
new_loop_calc_str = """        
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
                    bf_vals.append(bf)"""
code = code.replace(loop_calc_str, new_loop_calc_str)

# 4. Print Report
print_str = """    print(f"   {'Class / Metric':<20} | {'Dice Score':<14} | {'IoU':<12} | {'Precision':<12} | {'Recall':<12}")
    print("-" * 88)
    print(f"   {'U Lanh (Benign)':<20} | {dice_b:<14.4f} | {iou_b:<12.4f} | {prec_b:<12.4f} | {rec_b:<12.4f}")
    print(f"   {'U Ac (Malignant)':<20} | {dice_m:<14.4f} | {iou_m:<12.4f} | {prec_m:<12.4f} | {rec_m:<12.4f}")
    print("-" * 88)
    print(f"   {'TONG HOP (OVERALL)':<20} | {dice_avg:<14.4f} | {iou_avg:<12.4f} | {prec_avg:<12.4f} | {rec_avg:<12.4f}")
    print("=" * 88 + "\\n")"""
new_print_str = """    sum_b = summarize_hd95(hd95_vals[1])
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
    print("=" * 88 + "\\n")"""
code = code.replace(print_str, new_print_str)

# 5. CSV writing
csv_hdr = """                "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                "Recall_Overall", "Recall_Benign", "Recall_Malignant"
            ])"""
new_csv_hdr = """                "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                "Recall_Overall", "Recall_Benign", "Recall_Malignant",
                "HD95_Benign", "HD95_Malignant", "HD95_Tumor", "Boundary_F1"
            ])"""
code = code.replace(csv_hdr, new_csv_hdr)

csv_row = """            f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
            f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
            f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
            f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}"
        ])"""
new_csv_row = """            f"{dice_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
            f"{iou_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
            f"{prec_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
            f"{rec_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}",
            f"{sum_b['mean']:.1f}", f"{sum_m['mean']:.1f}", f"{sum_u['mean']:.1f}",
            f"{np.mean([v['f1'] for v in bf_vals]):.4f}" if bf_vals else "N/A"
        ])"""
code = code.replace(csv_row, new_csv_row)


with open("evaluate.py", "w", encoding="utf-8") as f:
    f.write(code)
