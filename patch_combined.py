import re

with open("create_combined_results.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Import seg_metrics
import_str = """from utils.images_utils import (
    colorize_mask, create_prediction_overlay, calculate_dice_score,
    calculate_iou_score, calculate_precision, calculate_recall
)"""
new_import_str = """from utils.images_utils import (
    colorize_mask, create_prediction_overlay, calculate_dice_score,
    calculate_iou_score, calculate_precision, calculate_recall
)
from utils.seg_metrics import hd95, boundary_f1, contour, boundary_dice, summarize_hd95"""
code = code.replace(import_str, new_import_str)

# 2. Add arrays for hd95 and bf
arr_str = """    overall_micro_recs = []

    global_tp = {1: 0, 2: 0}"""
new_arr_str = """    overall_micro_recs = []
    
    hd95_benign = []
    hd95_malignant = []
    hd95_tumor = []
    bf1_list = []
    
    global_tp = {1: 0, 2: 0}"""
code = code.replace(arr_str, new_arr_str)

# 3. Add calculation inside loop
calc_str = """        # Tính chỉ số cho ảnh hiện tại
        dice_b = calculate_dice_score((gt_mask == 1).astype(np.uint8), (pred_class == 1).astype(np.uint8))"""
new_calc_str = """        # Tính chỉ số cho ảnh hiện tại
        tol = getattr(cfg.BOUNDARY, "EVAL_TOL", 2)
        true_1 = (gt_mask == 1).astype(np.uint8)
        pred_1 = (pred_class == 1).astype(np.uint8)
        true_2 = (gt_mask == 2).astype(np.uint8)
        pred_2 = (pred_class == 2).astype(np.uint8)
        true_u = (gt_mask > 0).astype(np.uint8)
        pred_u = (pred_class > 0).astype(np.uint8)
        
        v_b, st_b = hd95(pred_1, true_1)
        v_m, st_m = hd95(pred_2, true_2)
        v_u, st_u = hd95(pred_u, true_u)
        
        hd95_benign.append((v_b, st_b))
        hd95_malignant.append((v_m, st_m))
        hd95_tumor.append((v_u, st_u))
        
        bf_res = None
        bd_dice = None
        if getattr(cfg.MODEL, "DUAL_DECODER", True):
            gt_b = contour(true_u)
            yp_b = boundary_pred_binary > 0
            bf_res = boundary_f1(yp_b, gt_b, tol=tol)
            bd_dice = boundary_dice(yp_b, gt_b)
            if bf_res is not None:
                bf1_list.append(bf_res)
        
        dice_b = calculate_dice_score(true_1, pred_1)"""
code = code.replace(calc_str, new_calc_str)

# 4. Update CSV logging for each image
csv_hdr_str = """                    "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                    "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                    "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                    "Recall_Overall", "Recall_Benign", "Recall_Malignant"
                ])"""
new_csv_hdr_str = """                    "Dice_Overall", "Dice_Benign(Lanh)", "Dice_Malignant(Ac)",
                    "IoU_Overall", "IoU_Benign(Lanh)", "IoU_Malignant(Ac)",
                    "Precision_Overall", "Precision_Benign", "Precision_Malignant",
                    "Recall_Overall", "Recall_Benign", "Recall_Malignant",
                    "HD95_Benign", "HD95_Malignant", "HD95_Tumor",
                    "Boundary_F1", "Boundary_Precision", "Boundary_Recall", "Boundary_Dice"
                ])"""
code = code.replace(csv_hdr_str, new_csv_hdr_str)

csv_row_str = """                f"{dice_micro_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
                f"{iou_micro_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
                f"{prec_micro_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
                f"{rec_micro_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}"
            ])"""
new_csv_row_str = """                f"{dice_micro_avg:.4f}", f"{dice_b:.4f}", f"{dice_m:.4f}",
                f"{iou_micro_avg:.4f}", f"{iou_b:.4f}", f"{iou_m:.4f}",
                f"{prec_micro_avg:.4f}", f"{prec_b:.4f}", f"{prec_m:.4f}",
                f"{rec_micro_avg:.4f}", f"{rec_b:.4f}", f"{rec_m:.4f}",
                f"{v_b if v_b is not None else st_b}", f"{v_m if v_m is not None else st_m}", f"{v_u if v_u is not None else st_u}",
                f"{bf_res['f1']:.4f}" if bf_res else "N/A",
                f"{bf_res['precision']:.4f}" if bf_res else "N/A",
                f"{bf_res['recall']:.4f}" if bf_res else "N/A",
                f"{bd_dice:.4f}" if bd_dice is not None else "N/A"
            ])"""
code = code.replace(csv_row_str, new_csv_row_str)

# 5. Fix Image titles
img_titles_str = """        font = ImageFont.load_default()
        
        texts = [
            f"Original Image",
            f"Ground Truth",
            f"Prediction (Dice: {dice_micro_avg:.4f})",
            f"Error Map",
            f"Boundary"
        ]"""
new_img_titles_str = """        font = ImageFont.load_default()
        
        hd_str = f"{v_u:.1f} px" if v_u is not None else "N/A"
        bf_str = f"BF1@{tol}px: {bf_res['f1']*100:.1f}%  (P {bf_res['precision']*100:.1f}% | R {bf_res['recall']*100:.1f}%)" if bf_res is not None else "N/A"
        
        texts = [
            f"Original Image",
            f"Ground Truth",
            f"Pred Mask  Dice: {dice_micro_avg:.4f} | HD95: {hd_str}",
            f"Error Map",
            f"Boundary Prediction  {bf_str}"
        ]"""
code = code.replace(img_titles_str, new_img_titles_str)

# 6. Overlay boundary GT on boundary pred
bound_draw_str = """        boundary_canvas = np.zeros_like(original)
        boundary_canvas[boundary_pred_binary > 0] = [255, 255, 0]  # Viền dự đoán màu vàng
        
        # Vẽ thêm viền GT màu xanh để so sánh nếu có
        boundary_canvas[boundary_gt_benign > 0] = [0, 255, 0]
        boundary_canvas[boundary_gt_malignant > 0] = [255, 0, 0]"""
new_bound_draw_str = """        boundary_canvas = np.zeros_like(original)
        # GT boundary (contour) = white transparent
        gt_contour = contour(gt_mask > 0)
        boundary_canvas[gt_contour] = [180, 180, 180]  # trắng mờ
        # Pred boundary = blue
        boundary_canvas[boundary_pred_binary > 0] = [0, 0, 255]"""
code = code.replace(bound_draw_str, new_bound_draw_str)

# 7. Final Summary Text
final_summary_str = """        f.write(f"  U Lành: {np.mean(benign_precs):.4f}\\n")
        f.write(f"  U Ác:   {np.mean(malignant_precs):.4f}\\n\\n")

        f.write(f"Recall trung bình:\\n")"""
new_final_summary_str = """        f.write(f"  U Lành: {np.mean(benign_precs):.4f}\\n")
        f.write(f"  U Ác:   {np.mean(malignant_precs):.4f}\\n\\n")
        
        sum_b = summarize_hd95(hd95_benign)
        sum_m = summarize_hd95(hd95_malignant)
        sum_u = summarize_hd95(hd95_tumor)
        
        f.write(f"HD95 (mean / median):\\n")
        f.write(f"  Tổng hợp: {sum_u['mean']:.1f} / {sum_u['median']:.1f} px\\n")
        f.write(f"  U Lành:   {sum_b['mean']:.1f} / {sum_b['median']:.1f} px\\n")
        f.write(f"  U Ác:     {sum_m['mean']:.1f} / {sum_m['median']:.1f} px\\n\\n")
        
        bf1_m = np.mean([v['f1'] for v in bf1_list]) if bf1_list else 0.0
        f.write(f"Boundary F1: {bf1_m:.4f}\\n\\n")

        f.write(f"Recall trung bình:\\n")"""
code = code.replace(final_summary_str, new_final_summary_str)

history_hdr_str = """                        "Dice_Micro", "Dice_Benign", "Dice_Malignant",
                        "IoU_Micro", "IoU_Benign", "IoU_Malignant",
                        "Prec_Micro", "Prec_Benign", "Prec_Malignant",
                        "Rec_Micro", "Rec_Benign", "Rec_Malignant"
                    ])"""
new_history_hdr_str = """                        "Dice_Micro", "Dice_Benign", "Dice_Malignant",
                        "IoU_Micro", "IoU_Benign", "IoU_Malignant",
                        "Prec_Micro", "Prec_Benign", "Prec_Malignant",
                        "Rec_Micro", "Rec_Benign", "Rec_Malignant",
                        "HD95_Micro", "HD95_Benign", "HD95_Malignant", "Boundary_F1"
                    ])"""
code = code.replace(history_hdr_str, new_history_hdr_str)

history_row_str = """                    f"{np.mean(overall_micro_precs):.4f}", f"{np.mean(benign_precs):.4f}", f"{np.mean(malignant_precs):.4f}",
                    f"{np.mean(overall_micro_recs):.4f}", f"{np.mean(benign_recs):.4f}", f"{np.mean(malignant_recs):.4f}"
                ])"""
new_history_row_str = """                    f"{np.mean(overall_micro_precs):.4f}", f"{np.mean(benign_precs):.4f}", f"{np.mean(malignant_precs):.4f}",
                    f"{np.mean(overall_micro_recs):.4f}", f"{np.mean(benign_recs):.4f}", f"{np.mean(malignant_recs):.4f}",
                    f"{sum_u['mean']:.1f}", f"{sum_b['mean']:.1f}", f"{sum_m['mean']:.1f}", f"{bf1_m:.4f}"
                ])"""
code = code.replace(history_row_str, new_history_row_str)


with open("create_combined_results.py", "w", encoding="utf-8") as f:
    f.write(code)
