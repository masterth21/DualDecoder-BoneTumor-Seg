import os
import csv
import glob
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig
from models.model import prepare_model
from utils.general_utils import join_paths


def calculate_dice(y_true, y_pred, classes=[1, 2]):
    """
    Tính Dice Score cho các lớp khối u (1: U lành, 2: U ác).
    Chỉ tính cho các lớp thực sự xuất hiện trong Ground Truth của ảnh đó,
    tránh trường hợp lớp không có bị tính 0.0 làm chia đôi oan uổng điểm toàn ảnh.
    """
    dice_scores = []
    for cls in classes:
        true_mask = (y_true == cls).astype(np.float32)
        pred_mask = (y_pred == cls).astype(np.float32)

        # Nếu lớp này không hề xuất hiện trong Ground Truth của bác sĩ thì bỏ qua
        if np.sum(true_mask) == 0:
            continue

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)

        dice_scores.append((2.0 * intersection) / (union + 1e-7))

    return np.mean(dice_scores) if len(dice_scores) > 0 else 1.0


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ================== CẤU HÌNH ĐƯỜNG DẪN TẬP ĐÁNH GIÁ (TEST / VAL) ==================
    # Ưu tiên lấy từ TEST_IMAGES_DIR / VAL_IMAGES_DIR (nếu truyền qua CLI)
    # hoặc DATASET.TEST (nếu có trong config), nếu không có sẽ lấy DATASET.VAL
    test_images_dir = getattr(cfg, "TEST_IMAGES_DIR", getattr(cfg, "VAL_IMAGES_DIR", None))
    test_masks_dir = getattr(cfg, "TEST_MASKS_DIR", getattr(cfg, "VAL_MASKS_DIR", None))

    if not test_images_dir:
        if hasattr(cfg.DATASET, "TEST") and getattr(cfg.DATASET.TEST, "IMAGES_PATH", None):
            test_images_dir = cfg.DATASET.TEST.IMAGES_PATH
            test_masks_dir = cfg.DATASET.TEST.MASK_PATH
        else:
            test_images_dir = cfg.DATASET.VAL.IMAGES_PATH
            test_masks_dir = cfg.DATASET.VAL.MASK_PATH

    val_images_dir = test_images_dir
    val_masks_dir = test_masks_dir

    # Tự động xác định thư mục boundary mask (nằm cạnh thư mục mask)
    val_boundary_dir = os.path.join(os.path.dirname(val_masks_dir), "boundary")

    if not os.path.isabs(val_images_dir):
        val_images_dir = join_paths(cfg.WORK_DIR, val_images_dir)
    if not os.path.isabs(val_masks_dir):
        val_masks_dir = join_paths(cfg.WORK_DIR, val_masks_dir)

    # Thư mục xuất kết quả: Mặc định là outputs/prediction_results hoặc truyền qua CLI: OUTPUT_DIR="duong_dan"
    output_dir = getattr(cfg, "OUTPUT_DIR", os.path.join(cfg.WORK_DIR, "outputs", "prediction_results"))
    masks_output_dir = os.path.join(output_dir, "predicted_masks")
    combined_output_dir = os.path.join(output_dir, "combined_plots")

    os.makedirs(masks_output_dir, exist_ok=True)
    os.makedirs(combined_output_dir, exist_ok=True)

    input_size = (cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH)

    # ================== TỰ ĐỘNG TÌM FILE WEIGHTS MỚI NHẤT ==================
    checkpoint_path = getattr(cfg, "CHECKPOINT_PATH", None)
    ckpt_dir = join_paths(cfg.WORK_DIR, cfg.CALLBACKS.MODEL_CHECKPOINT.PATH)

    if not checkpoint_path or not os.path.exists(checkpoint_path):
        pattern = os.path.join(ckpt_dir, "*model*")
        found = [f for f in glob.glob(pattern) if f.endswith(('.weights.h5', '.keras', '.hdf5', '.h5'))]
        if found:
            found.sort(key=os.path.getmtime, reverse=True)
            checkpoint_path = found[0]
        else:
            checkpoint_path = join_paths(ckpt_dir, f"{cfg.MODEL.WEIGHTS_FILE_NAME}.weights.h5")

    # Kiểm tra thư mục boundary tồn tại
    has_boundary_gt = os.path.isdir(val_boundary_dir)

    print("\n" + "=" * 80)
    print("🎨 TẠO MASK DỰ ĐOÁN VÀ ẢNH GHÉP TRỰC QUAN (COMBINED RESULTS — 5 CỘT)")
    print("=" * 80)
    print(f"Mô hình: {cfg.MODEL.TYPE}")
    print(f"Kích thước ảnh: {input_size[0]}x{input_size[1]}")
    print(f"Weights được nạp: {checkpoint_path}")
    print(f"Thư mục ảnh gốc: {val_images_dir}")
    print(f"Thư mục mask gốc: {val_masks_dir}")
    print(f"Thư mục boundary GT: {val_boundary_dir} ({'✅ Tồn tại' if has_boundary_gt else '⚠️ Chưa có — sẽ tạo on-the-fly'})")
    print(f"📁 Thư mục lưu Mask dự đoán: {masks_output_dir}")
    print(f"📁 Thư mục lưu Ảnh ghép trực quan: {combined_output_dir}")
    print("=" * 80 + "\n")

    assert os.path.exists(checkpoint_path), \
        f"Lỗi: Không tìm thấy file trọng số tại {checkpoint_path}! Vui lòng huấn luyện mô hình trước."

    print("Đang khởi tạo cấu trúc mô hình từ config...")
    model = prepare_model(cfg, training=False)

    # Keras 3 warm-up pass: Chạy 1 batch tensor rỗng để toàn bộ sub-layers (như Dense trong Attention) được build biến
    try:
        dummy_shape = (1, cfg.INPUT.HEIGHT, cfg.INPUT.WIDTH, cfg.INPUT.CHANNELS)
        _ = model(tf.zeros(dummy_shape, dtype=tf.float32), training=False)
    except Exception as e:
        print(f"[WARNING] Warm-up forward pass bỏ qua: {e}")

    print("Đang load trọng số (weights)...")
    try:
        if str(checkpoint_path).endswith(('.weights.h5', '.keras')):
            model.load_weights(checkpoint_path)
        else:
            model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    except (TypeError, ValueError):
        model.load_weights(checkpoint_path)
    print("✓ Load model thành công!\n")

    # Tìm các file mask (hỗ trợ .png, .jpeg, .jpg)
    valid_exts = ('.png', '.jpeg', '.jpg')
    mask_files = sorted([f for f in os.listdir(val_masks_dir) if f.lower().endswith(valid_exts)])

    num_samples = getattr(cfg, "NUM_SAMPLES", None)
    if num_samples is not None:
        mask_files = mask_files[:int(num_samples)]

    print(f"Đang xử lý {len(mask_files)} ảnh (Tạo Mask dự đoán + Ảnh ghép)...")

    # Cấu trúc lưu trữ tham số đánh giá chi tiết
    per_image_records = []

    benign_dices = []
    benign_ious = []
    benign_precs = []
    benign_recs = []

    malignant_dices = []
    malignant_ious = []
    malignant_precs = []
    malignant_recs = []

    overall_micro_dices = []
    overall_micro_ious = []
    overall_micro_precs = []
    overall_micro_recs = []
    
    hd95_benign = []
    hd95_malignant = []
    hd95_tumor = []
    bf1_list = []
    
    global_tp = {1: 0, 2: 0}
    global_fp = {1: 0, 2: 0}
    global_fn = {1: 0, 2: 0}

    for i, f in enumerate(tqdm(mask_files)):
        base = os.path.splitext(f)[0]

        # Tìm ảnh tương ứng với các đuôi khác nhau
        img_path = None
        for ext in valid_exts:
            candidate = os.path.join(val_images_dir, base + ext)
            if os.path.exists(candidate):
                img_path = candidate
                break

        if not img_path:
            continue

        mask_path = os.path.join(val_masks_dir, f)

        # 1. Đọc ảnh và mask
        original = np.array(Image.open(img_path).convert('RGB'))
        gt_mask = np.array(Image.open(mask_path).convert('L'))

        # Chuẩn hóa nhãn Ground Truth về [0: Nền, 1: U lành, 2: U ác]
        if np.any(gt_mask > 2):
            mapped_gt = np.zeros_like(gt_mask, dtype=np.uint8)
            mapped_gt[(gt_mask > 64) & (gt_mask <= 192)] = 1
            mapped_gt[gt_mask > 192] = 2
            gt_mask = mapped_gt

        # 1b. Đọc boundary mask ground truth
        boundary_gt = None
        if has_boundary_gt:
            boundary_path = os.path.join(val_boundary_dir, f"{base}.png")
            if os.path.exists(boundary_path):
                boundary_gt = np.array(Image.open(boundary_path).convert('L'))

        # Nếu chưa có boundary GT trên đĩa, tạo on-the-fly bằng morphological gradient
        # Tạo boundary per-class để phân biệt u lành vs u ác khi hiển thị
        kernel_size = getattr(cfg.BOUNDARY, "KERNEL_SIZE", 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        boundary_gt_benign = np.zeros_like(gt_mask, dtype=np.uint8)
        boundary_gt_malignant = np.zeros_like(gt_mask, dtype=np.uint8)
        for cls in [1, 2]:
            cls_mask = (gt_mask == cls).astype(np.uint8)
            if np.sum(cls_mask) == 0:
                continue
            dilated = cv2.dilate(cls_mask, kernel, iterations=1)
            eroded = cv2.erode(cls_mask, kernel, iterations=1)
            cls_boundary = (dilated - eroded).astype(np.uint8)
            if cls == 1:
                boundary_gt_benign = cls_boundary
            else:
                boundary_gt_malignant = cls_boundary
        # Nếu có file boundary GT đã lưu trên đĩa, đọc giá trị phân biệt nếu có
        if boundary_gt is not None:
            if np.any((boundary_gt > 64) & (boundary_gt <= 192)):
                boundary_gt_benign = ((boundary_gt > 64) & (boundary_gt <= 192)).astype(np.uint8)
                boundary_gt_malignant = (boundary_gt > 192).astype(np.uint8)

        # Resize ảnh về kích thước mô hình
        if original.shape[:2] != input_size:
            original = tf.image.resize(original, input_size).numpy().astype(np.uint8)
        if gt_mask.shape[:2] != input_size:
            gt_mask = tf.image.resize(gt_mask[..., np.newaxis], input_size, method='nearest')[..., 0].numpy()
        if boundary_gt_benign.shape[:2] != input_size:
            boundary_gt_benign = tf.image.resize(boundary_gt_benign[..., np.newaxis], input_size, method='nearest')[..., 0].numpy().astype(np.uint8)
        if boundary_gt_malignant.shape[:2] != input_size:
            boundary_gt_malignant = tf.image.resize(boundary_gt_malignant[..., np.newaxis], input_size, method='nearest')[..., 0].numpy().astype(np.uint8)

        # 2. Dự đoán qua mô hình
        img_input = np.expand_dims(original / 255.0, axis=0)
        preds = model.predict(img_input, verbose=0)

        # Trích xuất output chính xác dựa vào model.output_names
        if not isinstance(preds, (list, tuple)):
            preds = [preds]
        out_dict = dict(zip(model.output_names, preds))
        pred = out_dict["refined_output"]
        boundary_pred_raw = out_dict.get("boundary_output")

        pred_class = np.argmax(pred[0], axis=-1).astype(np.uint8)

        # Tạo boundary prediction: gộp tất cả kênh boundary thành 1 ảnh nhị phân
        if boundary_pred_raw is not None:
            bp = boundary_pred_raw[0]  # (H, W, C)
            # Lấy max trên tất cả kênh boundary, bỏ kênh 0 (nền) nếu có nhiều hơn 1 kênh
            if bp.shape[-1] > 1:
                bp_merged = np.max(bp[..., 1:], axis=-1)  # Gộp kênh u lành + u ác
            else:
                bp_merged = bp[..., 0]
            boundary_pred_binary = (bp_merged > 0.5).astype(np.uint8) * 255
        else:
            # Fallback: tạo boundary từ predicted mask nếu model không có boundary head
            kernel_size = getattr(cfg.BOUNDARY, "KERNEL_SIZE", 3)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            boundary_pred_binary = np.zeros_like(pred_class, dtype=np.uint8)
            for cls in [1, 2]:
                cls_mask = (pred_class == cls).astype(np.uint8)
                if np.sum(cls_mask) == 0:
                    continue
                dilated = cv2.dilate(cls_mask, kernel, iterations=1)
                eroded = cv2.erode(cls_mask, kernel, iterations=1)
                boundary_pred_binary = np.maximum(boundary_pred_binary, dilated - eroded)
            boundary_pred_binary = (boundary_pred_binary > 0).astype(np.uint8) * 255

        # 3. LƯU FILE MASK DỰ ĐOÁN (0: Nền, 128: U lành, 255: U ác)
        saved_mask_img = np.zeros_like(pred_class, dtype=np.uint8)
        saved_mask_img[pred_class == 1] = 128  # U lành
        saved_mask_img[pred_class == 2] = 255  # U ác
        Image.fromarray(saved_mask_img).save(os.path.join(masks_output_dir, f"{base}.png"))

        # 4. TÍNH TOÁN CÁC CHỈ SỐ ĐÁNH GIÁ (CẢ MACRO VÀ MICRO)
        true_b = (gt_mask == 1)
        pred_b = (pred_class == 1)
        true_m = (gt_mask == 2)
        pred_m = (pred_class == 2)
        true_tumor = (gt_mask > 0)
        pred_tumor = (pred_class > 0)

        tp_b = np.sum(true_b & pred_b)
        fp_b = np.sum((~true_b) & pred_b)
        fn_b = np.sum(true_b & (~pred_b))

        tp_m = np.sum(true_m & pred_m)
        fp_m = np.sum((~true_m) & pred_m)
        fn_m = np.sum(true_m & (~pred_m))

        # 4.1 U Lành (Benign - Class 1)
        if np.sum(true_b) > 0:
            d_b = (2.0 * tp_b) / (2.0 * tp_b + fp_b + fn_b + 1e-7)
            iou_b = tp_b / (tp_b + fp_b + fn_b + 1e-7)
            prec_b = tp_b / (tp_b + fp_b + 1e-7)
            rec_b = tp_b / (tp_b + fn_b + 1e-7)
            benign_dices.append(d_b)
            benign_ious.append(iou_b)
            benign_precs.append(prec_b)
            benign_recs.append(rec_b)
            dice_b_str = f"{d_b:.4f}"
        else:
            dice_b_str = "N/A"

        # 4.2 U Ác (Malignant - Class 2)
        if np.sum(true_m) > 0:
            d_m = (2.0 * tp_m) / (2.0 * tp_m + fp_m + fn_m + 1e-7)
            iou_m = tp_m / (tp_m + fp_m + fn_m + 1e-7)
            prec_m = tp_m / (tp_m + fp_m + 1e-7)
            rec_m = tp_m / (tp_m + fn_m + 1e-7)
            malignant_dices.append(d_m)
            malignant_ious.append(iou_m)
            malignant_precs.append(prec_m)
            malignant_recs.append(rec_m)
            dice_m_str = f"{d_m:.4f}"
        else:
            dice_m_str = "N/A"

        # 4.3 Macro Dice cho ảnh hiện tại (Trung bình các lớp u xuất hiện)
        present_dices = []
        if np.sum(true_b) > 0: present_dices.append(d_b)
        if np.sum(true_m) > 0: present_dices.append(d_m)
        d_macro_img = np.mean(present_dices) if len(present_dices) > 0 else 1.0

        # 4.4 Micro Dice cho ảnh hiện tại (Gộp toàn bộ pixel tổn thương)
        tp_micro_img = tp_b + tp_m
        fp_micro_img = fp_b + fp_m
        fn_micro_img = fn_b + fn_m

        d_micro_img = (2.0 * tp_micro_img) / (2.0 * tp_micro_img + fp_micro_img + fn_micro_img + 1e-7) if (tp_micro_img + fn_micro_img) > 0 else 1.0
        iou_micro_img = tp_micro_img / (tp_micro_img + fp_micro_img + fn_micro_img + 1e-7) if (tp_micro_img + fn_micro_img) > 0 else 1.0
        prec_micro_img = tp_micro_img / (tp_micro_img + fp_micro_img + 1e-7) if (tp_micro_img + fp_micro_img) > 0 else (1.0 if (tp_micro_img + fn_micro_img) == 0 else 0.0)
        rec_micro_img = tp_micro_img / (tp_micro_img + fn_micro_img + 1e-7) if (tp_micro_img + fn_micro_img) > 0 else 1.0

        overall_micro_dices.append(d_micro_img)
        overall_micro_ious.append(iou_micro_img)
        overall_micro_precs.append(prec_micro_img)
        overall_micro_recs.append(rec_micro_img)

        # Tích lũy ma trận nhầm lẫn toàn cục (Global TP, FP, FN)
        global_tp[1] += int(tp_b)
        global_fp[1] += int(fp_b)
        global_fn[1] += int(fn_b)

        global_tp[2] += int(tp_m)
        global_fp[2] += int(fp_m)
        global_fn[2] += int(fn_m)

        # Lưu bản ghi cho từng ảnh vào danh sách
        per_image_records.append([
            f"{base}.png", dice_b_str, dice_m_str,
            f"{d_macro_img:.4f}", f"{d_micro_img:.4f}",
            f"{iou_micro_img:.4f}", f"{prec_micro_img:.4f}",
            f"{rec_micro_img:.4f}", f"{d_micro_img:.4f}"
        ])

        # Điểm Dice hiển thị lên tiêu đề ảnh (Macro giữa 2 lớp u)
        dice_display = calculate_dice(gt_mask, pred_class, classes=[1, 2])

        # 5. XỬ LÝ HÌNH ẢNH GHÉP 5 CỘT (Original + Boundary GT + GT Mask + Boundary Pred + Pred Mask)

        # 5.1 Ảnh Boundary Ground Truth — ảnh xám (u lành: xám 128, u ác: trắng 255)
        boundary_gt_display = np.zeros(original.shape[:2], dtype=np.uint8)
        boundary_gt_display[boundary_gt_benign > 0] = 128    # U lành — xám
        boundary_gt_display[boundary_gt_malignant > 0] = 255  # U ác — trắng

        # 5.2 Ảnh Ground Truth Region Mask — ảnh xám (u lành: xám 128, u ác: trắng 255)
        gt_display = np.zeros(original.shape[:2], dtype=np.uint8)
        gt_display[gt_mask == 1] = 128    # U lành — xám
        gt_display[gt_mask == 2] = 255    # U ác — trắng

        # 5.3 Ảnh Boundary Prediction từ nhánh Boundary Decoder (xanh dương trên nền đen)
        boundary_pred_display = np.zeros_like(original)
        boundary_pred_mask = (boundary_pred_binary > 128) if boundary_pred_binary.max() > 1 else (boundary_pred_binary > 0)
        boundary_pred_display[boundary_pred_mask] = [0, 180, 255]  # Xanh dương sáng

        # 5.4 Ảnh Prediction Mask (Ám màu X-ray xanh dương, overlay màu đỏ)
        pred_display = original.copy()
        pred_display = (pred_display * [0.6, 0.6, 1.2]).clip(0, 255).astype(np.uint8)

        # Lớp overlay màu cho khối u (u lành: xanh lá, u ác: đỏ)
        color_overlay = np.zeros_like(pred_display)
        color_overlay[pred_class == 1] = [0, 200, 0]     # U lành — xanh lá
        color_overlay[pred_class == 2] = [220, 50, 50]    # U ác — đỏ

        # Blend màu overlay
        alpha = 0.5
        mask_indices = (pred_class == 1) | (pred_class == 2)
        if np.any(mask_indices):
            base_pixels = pred_display[mask_indices].astype(float)
            overlay_pixels = color_overlay[mask_indices].astype(float)
            blended = base_pixels * (1 - alpha) + overlay_pixels * alpha
            pred_display[mask_indices] = blended.astype(np.uint8)

        # 5.5 Vẽ hình 5 cột trực quan
        fig, axes = plt.subplots(1, 5, figsize=(25, 5), facecolor='white')
        fig.patch.set_linewidth(4)
        fig.patch.set_edgecolor('black')
        plt.subplots_adjust(wspace=0.08)

        axes[0].imshow(original)
        axes[0].set_title(f"Original X-ray\n({base})", fontsize=10, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(boundary_gt_display, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Boundary GT", fontsize=10, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(gt_display, cmap='gray', vmin=0, vmax=255)
        axes[2].set_title("Ground Truth Mask", fontsize=10, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(boundary_pred_display)
        axes[3].set_title("Boundary Prediction", fontsize=10, fontweight='bold', color='deepskyblue')
        axes[3].axis('off')

        axes[4].imshow(pred_display)
        axes[4].set_title(f"Pred Mask (Dice: {d_micro_img:.4f})", fontsize=10, fontweight='bold', color='red')
        axes[4].axis('off')

        # Lưu ảnh ghép
        combined_file = os.path.join(combined_output_dir, f"{base}_combined.png")
        plt.savefig(combined_file, bbox_inches='tight', pad_inches=0.1, dpi=120)
        plt.close(fig)

    # ================== TÍNH TOÁN VÀ XUẤT THAM SỐ TỔNG HỢP (CẢ MACRO & MICRO) ==================
    eps = 1e-7

    # --- A. ĐÁNH GIÁ TỪNG ẢNH (Image-level Means) ---
    mean_dice_b = np.mean(benign_dices) if len(benign_dices) > 0 else 0.0
    mean_iou_b = np.mean(benign_ious) if len(benign_ious) > 0 else 0.0
    mean_prec_b = np.mean(benign_precs) if len(benign_precs) > 0 else 0.0
    mean_rec_b = np.mean(benign_recs) if len(benign_recs) > 0 else 0.0

    mean_dice_m = np.mean(malignant_dices) if len(malignant_dices) > 0 else 0.0
    mean_iou_m = np.mean(malignant_ious) if len(malignant_ious) > 0 else 0.0
    mean_prec_m = np.mean(malignant_precs) if len(malignant_precs) > 0 else 0.0
    mean_rec_m = np.mean(malignant_recs) if len(malignant_recs) > 0 else 0.0

    # Image-level Macro (Trung bình 2 lớp)
    img_macro_dice = (mean_dice_b + mean_dice_m) / 2.0
    img_macro_iou = (mean_iou_b + mean_iou_m) / 2.0
    img_macro_prec = (mean_prec_b + mean_prec_m) / 2.0
    img_macro_rec = (mean_rec_b + mean_rec_m) / 2.0

    # Image-level Micro (Trung bình vùng u gộp)
    img_micro_dice = np.mean(overall_micro_dices) if len(overall_micro_dices) > 0 else 0.0
    img_micro_iou = np.mean(overall_micro_ious) if len(overall_micro_ious) > 0 else 0.0
    img_micro_prec = np.mean(overall_micro_precs) if len(overall_micro_precs) > 0 else 0.0
    img_micro_rec = np.mean(overall_micro_recs) if len(overall_micro_recs) > 0 else 0.0

    # --- B. ĐÁNH GIÁ TÍCH LŨY TOÀN BỘ TẬP DỮ LIỆU (Global Dataset-level) ---
    # 1. Từng lớp riêng biệt (Class-level Global)
    g_dice_b = (2.0 * global_tp[1] + eps) / (2.0 * global_tp[1] + global_fp[1] + global_fn[1] + eps)
    g_iou_b = (global_tp[1] + eps) / (global_tp[1] + global_fp[1] + global_fn[1] + eps)
    g_prec_b = (global_tp[1] + eps) / (global_tp[1] + global_fp[1] + eps)
    g_rec_b = (global_tp[1] + eps) / (global_tp[1] + global_fn[1] + eps)

    g_dice_m = (2.0 * global_tp[2] + eps) / (2.0 * global_tp[2] + global_fp[2] + global_fn[2] + eps)
    g_iou_m = (global_tp[2] + eps) / (global_tp[2] + global_fp[2] + global_fn[2] + eps)
    g_prec_m = (global_tp[2] + eps) / (global_tp[2] + global_fp[2] + eps)
    g_rec_m = (global_tp[2] + eps) / (global_tp[2] + global_fn[2] + eps)

    # 2. GLOBAL MACRO-AVERAGE (Cân bằng 50-50 giữa Lành và Ác)
    g_macro_dice = (g_dice_b + g_dice_m) / 2.0
    g_macro_iou = (g_iou_b + g_iou_m) / 2.0
    g_macro_prec = (g_prec_b + g_prec_m) / 2.0
    g_macro_rec = (g_rec_b + g_rec_m) / 2.0

    # 3. GLOBAL MICRO-AVERAGE (Gộp toàn bộ pixel TP, FP, FN của cả 2 lớp u)
    g_tp_micro = global_tp[1] + global_tp[2]
    g_fp_micro = global_fp[1] + global_fp[2]
    g_fn_micro = global_fn[1] + global_fn[2]

    g_micro_dice = (2.0 * g_tp_micro + eps) / (2.0 * g_tp_micro + g_fp_micro + g_fn_micro + eps)
    g_micro_iou = (g_tp_micro + eps) / (g_tp_micro + g_fp_micro + g_fn_micro + eps)
    g_micro_prec = (g_tp_micro + eps) / (g_tp_micro + g_fp_micro + eps)
    g_micro_rec = (g_tp_micro + eps) / (g_tp_micro + g_fn_micro + eps)

    eval_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Lưu file prediction_details_test.csv (chi tiết từng ảnh)
    csv_path = os.path.join(output_dir, "prediction_details_test.csv")
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "Image_Name", "Dice_Benign", "Dice_Malignant",
            "Macro_Dice", "Micro_Dice", "IoU_Micro", "Precision", "Recall", "F1_Score"
        ])
        writer.writerows(per_image_records)

    # 2. Trích xuất thông tin cấu hình Attention và mô hình
    dual_decoder = getattr(cfg.MODEL, "DUAL_DECODER", True)
    decoder_cfg = getattr(cfg.MODEL, "DECODER", {})
    decoder_type = str(decoder_cfg.get("TYPE", "unet")).lower()
    
    skip_cfg = getattr(cfg.MODEL, "SKIP_ATTENTION", None)
    if skip_cfg:
        region_cfg = getattr(skip_cfg, "REGION", {})
        bound_cfg = getattr(skip_cfg, "BOUNDARY", {})
        region_type = region_cfg.get("TYPE", "none") if hasattr(region_cfg, "get") else getattr(region_cfg, "TYPE", "none")
        bound_type = bound_cfg.get("TYPE", "none") if hasattr(bound_cfg, "get") else getattr(bound_cfg, "TYPE", "none")
        skip_summary = f"Reg:{region_type}|Bnd:{bound_type}"
    else:
        skip_summary = str(getattr(cfg.MODEL, "SKIP_ATTENTION_TYPE", "None"))
        
    bottleneck_attn = getattr(cfg.MODEL, "BOTTLENECK_ATTENTION", "None")

    if cfg.MODEL.TYPE == "dual_decoder_resnet":
        attention_summary = f"Skip: {skip_summary} | Bot: {bottleneck_attn} | Dec: {decoder_type} | Dual: {dual_decoder}"
    elif cfg.MODEL.TYPE == "hybrid_transunet":
        attention_summary = "Bottleneck: MultiHeadAttention (Transformer 8 heads)"
    else:
        attention_summary = "None (Standard CNN)"

    # 3. Ghi nối (append) vào file CSV LỊCH SỬ ĐÁNH GIÁ TỔNG HỢP (All Evaluations Master CSV)
    history_csv_headers = [
        "Timestamp", "Model_Type", "Backbone", "Attention_Summary",
        "Skip_Attention_Type", "Skip_Attention_Stages", "Bottleneck_Attention",
        "Weights_File", "Global_Micro_Dice", "Global_Macro_Dice",
        "Dice_Benign(Lanh)", "Dice_Malignant(Ac)", "Global_Micro_IoU",
        "Global_Macro_IoU", "Global_Micro_Precision", "Global_Micro_Recall",
        "Mean_Img_Micro_Dice", "Mean_Img_Macro_Dice", "Val_Dataset_Path", "Num_Images"
    ]

    history_row = [
        eval_time,
        cfg.MODEL.TYPE,
        getattr(cfg.MODEL.BACKBONE, 'TYPE', 'None'),
        attention_summary,
        skip_summary,
        "",
        bottleneck_attn,
        os.path.basename(checkpoint_path),
        f"{g_micro_dice:.4f}",
        f"{g_macro_dice:.4f}",
        f"{g_dice_b:.4f}",
        f"{g_dice_m:.4f}",
        f"{g_micro_iou:.4f}",
        f"{g_macro_iou:.4f}",
        f"{g_micro_prec:.4f}",
        f"{g_micro_rec:.4f}",
        f"{img_micro_dice:.4f}",
        f"{img_macro_dice:.4f}",
        val_images_dir,
        len(mask_files)
    ]

    # Lưu vào cả thư mục outputs chung và thư mục output_dir hiện tại nếu khác nhau
    target_history_paths = [
        os.path.join(cfg.WORK_DIR, "outputs", "all_evaluations_history.csv")
    ]
    if os.path.abspath(output_dir) != os.path.abspath(os.path.join(cfg.WORK_DIR, "outputs")):
        target_history_paths.append(os.path.join(output_dir, "all_evaluations_history.csv"))

    target_history_paths = list(dict.fromkeys(target_history_paths))

    for h_path in target_history_paths:
        os.makedirs(os.path.dirname(h_path), exist_ok=True)
        h_exists = os.path.exists(h_path)
        with open(h_path, mode='a', newline='', encoding='utf-8') as f_h:
            h_writer = csv.writer(f_h)
            if not h_exists:
                h_writer.writerow(history_csv_headers)
            h_writer.writerow(history_row)

    # 4. Lưu file summary_evaluation_test.txt (tổng hợp tham số chi tiết lần chạy này)
    txt_path = os.path.join(output_dir, "summary_evaluation_test.txt")
    summary_text = f"""
========================================================================================
📌 MÔ HÌNH (MODEL)        : {cfg.MODEL.TYPE} (Backbone: {getattr(cfg.MODEL.BACKBONE, 'TYPE', 'resnet34')})
📌 CẤU HÌNH ATTENTION     : {attention_summary}
📌 FILE WEIGHTS           : {checkpoint_path}
📌 TẬP DỮ LIỆU (DATASET)  : {val_images_dir} ({len(mask_files)} ảnh)
📌 THỜI GIAN ĐÁNH GIÁ     : {eval_time}
========================================================================================

----------------------------------------------------------------------------------------
I. ĐÁNH GIÁ TÍCH LŨY TOÀN BỘ TẬP DỮ LIỆU (GLOBAL DATASET-LEVEL - CHUẨN QUỐC TẾ MICCAI)
----------------------------------------------------------------------------------------
1. Từng lớp tổn thương (Per-Class Performance):
   - U Lành (Benign)    : Dice = {g_dice_b:.4f} ({g_dice_b * 100:.2f}%) | IoU = {g_iou_b:.4f} | Prec = {g_prec_b:.4f} | Rec = {g_rec_b:.4f}
   - U Ác (Malignant)   : Dice = {g_dice_m:.4f} ({g_dice_m * 100:.2f}%) | IoU = {g_iou_m:.4f} | Prec = {g_prec_m:.4f} | Rec = {g_rec_m:.4f}

2. Chỉ số MACRO-AVERAGE (Cân bằng 50-50 giữa U Lành và U Ác):
   - Global Macro Dice  : {g_macro_dice:.4f} ({g_macro_dice * 100:.2f}%)
   - Global Macro IoU   : {g_macro_iou:.4f} ({g_macro_iou * 100:.2f}%)
   - Global Macro Prec  : {g_macro_prec:.4f} ({g_macro_prec * 100:.2f}%)
   - Global Macro Rec   : {g_macro_rec:.4f} ({g_macro_rec * 100:.2f}%)

3. Chỉ số MICRO-AVERAGE (Tổng hợp toàn bộ pixel tổn thương không phân biệt lớp):
   - Global Micro Dice  : {g_micro_dice:.4f} ({g_micro_dice * 100:.2f}%)
   - Global Micro IoU   : {g_micro_iou:.4f} ({g_micro_iou * 100:.2f}%)
   - Global Micro Prec  : {g_micro_prec:.4f} ({g_micro_prec * 100:.2f}%)
   - Global Micro Rec   : {g_micro_rec:.4f} ({g_micro_rec * 100:.2f}%)

----------------------------------------------------------------------------------------
II. ĐÁNH GIÁ TRUNG BÌNH TỪNG ẢNH (PER-IMAGE MEAN METRICS)
----------------------------------------------------------------------------------------
1. Từng lớp tổn thương (Per-Class Mean):
   - Mean Dice U Lành   : {mean_dice_b:.4f} ({mean_dice_b * 100:.2f}%) | IoU = {mean_iou_b:.4f} | Prec = {mean_prec_b:.4f} | Rec = {mean_rec_b:.4f}
   - Mean Dice U Ác     : {mean_dice_m:.4f} ({mean_dice_m * 100:.2f}%) | IoU = {mean_iou_m:.4f} | Prec = {mean_prec_m:.4f} | Rec = {mean_rec_m:.4f}

2. Chỉ số MACRO-AVERAGE (Mean từng ảnh):
   - Mean Macro Dice    : {img_macro_dice:.4f} ({img_macro_dice * 100:.2f}%)
   - Mean Macro IoU     : {img_macro_iou:.4f} ({img_macro_iou * 100:.2f}%)
   - Mean Macro Prec    : {img_macro_prec:.4f} ({img_macro_prec * 100:.2f}%)
   - Mean Macro Rec     : {img_macro_rec:.4f} ({img_macro_rec * 100:.2f}%)

3. Chỉ số MICRO-AVERAGE (Mean vùng u từng ảnh):
   - Mean Micro Dice    : {img_micro_dice:.4f} ({img_micro_dice * 100:.2f}%)
   - Mean Micro IoU     : {img_micro_iou:.4f} ({img_micro_iou * 100:.2f}%)
   - Mean Micro Prec    : {img_micro_prec:.4f} ({img_micro_prec * 100:.2f}%)
   - Mean Micro Rec     : {img_micro_rec:.4f} ({img_micro_rec * 100:.2f}%)
========================================================================================
"""
    with open(txt_path, mode='w', encoding='utf-8') as f_txt:
        f_txt.write(summary_text)

    # In kết quả tổng hợp ra màn hình Terminal
    print("\n" + summary_text)
    print("=" * 88)
    print("✅ HOÀN THÀNH XUẤT TOÀN BỘ KẾT QUẢ, MASK VÀ BẢNG THAM SỐ (CẢ MACRO VÀ MICRO)!")
    print(f"   📁 1. Mask dự đoán riêng: {masks_output_dir}")
    print(f"   📁 2. Ảnh ghép 5 cột trực quan: {combined_output_dir}")
    print(f"         (Original | Boundary GT | GT Mask | Boundary Pred | Pred Mask)")
    print(f"   📊 3. File chi tiết từng ảnh (CSV): {csv_path}")
    print(f"   📝 4. File tổng hợp tham số (TXT):  {txt_path}")
    print(f"   📈 5. File LỊCH SỬ TẤT CẢ LẦN EVAL (CSV master): {target_history_paths[0]}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    main()