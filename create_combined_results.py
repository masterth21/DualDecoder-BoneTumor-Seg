import os
import glob
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
    Tính Dice Score cho các lớp khối u (1: U lành, 2: U ác)
    """
    dice_scores = []
    for cls in classes:
        true_mask = (y_true == cls).astype(np.float32)
        pred_mask = (y_pred == cls).astype(np.float32)

        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)

        if union == 0:
            continue

        dice_scores.append((2.0 * intersection) / (union + 1e-7))

    return np.mean(dice_scores) if len(dice_scores) > 0 else 1.0


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ================== CẤU HÌNH ĐƯỜNG DẪN ==================
    val_images_dir = getattr(cfg, "VAL_IMAGES_DIR", cfg.DATASET.VAL.IMAGES_PATH)
    val_masks_dir = getattr(cfg, "VAL_MASKS_DIR", cfg.DATASET.VAL.MASK_PATH)

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

    print("\n" + "=" * 80)
    print("🎨 TẠO MASK DỰ ĐOÁN VÀ ẢNH GHÉP TRỰC QUAN (COMBINED RESULTS)")
    print("=" * 80)
    print(f"Mô hình: {cfg.MODEL.TYPE}")
    print(f"Kích thước ảnh: {input_size[0]}x{input_size[1]}")
    print(f"Weights được nạp: {checkpoint_path}")
    print(f"Thư mục ảnh gốc: {val_images_dir}")
    print(f"Thư mục mask gốc: {val_masks_dir}")
    print(f"📁 Thư mục lưu Mask dự đoán: {masks_output_dir}")
    print(f"📁 Thư mục lưu Ảnh ghép trực quan: {combined_output_dir}")
    print("=" * 80 + "\n")

    assert os.path.exists(checkpoint_path), \
        f"Lỗi: Không tìm thấy file trọng số tại {checkpoint_path}! Vui lòng huấn luyện mô hình trước."

    print("Đang khởi tạo cấu trúc mô hình từ config...")
    model = prepare_model(cfg, training=False)

    print("Đang load trọng số (weights)...")
    model.load_weights(checkpoint_path, by_name=True, skip_mismatch=True)
    print("✓ Load model thành công!\n")

    # Tìm các file mask (hỗ trợ .png, .jpeg, .jpg)
    valid_exts = ('.png', '.jpeg', '.jpg')
    mask_files = sorted([f for f in os.listdir(val_masks_dir) if f.lower().endswith(valid_exts)])

    num_samples = getattr(cfg, "NUM_SAMPLES", None)
    if num_samples is not None:
        mask_files = mask_files[:int(num_samples)]

    print(f"Đang xử lý {len(mask_files)} ảnh (Tạo Mask dự đoán + Ảnh ghép)...")

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

        # Resize ảnh về kích thước mô hình
        if original.shape[:2] != input_size:
            original = tf.image.resize(original, input_size).numpy().astype(np.uint8)
        if gt_mask.shape[:2] != input_size:
            gt_mask = tf.image.resize(gt_mask[..., np.newaxis], input_size, method='nearest')[..., 0].numpy()

        # 2. Dự đoán qua mô hình
        img_input = np.expand_dims(original / 255.0, axis=0)
        preds = model.predict(img_input, verbose=0)

        # Trích xuất output chính xác cho Dual Decoder
        if isinstance(preds, (list, tuple)):
            if cfg.MODEL.TYPE == "dual_decoder_resnet":
                pred = preds[2]  # Output cuối cùng được tinh chỉnh bởi NaLaFormer (refined_output)
            else:
                pred = preds[0]
        else:
            pred = preds

        pred_class = np.argmax(pred[0], axis=-1).astype(np.uint8)

        # 3. LƯU FILE MASK DỰ ĐOÁN (0: Nền, 128: U lành, 255: U ác)
        saved_mask_img = np.zeros_like(pred_class, dtype=np.uint8)
        saved_mask_img[pred_class == 1] = 128  # U lành
        saved_mask_img[pred_class == 2] = 255  # U ác
        Image.fromarray(saved_mask_img).save(os.path.join(masks_output_dir, f"{base}.png"))

        # 4. Tính điểm Dice
        dice_score = calculate_dice(gt_mask, pred_class, classes=[1, 2])

        # 5. XỬ LÝ HÌNH ẢNH GHÉP (Original + GT + Prediction Overlay)
        # 5.1 Ảnh Ground Truth (Mask trắng trên nền đen)
        gt_display = np.zeros_like(original)
        gt_display[gt_mask == 1] = [255, 255, 255]
        gt_display[gt_mask == 2] = [255, 255, 255]

        # 5.2 Ảnh Prediction (Ám màu X-ray xanh dương, overlay màu đỏ)
        pred_display = original.copy()
        pred_display = (pred_display * [0.6, 0.6, 1.2]).clip(0, 255).astype(np.uint8)

        # Lớp overlay màu cho khối u (màu đỏ cam nổi bật)
        color_overlay = np.zeros_like(pred_display)
        color_overlay[pred_class == 1] = [220, 70, 70]   # U lành
        color_overlay[pred_class == 2] = [240, 50, 50]   # U ác

        # Blend màu overlay
        alpha = 0.5
        mask_indices = (pred_class == 1) | (pred_class == 2)
        if np.any(mask_indices):
            base_pixels = pred_display[mask_indices].astype(float)
            overlay_pixels = color_overlay[mask_indices].astype(float)
            blended = base_pixels * (1 - alpha) + overlay_pixels * alpha
            pred_display[mask_indices] = blended.astype(np.uint8)

        # 5.3 Vẽ hình 3 cột đúng theo mẫu chuẩn
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        fig.patch.set_linewidth(4)
        fig.patch.set_edgecolor('black')
        plt.subplots_adjust(wspace=0.05)

        axes[0].imshow(original)
        axes[0].set_title(f"Original X-ray ({base})", fontsize=12, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(gt_display)
        axes[1].set_title("Ground Truth (Bác sĩ)", fontsize=12, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(pred_display)
        axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})", fontsize=12, fontweight='bold')
        axes[2].axis('off')

        # Lưu ảnh ghép
        combined_file = os.path.join(combined_output_dir, f"{base}_combined.png")
        plt.savefig(combined_file, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    print(f"\n" + "=" * 80)
    print("✅ HOÀN THÀNH XUẤT TOÀN BỘ KẾT QUẢ!")
    print(f"   📁 1. Mask dự đoán riêng: {masks_output_dir}")
    print(f"   📁 2. Ảnh ghép 3 cột trực quan: {combined_output_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()