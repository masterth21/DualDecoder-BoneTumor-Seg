import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import các thư viện từ repo UNet-3-Plus (giống trong train.py)
import hydra
from omegaconf import DictConfig
from models.model import prepare_model

# Hàm tính Dice Score
def calculate_dice(y_true, y_pred, classes=[1, 2]):
    dice_scores = []
    for cls in classes:
        true_mask = (y_true == cls).astype(np.float32)
        pred_mask = (y_pred == cls).astype(np.float32)
        
        intersection = np.sum(true_mask * pred_mask)
        union = np.sum(true_mask) + np.sum(pred_mask)
        
        if union == 0:
            continue
            
        dice_scores.append((2. * intersection) / (union + 1e-7))
    
    return np.mean(dice_scores) if len(dice_scores) > 0 else 1.0

# Dùng Hydra để đọc config y hệt như lúc bạn chạy lệnh train
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # ================== CẤU HÌNH ==================
    # model_path = "/workspace/unet3p/checkpoint/model_unet3plus.hdf5"   
    # val_images_dir = "/workspace/unet3p/data_png/val/images"
    # val_masks_dir  = "/workspace/unet3p/data_png/val/labels"
    # output_dir     = "/workspace/unet3p/prediction_results_2"    
    model_path = "/workspace/unet3p/checkpoint/model_unet_plus_plus.hdf5"   
    val_images_dir = "/workspace/unet3p/data_png/val/images"
    val_masks_dir  = "/workspace/unet3p/data_png/val/labels"
    # ĐỔI TÊN THƯ MỤC ĐẦU RA ĐỂ DỄ TÌM
    output_dir     = "/workspace/unet3p/prediction_results_best_074"        
    
    input_size = (512, 512)
    os.makedirs(output_dir, exist_ok=True)

    print("Đang khởi tạo cấu trúc mô hình từ config...")
    # Khởi tạo mô hình thông qua prepare_model của tác giả (training=False để inference)
    model = prepare_model(cfg, training=False) 

    print("Đang load trọng số (weights)...")
    model.load_weights(model_path)
    print("Load model thành công!\n")

    mask_files = sorted([f for f in os.listdir(val_masks_dir) if f.endswith('.jpeg')])
    print(f"Đang tạo {len(mask_files)} ảnh ghép (Original + GT + Prediction)...")

    for i, f in enumerate(tqdm(mask_files)):
        base = os.path.splitext(f)[0]
        img_path = os.path.join(val_images_dir, base + '.jpeg')
        mask_path = os.path.join(val_masks_dir, f)
        
        if not os.path.exists(img_path):
            continue
        
        # Đọc ảnh và mask
        original = np.array(Image.open(img_path).convert('RGB'))
        gt_mask = np.array(Image.open(mask_path).convert('L'))
        
        # Resize
        if original.shape[:2] != input_size:
            original = tf.image.resize(original, input_size).numpy().astype(np.uint8)
        if gt_mask.shape[:2] != input_size:
            gt_mask = tf.image.resize(gt_mask[..., np.newaxis], input_size, method='nearest')[..., 0].numpy()
        
        # Dự đoán
        img_input = np.expand_dims(original / 255.0, axis=0)
        pred = model.predict(img_input, verbose=0)
        
        # Bắt trường hợp model dùng Deep Supervision (trả về list nhiều outputs)
        if isinstance(pred, list):
            pred = pred[0]
            
        pred_class = np.argmax(pred[0], axis=-1).astype(np.uint8)
        
        # Tính điểm Dice
        dice_score = calculate_dice(gt_mask, pred_class, classes=[1, 2])
        
        # ------------------ XỬ LÝ HÌNH ẢNH THEO ĐÚNG MẪU ------------------
        # 1. Ảnh Ground Truth (Mask trắng trên nền đen)
        gt_display = np.zeros_like(original)
        gt_display[gt_mask == 1] = [255, 255, 255]
        gt_display[gt_mask == 2] = [255, 255, 255]
        
        # 2. Ảnh Prediction (Ám màu X-ray xanh dương, overlay màu đỏ)
        pred_display = original.copy()
        
        # Tăng nhẹ kênh Blue, giảm kênh Red và Green để ra hiệu ứng xanh dương mờ
        pred_display = (pred_display * [0.6, 0.6, 1.2]).clip(0, 255).astype(np.uint8) 
        
        # Tạo lớp overlay màu
        color_overlay = np.zeros_like(pred_display)
        color_overlay[pred_class == 1] = [200, 80, 80]
        color_overlay[pred_class == 2] = [200, 80, 80] 
        
        # Trộn (blend) màu overlay vào vùng mô hình dự đoán
        alpha = 0.5
        mask_indices = (pred_class == 1) | (pred_class == 2)
        
        if np.any(mask_indices):
            base_pixels = pred_display[mask_indices].astype(float)
            overlay_pixels = color_overlay[mask_indices].astype(float)
            blended = base_pixels * (1 - alpha) + overlay_pixels * alpha
            pred_display[mask_indices] = blended.astype(np.uint8)
        
        # --- VẼ PLOT THEO MẪU ---
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')
        
        # Đóng khung viền đen dày
        fig.patch.set_linewidth(4)
        fig.patch.set_edgecolor('black')
        plt.subplots_adjust(wspace=0.05)
        
        # Original X-ray
        axes[0].imshow(original)
        axes[0].set_title(f"Original X-ray (ID: {i})", fontsize=12)
        axes[0].axis('off')
        
        # Ground Truth
        axes[1].imshow(gt_display)
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis('off')
        
        # Prediction
        axes[2].imshow(pred_display)
        axes[2].set_title(f"Prediction (Dice: {dice_score:.4f})", fontsize=12)
        axes[2].axis('off')
        
        # Lưu file
        plt.savefig(os.path.join(output_dir, f"{base}_combined.png"), bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)

    print(f"\n✅ Hoàn thành! Tất cả ảnh ghép được lưu tại: {output_dir}")

if __name__ == "__main__":
    main()