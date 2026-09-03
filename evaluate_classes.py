import os

import numpy as np

from PIL import Image

import tensorflow as tf

from tqdm import tqdm



model_path = "/workspace/unet3p/checkpoint/model_unet3plus.hdf5"

val_img_dir = "data_png/val/images"

val_mask_dir = "data_png/val/labels"

input_size = (512, 512)



print("Đang load mô hình tốt nhất...")

model = tf.keras.models.load_model(model_path, compile=False)

print("Load model thành công!\n")



def dice_per_class(y_true, y_pred, c, smooth=1e-6):

    yt = (y_true == c).astype(np.float32)

    yp = (y_pred == c).astype(np.float32)

    intersection = np.sum(yt * yp)

    return (2. * intersection + smooth) / (np.sum(yt) + np.sum(yp) + smooth)



dice_bg = []

dice_benign = []

dice_malignant = []



mask_files = [f for f in os.listdir(val_mask_dir) if f.endswith('.jpeg')]



print("Đang đánh giá trên tập Val...")



for f in tqdm(mask_files):

    base = os.path.splitext(f)[0]

    img_path = os.path.join(val_img_dir, base + '.jpeg')

    mask_path = os.path.join(val_mask_dir, f)

    

    if not os.path.exists(img_path):

        continue

    

    img = np.array(Image.open(img_path).convert('RGB')) / 255.0

    mask = np.array(Image.open(mask_path))

    

    # Resize nếu cần

    if img.shape[:2] != input_size:

        img = tf.image.resize(img, input_size).numpy()

    if mask.shape[:2] != input_size:

        mask = tf.image.resize(mask[..., np.newaxis], input_size, method='nearest')[..., 0].numpy()

    

    # Predict

    pred = model.predict(np.expand_dims(img, 0), verbose=0)[0]

    pred = np.argmax(pred, axis=-1)   # chuyển sang class index

    

    dice_bg.append(dice_per_class(mask, pred, 0))

    dice_benign.append(dice_per_class(mask, pred, 1))

    dice_malignant.append(dice_per_class(mask, pred, 2))



print("\n" + "="*60)
                                                                                                                                                                                        
print("KẾT QUẢ ĐÁNH GIÁ CHI TIẾT TRÊN TẬP VAL")

print("="*60)

print(f"Tổng số ảnh Val: {len(mask_files)}")

print(f"Dice Background : {np.mean(dice_bg):.4f}")

print(f"Dice U lành     : {np.mean(dice_benign):.4f}")

print(f"Dice U ác       : {np.mean(dice_malignant):.4f}")

print(f"Dice trung bình : {(np.mean(dice_bg) + np.mean(dice_benign) + np.mean(dice_malignant))/3:.4f}")

print("="*60)



print(f"\nNhận xét:")

print(f"- U lành Dice = {np.mean(dice_benign):.4f} → khá")

print(f"- U ác Dice   = {np.mean(dice_malignant):.4f} → yếu (đây là điểm cần cải thiện)")
