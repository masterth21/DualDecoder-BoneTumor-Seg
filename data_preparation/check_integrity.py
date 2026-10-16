import os
import cv2
import numpy as np
import pandas as pd
import hashlib
import glob

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

out = "D:/TumorBone/data/BTXRD_NEW"
df = pd.read_csv(os.path.join(out, "splits.csv"))

# b. Giao nhau image_id và md5
print("\n--- b. Kiểm tra giao nhau ---")
for ds in ["Split_full", "Split"]:
    print(f"Checking {ds}:")
    splits = {'train': set(), 'val': set(), 'test': set()}
    md5s = {'train': set(), 'val': set(), 'test': set()}
    for sp in ["train", "val", "test"]:
        img_dir = os.path.join(out, ds, sp, "images")
        if not os.path.exists(img_dir): continue
        for f in os.listdir(img_dir):
            splits[sp].add(os.path.splitext(f)[0])
            md5s[sp].add(get_md5(os.path.join(img_dir, f)))
    
    inter_img = splits['train'] & splits['val'] | splits['train'] & splits['test'] | splits['val'] & splits['test']
    inter_md5 = md5s['train'] & md5s['val'] | md5s['train'] & md5s['test'] | md5s['val'] & md5s['test']
    print(f"  Giao nhau image_id: {len(inter_img)}")
    print(f"  Giao nhau md5: {len(inter_md5)}")

# c. Mask properties
print("\n--- c. Kiểm tra Mask ---")
mask_errors = 0
for _, row in df.iterrows():
    img_id = row['image_id']
    sp = row['split']
    cls_val = row['class']
    
    # Check Split_full
    img_path = glob.glob(os.path.join(out, "Split_full", sp, "images", img_id + ".*"))[0]
    mask_path = os.path.join(out, "Split_full", sp, "masks", img_id + ".png")
    
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if img.shape[:2] != mask.shape[:2]:
        print(f"ERROR: Size mismatch {img_id}")
        mask_errors += 1
        
    uniq = set(np.unique(mask))
    if not uniq.issubset({0, 128, 255}):
        print(f"ERROR: Invalid values in {img_id} mask: {uniq}")
        mask_errors += 1
        
    if cls_val == 0 and uniq != {0}:
        print(f"ERROR: Normal image {img_id} has non-zero mask")
        mask_errors += 1
    if cls_val == 1 and 255 in uniq:
        print(f"ERROR: Benign image {img_id} has malignant mask")
        mask_errors += 1
    if cls_val == 2 and 128 in uniq:
        print(f"ERROR: Malignant image {img_id} has benign mask")
        mask_errors += 1
        
    if cls_val != 0 and len(uniq - {0}) == 0:
        print(f"ERROR: Tumor image {img_id} has empty mask")
        mask_errors += 1

print(f"Mask errors: {mask_errors}")

# d. Proportions
print("\n--- d. Kiểm tra tỉ lệ lớp ---")
total = len(df)
overall_prop = df['class'].value_counts() / total
for sp in ["train", "val", "test"]:
    sp_df = df[df['split'] == sp]
    sp_prop = sp_df['class'].value_counts() / len(sp_df)
    
    print(f"Split: {sp}")
    for c in [0, 1, 2]:
        o = overall_prop.get(c, 0)
        s = sp_prop.get(c, 0)
        diff = abs(o - s) * 100
        print(f"  Class {c}: Overall={o:.3f}, Split={s:.3f}, Diff={diff:.2f}%")
