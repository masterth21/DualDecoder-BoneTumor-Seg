import os
import argparse
import pandas as pd
import json
import cv2
import numpy as np
import hashlib
from sklearn.model_selection import train_test_split
from collections import defaultdict
import shutil
import warnings

def get_md5(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=str, required=True, help="Data root path")
    parser.add_argument("--out", type=str, required=True, help="Output path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    src = args.src
    out = args.out
    seed = args.seed

    images_dir = os.path.join(src, "images")
    ann_dir = os.path.join(src, "Annotations")
    xlsx_path = os.path.join(src, "dataset.xlsx")

    # 2.1 Check
    print("Reading dataset.xlsx...")
    df = pd.read_excel(xlsx_path)
    
    if len(df) != 3746:
        print(f"ERROR: Expected 3746 rows, got {len(df)}")
        return
    if df["image_id"].nunique() != len(df):
        print("ERROR: Duplicate image_id in xlsx")
        return

    # Mappings
    tumor_map = dict(zip(df['image_id'], df['tumor']))
    benign_map = dict(zip(df['image_id'], df['benign']))
    malignant_map = dict(zip(df['image_id'], df['malignant']))
    center_map = dict(zip(df['image_id'], df['center']))

    images_files = {os.path.splitext(f)[0]: f for f in os.listdir(images_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))}
    ann_files = {os.path.splitext(f)[0]: f for f in os.listdir(ann_dir) if f.lower().endswith('.json')}

    warnings_list = []

    # Verify everything
    data = []
    print("Verifying images and JSONs...")
    for index, row in df.iterrows():
        img_id = os.path.splitext(str(row['image_id']))[0]
        is_tumor = int(row['tumor'])
        
        if img_id not in images_files:
            print(f"ERROR: Image {img_id} not found in images/")
            return
            
        has_json = img_id in ann_files
        
        if is_tumor and not has_json:
            print(f"ERROR: {img_id} has tumor=1 but no JSON")
            return
        if not is_tumor and has_json:
            print(f"ERROR: {img_id} has tumor=0 but has JSON")
            return

        cls_val = 0
        if int(row['benign']) == 1:
            cls_val = 1
        elif int(row['malignant']) == 1:
            cls_val = 2
            
        data.append({
            'image_id': img_id,
            'image_file': images_files[img_id],
            'json_file': ann_files.get(img_id),
            'cls': cls_val,
            'center': int(row['center'])
        })

    print("Hashing images for deduplication...")
    md5_to_ids = defaultdict(list)
    for item in data:
        img_path = os.path.join(images_dir, item['image_file'])
        md5_to_ids[get_md5(img_path)].append(item['image_id'])

    dup_groups = [ids for ids in md5_to_ids.values() if len(ids) > 1]
    if dup_groups:
        print(f"Found {len(dup_groups)} groups of identical images.")
        for g in dup_groups:
            print(f"  Duplicate group: {g}")
            
    # Group item based on MD5 to avoid leakage
    # We will split unique MD5s, then expand to all image_ids
    unique_items = []
    for md5, ids in md5_to_ids.items():
        # Represent the group by the first item's class and center
        # We assume duplicates have the same class and center
        repr_id = ids[0]
        repr_item = next(item for item in data if item['image_id'] == repr_id)
        
        # Check if duplicates have conflicting classes
        cls_set = {next(i['cls'] for i in data if i['image_id'] == i_id) for i_id in ids}
        if len(cls_set) > 1:
            msg = f"WARNING: Duplicates {ids} have conflicting classes: {cls_set}. Using {repr_item['cls']}."
            print(msg)
            warnings_list.append(msg)
            
        unique_items.append({
            'md5': md5,
            'ids': ids,
            'cls': repr_item['cls'],
            'center': repr_item['center']
        })

    # Stratify unique items
    df_unique = pd.DataFrame(unique_items)
    df_unique['strat'] = df_unique['cls'].astype(str) + "_" + df_unique['center'].astype(str)
    
    # Check for tiny strata
    strata_counts = df_unique['strat'].value_counts()
    for strat, count in strata_counts.items():
        if count < 2:
            cls_val = strat.split("_")[0]
            print(f"Stratum {strat} too small ({count} items), merging centers for class {cls_val}")
            df_unique.loc[df_unique['cls'].astype(str) == cls_val, 'strat'] = cls_val

    print("Splitting Full dataset...")
    # Train = 70%, Val = 15%, Test = 15%
    # First split 70 / 30
    train_unique, temp_unique = train_test_split(df_unique, test_size=0.30, random_state=seed, stratify=df_unique['strat'])
    # Then split 30 into 15/15 -> 50/50 of temp
    val_unique, test_unique = train_test_split(temp_unique, test_size=0.50, random_state=seed, stratify=temp_unique['strat'])

    split_map = {}
    for _, row in train_unique.iterrows():
        for i_id in row['ids']: split_map[i_id] = 'train'
    for _, row in val_unique.iterrows():
        for i_id in row['ids']: split_map[i_id] = 'val'
    for _, row in test_unique.iterrows():
        for i_id in row['ids']: split_map[i_id] = 'test'

    # Create directories
    print("Preparing directories...")
    os.makedirs(out, exist_ok=True)
    check_dir = os.path.join(out, "_check")
    os.makedirs(check_dir, exist_ok=True)
    
    for ds in ["Split_full", "Split"]:
        for sp in ["train", "val", "test"]:
            os.makedirs(os.path.join(out, ds, sp, "images"), exist_ok=True)
            os.makedirs(os.path.join(out, ds, sp, "masks"), exist_ok=True) # Changed from mask to masks if that's standard, wait prompt says masks/

    # 2.2 Create Masks
    print("Creating masks and copying images...")
    
    records = []
    
    benign_labels = ["osteochondroma", "multiple osteochondromas", "simple bone cyst", "giant cell tumor", "osteofibroma", "synovial osteochondroma", "other bt"]
    malignant_labels = ["osteosarcoma", "other mt"]

    check_images_saved = {'train': 0, 'val': 0, 'test': 0}

    for item in data:
        img_id = item['image_id']
        sp = split_map[img_id]
        cls_val = item['cls']
        img_file = item['image_file']
        json_file = item['json_file']
        
        img_path = os.path.join(images_dir, img_file)
        
        # Read image to get shape exactly as OpenCV does
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"ERROR: Could not read image {img_path}")
            continue
        h_img, w_img = img.shape[:2]
        
        mask = np.zeros((h_img, w_img), dtype=np.uint8)
        num_polygons = 0
        
        if json_file:
            with open(os.path.join(ann_dir, json_file), 'r', encoding='utf-8') as f:
                jdata = json.load(f)
            
            w_json = jdata.get('imageWidth', w_img)
            h_json = jdata.get('imageHeight', h_img)
            
            scale_x = w_img / w_json if w_json else 1.0
            scale_y = h_img / h_json if h_json else 1.0
            if scale_x != 1.0 or scale_y != 1.0:
                msg = f"WARNING: {img_id} size mismatch (img: {w_img}x{h_img}, json: {w_json}x{h_json})"
                warnings_list.append(msg)
                
            shapes = jdata.get('shapes', [])
            polys = [s for s in shapes if s['shape_type'] == 'polygon']
            rects = [s for s in shapes if s['shape_type'] == 'rectangle']
            
            # Check labels against DB
            json_cls = 0
            for s in polys + rects:
                lbl = s['label'].lower().strip()
                if lbl in benign_labels:
                    json_cls = 1
                elif lbl in malignant_labels:
                    json_cls = 2
                    
            if json_cls != 0 and json_cls != cls_val:
                msg = f"WARNING: {img_id} label mismatch between JSON (cls={json_cls}) and XLSX (cls={cls_val}). Trusting XLSX."
                print(msg)
                warnings_list.append(msg)
                
            mask_val = 128 if cls_val == 1 else (255 if cls_val == 2 else 0)
            
            for p in polys:
                pts = np.array(p['points'], dtype=np.float32)
                pts[:, 0] *= scale_x
                pts[:, 1] *= scale_y
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], color=mask_val)
                num_polygons += 1
                
            if len(polys) == 0 and len(rects) > 0:
                msg = f"WARNING: {img_id} has rectangles but NO polygons. Filling rectangles."
                print(msg)
                warnings_list.append(msg)
                for r in rects:
                    pts = np.array(r['points'], dtype=np.float32)
                    pts[:, 0] *= scale_x
                    pts[:, 1] *= scale_y
                    pts = pts.astype(np.int32)
                    x1, y1 = np.min(pts[:, 0]), np.min(pts[:, 1])
                    x2, y2 = np.max(pts[:, 0]), np.max(pts[:, 1])
                    cv2.rectangle(mask, (x1, y1), (x2, y2), color=mask_val, thickness=-1)
                    num_polygons += 1
                    
        mask_pixels = np.count_nonzero(mask)
        
        # Save check overlays
        if cls_val != 0 and check_images_saved[sp] < 4:
            overlay = img.copy()
            overlay[mask == 128] = overlay[mask == 128] * 0.5 + np.array([0, 255, 0]) * 0.5
            overlay[mask == 255] = overlay[mask == 255] * 0.5 + np.array([0, 0, 255]) * 0.5
            cv2.imwrite(os.path.join(check_dir, f"{sp}_{img_id}_check.jpg"), overlay)
            check_images_saved[sp] += 1
            
        # Write to Split_full
        dest_img_full = os.path.join(out, "Split_full", sp, "images", img_file)
        dest_mask_full = os.path.join(out, "Split_full", sp, "masks", img_id + ".png")
        shutil.copy2(img_path, dest_img_full)
        cv2.imwrite(dest_mask_full, mask)
        
        # Write to Split if tumor
        if cls_val != 0:
            dest_img_tumor = os.path.join(out, "Split", sp, "images", img_file)
            dest_mask_tumor = os.path.join(out, "Split", sp, "masks", img_id + ".png")
            shutil.copy2(img_path, dest_img_tumor)
            cv2.imwrite(dest_mask_tumor, mask)
            
        records.append({
            'image_id': img_id,
            'split': sp,
            'class': cls_val,
            'center': item['center'],
            'n_polygons': num_polygons,
            'mask_pixels': mask_pixels
        })

    # Save CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(os.path.join(out, "splits.csv"), index=False)
    
    # Save README
    readme_path = os.path.join(out, "README_split.txt")
    with open(readme_path, "w", encoding='utf-8') as f:
        f.write(f"Seed: {seed}\n")
        f.write("Ratio: 70/15/15\n\n")
        
        f.write("=== Split_full ===\n")
        f.write(df_out.groupby(['split', 'class', 'center']).size().to_string())
        f.write("\n\n")
        
        f.write("=== Split ===\n")
        f.write(df_out[df_out['class'] != 0].groupby(['split', 'class', 'center']).size().to_string())
        f.write("\n\n")
        
        f.write("Warnings:\n")
        for w in warnings_list:
            f.write(w + "\n")
            
    print("\nDONE!")

if __name__ == "__main__":
    main()
