import os
import zipfile
import subprocess
import glob

def get_dir_size(path):
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total += os.path.getsize(fp)
    return total

print("Zipping directories...")
out_dir = "D:/TumorBone/data/BTXRD_NEW"
zip_path = os.path.join(out_dir, "BTXRD_v2.zip")

with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(out_dir):
        if "_check" in root:
            continue
        for file in files:
            if file == "BTXRD_v2.zip": continue
            file_path = os.path.join(root, file)
            arcname = os.path.relpath(file_path, out_dir)
            zipf.write(file_path, arcname)

size_mb = os.path.getsize(zip_path) / (1024 * 1024)
print(f"Zip size: {size_mb:.2f} MB")

with open(os.path.join(out_dir, "README_split.txt"), 'r', encoding='utf-8') as f:
    readme = f.read()

print("\n--- README_split.txt ---")
print(readme)
print("------------------------")
