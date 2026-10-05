"""
Script tạo Boundary Mask Ground Truth từ Region Mask cho toàn bộ tập dữ liệu.

Boundary được tính bằng morphological gradient: Dilate(mask) - Erode(mask)
sử dụng structuring element 3x3.

Cho mỗi thư mục mask (train/val/test), tạo thư mục boundary tương ứng:
  - BTXRD/Split/train/mask  →  BTXRD/Split/train/boundary
  - BTXRD/Split/val/mask    →  BTXRD/Split/val/boundary
  - BTXRD/Split/test/mask   →  BTXRD/Split/test/boundary

Boundary mask được lưu dưới dạng ảnh nhị phân (0: không phải boundary, 255: boundary).
"""

import os
import sys
import cv2
import numpy as np
from tqdm import tqdm


def generate_boundary_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Tạo boundary mask từ region segmentation mask.

    Args:
        mask (np.ndarray): Ảnh mask grayscale (H, W), giá trị 0/128/255 hoặc 0/1/2.
        kernel_size (int): Kích thước structuring element.

    Returns:
        np.ndarray: Boundary mask nhị phân (H, W), 0 hoặc 255.
    """
    # Chuẩn hóa về nhãn [0, 1, 2]
    if np.max(mask) > 2:
        mapped = np.zeros_like(mask, dtype=np.uint8)
        mapped[(mask > 64) & (mask <= 192)] = 1
        mapped[mask > 192] = 2
        mask = mapped

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    boundary = np.zeros_like(mask, dtype=np.uint8)

    # Tính boundary cho mỗi lớp khối u (u lành: 128 - xám, u ác: 255 - trắng)
    for cls in [1, 2]:
        cls_mask = (mask == cls).astype(np.uint8)
        if np.sum(cls_mask) == 0:
            continue
        dilated = cv2.dilate(cls_mask, kernel, iterations=1)
        eroded = cv2.erode(cls_mask, kernel, iterations=1)
        cls_boundary = (dilated - eroded).astype(np.uint8)
        val = 128 if cls == 1 else 255
        boundary[cls_boundary > 0] = val

    return boundary


def process_directory(mask_dir: str, boundary_dir: str, kernel_size: int = 3):
    """
    Xử lý toàn bộ thư mục mask, tạo boundary và lưu vào thư mục đích.

    Args:
        mask_dir (str): Đường dẫn thư mục chứa mask gốc.
        boundary_dir (str): Đường dẫn thư mục lưu boundary.
        kernel_size (int): Kích thước kernel morphological.
    """
    if not os.path.exists(mask_dir):
        print(f"[SKIP] Thư mục không tồn tại: {mask_dir}")
        return

    os.makedirs(boundary_dir, exist_ok=True)

    valid_exts = ('.png', '.jpg', '.jpeg')
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.lower().endswith(valid_exts)])

    if len(mask_files) == 0:
        print(f"[SKIP] Không tìm thấy file mask trong: {mask_dir}")
        return

    print(f"\n📂 Xử lý: {mask_dir}")
    print(f"   → Lưu boundary tại: {boundary_dir}")
    print(f"   → Số lượng mask: {len(mask_files)}")

    for f in tqdm(mask_files, desc="   Generating boundaries"):
        mask_path = os.path.join(mask_dir, f)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"   [WARNING] Không đọc được: {mask_path}")
            continue

        boundary = generate_boundary_mask(mask, kernel_size)

        # Lưu boundary mask (đổi thành .png)
        base = os.path.splitext(f)[0]
        out_path = os.path.join(boundary_dir, f"{base}.png")
        cv2.imwrite(out_path, boundary)

    print(f"   ✅ Hoàn thành! Đã tạo {len(mask_files)} boundary masks.")


def main():
    """
    Chạy tạo boundary mask cho tất cả split (train, val, test).
    Có thể chạy trực tiếp: python generate_boundary_masks.py [đường_dẫn_BTXRD]
    """
    # Xác định thư mục gốc BTXRD
    if len(sys.argv) > 1:
        btxrd_root = sys.argv[1]
    else:
        # Mặc định: cùng cấp với thư mục project
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        btxrd_root = os.path.join(project_dir, "BTXRD", "Split")

    print("=" * 70)
    print("🔬 TẠO BOUNDARY MASK GROUND TRUTH TỪ REGION MASK")
    print("=" * 70)
    print(f"Thư mục gốc: {btxrd_root}")

    # Xử lý cho 3 split: train, val, test
    splits = ["train", "val", "test"]
    for split in splits:
        mask_dir = os.path.join(btxrd_root, split, "mask")
        boundary_dir = os.path.join(btxrd_root, split, "boundary")
        process_directory(mask_dir, boundary_dir)

    print("\n" + "=" * 70)
    print("✅ HOÀN THÀNH TẠO TOÀN BỘ BOUNDARY MASK!")
    print("=" * 70)


if __name__ == "__main__":
    main()
