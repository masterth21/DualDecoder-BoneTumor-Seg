"""
Utility module for Boundary Ground Truth Generation.
Creates binary boundary masks from binary or multi-class region segmentation masks
using morphological gradient (Dilation - Erosion).
"""

import cv2
import numpy as np
import tensorflow as tf


def extract_boundary_np(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Extract boundary mask from binary or multi-class region segmentation mask using OpenCV (NumPy).
    
    Args:
        mask (np.ndarray): Input mask of shape (H, W) or (H, W, C).
                           Values can be binary (0/1 or 0/255) or one-hot channels.
        kernel_size (int): Size of structuring element kernel (default=3).
        
    Returns:
        np.ndarray: Boundary mask of same shape as input mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    
    if mask.ndim == 2:
        # Binary mask shape (H, W)
        mask_uint8 = (mask > 0.5).astype(np.uint8)
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        boundary = dilated - eroded
        return boundary.astype(np.float32)
        
    elif mask.ndim == 3:
        # Multi-channel or one-hot mask shape (H, W, C)
        boundary_channels = []
        for c in range(mask.shape[-1]):
            ch_mask = (mask[..., c] > 0.5).astype(np.uint8)
            dilated = cv2.dilate(ch_mask, kernel, iterations=1)
            eroded = cv2.erode(ch_mask, kernel, iterations=1)
            b_ch = dilated - eroded
            boundary_channels.append(b_ch)
        boundary = np.stack(boundary_channels, axis=-1)
        return boundary.astype(np.float32)
    else:
        raise ValueError(f"Unsupported mask dimension: {mask.ndim}")


def extract_boundary_tf(mask_tensor: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
    """
    Extract boundary mask from tensor using TensorFlow max_pool2d (dilation) and -max_pool2d(-mask) (erosion).
    
    Args:
        mask_tensor (tf.Tensor): Batch of masks with shape (B, H, W, C).
        kernel_size (int): Size of morphological pooling kernel (default=3).
        
    Returns:
        tf.Tensor: Boundary tensor of shape (B, H, W, C).
    """
    mask = tf.cast(mask_tensor, tf.float32)
    
    # Dilation via Max Pooling
    dilated = tf.nn.max_pool2d(
        mask,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    # Erosion via Min Pooling (Negative Max Pooling of Negative Tensor)
    eroded = -tf.nn.max_pool2d(
        -mask,
        ksize=[1, kernel_size, kernel_size, 1],
        strides=[1, 1, 1, 1],
        padding='SAME'
    )
    
    boundary = tf.clip_by_value(dilated - eroded, 0.0, 1.0)
    return boundary


def binary_tumor_tf(mask_onehot: tf.Tensor) -> tf.Tensor:
    """
    Gộp mask one-hot (B,H,W,C) thành u nhị phân (B,H,W,1): u = lành ∪ ác = max(kênh 1..C-1).
    Nếu chỉ có 1 kênh (mask nhị phân) thì trả nguyên kênh đó.
    """
    mask = tf.cast(mask_onehot, tf.float32)
    if mask.shape[-1] == 1:
        return mask
    return tf.reduce_max(mask[..., 1:], axis=-1, keepdims=True)


def extract_boundary_binary_tf(mask_onehot: tf.Tensor, kernel_size: int = 3) -> tf.Tensor:
    """Biên của u NHỊ PHÂN (lành ∪ ác): (B,H,W,C) one-hot -> (B,H,W,1)."""
    return extract_boundary_tf(binary_tumor_tf(mask_onehot), kernel_size=kernel_size)


def sdf_from_binary_np(tumor: np.ndarray, trunc: float = 20.0) -> np.ndarray:
    """
    Signed distance (đã cắt & chuẩn hoá) của 1 mask u nhị phân (H,W).
      d = outside - inside, trong đó
        inside  = distance_transform_edt(tumor)      (>0 bên trong u, khoảng cách tới pixel nền gần nhất)
        outside = distance_transform_edt(1 - tumor)  (>0 bên ngoài u, khoảng cách tới pixel u gần nhất)
      => d ÂM bên trong u, DƯƠNG bên ngoài, |d| ~ khoảng cách tới biên.
      sdf = clip(d, -trunc, trunc) / trunc ∈ [-1, 1].
    Ảnh không có u -> toàn +1. Ảnh toàn u -> toàn -1.
    """
    from scipy.ndimage import distance_transform_edt
    t = np.asarray(tumor) > 0.5
    if not t.any():
        return np.ones(t.shape, dtype=np.float32)
    if t.all():
        return -np.ones(t.shape, dtype=np.float32)
    inside = distance_transform_edt(t)
    outside = distance_transform_edt(~t)
    d = outside - inside
    return (np.clip(d, -trunc, trunc) / float(trunc)).astype(np.float32)


def sdf_from_onehot_np(mask_onehot, trunc: float = 20.0) -> np.ndarray:
    """
    Batch one-hot (B,H,W,C) -> SDF u nhị phân (B,H,W,1) float32.
    Luôn dùng u = lành ∪ ác (bất kể BOUNDARY.MODE).
    """
    m = np.asarray(mask_onehot, dtype=np.float32)
    tumor = m[..., 0] if m.shape[-1] == 1 else np.max(m[..., 1:], axis=-1)
    out = np.stack([sdf_from_binary_np(tumor[b], trunc) for b in range(tumor.shape[0])], axis=0)
    return out[..., None].astype(np.float32)
