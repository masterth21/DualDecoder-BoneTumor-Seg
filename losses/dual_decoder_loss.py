"""
Loss functions for Dual-Decoder (Region + Boundary) Architecture.
Includes Boundary BCE, Boundary Dice Loss, Region Dice/CE Loss, and Multi-Output Loss handlers.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def boundary_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Per-class Dice Loss specially computed for sparse boundary maps.
    Weights tumor boundaries heavily over background borders.
    Uses presence masking so absent boundaries do not add flat penalty.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate Dice per channel (axis=[1, 2] over Height & Width)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
    has_gt = tf.cast(tf.reduce_sum(y_true, axis=[1, 2]) > 0, tf.float32)
    sample_weights = class_weights * has_gt
    normalized_weights = sample_weights / (tf.reduce_sum(sample_weights, axis=-1, keepdims=True) + 1e-7)

    loss = tf.reduce_sum((1.0 - dice_per_class) * normalized_weights, axis=-1)
    return tf.reduce_mean(loss)


def boundary_bce_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss for boundary prediction with class weighting.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
    bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    weighted_bce = bce * class_weights
    return tf.reduce_mean(weighted_bce)


def boundary_bce_dice_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Combined Boundary Loss: alpha * Boundary_BCE + beta * Boundary_Dice.
    """
    bce = boundary_bce_loss(y_true, y_pred)
    dice = boundary_dice_loss(y_true, y_pred)
    return alpha * bce + beta * dice


def region_dice_ce_loss(y_true, y_pred, smooth=1e-6):
    """
    Region Loss: Weighted Soft Cross-Entropy + Per-Class Dice Loss for Region Segmentation.
    Uses presence masking on Dice Loss so absent classes do not add flat penalties,
    while Weighted Cross-Entropy strictly penalizes any false positive predictions.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 1. Per-Class Dice Loss with Presence Masking
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
    has_gt = tf.cast(tf.reduce_sum(y_true, axis=[1, 2]) > 0, tf.float32)
    sample_weights = class_weights * has_gt
    normalized_weights = sample_weights / (tf.reduce_sum(sample_weights, axis=-1, keepdims=True) + 1e-7)

    loss_dice = tf.reduce_mean(tf.reduce_sum((1.0 - dice_per_class) * normalized_weights, axis=-1))

    # 2. Weighted Cross-Entropy Loss (applies to all classes to suppress false positives)
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce_per_class = -y_true * tf.math.log(y_pred_clipped)
    loss_ce = tf.reduce_mean(tf.reduce_sum(ce_per_class * class_weights, axis=-1))

    return loss_dice + loss_ce


def make_binary_boundary_bce_dice_loss(pos_weight=1.0, alpha=0.5, beta=0.5, smooth=1e-6):
    """
    Loss biên NHỊ PHÂN 1 kênh (B,H,W,1): alpha * BCE(pos_weight) + beta * Dice.
    Không dùng class_weights 3 kênh. Dice tính theo mẫu, có presence masking
    (mẫu không có biên GT không cộng Dice — giống mode class), BCE vẫn phạt mọi FP.
    """
    pos_weight = float(pos_weight)

    def binary_boundary_bce_dice_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        p = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        bce = -(pos_weight * y_true * tf.math.log(p) + (1.0 - y_true) * tf.math.log(1.0 - p))
        loss_bce = tf.reduce_mean(bce)

        inter = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
        union = tf.reduce_sum(y_true, axis=[1, 2, 3]) + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        dice = (2.0 * inter + smooth) / (union + smooth)
        has_gt = tf.cast(tf.reduce_sum(y_true, axis=[1, 2, 3]) > 0, tf.float32)
        loss_dice = tf.reduce_mean((1.0 - dice) * has_gt)

        return alpha * loss_bce + beta * loss_dice

    return binary_boundary_bce_dice_loss


def make_sdf_loss(trunc=20.0, near_px=5.0, near_weight=5.0, huber_delta=0.1):
    """
    SmoothL1 (Huber, delta) giữa sdf_pred và sdf_gt (cả hai ∈ [-1,1]),
    trọng số pixel w = 1 + (near_weight - 1) * 1[|sdf_gt| * trunc <= near_px],
    trung bình có trọng số: sum(w * huber) / sum(w).
    """
    trunc = float(trunc); near_px = float(near_px)
    near_weight = float(near_weight); delta = float(huber_delta)

    def sdf_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        err = tf.abs(y_pred - y_true)
        huber = tf.where(err < delta, 0.5 * tf.square(err) / delta, err - 0.5 * delta)
        near = tf.cast(tf.abs(y_true) * trunc <= near_px, tf.float32)
        w = 1.0 + (near_weight - 1.0) * near
        return tf.reduce_sum(w * huber) / (tf.reduce_sum(w) + 1e-7)

    return sdf_loss


def make_cls_loss(class_weights=(1.0, 1.0, 1.0)):
    """
    Cross-entropy cấp ảnh có trọng số lớp cho 'cls_output' (B,3): 0 bình thường, 1 u lành, 2 u ác.
    Trung bình có trọng số: sum(w_y * CE) / sum(w_y).
    """
    w = tf.constant([float(v) for v in class_weights], dtype=tf.float32)

    def cls_loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(tf.cast(y_pred, tf.float32), 1e-7, 1.0)
        ce = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
        sw = tf.reduce_sum(y_true * w, axis=-1)
        return tf.reduce_sum(sw * ce) / (tf.reduce_sum(sw) + 1e-7)

    return cls_loss


def get_dual_decoder_losses(weights=None, boundary_mode="class", pos_weight=1.0, sdf_cfg=None, cls_cfg=None):
    """
    Returns loss dictionary and loss weights for Keras model compilation.
    
    Model outputs expected:
      - 'region_output': Intermediate Region Segmentation
      - 'boundary_output': Intermediate Boundary Detection
            boundary_mode="class"  -> boundary_bce_dice_loss (3 kênh, như cũ)
            boundary_mode="binary" -> make_binary_boundary_bce_dice_loss(pos_weight) (1 kênh)
      - 'refined_output': Final Refined Region Segmentation
      - 'sdf_output' (tuỳ chọn): SmoothL1 có trọng số gần biên. Trọng số lấy từ weights['sdf_output'].
            sdf_cfg: dict TRUNC / NEAR_PX / NEAR_WEIGHT / HUBER_DELTA.
    Loss 'sdf_output' chỉ có khi truyền sdf_cfg (không None); train script lọc theo model.output_names.
      - 'cls_output' (tuỳ chọn): CE cấp ảnh có trọng số lớp, khi truyền cls_cfg (dict CLASS_WEIGHTS).
    """
    if weights is None:
        weights = {
            'region_output': 0.5,
            'boundary_output': 1.0,
            'refined_output': 1.0
        }

    boundary_mode = str(boundary_mode).lower()
    if boundary_mode == "binary":
        boundary_loss = make_binary_boundary_bce_dice_loss(pos_weight=pos_weight)
    elif boundary_mode == "class":
        boundary_loss = boundary_bce_dice_loss
    else:
        raise ValueError(f"boundary_mode không hợp lệ: '{boundary_mode}' (class | binary)")

    losses = {
        'region_output': region_dice_ce_loss,
        'boundary_output': boundary_loss,
        'refined_output': region_dice_ce_loss
    }
    if sdf_cfg is not None:
        sdf_cfg = dict(sdf_cfg)
        losses['sdf_output'] = make_sdf_loss(
            trunc=sdf_cfg.get("TRUNC", 20.0),
            near_px=sdf_cfg.get("NEAR_PX", 5.0),
            near_weight=sdf_cfg.get("NEAR_WEIGHT", 5.0),
            huber_delta=sdf_cfg.get("HUBER_DELTA", 0.1),
        )
    if cls_cfg is not None:
        losses['cls_output'] = make_cls_loss(list(dict(cls_cfg).get("CLASS_WEIGHTS", [1.0, 1.0, 1.0])))

    return losses, weights
