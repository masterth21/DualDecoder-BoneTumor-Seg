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
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate Dice per channel (axis=[1, 2] over Height & Width)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
    loss = tf.reduce_sum((1.0 - dice_per_class) * class_weights, axis=-1)
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
    Phạt nặng việc chỉ dự đoán nền (background) để buộc mô hình tập trung phát hiện khối u.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # 1. Per-Class Dice Loss (tính riêng từng kênh u dọc theo [H, W])
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice_per_class = (2.0 * intersection + smooth) / (union + smooth)

    # Trọng số: Nền: 0.1, U Lành: 0.45, U Ác: 0.45
    class_weights = tf.constant([0.1, 0.45, 0.45], dtype=tf.float32)
    loss_dice = tf.reduce_mean(tf.reduce_sum((1.0 - dice_per_class) * class_weights, axis=-1))

    # 2. Weighted Cross-Entropy Loss
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    ce_per_class = -y_true * tf.math.log(y_pred_clipped)
    loss_ce = tf.reduce_mean(tf.reduce_sum(ce_per_class * class_weights, axis=-1))

    return loss_dice + loss_ce


def get_dual_decoder_losses(weights=None):
    """
    Returns loss dictionary and loss weights for Keras model compilation.
    
    Model outputs expected:
      - 'region_output': Intermediate Region Segmentation
      - 'boundary_output': Intermediate Boundary Detection
      - 'refined_output': Final Refined Region Segmentation
    """
    if weights is None:
        weights = {
            'region_output': 0.5,
            'boundary_output': 1.0,
            'refined_output': 1.0
        }

    losses = {
        'region_output': region_dice_ce_loss,
        'boundary_output': boundary_bce_dice_loss,
        'refined_output': region_dice_ce_loss
    }

    return losses, weights
