"""
UNet 3+ Loss
"""
import tensorflow as tf
import tensorflow.keras.backend as K
from .loss import focal_loss, ssim_loss, iou_loss

def unet3p_hybrid_loss(y_true, y_pred):
    """
    Hybrid loss proposed in
    UNET 3+ (https://arxiv.org/ftp/arxiv/papers/2004/2004.08790.pdf)
    """
    f_loss = focal_loss(y_true, y_pred)
    ms_ssim_loss = ssim_loss(y_true, y_pred)
    jacard_loss = iou_loss(y_true, y_pred)

    return f_loss + ms_ssim_loss + jacard_loss

def weighted_dice_loss(y_true, y_pred):
    """
    Weighted Dice Loss để trị mất cân bằng class.
    Trọng số: [Nền: 0.1, U Lành: 0.3, U Ác: 0.8]
    U ác thiểu số nên được đặt trọng số cao nhất.
    """
    weights = tf.constant([0.1, 0.3, 0.8], dtype=tf.float32)
    
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Tính Dice cho từng class (axis = [1, 2] tức là chiều Height, Width)
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    
    dice = (2. * intersection + 1e-7) / (union + 1e-7)
    
    # Loss = 1 - Dice, sau đó nhân với trọng số phạt
    loss = (1.0 - dice) * weights
    
    # Lấy trung bình loss của cả 3 class để backpropagate
    return tf.reduce_mean(loss)

def paper2_cedice_loss(y_true, y_pred):
    """
    CE-pp2 Loss: Combine Cross-Entropy and Dice Loss for segmentation.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    loss_dice = 1.0 - tf.reduce_mean(dice)

    # Cross Entropy Loss
    y_pred_clipped = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    loss_ce = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred_clipped), axis=-1))

    return loss_dice + loss_ce

def paper3_point_dice_loss(y_true, y_pred):
    """
    Point-wise Dice Loss.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1.0 - (2.0 * intersection + 1e-6) / (union + 1e-6)

def hybrid_boundary_loss(y_true, y_pred):
    """
    Hybrid Boundary Loss: Combine Region CE-Dice Loss with Boundary Dice Loss.
    """
    from utils.boundary_utils import extract_boundary_tf
    ce_dice = paper2_cedice_loss(y_true, y_pred)

    y_true_b = extract_boundary_tf(y_true)
    y_pred_b = extract_boundary_tf(y_pred)

    # Boundary Dice loss
    intersection = tf.reduce_sum(y_true_b * y_pred_b, axis=[1, 2])
    union = tf.reduce_sum(y_true_b, axis=[1, 2]) + tf.reduce_sum(y_pred_b, axis=[1, 2])
    dice_b = (2.0 * intersection + 1e-6) / (union + 1e-6)
    loss_dice_b = 1.0 - tf.reduce_mean(dice_b)

    return ce_dice + 0.5 * loss_dice_b

def hybrid_abedice_loss(y_true, y_pred):
    """
    Adaptive Boundary-Enhanced Dice Loss.
    """
    return hybrid_boundary_loss(y_true, y_pred)

def classification_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss for Classification (CGM module).
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    bce = - (y_true * tf.math.log(y_pred) + (1.0 - y_true) * tf.math.log(1.0 - y_pred))
    return tf.reduce_mean(bce)