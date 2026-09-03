"""
Loss functions for Dual-Decoder (Region + Boundary) Architecture.
Includes Boundary BCE, Boundary Dice Loss, Region Dice/CE Loss, and Multi-Output Loss handlers.
"""

import tensorflow as tf
import tensorflow.keras.backend as K


def boundary_dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice Loss specially computed for sparse boundary maps.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return K.mean(1.0 - dice)


def boundary_bce_loss(y_true, y_pred):
    """
    Binary Cross-Entropy Loss for boundary prediction.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # Clip probabilities to avoid log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    bce = - (y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred))
    return K.mean(bce)


def boundary_bce_dice_loss(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Combined Boundary Loss: alpha * Boundary_BCE + beta * Boundary_Dice.
    """
    bce = boundary_bce_loss(y_true, y_pred)
    dice = boundary_dice_loss(y_true, y_pred)
    return alpha * bce + beta * dice


def region_dice_ce_loss(y_true, y_pred, smooth=1e-6):
    """
    Region Loss: Soft Cross-Entropy + Dice Loss for Region Segmentation.
    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Dice Loss
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + smooth) / (union + smooth)
    loss_dice = 1.0 - K.mean(dice)

    # Cross Entropy Loss
    y_pred_clipped = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    loss_ce = K.mean(-K.sum(y_true * K.log(y_pred_clipped), axis=-1))

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
