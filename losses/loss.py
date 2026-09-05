"""
Implementation of different loss functions
"""
import tensorflow as tf
import tensorflow.keras.backend as K


def iou(y_true, y_pred, smooth=1.e-9):
    """
    Calculate intersection over union (IoU) between images.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
    union = K.sum(y_true, [1, 2, 3]) + K.sum(y_pred, [1, 2, 3])
    union = union - intersection
    iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
    return iou


def iou_loss(y_true, y_pred):
    """
    Jaccard / IoU loss
    """
    return 1 - iou(y_true, y_pred)


def focal_loss(y_true, y_pred):
    """
    Focal loss
    """
    gamma = 2.
    alpha = 4.
    epsilon = 1.e-9

    y_true_c = tf.convert_to_tensor(y_true, tf.float32)
    y_pred_c = tf.convert_to_tensor(y_pred, tf.float32)

    model_out = tf.add(y_pred_c, epsilon)
    ce = tf.multiply(y_true_c, -tf.math.log(model_out))
    weight = tf.multiply(y_true_c, tf.pow(
        tf.subtract(1., model_out), gamma)
                         )
    fl = tf.multiply(alpha, tf.multiply(weight, ce))
    reduced_fl = tf.reduce_max(fl, axis=-1)
    return tf.reduce_mean(reduced_fl)


def ssim_loss(y_true, y_pred, smooth=1.e-9):
    """
    Structural Similarity Index loss.
    Input shape should be Batch x Height x Width x #Classes (BxHxWxN).
    Using Mean as reduction type for batch values.
    """
    ssim_value = tf.image.ssim(y_true, y_pred, max_val=1)
    return K.mean(1 - ssim_value + smooth, axis=0)


class MacroDiceMetric(tf.keras.metrics.Metric):
    """
    Keras Metric that calculates Macro Dice score for foreground tumor classes (Benign & Malignant).
    Accumulates TP, FP, FN across all batches in the epoch:
    1. Prevents 0/0=1.0 empty-class artifact on individual images (starts at 0.0000 on Epoch 0).
    2. True representation of segmentation overlap (does not divide tumor dice in half).
    3. Correctly triggers ReduceLROnPlateau and ModelCheckpoint when the model actually improves.
    """

    def __init__(self, classes: int = 3, name: str = 'dice_coef', post_processed: bool = True, **kwargs):
        super(MacroDiceMetric, self).__init__(name=name, **kwargs)
        self.classes = classes
        self.post_processed = post_processed
        self.num_fg = classes - 1 if classes > 1 else 1
        self.tp = self.add_weight(name='tp', shape=(self.num_fg,), initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', shape=(self.num_fg,), initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', shape=(self.num_fg,), initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        if self.classes == 1:
            y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
            y_true_bin = tf.cast(y_true > 0.5, tf.float32)
            tp = tf.reduce_sum(y_true_bin * y_pred_bin)
            fp = tf.reduce_sum((1.0 - y_true_bin) * y_pred_bin)
            fn = tf.reduce_sum(y_true_bin * (1.0 - y_pred_bin))
            self.tp.assign_add([tp])
            self.fp.assign_add([fp])
            self.fn.assign_add([fn])
        else:
            y_true_cls = tf.math.argmax(y_true, axis=-1, output_type=tf.int32)
            y_pred_cls = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)

            y_true_1hot = tf.one_hot(y_true_cls, self.classes, dtype=tf.float32)
            y_pred_1hot = tf.one_hot(y_pred_cls, self.classes, dtype=tf.float32)

            # Foreground tumor classes: 1 (Benign), 2 (Malignant)
            y_t_fg = y_true_1hot[..., 1:]
            y_p_fg = y_pred_1hot[..., 1:]

            tp = tf.reduce_sum(y_t_fg * y_p_fg, axis=[0, 1, 2])
            fp = tf.reduce_sum((1.0 - y_t_fg) * y_p_fg, axis=[0, 1, 2])
            fn = tf.reduce_sum(y_t_fg * (1.0 - y_p_fg), axis=[0, 1, 2])

            self.tp.assign_add(tp)
            self.fp.assign_add(fp)
            self.fn.assign_add(fn)

    def result(self):
        eps = 1e-7
        dice_per_class = (2.0 * self.tp + eps) / (2.0 * self.tp + self.fp + self.fn + eps)
        return tf.reduce_mean(dice_per_class)

    def reset_state(self):
        self.tp.assign(tf.zeros_like(self.tp))
        self.fp.assign(tf.zeros_like(self.fp))
        self.fn.assign(tf.zeros_like(self.fn))


# Backwards compatibility alias
DiceCoefficient = MacroDiceMetric


class ClassDiceMetric(tf.keras.metrics.Metric):
    """
    Keras Metric that calculates Dice score for a specific class index
    (class_id=1 for Benign, class_id=2 for Malignant).
    Accumulates TP, FP, FN across batches in the epoch natively.
    """
    def __init__(self, class_id: int, classes: int = 3, name: str = 'dice_class', **kwargs):
        super(ClassDiceMetric, self).__init__(name=name, **kwargs)
        self.class_id = class_id
        self.classes = classes
        self.tp = self.add_weight(name='tp', initializer='zeros', dtype=tf.float32)
        self.fp = self.add_weight(name='fp', initializer='zeros', dtype=tf.float32)
        self.fn = self.add_weight(name='fn', initializer='zeros', dtype=tf.float32)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_cls = tf.math.argmax(y_true, axis=-1, output_type=tf.int32)
        y_pred_cls = tf.math.argmax(y_pred, axis=-1, output_type=tf.int32)

        true_c = tf.cast(tf.equal(y_true_cls, self.class_id), tf.float32)
        pred_c = tf.cast(tf.equal(y_pred_cls, self.class_id), tf.float32)

        tp = tf.reduce_sum(true_c * pred_c)
        fp = tf.reduce_sum((1.0 - true_c) * pred_c)
        fn = tf.reduce_sum(true_c * (1.0 - pred_c))

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        eps = 1e-7
        return (2.0 * self.tp + eps) / (2.0 * self.tp + self.fp + self.fn + eps)

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.fn.assign(0.0)

def bmt_boundary_aware_loss(y_true, y_pred):
    """
    Bone Metastatic/Tumor Boundary-Aware Loss.
    Combines standard region Dice loss with boundary Binary Cross-Entropy loss.
    """
    from utils.boundary_utils import extract_boundary_tf
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Region Dice Loss
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2])
    union = tf.reduce_sum(y_true, axis=[1, 2]) + tf.reduce_sum(y_pred, axis=[1, 2])
    dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
    loss_dice = 1.0 - tf.reduce_mean(dice)

    # Boundary Extraction
    y_true_b = extract_boundary_tf(y_true)
    y_pred_b = extract_boundary_tf(y_pred)

    # Boundary BCE Loss
    y_pred_b = tf.clip_by_value(y_pred_b, 1e-7, 1.0 - 1e-7)
    loss_bce_b = tf.reduce_mean(-tf.reduce_sum(y_true_b * tf.math.log(y_pred_b), axis=-1))

    return loss_dice + loss_bce_b
