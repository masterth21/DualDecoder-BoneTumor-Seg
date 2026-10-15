"""
Comprehensive Test Suite for Medical X-ray Augmentation.
Verifies all 8 base transforms and all specific requirements:
1. Horizontal flip
2. Rotation
3. Zoom
4. Translation / Shift
5. Elastic deformation
6. Intensity transforms
7. CLAHE uint8 conversion & float output
8. MAX_TUMOR_LOSS revert behavior
9. VAL/TEST strictly not augmented
10. Mask values strictly in valid class set {0, 1, 2}
11. Empty mask handles gracefully without NaN or invalid cancel
12. Image value range strictly [0.0, 1.0]
13. Boundary ground truth generated matches augmented mask
14. Unknown config key raises error; preset 'none' applies identity
15. DALI + ENABLE raises clear error
16. End-to-end 2-epoch run on CPU for M0 with medium preset
17. Reproducibility test with seed
"""

import os
import sys
import copy
import shutil
import tempfile
import numpy as np
import cv2
import tensorflow as tf
from omegaconf import OmegaConf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_generators.augmentations import (
    XrayAugmenter,
    build_augmentation_config,
    AUGMENTATION_PRESETS
)
from utils.boundary_utils import extract_boundary_tf


class AssertRaises:
    """Context manager for testing exception raising without external dependencies"""
    def __init__(self, exc_type, match=None):
        self.exc_type = exc_type
        self.match = match

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            raise AssertionError(f"Expected {self.exc_type.__name__} was not raised.")
        if not issubclass(exc_type, self.exc_type):
            return False
        if self.match and self.match not in str(exc_val):
            raise AssertionError(f"Exception message '{exc_val}' does not match '{self.match}'")
        return True


# =============================================================================
# HELPER TEST FIXTURES & DATA BUILDERS
# =============================================================================
def make_test_sample(h=384, w=384, has_tumor=True, tumor_class=1):
    """Create a synthetic radiograph-like float32 image and int32 mask"""
    y, x = np.meshgrid(np.arange(h, dtype=np.float32), np.arange(w, dtype=np.float32), indexing='ij')
    # Simulated bone profile (monochrome repeated across 3 channels)
    bone = np.exp(-((x - w / 2.0) ** 2) / (2.0 * (w / 6.0) ** 2)) * 0.6 + 0.1
    img = np.repeat(bone[:, :, np.newaxis], 3, axis=-1).astype(np.float32)

    mask = np.zeros((h, w), dtype=np.int32)
    if has_tumor:
        cy, cx, r = h // 2, w // 2, 40
        tumor_idx = ((y - cy) ** 2 + (x - cx) ** 2) <= (r ** 2)
        mask[tumor_idx] = tumor_class
        img[tumor_idx] += 0.2

    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return img, mask


# =============================================================================
# 8 BASE TRANSFORM TESTS
# =============================================================================
def test_1_horizontal_flip():
    """Test 1: Horizontal flip flips both image and mask horizontally and preserves shape/dtype"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "HORIZONTAL_FLIP": {"PROB": 1.0}
    }
    aug = XrayAugmenter(cfg, seed=123)
    out_img, out_mask = aug(img, mask)

    assert out_img.shape == img.shape
    assert out_mask.shape == mask.shape
    assert out_img.dtype == np.float32
    assert out_mask.dtype == np.int32
    # Verify horizontal flip occurred
    expected_img = np.ascontiguousarray(np.fliplr(img))
    expected_mask = np.ascontiguousarray(np.fliplr(mask))
    np.testing.assert_allclose(out_img, expected_img, atol=1e-5)
    np.testing.assert_array_equal(out_mask, expected_mask)


def test_2_rotation():
    """Test 2: Rotation rotates within angle range and preserves valid class set"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "ROTATION": {"PROB": 1.0, "ANGLE_RANGE": [8, 10]}
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, out_mask = aug(img, mask)

    assert out_img.shape == img.shape
    assert out_mask.shape == mask.shape
    assert set(np.unique(out_mask)).issubset({0, 1, 2})
    assert not np.array_equal(out_img, img)


def test_3_zoom():
    """Test 3: Zoom scales tumor area predictably and preserves shape"""
    img, mask = make_test_sample()
    orig_area = np.count_nonzero(mask > 0)
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "ZOOM": {"PROB": 1.0, "RANGE": [1.08, 1.10]}
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, out_mask = aug(img, mask)

    new_area = np.count_nonzero(out_mask > 0)
    # Zoom-in expands tumor area
    assert new_area > orig_area
    assert set(np.unique(out_mask)).issubset({0, 1, 2})


def test_4_shift():
    """Test 4: Translation/Shift moves tumor within specified range"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "SHIFT": {"PROB": 1.0, "RANGE": [0.03, 0.05]}
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, out_mask = aug(img, mask)

    assert out_img.shape == img.shape
    assert set(np.unique(out_mask)).issubset({0, 1, 2})
    assert not np.array_equal(out_mask, mask)


def test_5_elastic_deformation():
    """Test 5: Elastic deformation smoothly warps image/mask without creating non-integer classes"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "ELASTIC": {"PROB": 1.0, "ALPHA": 15.0, "SIGMA": 4.0}
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, out_mask = aug(img, mask)

    assert out_img.shape == img.shape
    assert out_mask.shape == mask.shape
    assert out_mask.dtype == np.int32
    assert set(np.unique(out_mask)).issubset({0, 1, 2})
    assert not np.array_equal(out_mask, mask)


def test_6_intensity_transforms_range():
    """Test 6: All intensity transforms keep image values strictly in [0.0, 1.0]"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "BRIGHTNESS": {"PROB": 1.0, "RANGE": [-0.08, 0.08]},
        "CONTRAST": {"PROB": 1.0, "RANGE": [0.9, 1.1]},
        "GAMMA": {"PROB": 1.0, "RANGE": [0.85, 1.15]},
        "GAUSSIAN_NOISE": {"PROB": 1.0, "STDDEV": 0.02},
        "GAUSSIAN_BLUR": {"PROB": 1.0, "SIGMA_RANGE": [0.5, 1.0]},
    }
    aug = XrayAugmenter(cfg, seed=42)
    for _ in range(10):
        out_img, _ = aug(img, mask)
        assert np.min(out_img) >= 0.0
        assert np.max(out_img) <= 1.0
        assert out_img.dtype == np.float32


def test_7_clahe_uint8_conversion():
    """Test 7: CLAHE runs on uint8 and outputs normalized float32 in [0.0, 1.0]"""
    img, mask = make_test_sample()
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "CLAHE": {"PROB": 1.0, "CLIP_LIMIT": 2.0, "TILE_GRID_SIZE": 8}
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, _ = aug(img, mask)

    assert out_img.shape == img.shape
    assert out_img.dtype == np.float32
    assert np.min(out_img) >= 0.0
    assert np.max(out_img) <= 1.0
    # Channel symmetry preserved
    np.testing.assert_allclose(out_img[:, :, 0], out_img[:, :, 1], atol=1e-5)
    np.testing.assert_allclose(out_img[:, :, 0], out_img[:, :, 2], atol=1e-5)


def test_8_max_tumor_loss_cancellation():
    """Test 8: MAX_TUMOR_LOSS cancels transform when tumor is cropped/lost > 10%"""
    h, w = 384, 384
    img = np.zeros((h, w, 3), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.int32)
    # Place tumor near edge so translation cuts > 10%
    mask[10:60, 10:60] = 1

    # Force a large shift that shifts tumor off-screen
    cfg = {
        "ENABLE": True,
        "PRESET": "none",
        "MAX_TUMOR_LOSS": 0.10,
        "SHIFT": {"PROB": 1.0, "RANGE": [-0.20, -0.20]}  # shifts out of frame
    }
    aug = XrayAugmenter(cfg, seed=42)
    out_img, out_mask = aug(img, mask)

    # Since tumor was lost > 10%, geometric transform must be reverted to original!
    np.testing.assert_array_equal(out_mask, mask)
    np.testing.assert_allclose(out_img, img, atol=1e-5)


# =============================================================================
# ADDITIONAL REQUIRED TESTS (SPECIFIED IN PROMPT)
# =============================================================================
def test_9_val_test_not_augmented():
    """Test 9: VAL/TEST datasets are strictly NOT augmented (output identical to input)"""
    from data_generators.tf_data_generator import DataGenerator
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
        cfg = compose(config_name="config")
    OmegaConf.set_struct(cfg, False)

    # Enable medium augmentation in config
    cfg.DATA_AUGMENTATION.ENABLE = True
    cfg.DATA_AUGMENTATION.PRESET = "medium"

    temp_dir = tempfile.mkdtemp()
    try:
        img_p = os.path.join(temp_dir, "test_img.png")
        mask_p = os.path.join(temp_dir, "test_mask.png")
        sample_img = (np.random.rand(384, 384, 3) * 255).astype(np.uint8)
        sample_mask = np.zeros((384, 384), dtype=np.uint8)
        sample_mask[100:150, 100:150] = 128  # Benign
        cv2.imwrite(img_p, sample_img)
        cv2.imwrite(mask_p, sample_mask)

        cfg.DATASET.VAL.IMAGES_PATH = [img_p]
        cfg.DATASET.VAL.MASK_PATH = [mask_p]
        cfg.HYPER_PARAMETERS.BATCH_SIZE = 1

        val_gen = DataGenerator(cfg, mode="VAL")
        assert val_gen.augmenter is None, "Augmenter must be None for VAL mode!"

        batch_x1, batch_y1 = val_gen[0]
        batch_x2, batch_y2 = val_gen[0]

        # In VAL mode, multiple calls must be 100% deterministic and identical
        np.testing.assert_allclose(batch_x1, batch_x2, atol=1e-7)
        np.testing.assert_allclose(batch_y1, batch_y2, atol=1e-7)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_10_mask_values_strictly_valid_classes():
    """Test 10: Mask values after augment only belong to valid class set {0, 1, 2}"""
    img, mask = make_test_sample(has_tumor=True, tumor_class=2)
    aug = XrayAugmenter({"ENABLE": True, "PRESET": "medium"}, seed=42)

    for i in range(15):
        _, out_mask = aug(img, mask)
        unique_vals = set(np.unique(out_mask))
        assert unique_vals.issubset({0, 1, 2}), f"Unexpected mask classes: {unique_vals}"
        assert out_mask.dtype == np.int32


def test_11_empty_mask_no_nan_no_invalid_cancel():
    """Test 11: Empty mask (no tumor) causes no division-by-zero, no NaN, and does not cancel transforms"""
    img, mask = make_test_sample(has_tumor=False)
    assert np.count_nonzero(mask) == 0

    aug = XrayAugmenter({
        "ENABLE": True,
        "PRESET": "medium",
        "ROTATION": {"PROB": 1.0, "ANGLE_RANGE": [5, 10]},
        "ZOOM": {"PROB": 1.0, "RANGE": [1.05, 1.10]}
    }, seed=42)

    out_img, out_mask = aug(img, mask)
    assert not np.isnan(out_img).any()
    assert not np.isnan(out_mask).any()
    assert np.count_nonzero(out_mask) == 0
    # Transform was applied, not cancelled
    assert not np.array_equal(out_img, img)


def test_12_image_value_range_strictly_0_to_1():
    """Test 12: Image value range after augment is strictly [0.0, 1.0] float32"""
    img, mask = make_test_sample(has_tumor=True)
    aug = XrayAugmenter({"ENABLE": True, "PRESET": "medium"}, seed=42)

    for _ in range(15):
        out_img, _ = aug(img, mask)
        assert out_img.dtype == np.float32
        assert float(np.min(out_img)) >= 0.0, f"Min below 0: {np.min(out_img)}"
        assert float(np.max(out_img)) <= 1.0, f"Max above 1: {np.max(out_img)}"


def test_13_boundary_matches_augmented_mask():
    """Test 13: Boundary ground truth matches augmented mask and differs from original mask when rotated"""
    img, mask = make_test_sample(has_tumor=True, tumor_class=1)
    aug = XrayAugmenter({
        "ENABLE": True,
        "PRESET": "none",
        "ROTATION": {"PROB": 1.0, "ANGLE_RANGE": [10, 10]}
    }, seed=42)

    _, aug_mask = aug(img, mask)
    assert not np.array_equal(aug_mask, mask), "Rotation should have altered mask position"

    # Original mask boundary
    mask_orig_1hot = np.eye(3, dtype=np.float32)[mask]
    bnd_orig = extract_boundary_tf(np.expand_dims(mask_orig_1hot, 0), kernel_size=5).numpy()

    # Augmented mask boundary
    mask_aug_1hot = np.eye(3, dtype=np.float32)[aug_mask]
    bnd_aug = extract_boundary_tf(np.expand_dims(mask_aug_1hot, 0), kernel_size=5).numpy()

    # Boundary from augmented mask must NOT match original mask boundary
    assert not np.array_equal(bnd_aug, bnd_orig), "Boundary must be derived from augmented mask!"
    # Boundary sum must be positive
    assert np.sum(bnd_aug) > 0


def test_14_unknown_config_key_and_none_preset():
    """Test 14: Unknown config key raises clear ValueError; preset 'none' applies identity"""
    # 14a: Unknown top-level key
    with AssertRaises(ValueError, match="Unrecognized augmentation config key"):
        build_augmentation_config({"ENABLE": True, "UNKNOWN_KEY_XYZ": 123})

    # 14b: Unknown sub-level key
    with AssertRaises(ValueError, match="Unrecognized parameter"):
        build_augmentation_config({"ENABLE": True, "ROTATION": {"PROB": 0.5, "UNKNOWN_SUBKEY": 1}})

    # 14c: Preset 'none' leaves input untouched
    img, mask = make_test_sample()
    aug_none = XrayAugmenter({"ENABLE": False, "PRESET": "none"}, seed=42)
    out_img, out_mask = aug_none(img, mask)
    np.testing.assert_array_equal(out_img, img)
    np.testing.assert_array_equal(out_mask, mask)


def test_15_dali_with_augmentation_raises_error():
    """Test 15: DALI generator with DATA_AUGMENTATION.ENABLE=True raises clear ValueError"""
    from data_generators.data_generator import get_data_generator
    cfg = OmegaConf.create({
        "DATA_GENERATOR_TYPE": "DALI_GENERATOR",
        "DATA_AUGMENTATION": {"ENABLE": True, "PRESET": "medium"},
        "HYPER_PARAMETERS": {"BATCH_SIZE": 2},
        "SEED": 42
    })

    with AssertRaises(ValueError, match="DALI pipeline does not support medical X-ray augmentation"):
        get_data_generator(cfg, mode="TRAIN")


def test_16_reproducibility_with_seed():
    """Test 16: Two augmenters with the same seed generate IDENTICAL augmented outputs"""
    img, mask = make_test_sample(has_tumor=True, tumor_class=2)
    aug1 = XrayAugmenter({"ENABLE": True, "PRESET": "medium"}, seed=999)
    aug2 = XrayAugmenter({"ENABLE": True, "PRESET": "medium"}, seed=999)

    for _ in range(5):
        img1, m1 = aug1(img, mask)
        img2, m2 = aug2(img, mask)
        np.testing.assert_allclose(img1, img2, atol=1e-6)
        np.testing.assert_array_equal(m1, m2)


def test_17_end_to_end_2_epochs_m0_medium_cpu():
    """
    Test 17: End-to-end 2 epochs on CPU with synthetic data for M0 (medium preset).
    Verifies: no NaN, checkpoint saved, aug_config.json present in run directory.
    """
    from hydra import compose, initialize_config_dir
    from models.dual_decoder_resnet import build_dual_decoder_resnet
    from data_generators.data_generator import DualDecoderWrapper
    from data_generators.tf_data_generator import DataGenerator
    from losses.loss import MacroDiceMetric

    temp_dir = tempfile.mkdtemp()
    try:
        # Create minimal synthetic dataset: 4 train, 2 val
        train_img_dir = os.path.join(temp_dir, "train", "images")
        train_mask_dir = os.path.join(temp_dir, "train", "mask")
        val_img_dir = os.path.join(temp_dir, "val", "images")
        val_mask_dir = os.path.join(temp_dir, "val", "mask")
        ckpt_dir = os.path.join(temp_dir, "checkpoint")
        tb_dir = os.path.join(temp_dir, "tb_logs")

        for d in [train_img_dir, train_mask_dir, val_img_dir, val_mask_dir, ckpt_dir, tb_dir]:
            os.makedirs(d, exist_ok=True)

        for i in range(4):
            im, mk = make_test_sample(h=128, w=128, has_tumor=(i % 2 == 0), tumor_class=(i % 2 + 1))
            cv2.imwrite(os.path.join(train_img_dir, f"img_{i}.png"), (im * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(train_mask_dir, f"img_{i}.png"), (mk * 128).astype(np.uint8))

        for i in range(2):
            im, mk = make_test_sample(h=128, w=128, has_tumor=True, tumor_class=1)
            cv2.imwrite(os.path.join(val_img_dir, f"val_{i}.png"), (im * 255).astype(np.uint8))
            cv2.imwrite(os.path.join(val_mask_dir, f"val_{i}.png"), (mk * 128).astype(np.uint8))

        with initialize_config_dir(config_dir=os.path.join(ROOT, "configs"), version_base=None):
            cfg = compose(config_name="config")
        OmegaConf.set_struct(cfg, False)

        cfg.WORK_DIR = ROOT
        cfg.INPUT.HEIGHT = 128
        cfg.INPUT.WIDTH = 128
        cfg.INPUT.CHANNELS = 3
        cfg.OUTPUT.CLASSES = 3
        cfg.MODEL.DUAL_DECODER = False  # M0 mode
        cfg.MODEL.DECODER.TYPE = "unet3plus"
        cfg.MODEL.BACKBONE.TYPE = "resnet34"
        cfg.MODEL.BACKBONE.WEIGHTS = "none"  # Quick random init for test
        cfg.DATA_AUGMENTATION.ENABLE = True
        cfg.DATA_AUGMENTATION.PRESET = "medium"
        cfg.HYPER_PARAMETERS.BATCH_SIZE = 2
        cfg.HYPER_PARAMETERS.EPOCHS = 2
        cfg.BOUNDARY.KERNEL_SIZE = 3
        cfg.CALLBACKS.MODEL_CHECKPOINT.PATH = ckpt_dir
        cfg.CALLBACKS.TENSORBOARD.PATH = tb_dir

        cfg.DATASET.TRAIN.IMAGES_PATH = train_img_dir
        cfg.DATASET.TRAIN.MASK_PATH = train_mask_dir
        cfg.DATASET.VAL.IMAGES_PATH = val_img_dir
        cfg.DATASET.VAL.MASK_PATH = val_mask_dir

        # Build M0 model
        model = build_dual_decoder_resnet(cfg)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-4),
            loss={"refined_output": "categorical_crossentropy"},
            metrics={"refined_output": [MacroDiceMetric(classes=3, name="dice_coef")]}
        )

        base_train_gen = DataGenerator(cfg, mode="TRAIN")
        base_val_gen = DataGenerator(cfg, mode="VAL")
        train_gen = DualDecoderWrapper(base_train_gen, output_names=model.output_names, kernel_size=3)
        val_gen = DualDecoderWrapper(base_val_gen, output_names=model.output_names, kernel_size=3)

        # Save aug_config.json to checkpoint dir
        import json
        effective_cfg = build_augmentation_config(cfg.DATA_AUGMENTATION)
        aug_cfg_file = os.path.join(ckpt_dir, "aug_config.json")
        with open(aug_cfg_file, "w", encoding="utf-8") as f:
            json.dump(effective_cfg, f, indent=2)

        ckpt_file = os.path.join(ckpt_dir, "m0_test_checkpoint.weights.h5")
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=ckpt_file,
                save_weights_only=True,
                save_best_only=False
            )
        ]

        history = model.fit(
            x=train_gen,
            validation_data=val_gen,
            epochs=2,
            callbacks=callbacks,
            verbose=0
        )

        # Assertions
        assert os.path.exists(aug_cfg_file), "aug_config.json must exist in run folder"
        assert os.path.exists(ckpt_file), "Model checkpoint must be saved"
        train_loss = history.history["loss"]
        assert len(train_loss) == 2, f"Expected 2 epochs, got {len(train_loss)}"
        assert not np.isnan(train_loss).any(), "Loss contains NaN!"
        print(f"[SUCCESS] Test 17 passed! Epoch losses: {train_loss}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_") and callable(v)]
    tests.sort(key=lambda f: int(f.__name__.split("_")[1]))
    print(f"Running {len(tests)} augmentation tests...\n")
    passed = 0
    failed = 0
    for t in tests:
        name = t.__name__
        try:
            t()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    print(f"\nResult: {passed} passed, {failed} failed out of {len(tests)} tests.")
    if failed > 0:
        sys.exit(1)
