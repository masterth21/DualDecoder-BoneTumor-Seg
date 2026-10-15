"""
Throughput and latency benchmark for X-ray Augmenter and DataGenerator.
Measures batch/s, samples/s across presets: none, light, medium,
and isolates component latencies (Elastic, CLAHE, Affine, Intensity).
"""

import time
import os
import sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_generators.augmentations import XrayAugmenter, AUGMENTATION_PRESETS
from tests.test_augmentations import make_test_sample


def benchmark_components(n_runs=100, size=384):
    print("=" * 60)
    print("BENCHMARK: Individual Augmentation Operations (ms per 384x384 image)")
    print("=" * 60)
    img, mask = make_test_sample(h=size, w=size, has_tumor=True, tumor_class=1)

    # 1. Flip
    aug_flip = XrayAugmenter({"ENABLE": True, "PRESET": "none", "HORIZONTAL_FLIP": {"PROB": 1.0}}, seed=42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = aug_flip(img, mask)
    t_flip = (time.perf_counter() - t0) / n_runs * 1000

    # 2. Affine (Rot + Zoom + Shift)
    aug_affine = XrayAugmenter({
        "ENABLE": True, "PRESET": "none",
        "ROTATION": {"PROB": 1.0, "ANGLE_RANGE": [-10, 10]},
        "ZOOM": {"PROB": 1.0, "RANGE": [0.9, 1.1]},
        "SHIFT": {"PROB": 1.0, "RANGE": [-0.05, 0.05]}
    }, seed=42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = aug_affine(img, mask)
    t_affine = (time.perf_counter() - t0) / n_runs * 1000

    # 3. Elastic deformation (coarse grid)
    aug_elastic = XrayAugmenter({
        "ENABLE": True, "PRESET": "none",
        "ELASTIC": {"PROB": 1.0, "ALPHA": 15.0, "SIGMA": 4.0}
    }, seed=42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = aug_elastic(img, mask)
    t_elastic = (time.perf_counter() - t0) / n_runs * 1000

    # 4. CLAHE
    aug_clahe = XrayAugmenter({
        "ENABLE": True, "PRESET": "none",
        "CLAHE": {"PROB": 1.0, "CLIP_LIMIT": 2.0, "TILE_GRID_SIZE": 8}
    }, seed=42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = aug_clahe(img, mask)
    t_clahe = (time.perf_counter() - t0) / n_runs * 1000

    # 5. Photometric (Brightness, Contrast, Gamma, Noise, Blur)
    aug_photo = XrayAugmenter({
        "ENABLE": True, "PRESET": "none",
        "BRIGHTNESS": {"PROB": 1.0, "RANGE": [-0.08, 0.08]},
        "CONTRAST": {"PROB": 1.0, "RANGE": [0.9, 1.1]},
        "GAMMA": {"PROB": 1.0, "RANGE": [0.85, 1.15]},
        "GAUSSIAN_NOISE": {"PROB": 1.0, "STDDEV": 0.01},
        "GAUSSIAN_BLUR": {"PROB": 1.0, "SIGMA_RANGE": [0.5, 1.0]}
    }, seed=42)
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = aug_photo(img, mask)
    t_photo = (time.perf_counter() - t0) / n_runs * 1000

    print(f"  Horizontal Flip:    {t_flip:6.2f} ms")
    print(f"  Combined Affine:    {t_affine:6.2f} ms")
    print(f"  Elastic (Coarse):   {t_elastic:6.2f} ms")
    print(f"  CLAHE:              {t_clahe:6.2f} ms")
    print(f"  Photometric bundle: {t_photo:6.2f} ms")
    print()


def benchmark_presets(batch_size=2, n_batches=100, size=384):
    print("=" * 60)
    print(f"BENCHMARK: TRAIN Generator Throughput (Batch Size = {batch_size}, Size = {size}x{size})")
    print("=" * 60)

    # Pre-generate sample pool of images & masks
    pool_size = 20
    images = []
    masks = []
    for i in range(pool_size):
        im, mk = make_test_sample(h=size, w=size, has_tumor=(i % 3 != 0), tumor_class=(i % 2 + 1))
        images.append(im)
        masks.append(mk)

    results = {}
    for preset in ["none", "light", "medium"]:
        cfg_dict = {"ENABLE": preset != "none", "PRESET": preset}
        augmenter = XrayAugmenter(cfg_dict, seed=42)

        # Warmup
        for b in range(5):
            for i in range(batch_size):
                idx = (b * batch_size + i) % pool_size
                _ = augmenter(images[idx], masks[idx])

        # Timed run
        t0 = time.perf_counter()
        total_samples = 0
        for b in range(n_batches):
            for i in range(batch_size):
                idx = (b * batch_size + i) % pool_size
                _ = augmenter(images[idx], masks[idx])
                total_samples += 1
        elapsed = time.perf_counter() - t0

        batches_per_sec = n_batches / elapsed
        samples_per_sec = total_samples / elapsed
        ms_per_batch = (elapsed / n_batches) * 1000
        ms_per_sample = (elapsed / total_samples) * 1000

        results[preset] = {
            "batches_per_sec": batches_per_sec,
            "samples_per_sec": samples_per_sec,
            "ms_per_batch": ms_per_batch,
            "ms_per_sample": ms_per_sample
        }

        print(f"Preset '{preset:<6}': {batches_per_sec:6.2f} batch/s | {samples_per_sec:6.2f} samples/s | {ms_per_batch:6.2f} ms/batch ({ms_per_sample:5.2f} ms/sample)")

    print("-" * 60)
    r_none = results["none"]["batches_per_sec"]
    r_light = results["light"]["batches_per_sec"]
    r_med = results["medium"]["batches_per_sec"]
    ratio = r_none / max(1e-5, r_med)
    print(f"Throughput Ratio (none / medium): {ratio:.2f}x")
    if ratio > 2.0:
        print(f"  -> Medium is {ratio:.2f}x slower than none: speedup proposal included.")
    else:
        print(f"  -> Medium is {ratio:.2f}x of none (within 2x budget): highly efficient.")
    print("=" * 60)
    return results


if __name__ == "__main__":
    benchmark_components()
    benchmark_presets(batch_size=2, n_batches=100, size=384)
