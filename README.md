# Dual-Decoder ResNet for Bone Tumor Segmentation (BTXRD)

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/Framework-TensorFlow%202.x-orange.svg)](https://tensorflow.org/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-green.svg)](https://hydra.cc/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end deep learning framework for precise bone tumor segmentation in radiographs (BTXRD dataset), featuring a **Dual-Decoder ResNet** architecture with **Skip-Connection Attention Gates**, **Boundary-Guided Feature Fusion**, and **On-the-Fly Boundary Generation**.

---

## 🌟 Key Highlights

- **Dual-Decoder Architecture**: Decoupled feature learning via:
  - **Region Decoder**: Captures semantic tumor regions (Benign / Malignant).
  - **Boundary Decoder**: Focuses explicitly on sharp tumor margins and contours.
- **Skip Connection Attention Gates**: Filters irrelevant background noise and selectively highlights salient encoder features before concatenation.
- **Boundary-Guided Fusion & Refinement**: Leverages high-frequency edge signals to refine blurred borders and eliminate false boundaries.
- **On-the-Fly Boundary Generation (`DualDecoderWrapper`)**: Computes ground truth boundary maps directly from standard polygon segmentation masks in real-time via morphological operations — **zero manual boundary preprocessing required**.
- **Multi-Task Objective**: Joint optimization using Weighted Region Dice-CE Loss and Boundary BCE-Dice Loss.

---

## 🏛️ Architecture Overview

