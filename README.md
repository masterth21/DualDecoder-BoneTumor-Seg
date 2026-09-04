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

```mermaid
flowchart TD
    In["📷 Input Radiograph (384x384x3)"] --> Enc["🧠 ResNet Backbone Encoder"]

    subgraph Skips ["Skip Connection Attention Gates"]
        S1["Attention Gate 1 (192x192)"]
        S2["Attention Gate 2 (96x96)"]
        S3["Attention Gate 3 (48x48)"]
        S4["Attention Gate 4 (24x24)"]
    end

    Enc --> S1
    Enc --> S2
    Enc --> S3
    Enc --> S4
    Enc --> Bottleneck["Bottleneck (12x12)"]

    Bottleneck --> RegDec["🟦 Region Decoder Branch<br/>(Semantic Tumor Area)"]
    Bottleneck --> BoundDec["🟩 Boundary Decoder Branch<br/>(Edge & Fine Contours)"]

    S1 -.-> RegDec
    S2 -.-> RegDec
    S3 -.-> RegDec
    S4 -.-> RegDec

    S1 -.-> BoundDec
    S2 -.-> BoundDec
    S3 -.-> BoundDec
    S4 -.-> BoundDec

    RegDec --> RegOut["🎯 Region Output<br/>(Auxiliary Loss)"]
    BoundDec --> BoundOut["🎯 Boundary Output<br/>(Auxiliary Loss)"]

    RegDec --> Fusion["⚡ Boundary-Guided Fusion Module"]
    BoundDec --> Fusion

    Fusion --> Refine["🔬 Residual Refinement Module"]
    RegOut -.-> Refine
    BoundOut -.-> Refine

    Refine --> FinalOut["🏆 Final Refined Mask<br/>(Background / Benign / Malignant)"]

    classDef default fill:#f9f9f9,stroke:#333,stroke-width:1px;
    classDef highlight fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    classDef final fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    class FinalOut final;
    class Fusion,Refine highlight;



---

## 📁 Repository Structure

```plaintext
├── configs/
│   └── config.yaml             # Experiment hyperparameters & dataset paths
├── data_generators/
│   ├── data_generator.py       # DualDecoderWrapper for on-the-fly targets
│   └── tf_data_generator.py    # Batch loader & augmentations
├── losses/
│   └── dual_decoder_loss.py    # Region & Boundary multi-task loss functions
├── models/
│   ├── dual_decoder_resnet.py  # Core Dual-Decoder ResNet architecture
│   └── model.py                # Model dispatcher
├── utils/
│   └── boundary_utils.py       # Morphological boundary extraction
├── train_dual_decoder.py       # Dedicated training pipeline
├── evaluate.py                 # Evaluation & metrics script
└── requirements.txt            # Python dependencies
