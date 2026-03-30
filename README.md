<div align="center">
  <h1>🦅 Aztec Detector Pipeline</h1>
  <p><strong>A high-performance dual-stage visual detection and classification system for complex objects in natural environments.</strong></p>
</div>

---

## 🔎 Overview

The **Aztec Detector** is an advanced Computer Vision pipeline built to precisely locate and classify specific components/objects (e.g. hardware or environmental anomalies) across diverse imaging conditions. 

By strategically chaining a highly-tuned regional object detector with a deep-feature image classifier, this project drastically minimizes false positives while ensuring state-of-the-art recall!

### ⚙️ The Dual-Stage Architecture

1. **Stage 1: Object Localization (YOLOv8)**  
   The image is first swept by a fine-tuned **YOLOv8** model (`ultralytics`), generating accurate bounding boxes and isolating regions of interest. YOLO is optimized here for high-speed, general-purpose localization.

2. **Stage 2: Target Classification (ResNet18)**  
   Each detected crop is passed to a dedicated **ResNet18** model (`torchvision`). This deep CNN extracts complex embeddings to classify the precise nature of the detected object, determining whether the patch is truly the target or a visual artifact.

---

## 🚀 Setup & Installation

### Requirements

- Python 3.9+
- A compatible GPU (CUDA or MPS/M1 environment) for high-speed inference.

```bash
git clone https://github.com/litvakda/aztec-detector.git
cd aztec-detector

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 🏃‍♂️ Running the Pipeline

To execute the detector on a batch of unseen images:

1. Place your target images inside `src/core/test_images/`.
2. Ensure you have the corresponding weights loaded into `/runs` and `/clf_models`. (Note: weights are omitted from source control due to limits; contact for weights).
3. Run the inference pipeline:

```bash
python src/core/pipeline.py
```

### 🧠 Core Modules

- `pipeline.py`: The entry point script coordinating both YOLO detection and ResNet classification.
- `train_classifier.py` / `pipeline_esrgan.py`: Scripts handling model training and advanced resolution preprocessing.
- `/preprocessing`: Additional tools for resizing, data augmentation, and dataset splitting required to shape complex input datasets for CNN consumption.

<div align="center">
  <i>Pushing constraints in real-world ML engineering.</i>
</div>
