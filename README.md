# Deepfake Frame Detector (ResNet-50 + Streamlit)

This project is a **deepfake image/frame detector** built with **PyTorch** and **Streamlit**. It takes a single face frame as input and classifies it as **real** or **fake** using a fine‑tuned ResNet‑50 convolutional neural network.

The project is designed as a portfolio‑ready end‑to‑end pipeline:
- Kaggle notebook for training and evaluation.
- Saved `.pth` model for inference.
- Streamlit web app for interactive demos.

---

## 1. Dataset

I use the Kaggle dataset:

- **Extracted Deepfake Frames**  
  https://www.kaggle.com/datasets/yuvrajpaikhot/extracted-deepfake-frames [web:18][web:19]

Key points:

- Frames extracted from real and deepfake videos.
- Organized into `train/`, `val/`, `test/` with subfolders for real and fake frames (e.g. `real_*`, `fake_*`).[web:18]
- Binary labels:  
  - `real_*` folders → real (label 0)  
  - `fake_*` folders → fake (label 1)

The training notebook uses `kagglehub.dataset_download` to pull the dataset inside a Kaggle environment.

---

## 2. Model and architecture

The detector is based on **ResNet‑50** from `torchvision.models`.[web:24][web:59]

- Backbone: 50‑layer residual network with bottleneck blocks and skip connections.[web:57][web:59]
- Final layer replaced by a custom head:
  - `Dropout(p=0.3)`
  - `Linear(in_features → 2)` for `[real, fake]`.
- Loss: `CrossEntropyLoss` with label smoothing.
- Optimizer: `AdamW` with weight decay and a StepLR scheduler.
- Input size: `224 × 224` RGB, normalized with ImageNet mean/std.[web:24][web:27]

The model is trained on frame‑level labels (no temporal/video information).

---

## 3. Training pipeline (Kaggle notebook)

The Kaggle notebook (`deepfake_resnet50_training.ipynb`) does:

1. **Data loading**
   - Download dataset via KaggleHub.
   - Collect image paths from `train/` and `val/` splits.
   - Assign binary labels (0=real, 1=fake).
   - Stratified `train_test_split` for train/validation.

2. **Preprocessing & augmentation**
   - Library: **Albumentations**.[web:23]
   - Train transforms:
     - Resize to `224 × 224`
     - Horizontal flip
     - Random brightness/contrast
     - Mild shift/scale/rotate
     - Gaussian blur
     - Normalize (ImageNet mean/std)
     - Convert to tensor
   - Validation transforms:
     - Resize + normalize + tensor.

3. **Training**
   - ResNet‑50 model with 2‑class head.
   - Cross‑entropy + label smoothing.
   - AdamW optimizer, StepLR scheduler.
   - Batch size: typically 16.
   - Save best model to `models/deepfake_resnet50_clean.pth` based on validation accuracy.[web:51][web:53]

4. **Evaluation**
   - Per‑epoch train & validation loss/accuracy.
   - Final classification report and confusion matrix on validation set.
   - Sanity check on a random validation image (print path, predicted class, and probabilities).

---

## 4. Streamlit app (local inference)

The `app.py` file provides a minimal Streamlit interface:

- Loads the trained model from:
  - `models/deepfake_resnet50_clean.pth`
- Uses the **same validation transform** as in training:
  - Resize to 224×224
  - ImageNet normalization
- UI features:
  - File uploader for `.jpg`, `.jpeg`, `.png`.
  - Displays uploaded image.
  - On “Analyze”, runs inference and shows:
    - Predicted label (`REAL` or `FAKE`)
    - Probabilities for each class
    - Simple progress bars for real/fake probabilities.

This allows quick demos on CPU, suitable for internship or CV presentation.[web:42][web:47][web:50]

---

## 5. Installation and usage

### 5.1. Environment setup (Anaconda)

```bash
# Create env
conda create -n deepfake-env python=3.10 -y
conda activate deepfake-env

# Install dependencies
pip install -r requirements.txt
