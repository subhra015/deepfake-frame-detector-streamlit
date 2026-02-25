import os
import numpy as np
from PIL import Image

import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------
# Config
# -------------------
CLASS_NAMES = ["real", "fake"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("models", "deepfake_resnet50_clean.pth")

# -------------------
# Transforms (same as validation in notebook)
# -------------------
def get_val_transform():
    return A.Compose(
        [
            A.Resize(IMAGE_SIZE, IMAGE_SIZE),
            A.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
            ToTensorV2(),
        ]
    )

# -------------------
# Model definition (must match training)
# -------------------
def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    # Use same architecture as training
    model = models.resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model

@st.cache_resource
def load_model():
    model = build_model(NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model

# -------------------
# Prediction helper
# -------------------
def predict_image(model, img_pil: Image.Image):
    # Convert PIL -> numpy -> transform -> tensor
    img = np.array(img_pil.convert("RGB"))
    transform = get_val_transform()
    img_t = transform(image=img)["image"].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(img_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    return pred_class, probs

# -------------------
# Streamlit UI
# -------------------
def main():
    st.title("Deepfake Frame Detector (ResNet-50)")
    st.write("Upload a face image/frame to check if it's **real** or **fake**.")

    model = load_model()

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Show the image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded image", use_column_width=True)

        if st.button("Analyze"):
            pred_class, probs = predict_image(model, img)

            prob_real = float(probs[0])
            prob_fake = float(probs[1])

            st.subheader(f"Prediction: {pred_class.upper()}")
            st.write(f"Real probability: {prob_real:.4f}")
            st.write(f"Fake probability: {prob_fake:.4f}")

            st.progress(int(prob_real * 100), text="Real")
            st.progress(int(prob_fake * 100), text="Fake")

if __name__ == "__main__":
    main()
