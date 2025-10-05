import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os

# ------------------ CONSTANTS ------------------
MODEL_PATH = "best_mri_classifier.h5"
DATA_DIR = "data/data"  # directory with subfolders = class names

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()
CLASS_NAMES = sorted(os.listdir(DATA_DIR))
st.sidebar.success(f"Loaded {len(CLASS_NAMES)} classes")
st.sidebar.write(CLASS_NAMES)

# ------------------ IMAGE PREPROCESS ------------------
def preprocess_image(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)
    return img

# ------------------ PREDICTION FUNCTION ------------------
def predict_image(image):
    img = preprocess_image(image)
    preds = model.predict(img)
    class_id = np.argmax(preds[0])
    confidence = preds[0][class_id] * 100
    return CLASS_NAMES[class_id], confidence

# ------------------ STREAMLIT UI ------------------
st.title("🧠 Brain Tumor MRI Classifier")
st.write("Upload an MRI scan and the model will predict the tumor type.")

uploaded_file = st.file_uploader("Choose an MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded MRI", use_container_width=True)

    # Predict
    with st.spinner("Analyzing MRI..."):
        predicted_class, confidence = predict_image(image)

    st.success(f"### 🩺 Prediction: `{predicted_class}`")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Optional: Add explanation
    st.caption("Model: EfficientNetV2B0 | Trained with TensorFlow")

