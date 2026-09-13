import os
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Chest X-Ray AI Assistant",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# App Title
# -----------------------------
st.title("🩺 Chest X-Ray AI Assistant")
st.write(
    "Upload a chest X-ray image for an AI-based prediction."
)

st.warning(
    "⚠️ This application is for educational purposes only. "
    "It is not a medical diagnosis."
)

# -----------------------------
# Find Model File
# -----------------------------
MODEL_FILES = [
    "model.keras",
    "chest_xray_model.keras",
    "pneumonia_model.keras",
    "best_model.keras",
    "model.h5",
    "chest_xray_model.h5",
    "pneumonia_model.h5",
    "best_model.h5"
]

def find_model_file():
    for file_name in MODEL_FILES:
        if os.path.exists(file_name):
            return file_name
    return None

MODEL_PATH = find_model_file()

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# -----------------------------
# Check Model
# -----------------------------
if MODEL_PATH is None:
    st.error("❌ Model file not found!")

    st.info(
        "Please upload your trained model file "
        "(.keras or .h5) to the GitHub repository."
    )

    st.write("Expected model file names:")
    st.code(
        "model.keras\n"
        "chest_xray_model.keras\n"
        "pneumonia_model.keras\n"
        "best_model.keras\n"
        "model.h5"
    )

    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success(f"✅ Model loaded successfully: {MODEL_PATH}")
except Exception as error:
    st.error("❌ Error while loading the model.")
    st.code(str(error))
    st.stop()

# -----------------------------
# Image Preprocessing
# -----------------------------
def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    return image_array

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.subheader("Uploaded X-Ray")
    st.image(
        image,
        caption="Chest X-Ray Image",
        use_container_width=True
    )

    if st.button("🔍 Predict"):
        with st.spinner("Analyzing image..."):
            processed_image = preprocess_image(image)
            prediction = model.predict(processed_image)

        st.subheader("Prediction Result")

        # Binary classification
        if prediction.shape[-1] == 1:
            probability = float(prediction[0][0])

            if probability >= 0.5:
                result = "Pneumonia"
                confidence = probability * 100
            else:
                result = "Normal"
                confidence = (1 - probability) * 100

        # Two-class classification
        else:
            predicted_class = int(np.argmax(prediction[0]))
            confidence = float(np.max(prediction[0])) * 100

            class_names = {
                0: "Normal",
                1: "Pneumonia"
            }

            result = class_names.get(
                predicted_class,
                f"Class {predicted_class}"
            )

        st.success(f"Prediction: {result}")
        st.info(f"Confidence: {confidence:.2f}%")
