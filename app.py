import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩻",
    layout="centered"
)

# Title
st.title("🩻 Chest X-Ray AI")
st.write("Upload a chest X-ray image to classify it as NORMAL or PNEUMONIA.")

# Load MobileNetV2 model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("chest_xray_mobilenet_model.keras")

model = load_model()

# Upload image
uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Chest X-Ray",
        use_container_width=True
    )

    # Preprocess image
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction) * 100

    st.subheader("Prediction Result")

    if result == "PNEUMONIA":
        st.error(f"Result: {result}")
    else:
        st.success(f"Result: {result}")

    st.info(f"Confidence: {confidence:.2f}%")

    st.warning(
        "Disclaimer: This AI result is for educational purposes only "
        "and must not be used as a medical diagnosis."
    )