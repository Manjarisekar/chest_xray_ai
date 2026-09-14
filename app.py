import os
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image


# ==================================================
# PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Chest X-Ray AI Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# CUSTOM CSS
# ==================================================

st.markdown(
    """
    <style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #1f4e79;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #555555;
        margin-bottom: 20px;
    }

    .result-box {
        padding: 20px;
        border-radius: 12px;
        background-color: #f0f7ff;
        border: 1px solid #c8e1ff;
        margin-top: 15px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff3cd;
        border: 1px solid #ffeeba;
        color: #664d03;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ==================================================
# MODEL CONFIGURATION
# ==================================================

MODEL_PATH = "chest_xray_model.keras"
IMAGE_SIZE = (224, 224)


# ==================================================
# SESSION STATE
# ==================================================

if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        return None

    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception as error:
        st.error(f"Model loading error: {error}")
        return None


model = load_trained_model()


# ==================================================
# SIDEBAR
# ==================================================

with st.sidebar:
    st.header("🩺 Chest X-Ray AI")

    selected_page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🔍 Prediction",
            "📋 Prediction History",
            "ℹ️ About Project"
        ]
    )

    st.divider()

    st.caption("Deep Learning-Based Medical Image Classification")


# ==================================================
# HOME PAGE
# ==================================================

if selected_page == "🏠 Home":

    st.markdown(
        '<div class="main-title">🩺 Chest X-Ray Disease Detection</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">AI-powered educational chest X-ray classification system</div>',
        unsafe_allow_html=True
    )

    st.info(
        "This application uses a trained deep learning model "
        "to classify chest X-ray images."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Input", "Chest X-Ray")

    with col2:
        st.metric("Model", "Deep Learning")

    with col3:
        st.metric("Output", "Normal / Pneumonia")

    st.subheader("How It Works")

    st.markdown(
        """
        1. Upload a chest X-ray image.
        2. The image is resized and normalized.
        3. The trained AI model analyzes the image.
        4. The application displays the predicted class.
        5. The confidence score is shown for educational reference.
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <b>Medical Disclaimer:</b> This application is intended only
        for educational and research purposes. It is not a medical
        diagnosis tool. Always consult a qualified medical professional.
        </div>
        """,
        unsafe_allow_html=True
    )


# ==================================================
# PREDICTION PAGE
# ==================================================

elif selected_page == "🔍 Prediction":

    st.title("🔍 Chest X-Ray Prediction")

    if model is None:
        st.error("❌ Trained model not found.")

        st.info(
            f"Please place '{MODEL_PATH}' in the same folder as app.py."
        )

        st.stop()

    st.success("✅ Trained model loaded successfully!")

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear chest X-ray image."
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        left_col, right_col = st.columns(2)

        with left_col:
            st.subheader("Uploaded X-Ray")
            st.image(
                image,
                caption="Input Chest X-Ray",
                use_container_width=True
            )

        with right_col:
            st.subheader("Image Information")
            st.write(f"**File name:** {uploaded_file.name}")
            st.write(f"**Image size:** {image.size}")
            st.write("**Image format:** RGB")

        st.divider()

        if st.button("🔍 Analyze X-Ray", type="primary"):

            with st.spinner("Analyzing image..."):

                try:
                    # Image preprocessing
                    processed_image = image.resize(IMAGE_SIZE)

                    image_array = np.array(processed_image)
                    image_array = image_array / 255.0
                    image_array = np.expand_dims(image_array, axis=0)

                    # Model prediction
                    prediction = model.predict(
                        image_array,
                        verbose=0
                    )

                    # Binary classification using sigmoid
                    if prediction.shape[-1] == 1:

                        probability = float(prediction[0][0])

                        if probability >= 0.5:
                            predicted_class = "Pneumonia"
                            confidence = probability * 100
                        else:
                            predicted_class = "Normal"
                            confidence = (1 - probability) * 100

                    # Two-class classification using softmax
                    elif prediction.shape[-1] == 2:

                        class_names = ["Normal", "Pneumonia"]

                        predicted_index = int(
                            np.argmax(prediction[0])
                        )

                        predicted_class = class_names[predicted_index]

                        confidence = (
                            float(prediction[0][predicted_index]) * 100
                        )

                    else:
                        st.error(
                            "This model output format is not supported."
                        )
                        st.stop()

                    # Save prediction history
                    history_record = {
                        "Date & Time": datetime.now().strftime(
                            "%d-%m-%Y %H:%M:%S"
                        ),
                        "File Name": uploaded_file.name,
                        "Prediction": predicted_class,
                        "Confidence": f"{confidence:.2f}%"
                    }

                    st.session_state.prediction_history.append(
                        history_record
                    )

                    # Display prediction result
                    st.subheader("Prediction Result")

                    st.markdown(
                        '<div class="result-box">',
                        unsafe_allow_html=True
                    )

                    if predicted_class == "Normal":
                        st.success(
                            f"Prediction: {predicted_class}"
                        )
                    else:
                        st.error(
                            f"Prediction: {predicted_class}"
                        )

                    st.progress(
                        min(max(confidence / 100, 0.0), 1.0)
                    )

                    st.write(
                        f"**Confidence Score:** {confidence:.2f}%"
                    )

                    st.markdown(
                        '</div>',
                        unsafe_allow_html=True
                    )

                    st.warning(
                        "This result is an AI prediction for educational "
                        "purposes and is not a medical diagnosis."
                    )

                except Exception as error:
                    st.error(f"Prediction error: {error}")


# ==================================================
# PREDICTION HISTORY PAGE
# ==================================================

elif selected_page == "📋 Prediction History":

    st.title("📋 Prediction History")

    history = st.session_state.prediction_history

    if len(history) == 0:
        st.info("No predictions have been made in this session yet.")

    else:
        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )

        if st.button("🗑️ Clear History"):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared.")
            st.rerun()


# ==================================================
# ABOUT PROJECT PAGE
# ==================================================

elif selected_page == "ℹ️ About Project":

    st.title("ℹ️ About the Project")

    st.subheader("Project Title")

    st.write(
        "Deep Learning-Based Chest X-Ray Classification "
        "for Pneumonia Detection"
    )

    st.subheader("Objective")

    st.write(
        "The objective of this project is to develop an AI-based "
        "application that analyzes chest X-ray images and classifies "
        "them into Normal and Pneumonia categories."
    )

    st.subheader("Technologies Used")

    st.markdown(
        """
        - Python
        - TensorFlow / Keras
        - NumPy
        - Pandas
        - Pillow
        - Streamlit
        - Deep Learning
        """
    )

    st.subheader("System Workflow")

    st.code(
        """
Chest X-Ray Image
        ↓
Image Preprocessing
        ↓
Trained Deep Learning Model
        ↓
Classification
        ↓
Prediction + Confidence Score
        """
    )

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ This project is developed for educational and research purposes.
        It must not be used as a substitute for professional medical advice.
        </div>
        """,
        unsafe_allow_html=True
    )