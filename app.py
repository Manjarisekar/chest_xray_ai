import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

st.set_page_config(
    page_title="Chest X-Ray AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 Chest X-Ray AI")
st.write("Upload a chest X-ray image to predict NORMAL or PNEUMONIA.")

st.warning(
    "This is an AI research prototype and not a medical diagnosis. "
    "Please consult a qualified medical professional."
)

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "chest_xray_mobilenet_model.keras"
    )

model = load_model()

# Find MobileNetV2 base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

# Find last convolutional layer
last_conv_layer = None

for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break


def generate_gradcam(image_array):

    processed_image = image_array / 127.5 - 1.0

    grad_model = tf.keras.models.Model(
        inputs=base_model.input,
        outputs=[
            last_conv_layer.output,
            base_model.output
        ]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(processed_image)
        loss = predictions[:, 0]

    gradients = tape.gradient(loss, conv_outputs)

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs @ pooled_gradients[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0)
    heatmap = heatmap / (tf.reduce_max(heatmap) + 1e-8)
    heatmap = heatmap.numpy()

    original = np.uint8(image_array[0])

    original = cv2.cvtColor(
        original,
        cv2.COLOR_RGB2BGR
    )

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    result = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    return result


uploaded_file = st.file_uploader(
    "Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))

    image_array = np.array(image)
    input_array = np.expand_dims(image_array, axis=0)

    st.subheader("Uploaded X-Ray")
    st.image(image, width="stretch")

    # Prediction
    processed_image = input_array / 127.5 - 1.0
    prediction = model.predict(processed_image, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "PNEUMONIA"
        confidence = prediction * 100
    else:
        result = "NORMAL"
        confidence = (1 - prediction) * 100

    st.subheader("Prediction Result")

    if result == "PNEUMONIA":
        st.error(f"Prediction: {result}")
    else:
        st.success(f"Prediction: {result}")

    st.write(f"Confidence: {confidence:.2f}%")

    # Grad-CAM
    st.subheader("Grad-CAM Explainability")
    st.caption(
        "Highlighted regions show areas that influenced the model's prediction."
    )

    gradcam_image = generate_gradcam(input_array)
    st.image(
        gradcam_image,
        caption="Grad-CAM Heatmap",
        width="stretch"
    )