import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model(
    "chest_xray_mobilenet_model.keras"
)

# Find MobileNetV2 base model
base_model = None

for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        base_model = layer
        break

if base_model is None:
    raise ValueError("Base model not found")

print("Base model:", base_model.name)

# Find last convolutional layer
last_conv_layer = None

for layer in reversed(base_model.layers):
    if isinstance(layer, tf.keras.layers.Conv2D):
        last_conv_layer = layer
        break

if last_conv_layer is None:
    raise ValueError("Convolutional layer not found")

print("Last convolutional layer:", last_conv_layer.name)


def make_gradcam(image_path):

    # Load and preprocess image
    image = tf.keras.utils.load_img(
        image_path,
        target_size=(224, 224)
    )

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    processed_image = image_array / 127.5 - 1.0

    # Grad-CAM model
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

    # Read original X-ray image
    original = cv2.imread(
        image_path,
        cv2.IMREAD_GRAYSCALE
    )

    original = cv2.resize(
        original,
        (224, 224)
    )

    # Convert grayscale to 3 channels
    original = cv2.cvtColor(
        original,
        cv2.COLOR_GRAY2BGR
    )

    # Resize heatmap
    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )

    heatmap_uint8 = np.uint8(255 * heatmap)

    heatmap_color = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET
    )

    print("Original shape:", original.shape)
    print("Heatmap shape:", heatmap_color.shape)

    # Combine image and heatmap
    superimposed_image = cv2.addWeighted(
        original,
        0.6,
        heatmap_color,
        0.4,
        0
    )

    # Save result
    cv2.imwrite(
        "gradcam_result.jpg",
        superimposed_image
    )

    # Display result
    plt.figure(figsize=(6, 6))
    plt.imshow(
        cv2.cvtColor(
            superimposed_image,
            cv2.COLOR_BGR2RGB
        )
    )
    plt.axis("off")
    plt.title("Grad-CAM Visualization")
    plt.show()

    print("Grad-CAM image saved as gradcam_result.jpg")


# Test image path
image_path = r"C:\Users\manj1\Downloads\archive\chest_xray\test\PNEUMONIA\person3_virus_15.jpeg"

make_gradcam(image_path)