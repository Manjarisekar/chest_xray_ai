import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Dataset path
test_path = r"C:\Users\manj1\Downloads\archive\chest_xray\test"

# Load test dataset
test_data = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Load MobileNetV2 model
model = tf.keras.models.load_model("chest_xray_mobilenet_model.keras")

# Get predictions
y_true = []
y_pred = []

for images, labels in test_data:
    predictions = model.predict(images, verbose=0)
    predicted_labels = (predictions >= 0.5).astype(int).flatten()

    y_true.extend(labels.numpy())
    y_pred.extend(predicted_labels)

# Evaluation
class_names = test_data.class_names

print("\n========== MOBILE NET MODEL EVALUATION ==========")
print("Class Names:", class_names)
print("Total Test Images:", len(y_true))

accuracy = np.mean(np.array(y_true) == np.array(y_pred)) * 100
print("\nAccuracy:")
print(f"{accuracy:.2f}%")

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    target_names=class_names
))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nEvaluation completed successfully!")