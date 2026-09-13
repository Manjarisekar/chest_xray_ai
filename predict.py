import tensorflow as tf
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array

# 1. Load trained model
model = tf.keras.models.load_model("chest_xray_model.keras")

# 2. Test image path
image_path = r"C:\Users\manj1\Downloads\archive\chest_xray\test\PNEUMONIA\person3_virus_15.jpeg"

# 3. Load and resize image
image = load_img(image_path, target_size=(224, 224))

# 4. Convert image into array
image_array = img_to_array(image)

# 5. Add batch dimension
image_array = np.expand_dims(image_array, axis=0)

# 6. Normalize pixel values
image_array = image_array / 255.0

# 7. Make prediction
prediction = model.predict(image_array)

# 8. Display result
probability = float(prediction[0][0])

if probability >= 0.5:
    result = "PNEUMONIA"
    confidence = probability * 100
else:
    result = "NORMAL"
    confidence = (1 - probability) * 100

print("--------------------------------")
print("Chest X-Ray Prediction")
print("--------------------------------")
print("Prediction:", result)
print("Confidence: {:.2f}%".format(confidence))
print("--------------------------------")
print("Note: This is an AI model prediction, not a medical diagnosis.")