import tensorflow as tf
from tensorflow.keras import layers, models

# Dataset paths
train_path = r"C:\Users\manj1\Downloads\archive\chest_xray\train"
val_path = r"C:\Users\manj1\Downloads\archive\chest_xray\val"

# Load datasets
train_data = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=True
)

val_data = tf.keras.utils.image_dataset_from_directory(
    val_path,
    image_size=(224, 224),
    batch_size=32,
    shuffle=False
)

# Improve performance
AUTOTUNE = tf.data.AUTOTUNE

train_data = train_data.prefetch(buffer_size=AUTOTUNE)
val_data = val_data.prefetch(buffer_size=AUTOTUNE)

# Load pretrained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights="imagenet"
)

# Freeze pretrained layers
base_model.trainable = False

# Build transfer learning model
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    layers.Rescaling(1.0 / 127.5, offset=-1),

    base_model,

    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(1, activation="sigmoid")
])

# Compile model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train model
history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5
)

# Save model
model.save("chest_xray_mobilenet_model.keras")

print("Transfer learning training completed!")
print("MobileNetV2 model saved successfully!")