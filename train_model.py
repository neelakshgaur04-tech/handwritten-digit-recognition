from pathlib import Path

import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

# MNIST contains 60,000 training images and 10,000 test images.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize pixel values from 0-255 to 0-1.
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# A small CNN is better suited to image recognition than a plain dense network.
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation="relu"),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(
    x_train[..., tf.newaxis],
    y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=128
)

test_loss, test_accuracy = model.evaluate(
    x_test[..., tf.newaxis], y_test, verbose=0
)

model.save(MODEL_DIR / "digit_model.keras")

print(f"Test accuracy: {test_accuracy * 100:.2f}%")
print(f"Model saved to: {MODEL_DIR / 'digit_model.keras'}")
