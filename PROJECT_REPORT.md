# Project Report

## Handwritten Digit Recognition Web Application Using Deep Learning

### 1. Introduction

Handwritten digit recognition is a common introductory computer-vision problem. In this project, a small convolutional neural network (CNN) is trained on the MNIST dataset and connected to a web application.

The main purpose is not only to train a model, but also to understand how an ML model can receive real user input and return a prediction through a backend API.

### 2. Objectives

1. Train a neural network to classify handwritten digits from 0 to 9.
2. Understand basic image preprocessing.
3. Connect a TensorFlow model with a Flask backend.
4. Build a simple browser-based drawing interface.
5. Return a prediction and confidence value to the user.

### 3. Dataset

The project uses the MNIST handwritten-digit dataset.

Each sample is a grayscale image of size **28 × 28 pixels** and belongs to one of ten classes: 0 through 9.

The training script loads the dataset directly through TensorFlow/Keras.

### 4. Model

A small CNN is used because convolutional layers are well suited to image data.

The model contains:

- Two convolutional layers
- Two max-pooling layers
- A flattening layer
- A dense hidden layer
- A dropout layer
- A 10-class softmax output

The model is compiled using the Adam optimizer and sparse categorical cross-entropy loss.

### 5. Image preprocessing

The browser canvas produces a PNG image. The Flask backend:

1. Decodes the Base64 image.
2. Converts it to grayscale.
3. Resizes it to 28 × 28 pixels.
4. Inverts the colors so the digit matches the MNIST-style representation.
5. Normalizes pixel values to the range 0–1.
6. Sends the processed image to the model.

### 6. System architecture

```text
+----------------------+
|       Browser        |
|  HTML/CSS/JavaScript |
+----------+-----------+
           |
           | POST image
           v
+----------------------+
|     Flask Backend    |
|      /predict API    |
+----------+-----------+
           |
           | preprocessed image
           v
+----------------------+
| TensorFlow / Keras   |
|      CNN Model       |
+----------+-----------+
           |
           | prediction
           v
+----------------------+
| Browser displays     |
| digit + confidence   |
+----------------------+
```

### 7. Result

After training, the program prints the test-set accuracy. The application then allows a user to draw a digit and receive a prediction.

The project deliberately does not hard-code an accuracy value in the documentation. The value printed by the current training run is the result that should be reported.

### 8. Limitations

- The model is trained on MNIST, while a user's handwriting may look different.
- Simple resizing can lose some spatial information.
- Confidence is the model's probability output; it is not a guarantee that the prediction is correct.
- The application is intended as a learning project rather than a production handwriting-recognition system.

### 9. Future scope

The application can be improved by using better image centering, data augmentation, a larger CNN, prediction-history storage, model comparison, and online deployment.

### 10. Learning outcomes

This project provides practical experience with:

- Dataset loading
- Image normalization
- CNN architecture
- Model training and evaluation
- Flask APIs
- Frontend-backend communication
- Basic GitHub project organization
