# Handwritten Digit Recognition — Deep Learning Web App

A small end-to-end machine learning project that recognizes handwritten digits (0–9).

The project connects **a trained TensorFlow/Keras model** to a **Flask backend** and a simple **HTML/CSS/JavaScript frontend**. The user draws a digit in the browser, the backend preprocesses the image, and the model returns the predicted digit and confidence.

## Project idea

The goal was to understand how a machine-learning model can be turned into an actual application instead of being used only inside a notebook.

### Application flow

```text
User draws a digit
        ↓
HTML Canvas
        ↓
JavaScript sends image
        ↓
Flask /predict endpoint
        ↓
Image preprocessing (28 × 28)
        ↓
CNN trained on MNIST
        ↓
Predicted digit + confidence
        ↓
Result shown in browser
```

## Technologies

- **Python** — application and ML code
- **TensorFlow / Keras** — neural network
- **MNIST** — handwritten digit dataset
- **Flask** — backend API and web server
- **HTML, CSS, JavaScript** — frontend
- **Pillow + NumPy** — image preprocessing

## Folder structure

```text
digit-ai/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── PROJECT_REPORT.md
├── .gitignore
├── model/
│   └── digit_model.keras   # created after training
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## Setup

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd digit-ai
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train_model.py
```

The first training run downloads MNIST and creates:

```text
model/digit_model.keras
```

### 5. Start the application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Model

The project uses a small **Convolutional Neural Network (CNN)**:

```text
28×28 grayscale image
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Conv2D
        ↓
MaxPooling
        ↓
Flatten
        ↓
Dense
        ↓
10-class Softmax
```

The model is trained using the MNIST training set and evaluated using the MNIST test set. The actual test accuracy is printed by `train_model.py`, rather than being claimed in advance.

## Important note about GitHub

The trained `.keras` model is generated locally by `train_model.py`. This keeps the repository lightweight and makes the training process reproducible.

If you decide to upload the trained model too, check GitHub's file-size limits first.

## Future improvements

- Improve preprocessing by centering the handwritten digit.
- Show the probability of all ten digits.
- Add a prediction history.
- Add a model-accuracy page.
- Compare a dense neural network with the CNN.
- Deploy the application online.

## Author

**Neelaksh Gaur**  
B.Tech CSE (AI/ML) — First Year

This project was created as a learning project to understand the connection between machine learning, backend development, and frontend interaction.
