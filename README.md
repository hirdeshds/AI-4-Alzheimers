# AI 4 Alzheimers

AI-based system for **early detection and assistance for Alzheimer’s patients** using Machine Learning and a backend API.

---

## 🏆 Project Overview

This project is an **end-to-end system** for detecting Alzheimer’s disease stages from MRI brain scans. It consists of:

1. **Machine Learning Module** – CNN-based model for classifying Alzheimer stages.
2. **Backend API (FastAPI)** – Serves predictions and stores patient information.
3. **Android App (Optional)** – Users can upload MRI images to get predictions (folder included, development in progress).

---

## 👥 Team Members

* **Mahek** – Machine Learning & Backend Integration
* **Hirdesh** – Machine Learning & Backend Integration
* **Dhruv** – Android App Development

---

## 🎯 Objective

* Detect Alzheimer stages from MRI scans.
* Provide predictions via an API for integration with frontend applications.
* Store patient information along with prediction results.

**Classification Classes:**

* Non Demented
* Very Mild Demented
* Mild Demented
* Moderate Demented

---

## 🧠 Machine Learning Module

### CNN Architecture

* Convolutional Layers + ReLU
* MaxPooling Layers
* Dropout Layers (for regularization)
* Fully Connected (Dense) Layers
* Softmax output for multi-class classification

**Loss Function:** Categorical Crossentropy
**Optimizer:** Adam
**Metrics:** Accuracy

### Training & Evaluation

* Epochs: 10–20
* Tracks training & validation accuracy and loss
* Generates confusion matrix and classification report
* Saves trained model as `cnn_model.h5`

**Tools & Libraries:** Python, TensorFlow/Keras, NumPy, OpenCV, Pillow, Scikit-learn, Matplotlib

---

## ⚡ Backend API (FastAPI)

### Features

* Accepts MRI images and returns Alzheimer stage prediction
* Stores patient info in SQLite database
* Health check endpoint

### Endpoints

* **POST /predict** – Upload MRI image, receive prediction JSON
* **POST /patient/save** – Save patient information and prediction
* **GET /ping** – Health check

**Technology Stack:** Python, FastAPI, Uvicorn, SQLite

---

## ▶️ How to Run

1. Install dependencies:

```bash
pip install -r Backend/requirements.txt
pip install -r ML/requirements.txt
```

2. Train the model (optional):

```bash
python Backend/ml/train_cnn.py
```

3. Evaluate the model:

```bash
python Backend/ml/evaluate.py
```

4. Run FastAPI backend:

```bash
uvicorn Backend/app:app --reload
```

5. Test API endpoints using Postman or any HTTP client.

---

## 🔗 Future Scope

* Hyperparameter tuning and transfer learning (VGG16, ResNet)
* Model explainability with Grad-CAM
* Full Android or Web frontend integration
* Deployment on cloud platforms (AWS, Heroku, Render)

---

## 📄 License

This project is intended for **academic and educational purposes only**.
