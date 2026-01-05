# AI-Based Alzheimer Detection System (Machine Learning)

This repository contains **only the Machine Learning / Deep Learning part** of the AI-Based Alzheimer Detection System. It focuses on dataset preparation, CNN model training, evaluation, and model saving for backend integration.

---


## 🎯 Objective

To develop a **CNN-based image classification model** that detects Alzheimer’s disease stages from MRI brain scans.

**Classification Classes:**

* Non Demented
* Very Mild Demented
* Mild Demented
* Moderate Demented

---

## 📂 Dataset Details

* MRI brain scan images
* Organized into 4 class folders
* Train–Validation Split: **80% / 20%**

### Preprocessing Steps

* Resize images to **128 × 128**
* Normalize pixel values (0–1)
* Optional data augmentation (rotation, zoom, flip)

---

## 🧠 Model Architecture

* Convolutional Layers + ReLU
* MaxPooling Layers
* Dropout (to reduce overfitting)
* Fully Connected (Dense) Layers
* Softmax Output Layer

**Loss Function:** Categorical Crossentropy
**Optimizer:** Adam
**Metrics:** Accuracy

---

## 🚀 Training & Evaluation

* Epochs: **10–20**
* Validation accuracy & loss tracking
* Confusion Matrix & Classification Report

---

## 📁 Repository Structure

```
ml_alzheimer_detection/
│
├── dataset/
│   ├── train/
│   │   ├── NonDemented/
│   │   ├── VeryMildDemented/
│   │   ├── MildDemented/
│   │   └── ModerateDemented/
│   └── val/
│       ├── NonDemented/
│       ├── VeryMildDemented/
│       ├── MildDemented/
│       └── ModerateDemented/
│
├── train_cnn.py          # CNN training script
├── evaluate.py           # Model evaluation & metrics
├── requirements.txt      # ML dependencies
├── saved_model/
│   └── cnn_model.h5      # Trained model
└── README.md
```

---

## 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* OpenCV
* Matplotlib
* Scikit-learn

---

## 📦 Deliverables

* `train_cnn.py` – CNN model training
* `evaluate.py` – Model evaluation
* `cnn_model.h5` – Trained Alzheimer detection model
* Preprocessed MRI dataset

---

## 👥 Contributors (ML Team)

* **Mahek** 
* **Hirdesh** 
## 🔗 Future Scope

* Hyperparameter tuning
* Transfer learning (VGG16, ResNet)
* Model explainability (Grad-CAM)
* Integration with FastAPI backend

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
python train_cnn.py
python evaluate.py
```

---

## 📄 License

This project is intended for academic and educational use only.
