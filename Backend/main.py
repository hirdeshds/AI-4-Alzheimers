from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import os

# Safety: disable GPU (Railway CPU only)
tf.config.set_visible_devices([], 'GPU')

app = FastAPI(title="Alzheimer MRI Classification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "model/alzheimer_Model_final.h5"

model = tf.keras.models.load_model(MODEL_PATH)

classes = [
    "Mild Impairment",
    "Moderate Impairment",
    "No Impairment",
    "Very Mild Impairment"
]

def is_mri_like(image: Image.Image) -> bool:
    img = np.array(image.convert("RGB"))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    diff_rg = np.mean(np.abs(r - g))
    diff_rb = np.mean(np.abs(r - b))
    diff_gb = np.mean(np.abs(g - b))

    return diff_rg < 15 and diff_rb < 15 and diff_gb < 15

def prepare_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image), axis=0)
    return preprocess_input(img_array)

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    try:
        image = Image.open(file.file)
        image.verify()
        file.file.seek(0)
        image = Image.open(file.file)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    if not is_mri_like(image):
        raise HTTPException(status_code=400, detail="Not a valid MRI")

    img = prepare_image(image)
    preds = model.predict(img)

    idx = int(np.argmax(preds))
    confidence = float(np.max(preds)) * 100

    return {
        "prediction": classes[idx],
        "confidence": round(confidence, 2)
    }
