from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import keras
import os
from tensorflow import keras
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables (initialized later)
MODEL = None
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

# Model path
MODEL_PATH = r"C:\Users\j_san\OneDrive\Desktop\AIML\Projects\potato-disesase\saved_models\1\1.keras"

@app.on_event("startup")  # Load model when FastAPI starts
def load_model():
    global MODEL  # Declare MODEL as a global variable
    MODEL = keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully!")

@app.get("/ping")
async def ping():
    return "Hello, I am finally creating end-to-end Deep Learning projects!"

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):  
    try:
        global MODEL  # Ensure we're using the globally loaded model

        # Read image
        image_data = await file.read()
        print(f"📂 Received file: {file.filename}, Size: {len(image_data)} bytes")

        image = read_file_as_image(image_data)
        print(f"🖼 Image shape after processing: {image.shape}")

        # Preprocess image
        img_batch = np.expand_dims(image, axis=0)
        print(f"📦 Image batch shape: {img_batch.shape}")

        # Ensure model is loaded
        if MODEL is None:
            raise RuntimeError("Model is not loaded yet!")

        # Make prediction
        prediction = MODEL.predict(img_batch)
        print(f"🔮 Raw Prediction Output: {prediction}")

        # Process output
        predicted_class = CLASS_NAMES[np.argmax(prediction[0])]
        confidence = float(np.max(prediction[0]))

        response = {
            "class": predicted_class,
            "confidence": confidence
        }
        print(f"✅ Response: {response}")
        return response

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
