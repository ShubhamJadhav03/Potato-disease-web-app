from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import requests
from io import BytesIO
from PIL import Image

app = FastAPI()

# TensorFlow Serving endpoint
TF_SERVING_URL = "http://localhost:8501/v1/models/potatoes_model:predict"

# Class labels
CLASS_NAMES = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

@app.get("/ping")
async def ping():
    return {"message": "Hello, I am finally creating end-to-end Deep Learning projects!"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):  
    try:
        # Read image
        image_data = await file.read()
        print(f"📂 Received file: {file.filename}, Size: {len(image_data)} bytes")

        # Convert image to NumPy array
        image = read_file_as_image(image_data)
        print(f"🖼 Image shape after processing: {image.shape}")

        # Preprocess image (adjust as per your model's input shape)
        img_batch = np.expand_dims(image, axis=0)  # Add batch dimension

        # Convert to JSON format for TensorFlow Serving
        data = {"instances": img_batch.tolist()}

        # Send request to TensorFlow Serving
        response = requests.post(TF_SERVING_URL, json=data)

        # Check for errors
        if response.status_code != 200:
            return {"error": "Failed to get prediction from TensorFlow Serving", "details": response.text}

        # Process the response
        prediction = response.json()["predictions"][0]
        predicted_class = CLASS_NAMES[np.argmax(prediction)]
        confidence = float(np.max(prediction))

        response_data = {
            "class": predicted_class,
            "confidence": confidence
        }

        print(f"✅ Response: {response_data}")
        return response_data

    except Exception as e:
        print("❌ Error:", e)
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
