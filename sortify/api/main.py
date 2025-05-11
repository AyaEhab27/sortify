import os
import zipfile
import requests
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_URL = "https://drive.google.com/uc?export=download&id=1P-JY8OTsuBnc4JCg_eLS2TqTtz8LTj3i"
MODEL_ZIP = "model.zip"
MODEL_DIR = "model"
MODEL_FILE = os.path.join(MODEL_DIR, "best_model_full (1).h5")


def download_model():
    if not os.path.exists(MODEL_FILE):
        print("Downloading model...")
        try:
            
            os.makedirs(MODEL_DIR, exist_ok=True)
            
            
            session = requests.Session()
            response = session.get(MODEL_URL, stream=True)
            response.raise_for_status()
            
            with open(MODEL_ZIP, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            
            with zipfile.ZipFile(MODEL_ZIP, 'r') as zip_ref:
                zip_ref.extractall('./model')
            
            
            os.remove(MODEL_ZIP)
            
            print("Model downloaded and extracted successfully.")
            
        except Exception as e:
            print(f"Error downloading model: {e}")
            raise


def load_model():
    try:
        model = tf.keras.models.load_model(MODEL_FILE)
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


download_model()
model = load_model()


WASTE_CATEGORIES = ["Plastic", "Glass", "Metal", "Cardboard", "Paper", "Trash"]

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        try:
            image = Image.open(io.BytesIO(contents)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        
        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        
        try:
            predictions = model.predict(image_array)
        except Exception as e:
            raise HTTPException(status_code=500, detail="Model prediction failed")
        
        
        results = []
        for i, (category, confidence) in enumerate(zip(WASTE_CATEGORIES, predictions[0])):
            results.append({
                "id": i,
                "name": category,
                "confidence": float(confidence),
                "icon": f"/assets/icon{i+1}.png"
            })
        
        
        results.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "top_prediction": results[0],
            "all_predictions": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
