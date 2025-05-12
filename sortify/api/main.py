import os
import subprocess  
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# MODEL_URL = "https://drive.google.com/uc?id=1P-JY8OTsuBnc4JCg_eLS2TqTtz8LTj3i"
MODEL_DIR = Path("model")
MODEL_PATH = MODEL_DIR / "best_model_full (1).h5"

# def download_model():
#     if not MODEL_PATH.exists():
#         MODEL_DIR.mkdir(exist_ok=True)
#         print("Downloading model...")
#         try:
#             subprocess.run([
#                 "wget",
#                 "--no-check-certificate",
#                 MODEL_URL,
#                 "-O", str(MODEL_PATH)
#             ], check=True)
#             print(f"Model downloaded! Size: {os.path.getsize(MODEL_PATH)} bytes")
#         except Exception as e:
#             print(f"Failed to download model: {e}")
#             raise

def load_model():
    try:
        if not MODEL_PATH.exists():
            raise Exception("Model file not found! Please upload it manually to /model/ directory.")
            
        model = tf.keras.models.load_model(MODEL_PATH)
        print("Model loaded successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise

model = load_model()


waste_categories = [
    "Plastic", "Glass", "Metal", "Cardboard", "Paper", "Trash"
]

# def preprocess_image(image):
#     """Preprocess image for model prediction"""
#     try:

#         if image.mode != 'RGB':
#             image = image.convert('RGB')
        

#         image = image.resize((224, 224))
#         image_array = np.array(image) / 255.0
#         image_array = np.expand_dims(image_array, axis=0)
#         return image_array
#     except Exception as e:
#         raise ValueError(f"Image processing error: {e}")

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):

    allowed_extensions = {'jpg', 'jpeg', 'png'}
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail="Invalid file type. Only JPG/JPEG/PNG allowed.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image.verify()
        image = Image.open(io.BytesIO(contents))

        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        try:
            image.verify()  
            image = Image.open(io.BytesIO(contents))  
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")
        

        image_array = preprocess_image(image)
        

        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        

        waste_categories = ["Plastic", "Glass", "Metal", "Cardboard", "Paper", "Trash"]
        
        return {
            "prediction": waste_categories[predicted_class],
            "confidence": confidence
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

