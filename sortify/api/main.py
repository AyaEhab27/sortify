import os
import gdown
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# Disable TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

app = FastAPI()

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_URL = "https://drive.google.com/uc?id=1-LrHSmkK33-PAC48fZVc7mRYsDMDsWFG"
MODEL_PATH = "best_model_full (1).h5"  


waste_categories = [
    "Plastic", "Glass", "Metal", "Cardboard", "Paper", "Trash"
]

def download_model():
    """Download model from Google Drive"""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print(f"Model downloaded! Size: {os.path.getsize(MODEL_PATH)/1e6:.2f} MB")
        except Exception as e:
            print(f"Failed to download model: {e}")
            raise

def load_model():
    """Load and compile the model"""
    try:
        download_model()
        model = tf.keras.models.load_model(MODEL_PATH)
        

        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model loaded and compiled successfully!")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


model = load_model()

def preprocess_image(image):
    """Basic image preprocessing"""
    try:

        if image.mode != 'RGB':
            image = image.convert('RGB')
        

        image = image.resize((224, 224))
        image_array = np.array(image) / 255.0
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Image processing error: {e}")

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    """Classify waste from uploaded image"""
    try:

        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        image.verify()
        image = Image.open(io.BytesIO(contents))  
        

        image_array = preprocess_image(image)
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        

        results = {
            "top_prediction": {
                "category": waste_categories[predicted_class],
                "confidence": float(predictions[0][predicted_class])
            },
            "all_predictions": [
                {
                    "category": category,
                    "confidence": float(confidence),
                    "category_id": idx
                } for idx, (category, confidence) in enumerate(zip(waste_categories, predictions[0]))
            ]
        }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/health")
async def health_check():
    """Service health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "model_info": {
            "input_shape": model.input_shape,
            "classes": waste_categories
        }
    }
