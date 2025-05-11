from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import tensorflow as tf
from PIL import Image
import io
import os
import gdown
import zipfile
import logging

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


logging.basicConfig(level=logging.INFO)

if not os.path.exists('model'):
    model_url = "https://drive.google.com/uc?export=download&id=1P-JY8OTsuBnc4JCg_eLS2TqTtz8LTj3i"  
    output = "model.zip"
    gdown.download(model_url, output, quiet=False)
    
    with zipfile.ZipFile(output, 'r') as zip_ref:
        zip_ref.extractall('.')
    
    os.remove(output)


model = tf.keras.models.load_model('./model/best_model_full (1).h5')  

waste_categories = [
    "Plastic", "Glass", "Metal", "Cardboard", "Paper", "Trash"
]

@app.post("/classify")
async def classify_waste(file: UploadFile = File(...)):
    try:
        
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        image = image.resize((224, 224))  
        
       
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        logging.info("Image pre-processing done.")

        # التنبؤ باستخدام النموذج
        predictions = model.predict(image_array)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])

        # إعداد النتائج لكل فئة من القمامة
        results = []
        for i, category in enumerate(waste_categories):
            results.append({
                "id": i,
                "name": category,
                "confidence": float(predictions[0][i]),
                "icon": f"/assests/icon{i+1}.png"  
            })

        
        results.sort(key=lambda x: x["confidence"], reverse=True)

        logging.info("Prediction successful.")

        return {
            "top_prediction": {
                "name": waste_categories[predicted_class],
                "confidence": confidence
            },
            "all_predictions": results
        }

    except Exception as e:
        logging.error(f"Error processing the image: {str(e)}")
        return {"error": "Failed to classify the image", "details": str(e)}
