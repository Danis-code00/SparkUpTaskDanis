import io
import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

app = FastAPI()

MODEL_PATH = r"D:\Personal (Academic)\SparkUpTaskDanis\Week 4\runs\detect\logo_final5\weights\best.pt" 

# --- NAME MAPPING ---
# This maps the ID numbers from your training to your new names
NAME_MAP = {
    0: "LogiSpark",
    1: "Nepvent",
    2: "SparkUp"
}

try:
    model = YOLO(MODEL_PATH) 
    print(f"✅ Custom Model loaded! Original Labels: {model.names}")
except Exception as e:
    print(f"❌ ERROR: Could not load model: {e}")
    model = None

@app.post("/detect")
async def detect_logo(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents))
    results = model(image, conf=0.25)

    detections = []
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls)
            # Use our NAME_MAP if the ID exists, otherwise use model's default name
            label_name = NAME_MAP.get(class_id, result.names[class_id])
            
            detections.append({
                "label": label_name,
                "confidence": round(float(box.conf), 3),
                "box": box.xyxy.tolist()[0]
            })

    return {"filename": file.filename, "detections": detections}

@app.post("/detect-visual")
async def detect_visual(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    results = model(img, conf=0.25)
    
    # Update the names inside the results object so the plot uses your new names
    for i, name in NAME_MAP.items():
        results[0].names[i] = name
        
    annotated_img = results[0].plot() 
    _, im_png = cv2.imencode(".png", annotated_img)
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")