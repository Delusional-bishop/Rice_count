from fastapi import FastAPI, UploadFile, File
from rice_counter import count_objects_in_video
from firebase_admin import credentials, firestore, initialize_app
from ultralytics import YOLO
import shutil
import uuid
import os
import json

app = FastAPI()

# Initialize Firebase
cred = credentials.Certificate("firebase_config.json")
initialize_app(cred)
db = firestore.client()

# Load model
model = YOLO("my_model.pt")

# Load class names
import yaml
with open("data.yaml", 'r') as f:
    class_names = yaml.safe_load(f)['names']

@app.get("/")
async def read_root():
    return {"message": "Backend is running!"}

@app.post("/upload-video/")
async def upload_video(file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    video_path = f"{video_id}.mp4"
    with open(video_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Run the model
    object_counts, total_count = count_objects_in_video(
        video_path=video_path,
        model=model,
        class_names=class_names,
        counting_line_y=None,
        output_path=None,
        tolerance=5,
        frame_skip=1
    )

    # Upload to Firebase
    result = {
        "total_count": total_count,
        "counts_by_class": dict(object_counts)
    }
    db.collection("results").document(video_id).set(result)

    os.remove(video_path)
    return {"status": "processed", "video_id": video_id, "results": result}
