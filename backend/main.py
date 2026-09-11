import os
import sys
import json
import uuid
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.inference import process_image_file

app = FastAPI(
    title="CattleWeightAI API",
    description="Precision Agritech AI - Biometric Livestock Weight & Morphometry Estimation API",
    version="2.4.0"
)

# Enable CORS for React frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "history.json")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "uploaded_images")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# Serve uploaded images statically
app.mount("/static/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []

def save_history_entry(entry):
    history = load_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CattleWeightAI Neural Engine",
        "version": "2.4.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/models/comparison")
def get_model_comparison():
    return {
        "dataset_size": 15,
        "models": [
            {
                "name": "DeepLabV3+-ResNet50",
                "type": "Pixel-wise Semantic Segmentation",
                "mae_kg": 4.93,
                "rmse_kg": 5.75,
                "r2_score": 0.9937,
                "status": "Recommended (Default)",
                "use_case": "Official BAIF field data recording & high-precision weight estimation."
            },
            {
                "name": "YOLOv8-Segmentation",
                "type": "Bounding Box + Polygon Masking",
                "mae_kg": 51.2,
                "rmse_kg": 64.1,
                "r2_score": 0.6812,
                "status": "Fast Preview",
                "use_case": "Fast mobile preview & real-time live video capture."
            }
        ],
        "proof_note": "Calibrated on BAIF Sahiwal / HF Cross dataset using scale-invariant vertical height & torso depth ratio features."
    }

@app.post("/api/predict")
async def predict_cattle_weight(
    file: UploadFile = File(None),
    left_file: UploadFile = File(None),
    right_file: UploadFile = File(None),
    rear_file: UploadFile = File(None),
    cattle_id: str = Form("TAG-105730429112"),
    model_engine: str = Form("deeplab"),
    side_name: str = Form("Left Side Profile")
):
    try:
        results_by_side = {}
        primary_file_bytes = None
        primary_data_url = None
        pred_id = str(uuid.uuid4())[:8]

        # 1. Process Left Side Profile
        if left_file:
            left_bytes = await left_file.read()
            primary_file_bytes = left_bytes
            res_left = process_image_file(left_bytes, side_name="Left Side Profile", model_engine=model_engine)
            results_by_side["left"] = res_left
            b64_img = base64.b64encode(left_bytes).decode("utf-8")
            primary_data_url = f"data:image/jpeg;base64,{b64_img}"

        # 2. Process Right Side Profile
        if right_file:
            right_bytes = await right_file.read()
            if not primary_file_bytes:
                primary_file_bytes = right_bytes
                b64_img = base64.b64encode(right_bytes).decode("utf-8")
                primary_data_url = f"data:image/jpeg;base64,{b64_img}"
            res_right = process_image_file(right_bytes, side_name="Right Side Profile", model_engine=model_engine)
            results_by_side["right"] = res_right

        # 3. Process Rear View
        if rear_file:
            rear_bytes = await rear_file.read()
            if not primary_file_bytes:
                primary_file_bytes = rear_bytes
                b64_img = base64.b64encode(rear_bytes).decode("utf-8")
                primary_data_url = f"data:image/jpeg;base64,{b64_img}"
            res_rear = process_image_file(rear_bytes, side_name="Rear View", model_engine=model_engine)
            results_by_side["rear"] = res_rear

        # Fallback to single 'file' parameter if multi-view parameters weren't passed
        if not results_by_side and file:
            single_bytes = await file.read()
            primary_file_bytes = single_bytes
            b64_img = base64.b64encode(single_bytes).decode("utf-8")
            primary_data_url = f"data:image/jpeg;base64,{b64_img}"
            res_single = process_image_file(single_bytes, side_name=side_name, model_engine=model_engine)
            results_by_side["left"] = res_single

        if not results_by_side:
            raise HTTPException(status_code=400, detail="No valid profile photos uploaded.")

        # Combine results across profiles
        side_weights = [r["weight_kg"] for k, r in results_by_side.items() if k in ["left", "right"]]
        if side_weights:
            avg_weight = round(float(np.mean(side_weights)), 1)
        else:
            avg_weight = list(results_by_side.values())[0]["weight_kg"]

        # Take primary profile result object for visualizer
        primary_res = results_by_side.get("left") or results_by_side.get("right") or list(results_by_side.values())[0]
        primary_res["weight_kg"] = avg_weight  # Angle-averaged weight

        # Log history entry
        log_entry = {
            "id": pred_id,
            "cattle_id": cattle_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
            "engine": model_engine,
            "weight": avg_weight,
            "confidence_pct": primary_res["confidence_pct"],
            "measurements": primary_res["measurements"]
        }
        save_history_entry(log_entry)

        return {
            "success": True,
            "id": pred_id,
            "cattle_id": cattle_id,
            "timestamp": log_entry["timestamp"],
            "image_data_url": primary_data_url,
            "prediction": primary_res,
            "predictions_by_view": results_by_side,
            "views_processed": list(results_by_side.keys())
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/history")
def get_prediction_history():
    return {"success": True, "history": load_history()}

@app.delete("/api/history/{entry_id}")
def delete_history_entry(entry_id: str):
    history = load_history()
    updated = [item for item in history if item["id"] != entry_id]
    with open(HISTORY_FILE, "w") as f:
        json.dump(updated, f, indent=2)
FRONTEND_DIST = os.path.join(PROJECT_ROOT, "frontend", "dist")
if os.path.exists(FRONTEND_DIST):
    # Mount assets folder if present
    assets_dir = os.path.join(FRONTEND_DIST, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        # Allow API endpoints to be handled first
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API endpoint not found")
        
        target_file = os.path.join(FRONTEND_DIST, full_path)
        if full_path and os.path.exists(target_file) and os.path.isfile(target_file):
            return FileResponse(target_file)
        
        index_file = os.path.join(FRONTEND_DIST, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"detail": "Frontend index.html missing"}
else:
    @app.get("/")
    def root_fallback():
        return {
            "status": "online",
            "message": "CattleWeightAI API Backend is running.",
            "note": "Frontend dist folder not detected."
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)
