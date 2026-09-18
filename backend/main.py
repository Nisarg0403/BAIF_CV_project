import os
import sys
import json
import uuid
import base64
import numpy as np
from datetime import datetime
from typing import Optional, List
import cv2

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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

@app.post("/api/predict_video")
async def predict_cattle_weight_video(
    video_file: UploadFile = File(...),
    cattle_id: str = Form("TAG-105730429112"),
    model_engine: str = Form("deeplab")
):
    """
    Innovation 1: Multi-frame Video Keyframe Selection Endpoint.
    Accepts video input, extracts optimal keyframe, and delegates to existing inference logic.
    Falls back to frame 0 on any exception.
    """
    fallback_warning = None
    try:
        video_bytes = await video_file.read()
        if not video_bytes:
            raise ValueError("Empty video file payload received.")

        from src.features.video_keyframe_selector import select_best_keyframe
        best_frame = select_best_keyframe(video_bytes, max_frames=90)
        
        # Encode numpy array frame to JPEG bytes
        is_success, buffer = cv2.imencode(".jpg", best_frame)
        if not is_success:
            raise ValueError("Could not encode extracted keyframe to JPEG format.")
        img_bytes = buffer.tobytes()

    except Exception as e:
        # Fallback handling: log exception and extract frame 0
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "experimental")
        os.makedirs(log_dir, exist_ok=True)
        fallback_log = os.path.join(log_dir, f"fallback_predict_video_{timestamp_str}.log")
        with open(fallback_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"Video keyframe selection failed: {e}\n")
        
        fallback_warning = "video_keyframe_fallback"
        
        # Extract frame 0 as fallback
        try:
            temp_vid = os.path.join(log_dir, f"temp_{timestamp_str}.mp4")
            with open(temp_vid, "wb") as f_tmp:
                f_tmp.write(video_bytes)
            cap = cv2.VideoCapture(temp_vid)
            ret, frame0 = cap.read()
            cap.release()
            if os.path.exists(temp_vid):
                os.remove(temp_vid)
            if ret and frame0 is not None:
                _, buffer = cv2.imencode(".jpg", frame0)
                img_bytes = buffer.tobytes()
            else:
                raise ValueError("Could not extract frame 0 fallback.")
        except Exception as fb_err:
            raise HTTPException(status_code=400, detail=f"Video processing & fallback failed: {fb_err}")

    # Delegate to existing process_image_file logic
    res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine)
    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    if fallback_warning:
        res["warning"] = fallback_warning

    log_entry = {
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "engine": model_engine,
        "weight": res["weight_kg"],
        "confidence_pct": res["confidence_pct"],
        "measurements": res["measurements"]
    }
    save_history_entry(log_entry)

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": log_entry["timestamp"],
        "image_data_url": data_url,
        "prediction": res,
        "warning": fallback_warning
    }

@app.post("/api/predict_kan")
async def predict_cattle_weight_kan(
    file: UploadFile = File(None),
    cattle_id: str = Form("TAG-105730429112"),
    model_engine: str = Form("deeplab")
):
    """
    Innovation 5: Isolated PyTorch B-spline KAN Regressor Endpoint.
    Falls back to XGBoost baseline with warning on any failure.
    """
    fallback_warning = None
    try:
        if not file:
            raise ValueError("No file uploaded.")
        img_bytes = await file.read()
        res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine)
        
        m = res["measurements"]
        X_feat = np.array([[m["body_length_cm"], m["chest_girth_cm"], m["silhouette_area_cm2"], m["withers_height_cm"]]])
        
        from src.models.kan_regressor import KANRegressor
        kan_reg = KANRegressor(in_features=4)
        kan_weight = float(kan_reg.predict(X_feat)[0])
        
        res["weight_kg"] = round(kan_weight, 1)
        res["model_predictions"]["PyTorch KAN Regressor"] = round(kan_weight, 1)

    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "experimental")
        os.makedirs(log_dir, exist_ok=True)
        fallback_log = os.path.join(log_dir, f"fallback_predict_kan_{timestamp_str}.log")
        with open(fallback_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"KAN prediction failed: {e}\n")
        
        fallback_warning = "kan_fallback"
        # Fallback to existing process_image_file
        if file:
            await file.seek(0)
            img_bytes = await file.read()
            res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine)
        res["warning"] = fallback_warning

    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "image_data_url": data_url,
        "prediction": res,
        "warning": fallback_warning
    }

@app.post("/api/predict_ensemble")
async def predict_cattle_weight_ensemble(
    file: UploadFile = File(None),
    cattle_id: str = Form("TAG-105730429112"),
    model_engine: str = Form("deeplab"),
    alpha: float = Form(1.0)
):
    """
    Innovation 5: Weighted XGBoost + KAN Ensemble Endpoint (final_weight = alpha * XGB + (1-alpha) * KAN).
    Default alpha = 1.0 preserves pure XGBoost baseline. Falls back to pure XGBoost on failure.
    """
    fallback_warning = None
    try:
        if not file:
            raise ValueError("No file uploaded.")
        img_bytes = await file.read()
        res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine)
        
        xgb_weight = res["weight_kg"]
        m = res["measurements"]
        X_feat = np.array([[m["body_length_cm"], m["chest_girth_cm"], m["silhouette_area_cm2"], m["withers_height_cm"]]])
        
        from src.models.kan_regressor import KANRegressor
        kan_reg = KANRegressor(in_features=4)
        ensemble_wt = kan_reg.ensemble_predict(xgb_weight, X_feat, alpha=alpha)
        
        res["weight_kg"] = round(ensemble_wt, 1)
        res["model_predictions"][f"Ensemble (alpha={alpha})"] = round(ensemble_wt, 1)

    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "experimental")
        os.makedirs(log_dir, exist_ok=True)
        fallback_log = os.path.join(log_dir, f"fallback_predict_ensemble_{timestamp_str}.log")
        with open(fallback_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"Ensemble prediction failed: {e}\n")
        
        fallback_warning = "ensemble_fallback"
        if file:
            await file.seek(0)
            img_bytes = await file.read()
            res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine)
        res["warning"] = fallback_warning

    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "image_data_url": data_url,
        "prediction": res,
        "warning": fallback_warning
    }

@app.post("/api/predict_dual_angle")
async def predict_cattle_weight_dual_angle(
    side_file: UploadFile = File(None),
    rear_file: UploadFile = File(None),
    left_file: UploadFile = File(None),
    file: UploadFile = File(None),
    cattle_id: str = Form("TAG-105730429112"),
    model_engine: str = Form("deeplab")
):
    """
    Innovation 3: Dual-Angle Guided Capture (Side + 45° Rear View) Endpoint.
    Combines side profile silhouette with 45° rear view barrel width via cross-attention.
    Falls back to side profile only and logs to logs/experimental/fallback_predict_dual_angle_<timestamp>.log if rear view is missing or fails.
    """
    fallback_warning = None
    side_upload = side_file or left_file or file
    
    if not side_upload:
        raise HTTPException(status_code=400, detail="No side profile photo uploaded.")

    try:
        side_bytes = await side_upload.read()
        if not side_bytes:
            raise ValueError("Empty side file payload received.")

        res_side = process_image_file(side_bytes, side_name="Left Side Profile", model_engine=model_engine)

        # Attempt Dual-Angle Fusion if rear_file is provided
        if rear_file:
            rear_bytes = await rear_file.read()
            if rear_bytes:
                rear_arr = np.asarray(bytearray(rear_bytes), dtype=np.uint8)
                rear_img = cv2.imdecode(rear_arr, 1)
                if rear_img is not None:
                    # Segment rear view
                    from backend.inference import get_deeplab_segmenter, get_yolo_segmenter
                    segmenter = get_yolo_segmenter() if model_engine.lower() == "yolo" else get_deeplab_segmenter()
                    
                    rh, rw, _ = rear_img.shape
                    if max(rh, rw) > 800:
                        scale = 800.0 / max(rh, rw)
                        small_rear = cv2.resize(rear_img, (int(rw * scale), int(rh * scale)))
                    else:
                        small_rear = rear_img.copy()
                    
                    rear_mask = segmenter.segment(small_rear)
                    
                    from src.models.dual_angle_fusion import DualAngleFusionModule
                    fusion_module = DualAngleFusionModule()
                    fused_res = fusion_module.process_dual_views(res_side, rear_mask=rear_mask)
                    
                    res_side["weight_kg"] = fused_res["fused_weight_kg"]
                    res_side["dual_angle_fusion"] = fused_res
                    if "model_predictions" in res_side:
                        res_side["model_predictions"]["Dual-Angle Cross-Attention Fusion"] = fused_res["fused_weight_kg"]
            else:
                raise ValueError("Rear view payload empty.")
        else:
            raise ValueError("Rear view photo not provided for dual-angle fusion.")

    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = os.path.join(PROJECT_ROOT, "logs", "experimental")
        os.makedirs(log_dir, exist_ok=True)
        fallback_log = os.path.join(log_dir, f"fallback_predict_dual_angle_{timestamp_str}.log")
        with open(fallback_log, "w", encoding="utf-8") as f_log:
            f_log.write(f"Dual angle fusion failed or rear image missing: {e}\n")
        
        fallback_warning = "dual_angle_fallback"
        if side_upload:
            await side_upload.seek(0)
            side_bytes = await side_upload.read()
            res_side = process_image_file(side_bytes, side_name="Left Side Profile", model_engine=model_engine)
        res_side["warning"] = fallback_warning

    b64_img = base64.b64encode(side_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    log_entry = {
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "engine": model_engine,
        "weight": res_side["weight_kg"],
        "confidence_pct": res_side["confidence_pct"],
        "measurements": res_side["measurements"]
    }
    save_history_entry(log_entry)

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": log_entry["timestamp"],
        "image_data_url": data_url,
        "prediction": res_side,
        "warning": fallback_warning
    }




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
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("backend.main:app", host="0.0.0.0", port=port)
