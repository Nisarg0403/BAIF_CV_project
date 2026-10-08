import os
import sys

os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import json
import uuid
import base64
import re
import numpy as np
from datetime import datetime
from typing import Optional, List
import cv2

from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from pathlib import Path

def clean_cattle_tag(raw_tag: Optional[str]) -> str:
    """
    Cleans a tag string by:
    1. Stripping file extensions (.jpg, .jpeg, .png, .mp4, .mov, .webm, etc.)
    2. Stripping view/part/number suffixes (_Back_1, _Back, _LL_1, -LR, _left_2, _right_1, _1, _2, etc.)
    3. Stripping trailing/leading whitespace.
    """
    if not raw_tag:
        return ""
    
    # 1. Strip extension and whitespace using stdlib Path.stem
    tag_str = Path(str(raw_tag).strip()).stem.strip()
    
    # 2. Iteratively strip view/part/number suffixes anchored at the end of the string
    suffix_pattern = re.compile(r'[_\-\s]+(?:LL|LR|back|rear|left|right|side|front|profile)(?:[_\-\s]+\d+)?\s*$', re.IGNORECASE)
    idx_pattern = re.compile(r'[_\-\s]+\d{1,2}\s*$', re.IGNORECASE)
    
    while True:
        new_tag = suffix_pattern.sub('', tag_str).strip()
        new_tag = idx_pattern.sub('', new_tag).strip()
        if new_tag == tag_str or len(new_tag) == 0:
            break
        tag_str = new_tag

    GENERIC_STEMS = {
        "image", "photo", "upload", "file", "blob", "left", "right", "rear", "back", "side", "front",
        "video", "frame", "captured", "input", "test", "temp", "sample", "unknown", "cattle_snap",
        "ll", "lr", "side_file", "left_file", "right_file", "rear_file", "profile", "tag"
    }

    if tag_str.lower() in GENERIC_STEMS or len(tag_str) == 0:
        return ""

    return tag_str

def derive_cattle_tag_from_request(
    form_cattle_id: Optional[str],
    *files: Optional[UploadFile]
) -> str:
    """
    Auto-extracts cattle tag ID from uploaded image/video filenames or form_cattle_id.
    Strips file extensions and part suffixes (_LL, _LR, _back, _side, etc.).
    """
    GENERIC_STEMS = {
        "image", "photo", "upload", "file", "blob", "left", "right", "rear", "back", "side", "front",
        "video", "frame", "captured", "input", "test", "temp", "sample", "unknown", "cattle_snap",
        "ll", "lr", "side_file", "left_file", "right_file", "rear_file", "profile", "tag"
    }

    # 1. Check filenames from uploaded files first
    for file in files:
        if file and file.filename:
            stem = clean_cattle_tag(file.filename)
            if stem and stem.lower() not in GENERIC_STEMS:
                return stem

    # 2. Check form_cattle_id if passed
    if form_cattle_id and form_cattle_id.strip():
        cleaned_form = clean_cattle_tag(form_cattle_id)
        if cleaned_form and cleaned_form.lower() not in GENERIC_STEMS:
            return cleaned_form

    # 3. Dynamic fallback tag if no tag provided or extractable
    return f"TAG-{int(datetime.now().timestamp()) % 100000:05d}"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.inference import process_image_file, get_effective_feature_flags
from backend.database import (
    save_prediction_record,
    get_all_predictions,
    delete_prediction_record,
    save_ground_truth_feedback,
    is_supabase_enabled
)

def extract_request_feature_flags(
    x_feature_flags: Optional[str] = None,
    x_enable_unwarp: Optional[str] = None,
    x_enable_exif: Optional[str] = None,
    x_enable_xai: Optional[str] = None,
    x_enable_kan: Optional[str] = None,
    x_enable_video: Optional[str] = None,
    x_enable_dual_angle: Optional[str] = None,
    form_flags: Optional[str] = None
) -> dict:
    overrides = {}
    
    for flag_str in [x_feature_flags, form_flags]:
        if flag_str:
            if flag_str.startswith("{"):
                try:
                    overrides.update(json.loads(flag_str))
                except Exception:
                    pass
            else:
                parts = flag_str.split(",")
                for part in parts:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        overrides[k.strip().upper()] = v.strip().lower() in ("true", "1", "yes")
    
    if x_enable_unwarp is not None:
        overrides["ENABLE_PERSPECTIVE_UNWARP"] = x_enable_unwarp.lower() in ("true", "1", "yes")
    if x_enable_exif is not None:
        overrides["ENABLE_EXIF_CALIBRATION"] = x_enable_exif.lower() in ("true", "1", "yes")
    if x_enable_xai is not None:
        overrides["ENABLE_XAI_CARDS"] = x_enable_xai.lower() in ("true", "1", "yes")
    if x_enable_kan is not None:
        overrides["ENABLE_KAN"] = x_enable_kan.lower() in ("true", "1", "yes")
    if x_enable_video is not None:
        overrides["ENABLE_VIDEO_KEYFRAME"] = x_enable_video.lower() in ("true", "1", "yes")
    if x_enable_dual_angle is not None:
        overrides["ENABLE_DUAL_ANGLE"] = x_enable_dual_angle.lower() in ("true", "1", "yes")

    return overrides

import logging
logger = logging.getLogger("CattleWeightAI")

def audit_log_flags(endpoint: str, effective_flags: dict):
    logger.info(f"Endpoint: {endpoint} | Effective Flags: {effective_flags}")

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

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "service": "CattleWeightAI Neural Engine",
        "version": "2.4.0",
        "database_backend": "Supabase Cloud DB" if is_supabase_enabled() else "Local JSON Fallback",
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
    cattle_id: str = Form(""),
    model_engine: str = Form("deeplab"),
    side_name: str = Form("Left Side Profile"),
    feature_flags_form: Optional[str] = Form(None),
    x_feature_flags: Optional[str] = Header(None),
    x_enable_unwarp: Optional[str] = Header(None),
    x_enable_exif: Optional[str] = Header(None),
    x_enable_xai: Optional[str] = Header(None),
    x_enable_kan: Optional[str] = Header(None),
    x_enable_video: Optional[str] = Header(None),
    x_enable_dual_angle: Optional[str] = Header(None)
):
    try:
        request_overrides = extract_request_feature_flags(
            x_feature_flags=x_feature_flags,
            x_enable_unwarp=x_enable_unwarp,
            x_enable_exif=x_enable_exif,
            x_enable_xai=x_enable_xai,
            x_enable_kan=x_enable_kan,
            x_enable_video=x_enable_video,
            x_enable_dual_angle=x_enable_dual_angle,
            form_flags=feature_flags_form
        )
        effective_flags = get_effective_feature_flags(request_overrides)
        audit_log_flags("/api/predict", effective_flags)

        derived_cattle_id = derive_cattle_tag_from_request(cattle_id, left_file, right_file, rear_file, file)

        results_by_side = {}
        primary_file_bytes = None
        primary_data_url = None
        pred_id = str(uuid.uuid4())[:8]

        # 1. Process Left Side Profile
        if left_file:
            left_bytes = await left_file.read()
            primary_file_bytes = left_bytes
            res_left = process_image_file(left_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
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
            res_right = process_image_file(right_bytes, side_name="Right Side Profile", model_engine=model_engine, feature_flags=effective_flags)
            results_by_side["right"] = res_right

        # 3. Process Rear View
        if rear_file:
            rear_bytes = await rear_file.read()
            if not primary_file_bytes:
                primary_file_bytes = rear_bytes
                b64_img = base64.b64encode(rear_bytes).decode("utf-8")
                primary_data_url = f"data:image/jpeg;base64,{b64_img}"
            res_rear = process_image_file(rear_bytes, side_name="Rear View", model_engine=model_engine, feature_flags=effective_flags)
            results_by_side["rear"] = res_rear

        # Fallback to single 'file' parameter if multi-view parameters weren't passed
        if not results_by_side and file:
            single_bytes = await file.read()
            primary_file_bytes = single_bytes
            b64_img = base64.b64encode(single_bytes).decode("utf-8")
            primary_data_url = f"data:image/jpeg;base64,{b64_img}"
            res_single = process_image_file(single_bytes, side_name=side_name, model_engine=model_engine, feature_flags=effective_flags)
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

        # Log history entry to Database / Storage
        log_entry = save_prediction_record(
            pred_id=pred_id,
            cattle_id=derived_cattle_id,
            engine=model_engine,
            weight_kg=avg_weight,
            confidence_pct=primary_res["confidence_pct"],
            measurements=primary_res["measurements"],
            predictions_by_view=results_by_side,
            image_bytes=primary_file_bytes,
            image_filename=f"pred_{pred_id}.jpg"
        )

        return {
            "success": True,
            "id": pred_id,
            "cattle_id": derived_cattle_id,
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
    cattle_id: str = Form(""),
    model_engine: str = Form("deeplab"),
    feature_flags_form: Optional[str] = Form(None),
    x_feature_flags: Optional[str] = Header(None),
    x_enable_unwarp: Optional[str] = Header(None),
    x_enable_exif: Optional[str] = Header(None),
    x_enable_xai: Optional[str] = Header(None),
    x_enable_kan: Optional[str] = Header(None),
    x_enable_video: Optional[str] = Header(None),
    x_enable_dual_angle: Optional[str] = Header(None)
):
    """
    Innovation 1: Multi-frame Video Keyframe Selection Endpoint.
    Accepts video input, extracts optimal keyframe, and delegates to existing inference logic.
    Falls back to frame 0 on any exception.
    """
    request_overrides = extract_request_feature_flags(
        x_feature_flags=x_feature_flags,
        x_enable_unwarp=x_enable_unwarp,
        x_enable_exif=x_enable_exif,
        x_enable_xai=x_enable_xai,
        x_enable_kan=x_enable_kan,
        x_enable_video=x_enable_video,
        x_enable_dual_angle=x_enable_dual_angle,
        form_flags=feature_flags_form
    )
    effective_flags = get_effective_feature_flags(request_overrides)
    audit_log_flags("/api/predict_video", effective_flags)

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
    res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    if fallback_warning:
        res["warning"] = fallback_warning

    derived_cattle_id = derive_cattle_tag_from_request(cattle_id, video_file)

    log_entry = save_prediction_record(
        pred_id=pred_id,
        cattle_id=derived_cattle_id,
        engine=model_engine,
        weight_kg=res["weight_kg"],
        confidence_pct=res["confidence_pct"],
        measurements=res["measurements"],
        image_bytes=img_bytes,
        image_filename=f"pred_vid_{pred_id}.jpg"
    )

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": derived_cattle_id,
        "timestamp": log_entry["timestamp"],
        "image_data_url": data_url,
        "prediction": res,
        "warning": fallback_warning
    }

@app.post("/api/predict_kan")
async def predict_cattle_weight_kan(
    file: UploadFile = File(None),
    cattle_id: str = Form(""),
    model_engine: str = Form("deeplab"),
    feature_flags_form: Optional[str] = Form(None),
    x_feature_flags: Optional[str] = Header(None),
    x_enable_unwarp: Optional[str] = Header(None),
    x_enable_exif: Optional[str] = Header(None),
    x_enable_xai: Optional[str] = Header(None),
    x_enable_kan: Optional[str] = Header(None),
    x_enable_video: Optional[str] = Header(None),
    x_enable_dual_angle: Optional[str] = Header(None)
):
    """
    Innovation 5: Isolated PyTorch B-spline KAN Regressor Endpoint.
    Falls back to XGBoost baseline with warning on any failure.
    """
    request_overrides = extract_request_feature_flags(
        x_feature_flags=x_feature_flags,
        x_enable_unwarp=x_enable_unwarp,
        x_enable_exif=x_enable_exif,
        x_enable_xai=x_enable_xai,
        x_enable_kan=x_enable_kan,
        x_enable_video=x_enable_video,
        x_enable_dual_angle=x_enable_dual_angle,
        form_flags=feature_flags_form
    )
    effective_flags = get_effective_feature_flags(request_overrides)
    audit_log_flags("/api/predict_kan", effective_flags)

    fallback_warning = None
    try:
        if not file:
            raise ValueError("No file uploaded.")
        img_bytes = await file.read()
        res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
        
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
            res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
        res["warning"] = fallback_warning

    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    derived_cattle_id = derive_cattle_tag_from_request(cattle_id, file)

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": derived_cattle_id,
        "timestamp": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
        "image_data_url": data_url,
        "prediction": res,
        "warning": fallback_warning
    }

@app.post("/api/predict_ensemble")
async def predict_cattle_weight_ensemble(
    file: UploadFile = File(None),
    cattle_id: str = Form(""),
    model_engine: str = Form("deeplab"),
    alpha: float = Form(1.0),
    feature_flags_form: Optional[str] = Form(None),
    x_feature_flags: Optional[str] = Header(None),
    x_enable_unwarp: Optional[str] = Header(None),
    x_enable_exif: Optional[str] = Header(None),
    x_enable_xai: Optional[str] = Header(None),
    x_enable_kan: Optional[str] = Header(None),
    x_enable_video: Optional[str] = Header(None),
    x_enable_dual_angle: Optional[str] = Header(None)
):
    """
    Innovation 5: Weighted XGBoost + KAN Ensemble Endpoint (final_weight = alpha * XGB + (1-alpha) * KAN).
    Default alpha = 1.0 preserves pure XGBoost baseline. Falls back to pure XGBoost on failure.
    """
    request_overrides = extract_request_feature_flags(
        x_feature_flags=x_feature_flags,
        x_enable_unwarp=x_enable_unwarp,
        x_enable_exif=x_enable_exif,
        x_enable_xai=x_enable_xai,
        x_enable_kan=x_enable_kan,
        x_enable_video=x_enable_video,
        x_enable_dual_angle=x_enable_dual_angle,
        form_flags=feature_flags_form
    )
    effective_flags = get_effective_feature_flags(request_overrides)
    audit_log_flags("/api/predict_ensemble", effective_flags)

    fallback_warning = None
    try:
        if not file:
            raise ValueError("No file uploaded.")
        img_bytes = await file.read()
        res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
        
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
            res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
        res["warning"] = fallback_warning

    b64_img = base64.b64encode(img_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    derived_cattle_id = derive_cattle_tag_from_request(cattle_id, file)

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": derived_cattle_id,
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
    cattle_id: str = Form(""),
    model_engine: str = Form("deeplab"),
    feature_flags_form: Optional[str] = Form(None),
    x_feature_flags: Optional[str] = Header(None),
    x_enable_unwarp: Optional[str] = Header(None),
    x_enable_exif: Optional[str] = Header(None),
    x_enable_xai: Optional[str] = Header(None),
    x_enable_kan: Optional[str] = Header(None),
    x_enable_video: Optional[str] = Header(None),
    x_enable_dual_angle: Optional[str] = Header(None)
):
    """
    Innovation 3: Dual-Angle Guided Capture (Side + 45° Rear View) Endpoint.
    Combines side profile silhouette with 45° rear view barrel width via cross-attention.
    Falls back to side profile only and logs to logs/experimental/fallback_predict_dual_angle_<timestamp>.log if rear view is missing or fails.
    """
    request_overrides = extract_request_feature_flags(
        x_feature_flags=x_feature_flags,
        x_enable_unwarp=x_enable_unwarp,
        x_enable_exif=x_enable_exif,
        x_enable_xai=x_enable_xai,
        x_enable_kan=x_enable_kan,
        x_enable_video=x_enable_video,
        x_enable_dual_angle=x_enable_dual_angle,
        form_flags=feature_flags_form
    )
    effective_flags = get_effective_feature_flags(request_overrides)
    audit_log_flags("/api/predict_dual_angle", effective_flags)

    fallback_warning = None
    side_upload = side_file or left_file or file
    
    if not side_upload:
        raise HTTPException(status_code=400, detail="No side profile photo uploaded.")

    try:
        side_bytes = await side_upload.read()
        if not side_bytes:
            raise ValueError("Empty side file payload received.")

        res_side = process_image_file(side_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)

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
            res_side = process_image_file(side_bytes, side_name="Left Side Profile", model_engine=model_engine, feature_flags=effective_flags)
        res_side["warning"] = fallback_warning

    b64_img = base64.b64encode(side_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64_img}"
    pred_id = str(uuid.uuid4())[:8]

    derived_cattle_id = derive_cattle_tag_from_request(cattle_id, side_file, left_file, file)

    log_entry = save_prediction_record(
        pred_id=pred_id,
        cattle_id=derived_cattle_id,
        engine=model_engine,
        weight_kg=res_side["weight_kg"],
        confidence_pct=res_side["confidence_pct"],
        measurements=res_side["measurements"],
        image_bytes=side_bytes,
        image_filename=f"pred_dual_{pred_id}.jpg"
    )

    return {
        "success": True,
        "id": pred_id,
        "cattle_id": derived_cattle_id,
        "timestamp": log_entry["timestamp"],
        "image_data_url": data_url,
        "prediction": res_side,
        "warning": fallback_warning
    }




@app.get("/api/history")
def get_prediction_history():
    return {"success": True, "history": get_all_predictions()}

@app.delete("/api/history/{entry_id}")
def delete_history_entry_endpoint(entry_id: str):
    success = delete_prediction_record(entry_id)
    return {"success": success, "deleted_id": entry_id}

class GroundTruthRequest(BaseModel):
    cattle_id: str
    actual_weight_kg: float
    measured_by: Optional[str] = "BAIF Technician"
    notes: Optional[str] = None

@app.post("/api/ground_truth")
def submit_ground_truth(req: GroundTruthRequest):
    """
    Submits scale-weighed cattle ground truth data from BAIF field agents.
    Stored in DB for dataset accumulation and model retraining.
    """
    cleaned_id = clean_cattle_tag(req.cattle_id)
    success = save_ground_truth_feedback(
        cattle_id=cleaned_id,
        actual_weight_kg=req.actual_weight_kg,
        measured_by=req.measured_by or "BAIF Technician",
        notes=req.notes or ""
    )
    if not success:
        raise HTTPException(status_code=500, detail="Failed to save ground truth record.")
    return {"success": True, "message": f"Ground truth recorded for cattle {cleaned_id}"}
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
