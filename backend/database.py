import os
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_FILE = os.path.join(PROJECT_ROOT, "data", "processed", "history.json")
UPLOADS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "uploaded_images")

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)

# Load environment variables from .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Supabase Credentials from Environment Variables
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

supabase_client = None

if SUPABASE_URL and SUPABASE_KEY:
    try:
        from supabase import create_client
        supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[SUCCESS] Supabase client initialized successfully.")
    except Exception as e:
        print(f"[WARNING] Supabase client initialization failed: {e}. Falling back to local storage.")
else:
    print("[INFO] SUPABASE_URL / SUPABASE_KEY missing. Operating in local JSON fallback mode.")


def is_supabase_enabled() -> bool:
    """Returns True if Supabase client is connected and ready."""
    return supabase_client is not None


# ==========================================
# Local JSON Fallback Helpers
# ==========================================
def load_local_history() -> List[Dict[str, Any]]:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_local_history_entry(entry: Dict[str, Any]):
    history = load_local_history()
    history.insert(0, entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def delete_local_history_entry(entry_id: str):
    history = load_local_history()
    updated = [item for item in history if item.get("id") != entry_id]
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2)


# ==========================================
# Supabase & Storage Operations
# ==========================================
def upload_media_to_supabase(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> Optional[str]:
    """
    Uploads a media file (image or video) to Supabase Storage bucket 'cattle-media'.
    Returns public CDN URL on success, or None on error/missing client.
    """
    if not is_supabase_enabled():
        return None
    
    try:
        bucket_name = "cattle-media"
        filepath = f"uploads/{datetime.now().strftime('%Y%m')}/{filename}"
        
        # Upload byte stream
        supabase_client.storage.from_(bucket_name).upload(
            path=filepath,
            file=file_bytes,
            file_options={"content-type": content_type, "x-upsert": "true"}
        )
        
        # Get public URL
        public_url = supabase_client.storage.from_(bucket_name).get_public_url(filepath)
        return public_url
    except Exception as e:
        print(f"[WARNING] Supabase storage upload warning: {e}")
        return None


def save_prediction_record(
    pred_id: str,
    cattle_id: str, # Represents the tag_number string extracted from image (e.g. 105729971323)
    engine: str,
    weight_kg: float,
    confidence_pct: float,
    measurements: Dict[str, Any],
    predictions_by_view: Optional[Dict[str, Any]] = None,
    image_bytes: Optional[bytes] = None,
    image_filename: Optional[str] = None
) -> Dict[str, Any]:
    """
    Saves a prediction record.
    Uses Supabase if available; falls back to history.json.
    """
    tag_number = cattle_id.strip()
    timestamp_str = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p")
    local_iso_str = datetime.now().astimezone().isoformat()
    image_url = None

    # Handle image storage
    if image_bytes:
        if not image_filename:
            image_filename = f"pred_{pred_id}.jpg"
        
        # Try uploading to Supabase Storage
        if is_supabase_enabled():
            image_url = upload_media_to_supabase(image_bytes, image_filename, "image/jpeg")

        # Fallback local image write if local or storage upload failed
        local_path = os.path.join(UPLOADS_DIR, image_filename)
        try:
            with open(local_path, "wb") as f:
                f.write(image_bytes)
            if not image_url:
                image_url = f"/static/uploads/{image_filename}"
        except Exception as e:
            print(f"Warning writing local image: {e}")

    log_entry = {
        "id": pred_id,
        "cattle_id": tag_number,
        "tag_number": tag_number,
        "timestamp": timestamp_str,
        "engine": engine,
        "weight": weight_kg,
        "confidence_pct": confidence_pct,
        "measurements": measurements or {},
        "image_url": image_url
    }

    # Extract individual measurement values for independent SQL columns
    m = measurements or {}
    body_length = m.get("body_length_cm")
    chest_girth = m.get("chest_girth_cm")
    withers_height = m.get("withers_height_cm")
    stature_height = m.get("stature_height_cm")
    silhouette_area = m.get("silhouette_area_cm2")

    # 1. Save to Supabase Cloud Database if available
    if is_supabase_enabled():
        try:
            # Query existing cattle UUID by tag_number or insert new cattle
            cattle_res = supabase_client.table("cattle").select("cattle_id").eq("tag_number", tag_number).execute()
            if cattle_res.data and len(cattle_res.data) > 0:
                db_cattle_id = cattle_res.data[0]["cattle_id"]
            else:
                ins_res = supabase_client.table("cattle").insert({
                    "tag_number": tag_number,
                    "breed": "HF",
                    "created_at": local_iso_str
                }).execute()
                db_cattle_id = ins_res.data[0]["cattle_id"] if ins_res.data else None

            # Insert prediction record into public.predictions with independent columns
            db_record = {
                "id": pred_id,
                "cattle_id": db_cattle_id,
                "tag_number": tag_number,
                "engine": engine,
                "estimated_weight": weight_kg,
                "confidence_pct": confidence_pct,
                "chest_girth": chest_girth,
                "body_length": body_length,
                "height_at_withers": withers_height,
                "height_at_stature": stature_height,
                "silhouette_area": silhouette_area,
                "image_url": image_url,
                "created_at": local_iso_str
            }
            res_ins = supabase_client.table("predictions").insert(db_record).execute()
            print(f"[SUCCESS] Saved prediction {pred_id} to Supabase Database: {res_ins}")
        except Exception as e:
            print(f"[ERROR] Failed to insert record into Supabase: {e}. Writing to local JSON as fallback.")

    # 2. Always maintain local history JSON as backup/cache
    save_local_history_entry(log_entry)
    return log_entry


def get_all_predictions(limit: int = 50) -> List[Dict[str, Any]]:
    """
    Fetches prediction history. Prefers Supabase Cloud DB; falls back to local history.json.
    """
    if is_supabase_enabled():
        try:
            res = (
                supabase_client.table("predictions")
                .select("*")
                .order("created_at", desc=True)
                .limit(limit)
                .execute()
            )
            rows = res.data or []
            
            # Map Supabase DB rows to frontend-expected fields
            formatted = []
            for r in rows:
                weight_val = float(r.get("estimated_weight") or r.get("weight_kg") or 0.0)
                body_len = float(r.get("body_length") or r.get("body_length_cm") or 0.0)
                chest_g = float(r.get("chest_girth") or r.get("chest_girth_cm") or 0.0)
                withers_h = float(r.get("height_at_withers") or r.get("withers_height_cm") or 0.0)
                stature_h = float(r.get("height_at_stature") or r.get("stature_height_cm") or 0.0)
                sil_area = float(r.get("silhouette_area") or r.get("silhouette_area_cm2") or 0.0)

                created_dt = r.get("created_at")
                if created_dt:
                    try:
                        ts_str = datetime.fromisoformat(created_dt.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %I:%M %p")
                    except Exception:
                        ts_str = created_dt
                else:
                    ts_str = datetime.now().astimezone().strftime("%Y-%m-%d %I:%M %p")

                tag_num = r.get("tag_number") or r.get("cattle_id") or ""
                formatted.append({
                    "id": r.get("id"),
                    "cattle_id": tag_num,
                    "tag_number": tag_num,
                    "timestamp": ts_str,
                    "engine": r.get("engine"),
                    "weight": weight_val,
                    "estimated_weight": weight_val,
                    "confidence_pct": float(r.get("confidence_pct") or 90.0),
                    "chest_girth": chest_g,
                    "body_length": body_len,
                    "height_at_withers": withers_h,
                    "height_at_stature": stature_h,
                    "silhouette_area": sil_area,
                    "image_url": r.get("image_url")
                })
            return formatted
        except Exception as e:
            print(f"[WARNING] Failed to fetch predictions from Supabase: {e}. Falling back to local history.")

    return load_local_history()[:limit]


def delete_prediction_record(entry_id: str) -> bool:
    """
    Deletes a prediction record from Supabase and local history.
    """
    if is_supabase_enabled():
        try:
            supabase_client.table("predictions").delete().eq("id", entry_id).execute()
            print(f"[SUCCESS] Deleted record {entry_id} from Supabase.")
        except Exception as e:
            print(f"[WARNING] Error deleting from Supabase: {e}")

    delete_local_history_entry(entry_id)
    return True


def save_ground_truth_feedback(cattle_id: str, actual_weight_kg: float, measured_by: str = "BAIF Agent", notes: str = "") -> bool:
    """
    Ground truth feedback recording (disabled per current database schema).
    """
    return True


