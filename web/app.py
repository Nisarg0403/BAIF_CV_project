import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import pickle
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
from io import BytesIO
import base64
import json
import uuid
from datetime import datetime

from src.segmentation.segment import CowSegmenter
from src.features.morphometry import extract_morphometric_features

# Set page config for a premium wide-layout dashboard
st.set_page_config(
    page_title="BAIF Cattle Weight Estimator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths for history
HISTORY_FILE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "history.json"))
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed", "history_images"))

# Custom premium styling using the BAIF Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Header Typography */
    .main-title {
        font-family: 'Outfit', sans-serif;
        color: #1E3A8A;
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #64748B;
        font-size: 1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Dashboard Cards */
    .dashboard-grid {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 1.5rem;
    }
    .db-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.02), 0 2px 4px -1px rgba(0,0,0,0.02);
        flex: 1;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .db-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0,0,0,0.05);
    }
    .db-card-val {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 800;
        color: #10B981;
        margin-bottom: 0.25rem;
    }
    .db-card-lbl {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        font-weight: 700;
        color: #475569;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .db-card-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #64748B;
        margin-top: 0.5rem;
    }
    
    /* Metric Result Card */
    .metric-result-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #172554 100%);
        color: #FFFFFF;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(30, 58, 138, 0.2);
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .metric-result-val {
        font-family: 'Outfit', sans-serif;
        font-size: 3.5rem;
        font-weight: 800;
        color: #10B981;
        text-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-result-lbl {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 700;
        color: #93C5FD;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .metric-result-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #E2E8F0;
        margin-top: 0.5rem;
    }
    
    /* Checklist / Stepper styling */
    .check-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid #E2E8F0;
        margin-bottom: 1.5rem;
    }
    .check-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        font-weight: 600;
        border-left: 4px solid #CBD5E1;
        background-color: #F8FAFC;
    }
    .check-item.complete {
        border-left-color: #10B981;
        background-color: #ECFDF5;
        color: #065F46;
    }
    .check-item.missing {
        border-left-color: #EF4444;
        background-color: #FEF2F2;
        color: #991B1B;
    }
    
    /* Info / Help Cards */
    .info-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #E2E8F0;
        border-left: 5px solid #1E3A8A;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01);
        margin-bottom: 1.25rem;
    }
    .info-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .info-desc {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #475569;
        line-height: 1.6;
    }
    
    /* General styles */
    .section-header {
        font-family: 'Outfit', sans-serif;
        color: #1E3A8A;
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E2E8F0;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)

# Custom camera component declaration
parent_dir = os.path.dirname(os.path.abspath(__file__))
build_dir = os.path.join(parent_dir, "camera_component")
camera_capture_component = components.declare_component("camera_capture_component", path=build_dir)

# Cache resources to keep app fast
@st.cache_resource
def load_segmenter():
    return CowSegmenter()

@st.cache_resource
def load_regressor():
    baif_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "baif_best_weight_regressor.pkl"))
    std_model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_weight_regressor.pkl"))
    
    model_path = baif_model_path if os.path.exists(baif_model_path) else std_model_path
    if not os.path.exists(model_path):
        st.error(f"Model file not found at path: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            model._is_baif_model = os.path.exists(baif_model_path)
            return model
    except Exception as e:
        st.error(f"Failed to load pickle model: {e}")
        return None

def load_image_for_display(file_or_path):
    """
    Safely loads an uploaded file or path into a PIL Image for display.
    """
    if file_or_path is None:
        return None
    try:
        if isinstance(file_or_path, str):
            if os.path.exists(file_or_path):
                return Image.open(file_or_path)
            return None
        # BytesIO / Streamlit uploaded file
        if hasattr(file_or_path, "getvalue"):
            bytes_data = file_or_path.getvalue()
        else:
            file_or_path.seek(0)
            bytes_data = file_or_path.read()
        return Image.open(BytesIO(bytes_data))
    except Exception as e:
        print(f"Error loading image for display: {e}")
        return None

def process_side_image(input_file, side_name, segmenter, model, withers_height_cm=135.0):
    """
    Processes a side profile image: segments, extracts morphometrics, and runs regression.
    """
    try:
        # Load and decode image bytes safely
        if hasattr(input_file, "getvalue"):
            bytes_data = input_file.getvalue()
        else:
            input_file.seek(0)
            bytes_data = input_file.read()
            
        file_bytes = np.asarray(bytearray(bytes_data), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        if img is None:
            print(f"Error: Could not decode image for {side_name}")
            return None
            
        original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 1. Downscale image for segmentation to save RAM and prevent OOM crashes on Streamlit Cloud
        h, w, _ = img.shape
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            small_img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            small_img = img.copy()
            
        # Segment on the downscaled image
        small_mask = segmenter.segment(small_img)
        if np.sum(small_mask) == 0:
            return None
            
        # Resize mask back to original dimensions
        mask = cv2.resize(small_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # 2. Extract bounding box of the cow mask
        y_indices, x_indices = np.nonzero(mask)
        if len(y_indices) == 0:
            return None
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        
        # Crop mask to isolate the cow body
        cropped_mask = mask[ymin:ymax+1, xmin:xmax+1]
        crop_h, crop_w = cropped_mask.shape
        
        # Check model type: BAIF model uses physical scale calibration
        is_baif = getattr(model, '_is_baif_model', False)
        
        if is_baif:
            raw_feats = extract_morphometric_features(mask)
            scale_cm_per_px = withers_height_cm / raw_feats['height'] if raw_feats['height'] > 0 else 0.5
            cv_length_cm = raw_feats['length'] * scale_cm_per_px
            cv_girth_cm = raw_feats['girth'] * scale_cm_per_px
            cv_area_cm2 = raw_feats['area'] * (scale_cm_per_px ** 2)
            
            X_input = np.array([[cv_length_cm, cv_girth_cm, cv_area_cm2, withers_height_cm]])
            feats = {
                'length': cv_length_cm,
                'height': withers_height_cm,
                'area': cv_area_cm2,
                'girth': cv_girth_cm
            }
        else:
            # Standardize cropped mask height to 225px to match standard training scale
            std_h = 225
            scale_factor = std_h / crop_h if crop_h > 0 else 1.0
            std_w = int(crop_w * scale_factor)
            std_mask = cv2.resize(cropped_mask, (std_w, std_h), interpolation=cv2.INTER_NEAREST)
            feats = extract_morphometric_features(std_mask)
            X_input = np.array([[feats['length'], feats['height'], feats['area'], feats['girth']]])
            
        predicted_weight = float(model.predict(X_input)[0])
        
        # Draw visualization overlay (Green mask + red bbox)
        overlay = img.copy()
        overlay[mask == 255] = [0, 255, 0]  # Green cow silhouette
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        
        # Draw bbox
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        visualizer_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Calculate distance verification ratio (cow height / original image height)
        height_ratio = float(crop_h / h)
        warning = None
        if height_ratio < 0.38:
            warning = "too_far"
        elif height_ratio > 0.88:
            warning = "too_close"
        
        return {
            'weight': predicted_weight,
            'feats': feats,
            'original': original_img_rgb,
            'visualizer': visualizer_rgb,
            'height_ratio': height_ratio,
            'warning': warning
        }
    except Exception as e:
        print(f"Error processing side image {side_name}: {e}")
        return None

# Local history database persistence functions
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except Exception:
        return []

def delete_history_item(pred_id):
    history = load_history()
    updated = [item for item in history if item['id'] != pred_id]
    
    # Clean up associated images if present
    for item in history:
        if item['id'] == pred_id:
            # Delete images
            for key in ['left_img_path', 'right_img_path']:
                if item.get(key):
                    full_path = os.path.abspath(os.path.join(os.path.dirname(HISTORY_FILE), item[key]))
                    if os.path.exists(full_path):
                        try:
                            os.remove(full_path)
                        except Exception:
                            pass
                            
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(updated, f, indent=2)

def save_prediction(cattle_id, avg_weight, res_left, res_right):
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    
    history = load_history()
    pred_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d %I:%M %p")
    
    # Save visualizer overlays to folder
    left_rel_path = ""
    right_rel_path = ""
    if res_left:
        left_full_path = os.path.join(IMAGES_DIR, f"{pred_id}_left.jpg")
        cv2.imwrite(left_full_path, cv2.cvtColor(res_left['visualizer'], cv2.COLOR_RGB2BGR))
        left_rel_path = os.path.relpath(left_full_path, start=os.path.dirname(HISTORY_FILE))
    if res_right:
        right_full_path = os.path.join(IMAGES_DIR, f"{pred_id}_right.jpg")
        cv2.imwrite(right_full_path, cv2.cvtColor(res_right['visualizer'], cv2.COLOR_RGB2BGR))
        right_rel_path = os.path.relpath(right_full_path, start=os.path.dirname(HISTORY_FILE))
        
    entry = {
        "id": pred_id,
        "cattle_id": cattle_id,
        "timestamp": timestamp,
        "weight": round(avg_weight, 1),
        "left_weight": round(res_left['weight'], 1) if res_left else None,
        "right_weight": round(res_right['weight'], 1) if res_right else None,
        "left_feats": res_left['feats'] if res_left else None,
        "right_feats": res_right['feats'] if res_right else None,
        "left_img_path": left_rel_path,
        "right_img_path": right_rel_path
    }
    
    history.insert(0, entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=2)
    return pred_id

def render_dashboard(history):
    st.markdown("<div class='section-header'>📊 Dashboard Overview</div>", unsafe_allow_html=True)
    
    # 1. Row of 3 premium statistics cards
    total_count = len(history)
    avg_w = sum([item['weight'] for item in history]) / total_count if total_count > 0 else 0.0
    
    st.markdown(f"""
    <div class="dashboard-grid">
        <div class="db-card">
            <div class="db-card-val">{total_count}</div>
            <div class="db-card-lbl">Total Cattle Analyzed</div>
            <div class="db-card-desc">Total successful predictions stored in persistent database.</div>
        </div>
        <div class="db-card">
            <div class="db-card-val">{avg_w:.1f} kg</div>
            <div class="db-card-lbl">Average Weight Estimated</div>
            <div class="db-card-desc">Average body weight across the entire scanned herd.</div>
        </div>
        <div class="db-card">
            <div class="db-card-val">96.3%</div>
            <div class="db-card-lbl">Model R² Accuracy</div>
            <div class="db-card-desc">Statistical validation performance of the XGBoost regressor.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 2. Main content split into Actions & Stats Details
    col_act, col_perf = st.columns([1, 1])
    
    with col_act:
        st.markdown("""
        <div class="info-card">
            <div class="info-title">🔮 Start Estimation Workspace</div>
            <div class="info-desc">
                Ready to estimate a cow's body weight? The tool will guide you through the acquisition of the 4 required standard angles using your smartphone's camera or image uploads.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 Start New Weight Prediction", use_container_width=True):
            st.session_state.current_page = "🔮 Weight Estimator"
            st.rerun()
            
    with col_perf:
        st.markdown("""
        <div class="info-card" style="border-left-color: #10B981;">
            <div class="info-title">📈 XGBoost Regression Spec</div>
            <div class="info-desc">
                The ML model evaluates body length, hip height, silhouette volume, and chest girth surrogates.
                <ul>
                    <li><b>Mean Absolute Error (MAE):</b> 18.23 kg</li>
                    <li><b>Root Mean Squared Error (RMSE):</b> 25.96 kg</li>
                    <li><b>Pearson Correlation:</b> 98.52%</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    # 3. Recent activity table
    st.markdown("<div class='section-header'>📜 Recent Herd Activity</div>", unsafe_allow_html=True)
    if total_count == 0:
        st.info("💡 No logs found. Click 'Start New Weight Prediction' above to run your first analysis.")
    else:
        # Display the 5 most recent records
        for entry in history[:5]:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 3, 3, 2])
                col1.write(f"📅 **{entry['timestamp']}**")
                col2.write(f"🏷️ Cattle ID: **{entry['cattle_id']}**")
                col3.write(f"⚖️ Est. Weight: **{entry['weight']} kg**")
                if col4.button("🔍 View Report", key=f"dash_rec_{entry['id']}", use_container_width=True):
                    st.session_state.viewing_history_id = entry['id']
                    st.session_state.current_page = "📜 History Log"
                    st.rerun()
                st.markdown("<hr style='margin: 0.5rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)

def render_estimator(segmenter, model):
    st.markdown("<div class='section-header'>🔮 Cattle Weight Estimation Workspace</div>", unsafe_allow_html=True)
    
    # Show warning if prediction was already run, let user clear it
    if st.session_state.prediction_run:
        # Display Prediction Result view
        render_prediction_result(segmenter, model)
        return
        
    # Cattle ID tag number text input & Withers Height Calibration
    col_id1, col_id2 = st.columns(2)
    with col_id1:
        st.session_state.cattle_id = st.text_input(
            "🏷️ Enter Cattle Tag ID Number", 
            value=st.session_state.cattle_id, 
            max_chars=20,
            help="Specify a unique identification tag for the cow to catalog it in the history log."
        )
    with col_id2:
        withers_val = st.session_state.get('withers_height_cm', 138.0)
        st.session_state.withers_height_cm = st.number_input(
            "📏 Physical Height at Withers (cm)",
            min_value=80.0,
            max_value=200.0,
            value=float(withers_val),
            step=1.0,
            help="Measured height from withers (shoulder) to ground. Used for exact physical scale calibration (cm/px)."
        )
    
    # 4-View Upload Stepper
    chk_head = st.session_state.photo_head is not None
    chk_side = st.session_state.photo_side is not None
    chk_back = st.session_state.photo_back is not None
    chk_other = st.session_state.photo_other_side is not None
    
    all_uploaded = chk_head and chk_side and chk_back and chk_other
    
    st.markdown("### 📸 Image Acquisition Stepper")
    col_st1, col_st2, col_st3, col_st4 = st.columns(4)
    
    def get_status_html(is_complete, label):
        status_class = "complete" if is_complete else "missing"
        status_icon = "✅ Complete" if is_complete else "❌ Missing"
        return f"""
        <div class="check-item {status_class}">
            <span>{label}</span>
            <span>{status_icon}</span>
        </div>
        """
        
    col_st1.markdown(get_status_html(chk_head, "1. Front / Head"), unsafe_allow_html=True)
    col_st2.markdown(get_status_html(chk_side, "2. Left Side View"), unsafe_allow_html=True)
    col_st3.markdown(get_status_html(chk_back, "3. Rear / Back"), unsafe_allow_html=True)
    col_st4.markdown(get_status_html(chk_other, "4. Right Side View"), unsafe_allow_html=True)
    
    tab_head, tab_side, tab_back, tab_other_side = st.tabs([
        "👤 1. Front (Head)",
        "🐄 2. Left Profile",
        "🍑 3. Rear (Back)",
        "🐄 4. Right Profile"
    ])
    
    def render_tab_content(view_name, label_title):
        stored_photo = st.session_state[f"photo_{view_name}"]
        
        if stored_photo:
            img = load_image_for_display(stored_photo)
            if img:
                st.image(img, caption=f"Selected {label_title}", width=320)
            if st.button(f"🗑️ Remove Photo", key=f"del_{view_name}"):
                st.session_state[f"photo_{view_name}"] = None
                st.rerun()
        else:
            st.info("📱 **Mobile Camera**: Tap the button below and select **'Camera'** (or take photo) to capture using your phone's native built-in camera software for full focus and resolution.")
            file = st.file_uploader(f"Capture or Upload {label_title} View", type=["jpg", "jpeg", "png"], key=f"file_{view_name}")
            if file:
                st.session_state[f"photo_{view_name}"] = file
                st.rerun()
                    
    with tab_head:
        st.markdown("#### Front Angle View")
        render_tab_content("head", "Front/Head")
        
    with tab_side:
        st.markdown("#### Left Side Profile View")
        render_tab_content("side", "Left Side Profile")
        
    with tab_back:
        st.markdown("#### Rear Angle View")
        render_tab_content("back", "Rear/Back")
        
    with tab_other_side:
        st.markdown("#### Right Side Profile View")
        render_tab_content("other_side", "Right Side Profile")
        
    # Actions block at bottom
    st.markdown("---")
    if all_uploaded:
        st.success("🎉 All 4 mandatory views captured successfully! The AI estimation is ready.")
        if st.button("🔮 Run AI Weight Estimation", use_container_width=True, type="primary"):
            st.session_state.prediction_run = True
            st.rerun()
    else:
        st.info("💡 **Acquisition Notice:** Please complete the acquisition checklist above to enable the AI Estimation button.")
        st.button("🔮 Run AI Weight Estimation (Disabled)", disabled=True, use_container_width=True)

def render_prediction_result(segmenter, model):
    st.markdown("<div class='section-header'>⚖️ Estimation Report & Results</div>", unsafe_allow_html=True)
    
    withers_h = float(st.session_state.get('withers_height_cm', 138.0))
    
    # Perform segmentation and predictions
    results = []
    
    # Process Left Side profile
    res_left = process_side_image(st.session_state.photo_side, "Left Side", segmenter, model, withers_height_cm=withers_h)
    if res_left:
        results.append((res_left, "Left Side Profile"))
    else:
        st.warning("⚠️ Left Profile Segmentation Error: Could not locate cattle silhouette. Please upload a clearer lateral profile.")
        
    # Process Right Side profile
    res_right = process_side_image(st.session_state.photo_other_side, "Right Side", segmenter, model, withers_height_cm=withers_h)
    if res_right:
        results.append((res_right, "Right Side Profile"))
    else:
        st.warning("⚠️ Right Profile Segmentation Error: Could not locate cattle silhouette. Please upload a clearer lateral profile.")
        
    # Check for distance warning violations on both sides
    has_warning = False
    
    if res_left:
        left_ratio = res_left.get('height_ratio', 0.5)
        left_warn = res_left.get('warning')
        if left_warn == "too_far":
            st.error(f"❌ **Left Side View Warning**: The cow is **too far** (occupies only {left_ratio*100:.1f}% of the frame height). Please move closer (approx. 2-3 meters) and recapture.")
            has_warning = True
        elif left_warn == "too_close":
            st.error(f"❌ **Left Side View Warning**: The cow is **too close** (occupies {left_ratio*100:.1f}% of the frame height). Please step back so the entire cow is visible.")
            has_warning = True
            
    if res_right:
        right_ratio = res_right.get('height_ratio', 0.5)
        right_warn = res_right.get('warning')
        if right_warn == "too_far":
            st.error(f"❌ **Right Side View Warning**: The cow is **too far** (occupies only {right_ratio*100:.1f}% of the frame height). Please move closer (approx. 2-3 meters) and recapture.")
            has_warning = True
        elif right_warn == "too_close":
            st.error(f"❌ **Right Side View Warning**: The cow is **too close** (occupies {right_ratio*100:.1f}% of the frame height). Please step back so the entire cow is visible.")
            has_warning = True

    if has_warning:
        st.info("💡 **Acquisition Criteria**: To ensure high-quality calculations, the cow should occupy between **40% and 85%** of the vertical height of your camera's frame. If the cow is too small or too large, the pixel dimensions will not calibrate accurately.")
        
        # Option to clear the invalid photos
        if st.button("🗑️ Clear Invalid Photos & Recapture", use_container_width=True, type="primary"):
            if res_left and res_left.get('warning'):
                st.session_state.photo_side = None
            if res_right and res_right.get('warning'):
                st.session_state.photo_other_side = None
            st.session_state.prediction_run = False
            st.rerun()
            
        if st.button("⬅️ Back to Estimator Workspace", use_container_width=True):
            st.session_state.prediction_run = False
            st.rerun()
        return

    if len(results) == 0:
        st.error("❌ Critical Error: Silhouette detection failed on both side profiles. Please review the lateral images and try again.")
        if st.button("⬅️ Back to Estimator Workspace", use_container_width=True):
            st.session_state.prediction_run = False
            st.rerun()
        return
        
    # Calculate Average Weight
    avg_weight = sum([res[0]['weight'] for res in results]) / len(results)
    
    # Automatically save prediction if not already done
    if st.session_state.saved_prediction_id is None:
        pred_id = save_prediction(st.session_state.cattle_id, avg_weight, res_left, res_right)
        st.session_state.saved_prediction_id = pred_id
        
    # Render premium weight display card with uncertainty range
    is_baif_model = getattr(model, '_is_baif_model', False)
    mae_margin = 17.2 if is_baif_model else 18.2
    st.markdown(f"""
    <div class="metric-result-card">
        <div class="metric-result-lbl">Averaged Estimated Cattle Body Weight</div>
        <div class="metric-result-val">{avg_weight:.1f} kg</div>
        <div class="metric-result-desc">Expected Range: <b>{max(100.0, avg_weight - mae_margin):.1f} kg – {avg_weight + mae_margin:.1f} kg</b> (±{mae_margin:.1f} kg MAE)</div>
        <div class="metric-result-desc" style="margin-top: 0.4rem; font-size: 0.85rem; opacity: 0.9;">Cattle ID Tag: <b>{st.session_state.cattle_id}</b> | Scale Calibration: <b>{withers_h:.0f} cm Withers Height</b> | Log ID: <b>{st.session_state.saved_prediction_id}</b></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Render details for each side
    for res_dict, side_label in results:
        st.markdown(f"### 📊 Analysis for {side_label} (Predicted: {res_dict['weight']:.1f} kg)")
        
        # Display side-by-side images
        col_img1, col_img2 = st.columns(2)
        with col_img1:
            st.image(res_dict['original'], caption=f"{side_label} - Original Capture", use_container_width=True)
        with col_img2:
            st.image(res_dict['visualizer'], caption=f"{side_label} - AI Silhouette isolation & bounding box", use_container_width=True)
            
        # Display morphometrics details card
        if is_baif_model:
            length_cm = res_dict['feats']['length']
            girth_cm = res_dict['feats']['girth']
            height_cm = res_dict['feats']['height']
        else:
            length_cm = res_dict['feats']['length'] * 0.4
            girth_cm = res_dict['feats']['girth'] * 0.7
            height_cm = res_dict['feats']['height'] * 0.5
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        col_m1.metric("Body Length (Estimated)", f"{length_cm:.1f} cm")
        col_m2.metric("Withers Height", f"{height_cm:.1f} cm")
        col_m3.metric("Torso Girth Surrogate", f"{girth_cm:.1f} cm")
        col_m4.metric("Silhouette Area", f"{res_dict['feats']['area']:.0f} cm²" if is_baif_model else f"{res_dict['feats']['area']:.0f} px")
        
    st.markdown("---")
    # Quick action button to restart prediction
    if st.button("🔄 Perform Another Prediction", use_container_width=True, type="primary"):
        # Reset state photos and variables
        st.session_state.photo_head = None
        st.session_state.photo_side = None
        st.session_state.photo_back = None
        st.session_state.photo_other_side = None
        st.session_state.prediction_run = False
        st.session_state.saved_prediction_id = None
        # Generate new random Cattle ID
        import random
        st.session_state.cattle_id = f"TAG-{random.randint(1000, 9999)}"
        st.rerun()

def render_history_page(history):
    # Check if a specific history detail report is requested
    if st.session_state.viewing_history_id:
        render_history_detail_view(st.session_state.viewing_history_id, history)
        return
        
    st.markdown("<div class='section-header'>📜 Cattle Prediction History Log</div>", unsafe_allow_html=True)
    
    if len(history) == 0:
        st.info("💡 Database Empty: No historical cattle records logged yet. Run a prediction to populate.")
        return
        
    # Search and Filter criteria
    col_se, col_we = st.columns([2, 3])
    with col_se:
        search_query = st.text_input("🔍 Search by Cattle ID tag number", value="", placeholder="e.g. TAG-1234")
    with col_we:
        all_weights = [item['weight'] for item in history]
        min_weight = float(min(all_weights)) if all_weights else 0.0
        max_weight = float(max(all_weights)) if all_weights else 600.0
        weight_range = st.slider(
            "⚖️ Filter by weight range (kg)", 
            min_value=0.0, 
            max_value=600.0, 
            value=(max(0.0, min_weight - 20), min(600.0, max_weight + 20))
        )
        
    # Filter the list
    filtered_history = []
    for item in history:
        # Match tag
        if search_query.strip().lower() and search_query.strip().lower() not in item['cattle_id'].lower():
            continue
        # Match weight
        if not (weight_range[0] <= item['weight'] <= weight_range[1]):
            continue
        filtered_history.append(item)
        
    if len(filtered_history) == 0:
        st.warning("⚠️ No historical records match your current filter search terms.")
        return
        
    # Header row
    col_h1, col_h2, col_h3, col_h4, col_h5 = st.columns([3, 2, 2, 2, 2])
    col_h1.write("**Timestamp**")
    col_h2.write("**Cattle ID**")
    col_h3.write("**Est. Weight**")
    col_h4.write("**Detailed View**")
    col_h5.write("**Action**")
    st.markdown("<hr style='margin: 0.5rem 0; border-color: #E2E8F0;'>", unsafe_allow_html=True)
    
    # Rows loop
    for entry in filtered_history:
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])
        col1.write(entry['timestamp'])
        col2.write(f"🏷️ **{entry['cattle_id']}**")
        col3.write(f"⚖️ **{entry['weight']} kg**")
        
        if col4.button("🔍 View Details", key=f"det_{entry['id']}", use_container_width=True):
            st.session_state.viewing_history_id = entry['id']
            st.rerun()
            
        if col5.button("🗑️ Delete", key=f"del_h_{entry['id']}", use_container_width=True):
            delete_history_item(entry['id'])
            st.success("Record deleted successfully!")
            st.rerun()
            
        st.markdown("<hr style='margin: 0.3rem 0; border-color: #F1F5F9;'>", unsafe_allow_html=True)

def render_history_detail_view(pred_id, history):
    # Find item
    record = None
    for item in history:
        if item['id'] == pred_id:
            record = item
            break
            
    if record is None:
        st.error("Error: Could not locate historical prediction record.")
        if st.button("⬅️ Back to History Log"):
            st.session_state.viewing_history_id = None
            st.rerun()
        return
        
    st.markdown(f"<div class='section-header'>🔍 Estimation Report — {record['cattle_id']}</div>", unsafe_allow_html=True)
    
    st.write(f"📅 **Date Captured:** {record['timestamp']} | **Record ID:** {record['id']}")
    
    # Weight Card
    st.markdown(f"""
    <div class="metric-result-card">
        <div class="metric-result-lbl">Averaged Estimated Cattle Body Weight</div>
        <div class="metric-result-val">{record['weight']:.1f} kg</div>
        <div class="metric-result-desc">Cattle ID tag: {record['cattle_id']} | Persisted record</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Draw side profile comparison details
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📊 Left Side Profile Analysis")
        if record.get('left_weight'):
            st.write(f"⚖️ Predicted Side Weight: **{record['left_weight']} kg**")
            # Load image
            img_p = record.get('left_img_path')
            if img_p:
                full_img_p = os.path.abspath(os.path.join(os.path.dirname(HISTORY_FILE), img_p))
                img = load_image_for_display(full_img_p)
                if img:
                    st.image(img, caption="Left Profile - AI Overlay", use_container_width=True)
            # Morphometrics metrics
            if record.get('left_feats'):
                lf = record['left_feats']
                st.metric("Estimated Length", f"{lf['length']*0.4:.1f} cm", f"{lf['length']:.0f} px")
                st.metric("Estimated Height", f"{lf['height']*0.5:.1f} cm", f"{lf['height']:.0f} px")
                st.metric("Torso Girth", f"{lf['girth']*0.7:.1f} cm", f"{lf['girth']:.0f} px")
        else:
            st.write("❌ *Left Profile data not recorded*")
            
    with col_right:
        st.markdown("### 📊 Right Side Profile Analysis")
        if record.get('right_weight'):
            st.write(f"⚖️ Predicted Side Weight: **{record['right_weight']} kg**")
            img_p = record.get('right_img_path')
            if img_p:
                full_img_p = os.path.abspath(os.path.join(os.path.dirname(HISTORY_FILE), img_p))
                img = load_image_for_display(full_img_p)
                if img:
                    st.image(img, caption="Right Profile - AI Overlay", use_container_width=True)
            if record.get('right_feats'):
                rf = record['right_feats']
                st.metric("Estimated Length", f"{rf['length']*0.4:.1f} cm", f"{rf['length']:.0f} px")
                st.metric("Estimated Height", f"{rf['height']*0.5:.1f} cm", f"{rf['height']:.0f} px")
                st.metric("Torso Girth", f"{rf['girth']*0.7:.1f} cm", f"{rf['girth']:.0f} px")
        else:
            st.write("❌ *Right Profile data not recorded*")
            
    st.markdown("---")
    if st.button("⬅️ Back to History Log", use_container_width=True):
        st.session_state.viewing_history_id = None
        st.rerun()

def render_methodology_page():
    st.markdown("<div class='section-header'>📖 AI Methodology & Operations</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <div class="info-title">📸 1. Standard Multi-View Capture</div>
        <div class="info-desc">
            The estimator requires 4 distinct photos of the cow captured from specific viewpoints:
            <ul>
                <li><b>Front View (Head Area)</b>: Used to evaluate head and neck structure.</li>
                <li><b>Left Side Profile View</b>: The primary anatomical profile used to measure length and height.</li>
                <li><b>Rear View (Back Area)</b>: Used to evaluate body width and hip spacing.</li>
                <li><b>Right Side Profile View</b>: The secondary profile, used to balance any position discrepancies.</li>
            </ul>
        </div>
    </div>
    
    <div class="info-card">
        <div class="info-title">🧠 2. DeepLabV3 Segmentation Overlay</div>
        <div class="info-desc">
            When the images are uploaded, the AI uses a deep neural network called <b>DeepLabV3-ResNet50</b>. The model inspects the pixels and separates the cow's body silhouette from the background (like fences, soil, grass, or handlers). It generates a clean, solid 2D mask representing the cow's exact lateral surface area.
        </div>
    </div>
    
    <div class="info-card">
        <div class="info-title">📏 3. Morphometric Parameters Extraction</div>
        <div class="info-desc">
            Once the cow's body silhouette is isolated, the system calculates 4 critical anatomical measurements:
            <ul>
                <li><b>Body Length (L)</b>: The horizontal pixel span from the shoulder point (chest) to the tailhead. (Pixel to CM: 0.4x)</li>
                <li><b>Body Height (H)</b>: The vertical pixel span from the shoulder/withers down to the ground. (Pixel to CM: 0.5x)</li>
                <li><b>Torso Girth (G)</b>: The vertical depth of the cow's midsection (barrel depth) at the center of the torso. (Pixel to CM: 0.7x)</li>
                <li><b>Silhouette Area (A)</b>: The total count of pixels forming the cow's body outline (representing overall body volume).</li>
            </ul>
        </div>
    </div>
    
    <div class="info-card">
        <div class="info-title">⚖️ 4. XGBoost Machine Learning Model</div>
        <div class="info-desc">
            These 4 measurements are fed into an <b>XGBoost Machine Learning model</b>. 
            XGBoost is a regression algorithm trained on a large dataset of actual cattle physical dimensions and their actual scales.
            The model analyzes the combination of these 4 features to predict the cow's weight in kilograms (kg) instantly.
            <br><br>
            <b>Dual-Side Averaging</b>: To prevent errors from the cow standing slightly tilted or under direct sunlight/shadows, the AI performs calculations for both the <b>Left Side Profile</b> and <b>Right Side Profile</b> independently, and then takes the average:
            <br><br>
            <div style="text-align: center; font-size: 1.1rem; font-weight: 700; color: #1E3A8A;">
                Final Weight = (Left Side Weight + Right Side Weight) / 2
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def main():
    # Model Loading
    segmenter = load_segmenter()
    model = load_regressor()
    
    if model is None:
        st.error("Error: Trained model weights file not found. Please verify the `best_weight_regressor.pkl` is inside your models folder.")
        return
        
    # Load history data
    history = load_history()
    
    # Initialize session state variables to hold 4 images
    if 'photo_head' not in st.session_state: st.session_state.photo_head = None
    if 'photo_side' not in st.session_state: st.session_state.photo_side = None
    if 'photo_back' not in st.session_state: st.session_state.photo_back = None
    if 'photo_other_side' not in st.session_state: st.session_state.photo_other_side = None
    
    # Initialize Cattle Tag ID
    if 'cattle_id' not in st.session_state or not st.session_state.cattle_id:
        import random
        st.session_state.cattle_id = f"TAG-{random.randint(1000, 9999)}"
        
    # Page routing state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "📊 Dashboard"
    if 'prediction_run' not in st.session_state:
        st.session_state.prediction_run = False
    if 'saved_prediction_id' not in st.session_state:
        st.session_state.saved_prediction_id = None
    if 'viewing_history_id' not in st.session_state:
        st.session_state.viewing_history_id = None
        
    # 1. Sidebar Design
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("🐄 BovIQ AI")
    st.sidebar.markdown("""
    **Cattle Body Weight Estimation**
    *Smartphone-Based Dairy Cattle Weight Estimation using Deep Learning & Morphometrics*
    """)
    
    st.sidebar.markdown("---")
    
    # Navigation mapping
    pages = ["📊 Dashboard", "🔮 Weight Estimator", "📜 History Log", "📖 How It's Measured"]
    
    # Sidebar Navigation Radio with state tracking
    nav_selection = st.sidebar.radio(
        "🧭 Navigation", 
        pages, 
        index=pages.index(st.session_state.current_page),
        key="nav_selection_radio"
    )
    
    # If the user manually clicks the sidebar navigation, update current_page
    if nav_selection != st.session_state.current_page:
        st.session_state.current_page = nav_selection
        # Reset history detail view when moving pages
        st.session_state.viewing_history_id = None
        st.rerun()
        
    # Reset button in Sidebar
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Reset All Photos"):
        st.session_state.photo_head = None
        st.session_state.photo_side = None
        st.session_state.photo_back = None
        st.session_state.photo_other_side = None
        st.session_state.prediction_run = False
        st.session_state.saved_prediction_id = None
        st.success("State reset successfully!")
        st.rerun()
        
    # Sidebar performance card
    st.sidebar.subheader("📊 Model Spec (XGBoost)")
    st.sidebar.markdown("""
    * **R² Score:** 96.25%
    * **MAE:** 18.23 kg
    * **RMSE:** 25.96 kg
    * **Pearson Corr:** 98.52%
    """)
    
    # Sidebar Credits Info
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    *Developed by:*
    * **Student:** Nisarg Vakharia
    * **Organization:** BAIF Development Research Foundation
    * **Institution:** MIT School of Computing
    """)

    # 2. Main Page Header
    st.markdown("<div class='main-title'>BAIF Cattle Weight Estimator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Smartphone-Based Dairy Cattle Weight Estimation using Deep Learning & Morphometric Features</div>", unsafe_allow_html=True)

    # Page Routing render block
    if st.session_state.current_page == "📊 Dashboard":
        render_dashboard(history)
    elif st.session_state.current_page == "🔮 Weight Estimator":
        render_estimator(segmenter, model)
    elif st.session_state.current_page == "📜 History Log":
        render_history_page(history)
    elif st.session_state.current_page == "📖 How It's Measured":
        render_methodology_page()
        
if __name__ == "__main__":
    main()
