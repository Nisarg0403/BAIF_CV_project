import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import pickle
import numpy as np
import streamlit as st
from PIL import Image

from src.segmentation.segment import CowSegmenter
from src.features.morphometry import extract_morphometric_features

# Set page config for a premium wide-layout dashboard
st.set_page_config(
    page_title="BAIF Cattle Weight Estimator",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1E3A8A;
        font-size: 2.8rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #4B5563;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 12px;
        padding: 1.5rem;
        border-left: 6px solid #10B981;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    .metric-val {
        font-size: 3rem;
        font-weight: 800;
        color: #10B981;
        text-align: center;
    }
    .metric-lbl {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Cache resources to keep app fast
@st.cache_resource
def load_segmenter():
    return CowSegmenter()

@st.cache_resource
def load_regressor():
    model_path = r"c:\Users\NISARG\BAIF\models\best_weight_regressor.pkl"
    if not os.path.exists(model_path):
        return None
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def main():
    # 1. Sidebar Design
    st.sidebar.markdown("<br>", unsafe_allow_html=True)
    st.sidebar.title("🐄 Project Info")
    st.sidebar.markdown("""
    **Cattle Body Weight Estimation Dashboard**
    
    *Developed by:*
    * **Student:** Nisarg Vakharia
    * **Organization:** BAIF Development Research Foundation
    * **Institution:** MIT School of Computing
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Model Performance (XGBoost)")
    st.sidebar.markdown("""
    * **MAE:** 18.23 kg
    * **RMSE:** 25.96 kg
    * **$R^2$ Score:** 96.25%
    * **Pearson Correlation:** 98.52%
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("📸 Image Acquisition Protocol")
    st.sidebar.markdown("""
    Ensure your cow image follows the standardized capture geometry:
    1. **Lateral (side) view** of the cow.
    2. Cow must be standing naturally.
    3. The entire body must be visible (no cut-off head/legs).
    4. Perpendicular camera angle (no extreme tilt).
    """)

    # 2. Main Page Header
    st.markdown("<div class='main-title'>BAIF Cattle Weight Estimator</div>", unsafe_allow_html=True)
    st.markdown("<div class='sub-title'>Smartphone-Based Dairy Cattle Weight Estimation using Deep Learning & Morphometric Features</div>", unsafe_allow_html=True)

    # 3. Model Loading
    segmenter = load_segmenter()
    model = load_regressor()
    
    if model is None:
        st.error("Error: Trained model weights file not found. Please run `python -m src.models.train` in the workspace first.")
        return

    # 4. File Uploader & Camera Tabs
    tab1, tab2 = st.tabs(["📁 Upload Image", "📸 Take Live Photo"])
    
    input_file = None
    with tab1:
        uploaded_file = st.file_uploader("Upload an image of a dairy cow...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            input_file = uploaded_file
            
    with tab2:
        st.info("💡 **Camera Capture Protocol Checklist:**\n"
                "* **Perpendicular View**: Stand perpendicular to the side of the cow.\n"
                "* **Standardized Distance**: Position yourself at the standard distance (e.g. 2.5m).\n"
                "* **Full Body Visible**: Head, legs, and tail must all be inside the camera frame.\n"
                "* **Standing Posture**: Cow must be standing naturally, not sitting or bending.")
        
        # HTML/CSS Visual Alignment Legend
        st.markdown("""
        <div style="border: 2px dashed #3B82F6; border-radius: 8px; padding: 12px; background-color: #EFF6FF; margin-bottom: 15px; font-family: 'Inter', sans-serif;">
            <h4 style="color: #1D4ED8; margin-top: 0; margin-bottom: 5px; font-size: 1rem; font-weight: 700;">📸 Live Camera Alignment Guide</h4>
            <p style="font-size: 0.85rem; color: #1E3A8A; margin-bottom: 10px;">Position the cow inside the camera preview to line up with these zones:</p>
            <div style="display: flex; justify-content: space-between; text-align: center; font-weight: 800; font-family: monospace; font-size: 0.95rem; color: #1E3A8A; background-color: #DBEAFE; padding: 10px; border-radius: 6px; border: 1px solid #BFDBFE;">
                <div style="flex: 1; border-right: 1px solid #93C5FD;">[ ZONE 1: HEAD ]</div>
                <div style="flex: 1.5; border-right: 1px solid #93C5FD; color: #047857; background-color: #D1FAE5; margin: 0 4px; border-radius: 3px;">[ ZONE 2: TORSO/GIRTH ]</div>
                <div style="flex: 1;">[ ZONE 3: RUMP ]</div>
            </div>
            <p style="font-size: 0.8rem; color: #4B5563; margin-top: 8px; margin-bottom: 0; text-align: center; font-weight: 500;">
                ↔ <i>Adjust distance so the complete cow spans exactly across Zone 1 to Zone 3</i> ↔
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        camera_file = st.camera_input("Snap a picture of the cow...")
        if camera_file is not None:
            input_file = camera_file
            
    if input_file is not None:
        # Load and decode image
        file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        original_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Show loading spinner
        with st.spinner("AI Processing: Segmenting cattle & estimating weight..."):
            # A. Segment
            mask = segmenter.segment(img)
            
            # Check if mask is empty
            if np.sum(mask) == 0:
                st.warning("⚠️ Could not detect/segment any cattle in the uploaded image. Please ensure the cow occupies a clear portion of the frame.")
                return
                
            # B. Extract features
            feats = extract_morphometric_features(mask)
            
            # C. Predict Weight
            X_input = np.array([[feats['length'], feats['height'], feats['area'], feats['girth']]])
            predicted_weight = float(model.predict(X_input)[0])
            
            # D. Calibrate pixels to cm for display
            length_cm = feats['length'] * 0.4
            girth_cm = feats['girth'] * 0.7
            height_cm = feats['height'] * 0.5
            
            # E. Draw visualization overlay (Green mask + red bbox)
            overlay = img.copy()
            overlay[mask == 255] = [0, 255, 0]  # Green cow silhouette
            cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
            
            # Draw bbox
            y_indices, x_indices = np.nonzero(mask)
            xmin, xmax = np.min(x_indices), np.max(x_indices)
            ymin, ymax = np.min(y_indices), np.max(y_indices)
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
            
            # Convert visualizer to RGB
            visualizer_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 5. Display Weight Result
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-lbl">Estimated Cattle Body Weight</div>
            <div class="metric-val">{predicted_weight:.1f} kg</div>
        </div>
        """, unsafe_allow_html=True)
        
        # 6. Display Images Columns
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("📷 Original Image")
            st.image(original_img_rgb, use_container_width=True)
        with col2:
            st.subheader("🤖 AI Segmentation & Detection Overlay")
            st.image(visualizer_rgb, use_container_width=True)
            
        # 7. Display Morphometric Features Card
        st.markdown("---")
        st.subheader("📏 Extracted Morphometric Features")
        
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Body Length (Estimated)", f"{length_cm:.1f} cm", f"{feats['length']:.0f} px")
        m_col2.metric("Body Height (Estimated)", f"{height_cm:.1f} cm", f"{feats['height']:.0f} px")
        m_col3.metric("Torso Girth (Estimated)", f"{girth_cm:.1f} cm", f"{feats['girth']:.0f} px")
        m_col4.metric("Silhouette Area", f"{feats['area'] / 1000:.1f}k px²", None)

if __name__ == "__main__":
    main()
