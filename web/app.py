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
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "best_weight_regressor.pkl"))
    if not os.path.exists(model_path):
        st.error(f"Model file not found at path: {model_path}")
        return None
    try:
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Failed to load pickle model: {e}")
        return None

def load_image_for_display(uploaded_file):
    """
    Safely loads an uploaded file into a PIL Image for displaying, without breaking stream state.
    """
    if uploaded_file is None:
        return None
    try:
        if hasattr(uploaded_file, "getvalue"):
            bytes_data = uploaded_file.getvalue()
        else:
            uploaded_file.seek(0)
            bytes_data = uploaded_file.read()
        import io
        return Image.open(io.BytesIO(bytes_data))
    except Exception as e:
        print(f"Error loading image for display: {e}")
        return None

def process_side_image(input_file, side_name, segmenter, model):
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
        
        # Segment
        mask = segmenter.segment(img)
        if np.sum(mask) == 0:
            return None
            
        # Extract features
        feats = extract_morphometric_features(mask)
        
        # Predict Weight
        X_input = np.array([[feats['length'], feats['height'], feats['area'], feats['girth']]])
        predicted_weight = float(model.predict(X_input)[0])
        
        # Draw visualization overlay (Green mask + red bbox)
        overlay = img.copy()
        overlay[mask == 255] = [0, 255, 0]  # Green cow silhouette
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        
        # Draw bbox
        y_indices, x_indices = np.nonzero(mask)
        xmin, xmax = np.min(x_indices), np.max(x_indices)
        ymin, ymax = np.min(y_indices), np.max(y_indices)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
        visualizer_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            'weight': predicted_weight,
            'feats': feats,
            'original': original_img_rgb,
            'visualizer': visualizer_rgb
        }
    except Exception as e:
        print(f"Error processing side image {side_name}: {e}")
        return None

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
        
    # Initialize session state variables to hold 4 images
    if 'photo_head' not in st.session_state: st.session_state.photo_head = None
    if 'photo_side' not in st.session_state: st.session_state.photo_side = None
    if 'photo_back' not in st.session_state: st.session_state.photo_back = None
    if 'photo_other_side' not in st.session_state: st.session_state.photo_other_side = None

    # Reset button in Sidebar
    if st.sidebar.button("🗑️ Reset All Photos"):
        st.session_state.photo_head = None
        st.session_state.photo_side = None
        st.session_state.photo_back = None
        st.session_state.photo_other_side = None
        st.rerun()

    # 4. Multi-View Acquisition Workspace
    st.markdown("### 📸 Multi-View Cattle Image Acquisition")
    st.write("Capture or upload the 4 required views of the cattle. Tapping 'Browse files' on mobile will open your phone's native camera.")
    
    tab_head, tab_side, tab_back, tab_other_side = st.tabs([
        "👤 1. Head Area (Front)",
        "🐄 2. Side View (Left)",
        "🍑 3. Back Area (Rear)",
        "🐄 4. Other Side (Right)"
    ])
    
    with tab_head:
        st.markdown("#### 1. Head Area (Front View)")
        up_head = st.file_uploader("Capture/Upload Front/Head View", type=["jpg", "png", "jpeg"], key="up_head")
        if up_head: st.session_state.photo_head = up_head
        
        if st.session_state.photo_head:
            st.success("✅ Front/Head view recorded.")
            img_h = load_image_for_display(st.session_state.photo_head)
            if img_h:
                st.image(img_h, width=280)
            
    with tab_side:
        st.markdown("#### 2. Side View (Left Profile)")
        up_side = st.file_uploader("Capture/Upload Left Side View", type=["jpg", "png", "jpeg"], key="up_side")
        if up_side: st.session_state.photo_side = up_side
        
        if st.session_state.photo_side:
            st.success("✅ Left Side view recorded.")
            img_s = load_image_for_display(st.session_state.photo_side)
            if img_s:
                st.image(img_s, width=280)
            
    with tab_back:
        st.markdown("#### 3. Back Area (Rear View)")
        up_back = st.file_uploader("Capture/Upload Rear/Back View", type=["jpg", "png", "jpeg"], key="up_back")
        if up_back: st.session_state.photo_back = up_back
        
        if st.session_state.photo_back:
            st.success("✅ Rear/Back view recorded.")
            img_b = load_image_for_display(st.session_state.photo_back)
            if img_b:
                st.image(img_b, width=280)
            
    with tab_other_side:
        st.markdown("#### 4. Other Side View (Right Profile)")
        up_other = st.file_uploader("Capture/Upload Right Side View", type=["jpg", "png", "jpeg"], key="up_other")
        if up_other: st.session_state.photo_other_side = up_other
        
        if st.session_state.photo_other_side:
            st.success("✅ Right Side view recorded.")
            img_os = load_image_for_display(st.session_state.photo_other_side)
            if img_os:
                st.image(img_os, width=280)

    # 5. Check if all 4 photos have been provided
    all_uploaded = (
        st.session_state.photo_head is not None and
        st.session_state.photo_side is not None and
        st.session_state.photo_back is not None and
        st.session_state.photo_other_side is not None
    )
    
    if all_uploaded:
        st.success("🎉 All 4 views captured! AI is running the weight estimation...")
        
        # Run prediction execution block
        results = []
        
        # Process Left Side profile
        res_left = process_side_image(st.session_state.photo_side, "Left Side", segmenter, model)
        if res_left:
            results.append((res_left, "Left Side View"))
        else:
            st.warning("⚠️ Could not detect/segment cattle in the Left Side photo. Please ensure it has a clear side profile.")
            
        # Process Right Side profile
        res_right = process_side_image(st.session_state.photo_other_side, "Right Side", segmenter, model)
        if res_right:
            results.append((res_right, "Right Side View"))
        else:
            st.warning("⚠️ Could not detect/segment cattle in the Right Side photo. Please ensure it has a clear side profile.")

        # Show estimation result if at least one side profile is available
        if len(results) > 0:
            st.markdown("---")
            # Average predicted weight from available side views
            avg_weight = sum([res[0]['weight'] for res in results]) / len(results)
            
            # Display Weight Result Card
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-lbl">Averaged Estimated Cattle Body Weight</div>
                <div class="metric-val">{avg_weight:.1f} kg</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Display analysis metrics and overlays for each processed side
            for res_dict, side_label in results:
                st.markdown(f"### 📊 Analysis for {side_label} (Weight: {res_dict['weight']:.1f} kg)")
                
                # Display Images Columns
                col1, col2 = st.columns(2)
                with col1:
                    st.image(res_dict['original'], caption=f"{side_label} - Original", use_container_width=True)
                with col2:
                    st.image(res_dict['visualizer'], caption=f"{side_label} - AI Segmentation Overlay", use_container_width=True)
                    
                # Display Morphometric Features Card
                length_cm = res_dict['feats']['length'] * 0.4
                girth_cm = res_dict['feats']['girth'] * 0.7
                height_cm = res_dict['feats']['height'] * 0.5
                
                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Body Length (Estimated)", f"{length_cm:.1f} cm", f"{res_dict['feats']['length']:.0f} px")
                m_col2.metric("Body Height (Estimated)", f"{height_cm:.1f} cm", f"{res_dict['feats']['height']:.0f} px")
                m_col3.metric("Torso Girth (Estimated)", f"{girth_cm:.1f} cm", f"{res_dict['feats']['girth']:.0f} px")
                m_col4.metric("Silhouette Area", f"{res_dict['feats']['area'] / 1000:.1f}k px²", None)
    else:
        st.info("💡 **Acquisition Progress:** Please capture or upload all 4 required views to enable weight estimation.")
        
        # Display completion checklist
        chk_head = "✅ Recorded" if st.session_state.photo_head else "❌ Missing"
        chk_side = "✅ Recorded" if st.session_state.photo_side else "❌ Missing"
        chk_back = "✅ Recorded" if st.session_state.photo_back else "❌ Missing"
        chk_other = "✅ Recorded" if st.session_state.photo_other_side else "❌ Missing"
        
        st.markdown(f"""
        **Cattle Capture Checklist:**
        * 👤 **Front View (Head Area)**: {chk_head}
        * 🐄 **Left Side View**: {chk_side}
        * 🍑 **Rear View (Back Area)**: {chk_back}
        * 🐄 **Right Side View**: {chk_other}
        """)

    # 6. Display 4-View Capture Gallery
    st.markdown("---")
    st.subheader("🖼️ 4-View Capture Gallery")
    g_col1, g_col2, g_col3, g_col4 = st.columns(4)
    
    with g_col1:
        st.write("**1. Head Area (Front)**")
        img_head = load_image_for_display(st.session_state.photo_head)
        if img_head:
            st.image(img_head, use_container_width=True)
        else:
            st.write("❌ *Not Captured/Uploaded*")
            
    with g_col2:
        st.write("**2. Left Side View**")
        img_side = load_image_for_display(st.session_state.photo_side)
        if img_side:
            st.image(img_side, use_container_width=True)
        else:
            st.write("❌ *Not Captured/Uploaded*")
            
    with g_col3:
        st.write("**3. Back Area (Rear)**")
        img_back = load_image_for_display(st.session_state.photo_back)
        if img_back:
            st.image(img_back, use_container_width=True)
        else:
            st.write("❌ *Not Captured/Uploaded*")
            
    with g_col4:
        st.write("**4. Right Side View**")
        img_other = load_image_for_display(st.session_state.photo_other_side)
        if img_other:
            st.image(img_other, use_container_width=True)
        else:
            st.write("❌ *Not Captured/Uploaded*")

if __name__ == "__main__":
    main()
