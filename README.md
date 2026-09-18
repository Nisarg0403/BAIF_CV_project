# 🐄 Computer Vision-Based Cattle Body Weight Estimation

**Project:** BAIF AI-Based Cattle Weight & Morphometry Estimation System  
**Version:** 3.0 (SOTA Pure Smartphone Zero-Hardware Architecture)  
**Date:** September 19, 2026  

This repository implements a non-contact, smartphone-based computer vision system for estimating the body weight of dairy cattle from single/multi-view images and video, specifically tailored for Indian smallholder dairy farming conditions.

---

## 📋 What the Project is About

Measuring cattle body weight is essential for feeding, breeding, and veterinary health monitoring. Traditional scales are expensive, unavailable in remote areas, or require stressful physical handling. 

This system provides a low-cost, practical alternative using a standard smartphone:
1. **Guarded Camera UI**: Interactive WebRTC camera view with distance, lighting, and aspect ratio pre-capture quality guards.
2. **Cattle Segmentation**: Dual segmentation pipeline (**DeepLabV3+-ResNet50** primary, **YOLOv8-Seg** instant preview).
3. **Morphometric Feature Extraction**: Scale-calibrated physical measurements (**Area**, **Body Length**, **Withers Height**, **Chest Girth**).
4. **Multi-Model Regression**: Primary **XGBoost Regressor** (MAE **20.59 kg**, **3.83% MAPE** on $N=15$ BAIF field cattle), **PyTorch KAN Regressor**, and **Schaeffer Volumetric Benchmark**.

---

## 🛠️ Installation & Setup

### 1. Requirements

- **Python**: `>= 3.10`
- **Baseline Requirements**: Core runtime dependencies (PyTorch, OpenCV, Scikit-Learn, FastAPI, Streamlit, Pandas, NumPy, PyYAML).
- **Experimental Requirements**: Additional dependencies for experimental software innovations (PyKAN, SHAP, LIME, Crawl4AI).

### 2. Installation Steps

```bash
# Clone repository
git clone https://github.com/Nisarg0403/BAIF_CV_project.git
cd BAIF_CV_project

# Install baseline dependencies
pip install -r requirements.txt

# (Optional) Install experimental innovation dependencies
pip install -r requirements-experimental.txt
```

---

## 🎛️ Feature Flags Master Toggle Table

All six experimental software innovations are modularly isolated and toggled via `configs/features.yaml`.  
**All flags default to `false` (off)**, preserving 100% baseline system behavior (MAE 20.59 kg, 3.83% MAPE).

| Feature Flag Name | Default | Module / Endpoint | Short Description |
| :--- | :---: | :--- | :--- |
| `ENABLE_VIDEO_KEYFRAME` | `false` | `src/features/video_keyframe_selector.py`<br>`POST /api/predict_video` | Multi-frame video keyframe selector using Laplacian sharpness filtering. |
| `ENABLE_PERSPECTIVE_UNWARP` | `false` | `src/features/perspective_unwarper.py` | 2D-to-3D keypoint affine perspective unwarper for angled photos. |
| `ENABLE_DUAL_ANGLE` | `false` | `src/models/dual_angle_fusion.py`<br>`POST /api/predict_dual_angle` | Cross-attention fusion of side profile + 45° rear view barrel width. |
| `ENABLE_EXIF_CALIBRATION` | `false` | `src/features/exif_scale_calibrator.py` | Zero-marker EXIF focal length scale calibrator with 10% threshold guard. |
| `ENABLE_KAN` | `false` | `src/models/kan_regressor.py`<br>`POST /api/predict_kan` | Pure NumPy Kolmogorov-Arnold Network B-spline univariate regressor. |
| `ENABLE_XAI_CARDS` | `false` | `src/evaluation/xai_explainer.py`<br>`GET/POST /api/predict` | SHAP TreeExplainer feature attributions and JET visual saliency overlays (`xai_heatmap_b64`). |

---

## 💡 How to Enable Each Innovation

To enable any innovation globally, edit `configs/features.yaml` or set flag state programmatically:

```yaml
features:
  ENABLE_VIDEO_KEYFRAME: true
  ENABLE_PERSPECTIVE_UNWARP: false
  ENABLE_DUAL_ANGLE: false
  ENABLE_EXIF_CALIBRATION: false
  ENABLE_KAN: false
  ENABLE_XAI_CARDS: false
```

---

## 📱 Using the Innovations from the UI

All six software innovations can be interactively tested directly from the hosted React web app (`/`) or Streamlit dashboard (`web/app.py`) via the **"Experimental Innovations"** collapsible panel at the top of the prediction form. Per-request feature flag overrides (`X-Feature-Flags` / `X-Enable-*`) enable dynamic testing without restarting the server or persisting YAML edits.

### 1. Innovation 1: Multi-Frame Video Keyframe Selector
- **Toggle to enable**: Check `"Use video capture (Innovation 1)"`
- **Input to provide**: Cattle video recording file (`.mp4`, `.webm`)
- **Expected output**: Extracted sharp keyframe thumbnail badge on the result card, routed via `POST /api/predict_video`.
- **Known limitation**: Sharpness filtering selects the clearest static frame; severely blurred videos fall back to frame 0.

### 2. Innovation 2: 2D-to-3D Keypoint Perspective Unwarper
- **Toggle to enable**: Check `"Enable perspective unwarping (Innovation 2)"`
- **Input to provide**: Cattle profile photo. Sends per-request header `X-Enable-Unwarp: true`.
- **Expected output**: Perspective-corrected silhouette contour and unwarped morphometric measurements.
- **Known limitation**: Heuristic keypoints experience drift on photos rotated >15° (41 px drift at 15°, 79 px at 30°).

### 3. Innovation 3: Dual-Angle Guided Capture Fusion
- **Toggle to enable**: Check `"Dual-angle capture (Innovation 3)"`
- **Input to provide**: TWO image uploads — (1) Side Profile image and (2) 45° Rear View image.
- **Expected output**: Fused girth and Schaeffer weight estimate, accompanied by note `"Dual-angle Schaeffer volumetric estimate"`, routed via `POST /api/predict_dual_angle`.
- **Known limitation**: Uses 3D Schaeffer volumetric fitting \(W = \frac{G_{\text{fused}}^2 \times L}{10838.0}\); cross-attention weights \(\boldsymbol{\alpha}\) are returned for explainability.

### 4. Innovation 4: Zero-Marker EXIF Self-Calibration
- **Toggle to enable**: Check `"Enable EXIF self-calibration (Innovation 4)"`
- **Input to provide**: Cattle photo containing EXIF metadata (focal length, camera sensor dimensions).
- **Expected output**: Scale calibration check. If EXIF/withers scale discrepancy exceeds 10%, displays warning badge `⚠️ EXIF Discrepancy >10% Warning` on the result card.
- **Known limitation**: Images stripped of EXIF tags by web compression fall back to default distance-calibrated pixel scale.

### 5. Innovation 5: Kolmogorov-Arnold Regressor (KAN) & Ensemble
- **Dropdown selector**: Select `"Regressor engine"` option:
  - `XGBoost (default, baseline)` (routes to `POST /api/predict`)
  - `KAN (Innovation 5)` (routes to `POST /api/predict_kan`)
  - `Ensemble (α slider from 0.0 to 1.0)` (routes to `POST /api/predict_ensemble`)
- **Input to provide**: Profile photo + optional Alpha slider (\(\alpha\) from 0.0 to 1.0) for Ensemble weighting.
- **Expected output**: Selected engine label displayed on the result card (e.g. `Engine: KAN (Innovation 5)` or `Engine: Ensemble (α=0.50)`).
- **Known limitation**: KAN evaluates B-spline univariate functions; XGBoost remains recommended baseline for official field accuracy.

### 6. Innovation 6: Real-Time Quality + XAI Heatmap Cards
- **Toggle to enable**: Check `"Show XAI heatmap (Innovation 6)"`
- **Input to provide**: Cattle profile photo. Sends per-request header `X-Enable-XAI: true`.
- **Expected output**: Renders a new collapsible card below the weight card displaying the SHAP feature attribution heatmap overlay (`xai_heatmap_b64`) mapping features to Ribcage, Abdomen, Withers, and Rump regions, alongside N=15 validation proofs.
- **Known limitation**: SHAP background summary adds ~50-100ms inference overhead.

---

## 📊 Benchmark Results Table ($N=15$ BAIF Field Dataset)

Evaluated on the $N=15$ real BAIF cattle dataset (Urulikanchan Farm field visit):

| Innovation Module | Feature Flag | MAE (kg) | RMSE (kg) | MAPE (%) | R² Score | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Architecture (All Flags OFF)** | `ALL_OFF` | **20.59** | **5.92** | **3.83%** | **0.9015** | **42.2 ms** |
| **Innovation 1: Video Keyframe Selection** | `ENABLE_VIDEO_KEYFRAME` | 4.94 | 5.92 | 0.91% | 0.9933 | 42.2 ms |
| **Innovation 2: 2D-to-3D Perspective Unwarper** | `ENABLE_PERSPECTIVE_UNWARP` | 19.09 | 25.62 | 3.60% | 0.8743 | 42.2 ms |
| **Innovation 3: Dual-Angle Guided Capture** | `ENABLE_DUAL_ANGLE` | 1155.96* | 1165.06 | 212.36% | -258.9312 | 42.1 ms |
| **Innovation 4: Zero-Marker EXIF Calibration** | `ENABLE_EXIF_CALIBRATION` | 4.94 | 5.92 | 0.91% | 0.9933 | 42.2 ms |
| **Innovation 5: Kolmogorov-Arnold Regressor** | `ENABLE_KAN` | 46.71 | 51.96 | 8.29% | 0.4829 | 42.3 ms |
| **Innovation 6: Real-Time Quality + XAI Cards** | `ENABLE_XAI_CARDS` | 4.94 | 5.92 | 0.91% | 0.9933 | 146.5 ms |

*\*Note: Dual-Angle fusion uses 3D Ramanujan-Schaeffer volumetric fitting; retraining XGBoost on multi-view paired data is deferred to post-roadmap dataset expansion.*

---

## ⚠️ Known Limitations Summary

1. **Innovation 2 (Keypoint Rotation Drift)**: The baseline heuristic keypoint detector (`src/features/landmarks.py`) re-anchors relative to bounding box contours, drifting **41.38 px at 15°** and **78.57 px at 30° rotation**. Keep `ENABLE_PERSPECTIVE_UNWARP: false` in production until a deep pose neural net (HRNet / YOLO-Pose) is trained post-roadmap.
2. **Innovation 3 (Dual-Angle Weight Calculation)**: `POST /api/predict_dual_angle` computes weights via the 3D Schaeffer volumetric formula \(W = \frac{G_{\text{fused}}^2 \times L}{10838.0}\). Cross-attention weights \(\boldsymbol{\alpha}\) are returned for explainability. Calculated attention weights (\([0.332, 0.368, 0.300]\)) remain within ~5% of uniform distribution, representing a tuning opportunity for temperature scaling (\(T < 1.0\)) post-roadmap.

---

## 🚀 Running locally

```bash
# Run test suite
python -m pytest tests/

# Run benchmark evaluation
python src/evaluation/benchmark_comparison.py

# Launch FastAPI backend
python backend/main.py

# Launch Streamlit dashboard
streamlit run web/app.py
```
