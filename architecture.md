# 🏗️ End-to-End System Architecture (`architecture.md`)
**Project:** BAIF AI-Based Cattle Weight & Morphometry Estimation System  
**Version:** 3.0 (SOTA Pure Smartphone Zero-Hardware Architecture)  
**Date:** September 19, 2026  

---

## 1. Complete System Architecture Diagram

```mermaid
flowchart TD
    %% Custom Camera UI Subsystem
    subgraph S1["📱 Custom Guarded Camera UI (WebRTC / HTML5)"]
        UI_CAM["WebRTC Custom Camera Stream"] --> Frame_Buffer["Multi-Frame Buffer\n(Laplacian Sharpness Filter)"]
        UI_CAM --> Light_Check["Luminance Check\n(L = 0.299R + 0.587G + 0.114B)"]
        UI_CAM --> Frame_Ratio["Occupancy Ratio Check\n(Height Ratio = Mask_H / Frame_H)"]
        UI_CAM --> Angle_Check["Orthogonality Check\n(Aspect Ratio = W / H)"]
        
        Light_Check -- "L < 45" --> Torch_Toggle["🔦 Torch Auto-Prompt / Toggle"]
        Frame_Ratio -- "Ratio < 0.40" --> HUD_Far["⚠️ TOO FAR Warning\n(Capture Disabled)"]
        Frame_Ratio -- "Ratio > 0.85" --> HUD_Close["🛑 TOO CLOSE Warning\n(Capture Disabled)"]
        Frame_Ratio -- "0.55 <= Ratio <= 0.80" --> HUD_Green["✅ OPTIMAL DISTANCE\n(Capture Enabled)"]
        Angle_Check -- "AR < 1.1" --> HUD_Angle["⚠️ ANGLED VIEW\n(Rotate to Side Profile)"]
        
        HUD_Green --> Trigger["Auto-Capture / User Shutter Tap"]
        Trigger --> Optimal_Frame["Optimal Frame Selected\n(Clean RGB Image + EXIF)"]
    end

    %% Ingestion & Preprocessing API
    subgraph S2["🚀 API Gateway & Preprocessing Layer (FastAPI Backend)"]
        Optimal_Frame --> API["FastAPI Gateway (/api/predict)"]
        
        %% Innovation 1 Branch
        API -. "ENABLE_VIDEO_KEYFRAME [Optional]" .-> Video_Keyframe["Video Keyframe Selector\n(src/features/video_keyframe_selector.py)"]
        Video_Keyframe --> Preproc["Image Normalization & Resizing"]
        API --> Preproc

        %% Innovation 4 Branch
        Preproc -. "ENABLE_EXIF_CALIBRATION [Optional]" .-> EXIF_Calib["Zero-Marker EXIF Self-Calibrator\n(src/features/exif_scale_calibrator.py)"]
        EXIF_Calib --> Scale_Validation["Focal Scale Cross-Validation\n(Flags >10% Discrepancy)"]
    end

    %% AI Segmentation & Keypoint Engine
    subgraph S3["🧠 Dual Segmentation & Pose Estimation Pipeline"]
        Preproc --> DeepLab["DeepLabV3+-ResNet50\n(Pixel Semantic Segmentation)"]
        Preproc --> YOLO_Seg["YOLOv8-Seg / YOLOv11\n(Fast Instance Masking)"]
        Preproc --> Keypoint_Net["2D Keypoint Pose Estimator\n(8 Anatomical Landmarks)"]
        
        DeepLab --> Mask_Fusion["Mask Fusion & Contour Cleaning\n(Torso Silhouette Extraction)"]
        YOLO_Seg --> Mask_Fusion
        
        Keypoint_Net --> Land_Pts["Landmarks:\nWithers, Hook, Pin, Shoulder,\nKnee, Stifle, Muzzle, Hoof"]
    end

    %% Geometric Geometry & Scale Normalization
    subgraph S4["📐 Geometry & Perspective Normalization Engine"]
        Mask_Fusion --> Morph_Ext["2D Morphometry Extractor\n(Pixel Length, Withers Height, Silhouette Area)"]
        Land_Pts --> Yaw_Pitch["3D Camera Angle Estimator\n(Yaw theta, Pitch phi)"]
        
        %% Innovation 2 Branch
        Yaw_Pitch -. "ENABLE_PERSPECTIVE_UNWARP [Optional/Deferred]" .-> Unwarp["Affine Perspective Unwarper\n(src/features/perspective_unwarper.py)"]
        Unwarp --> Morph_Corr["Corrected Morphometry\n(Orthogonal Length L_cm, Area A_cm2)"]
        Morph_Ext --> Morph_Corr
        
        Scale_Validation --> Morph_Corr
    end

    %% Multi-Model Weight Regression
    subgraph S5["🤖 Regressor Engine & Volumetric Fitting"]
        Morph_Corr --> Feature_Vector["Extracted Feature Vector:\n[L_cm, Withers_H_cm, Area_cm2, Aspect_Ratio, Scale_Factor]"]
        
        Feature_Vector --> XGB["XGBoost Regressor\n(Primary Field Model: 20.59 kg MAE)"]
        
        %% Innovation 5 Branch
        Feature_Vector -. "ENABLE_KAN [Optional]" .-> KAN["Kolmogorov-Arnold Network\n(src/models/kan_regressor.py)"]
        
        %% Innovation 3 Branch
        Feature_Vector -. "ENABLE_DUAL_ANGLE [Optional]" .-> Dual_Angle["Dual-Angle Cross-Attention Fusion\n(src/models/dual_angle_fusion.py)"]
        Dual_Angle --> Ramanujan["3D Ellipsoid Ramanujan Girth G_fused"]
        Ramanujan --> Schaeffer["Schaeffer Volumetric Formula: W = G^2 * L / 10838"]

        XGB --> Ensemble["Weighted Model Ensemble / Final Prediction"]
        KAN --> Ensemble
        Schaeffer --> Ensemble
    end

    %% Explainable AI & Client Presentation
    subgraph S6["📊 Explainable AI (XAI) & Client Dashboard"]
        Ensemble --> Weight_Est["Final Estimated Weight (kg)\n(±2.5% Target Error)"]
        
        %% Innovation 6 Branch
        Ensemble -. "ENABLE_XAI_CARDS [Optional]" .-> XAI["XAI Explainability Engine\n(src/evaluation/xai_explainer.py)"]
        XAI --> Heatmap_Overlay["JET Visual Saliency Overlay (xai_heatmap_b64)\n(Ribcage, Abdomen, Withers, Rump)"]
        
        Weight_Est --> React_UI["React / Streamlit User Dashboard"]
        Heatmap_Overlay --> React_UI
        Morph_Corr --> React_UI
        
        React_UI --> Card1["Weight & Error Margin Card\n(Predicted: 584 kg ±15 kg)"]
        React_UI --> Card2["Morphometric Breakdown\n(Length, Height, Girth, Area)"]
        React_UI --> Card3["SHAP Feature Importance Overlay\n(Ribcage, Abdomen, Hindquarters)"]
        React_UI --> Card4["Validation Proof Card\n(N=15 BAIF Field Proven)"]
    end
```

---

## 2. Component Layer Breakdown

### Layer 1: Guarded Custom Camera UI (`Frontend / WebRTC`)
- **Real-Time Viewfinder Feed**: HTML5 `navigator.mediaDevices.getUserMedia` with video stream canvas mirroring.
- **Pre-Capture Quality Guards**:
  1. **Distance Guard**: Calculates real-time bounding box height ratio ($\text{Height Ratio} = \frac{\text{Mask Height}}{\text{Frame Height}}$). Enables capture only within $[0.55 - 0.80]$ range (~2.0m – 2.5m distance).
  2. **Luminance & Torch Control**: Calculates canvas mean pixel brightness ($L$). Triggers automatic torch toggle prompt via `MediaStreamTrack.applyConstraints({ torch: true })` if $L < 45$.
  3. **Aspect Ratio Angle Guard**: Checks $\text{Aspect Ratio} = \frac{\text{Width}}{\text{Height}}$. Flags warning if animal is standing angled head-on/tail-on ($AR < 1.1$).
  4. **Multi-Frame Sharpness Buffer**: Stores 5 continuous frames in a circular canvas array and uses Laplacian variance blur detection to pick the sharpest frame.

### Layer 2: API Gateway & Data Ingestion (`FastAPI Backend`)
- **Endpoint `/api/v1/predict`**: Receives RGB image binary payload + EXIF headers.
- **Innovation 1 (`ENABLE_VIDEO_KEYFRAME`)**: `POST /api/predict_video` extracts optimal frame from video stream.
- **Innovation 4 (`ENABLE_EXIF_CALIBRATION`)**: Zero-marker EXIF self-calibrator parses camera focal length ($f_{\text{mm}}$) and sensor dimensions, cross-validating against withers scale factor with a $10\%$ discrepancy threshold.

### Layer 3: Dual Segmentation & Keypoint Pipeline (`PyTorch / ONNX`)
- **DeepLabV3+-ResNet50**: Performs pixel-level semantic classification for torso silhouette extraction (MAE ±4.93 kg standalone proof).
- **YOLOv8-Seg / YOLOv11**: Computes fast bounding polygon masks for instant preview (~50ms).
- **2D Animal Keypoint Detector**: Identifies 8 key anatomical landmarks (*Withers, Hook/Hip, Pin Bone, Shoulder, Knee, Stifle, Muzzle, Hoof*).

### Layer 4: Geometry Correction & Scale Engine (`Morphometry Core`)
- **Innovation 2 (`ENABLE_PERSPECTIVE_UNWARP`)**: Affine perspective unwarper estimates 3D camera yaw/pitch angles ($\theta_{\text{yaw}}, \phi_{\text{pitch}}$) from landmark ratios and restores 90° orthogonal profile.
- **Self-Calibrating Scale Engine**: Converts pixel metrics to physical centimeters ($\text{cm/px} = \frac{\text{Withers Height (cm)}}{\text{Mask Height (px)}}$).

### Layer 5: Multi-Model Regression Engine (`Scikit-Learn / PyTorch KAN`)
- **Primary Model (XGBoost Regressor)**: Fitted on BAIF field data, achieving 3.83% MAPE / 20.59 kg MAE.
- **Innovation 3 (`ENABLE_DUAL_ANGLE`)**: `POST /api/predict_dual_angle` fuses side profile silhouette with 45° rear view barrel width via cross-attention and 3D Ramanujan-Schaeffer volumetric fitting ($W = \frac{G_{\text{fused}}^2 \times L}{10838.0}$).
- **Innovation 5 (`ENABLE_KAN`)**: Pure NumPy Kolmogorov-Arnold Network B-spline univariate edge activation regressor for physical volumetric equations.

### Layer 6: Explainable AI & Client Presentation (`React + Vite / Streamlit`)
- **Innovation 6 (`ENABLE_XAI_CARDS`)**: SHAP `TreeExplainer` feature attributions mapped onto visual saliency heatmaps over anatomical regions (Ribcage, Abdomen, Withers, Rump), returned as Base64 JPEG (`xai_heatmap_b64`).

---

## 3. Deferred Work

### 1. Keypoint Detector Upgrade (Innovation 2 Perspective Unwarper Dependency)
- **Status**: Deferred to post-roadmap implementation.
- **Rationale**: Pre-flight audit revealed that the current heuristic landmark detector (`src/features/landmarks.py`) re-anchors search windows relative to silhouette bounding boxes, experiencing coordinate drift of **41.38 pixels at 15° rotation** and **78.57 pixels at 30° rotation**.
- **Upgrade Requirement**: Requires a trained deep keypoint neural network (e.g. HRNet, YOLOv8-Pose, or MobileNet-Pose) trained on annotated livestock anatomical landmarks.
- **Production Status**: `ENABLE_PERSPECTIVE_UNWARP` defaults to `false` in `configs/features.yaml` until the keypoint model upgrade is completed.

### 2. Dual-Angle XGBoost Retraining (Innovation 3 Multi-View Dependency)
- **Status**: Deferred to post-roadmap dataset expansion.
- **Rationale**: Retraining an XGBoost model specifically on dual-angle paired features requires a multi-view paired side+rear livestock dataset ($N > 100$).
- **Current Production Path**: 3D Ramanujan-Schaeffer volumetric integration is the active baseline production path for `POST /api/predict_dual_angle`.
