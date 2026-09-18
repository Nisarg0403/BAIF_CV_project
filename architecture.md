# 🏗️ End-to-End System Architecture (`architecture.md`)
**Project:** BAIF AI-Based Cattle Weight & Morphometry Estimation System  
**Version:** 3.0 (SOTA Pure Smartphone Zero-Hardware Architecture)  
**Date:** September 18, 2026  

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
        Optimal_Frame --> API["FastAPI Endpoint /predict"]
        API --> EXIF["EXIF Metadata Extractor\n(FocalLength, SensorWidth, Resolution)"]
        API --> Preproc["Image Normalization & Resizing"]
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
        
        Yaw_Pitch --> Unwarp["Affine Perspective Unwarper\n(Restores 90° Orthogonal Profile)"]
        Unwarp --> Morph_Corr["Corrected Morphometry\n(Orthogonal Length L_cm, Area A_cm2)"]
        
        EXIF --> Scale_Engine["Self-Calibrating Scale Engine\n(cm/px = Withers_Height_cm / Mask_Height_px)"]
        Scale_Engine --> Morph_Corr
    end

    %% Multi-Model Weight Regression
    subgraph S5["🤖 Regressor Engine & Volumetric Fitting"]
        Morph_Corr --> Feature_Vector["Extracted Feature Vector:\n[L_cm, Withers_H_cm, Area_cm2, Aspect_Ratio, Scale_Factor]"]
        
        Feature_Vector --> XGB["XGBoost Regressor\n(Primary Field Model)"]
        Feature_Vector --> KAN["Kolmogorov-Arnold Network\n(KAN Volumetric Regressor: W ~ G^2 * L)"]
        Feature_Vector --> Schaeffer["Schaeffer Volumetric Formula\n(Baseline Benchmark)"]
        
        XGB --> Ensemble["Weighted Model Ensemble"]
        KAN --> Ensemble
    end

    %% Explainable AI & Client Presentation
    subgraph S6["📊 Explainable AI (XAI) & Client Dashboard"]
        Ensemble --> Weight_Est["Final Estimated Weight (kg)\n(±2.5% Target Error)"]
        Mask_Fusion --> XAI["XAI Explainability Engine\n(LIME / SHAP Region Heatmap)"]
        
        Weight_Est --> React_UI["React / Streamlit User Dashboard"]
        XAI --> React_UI
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
- **EXIF Extractor**: Parses camera focal length ($f$), sensor width, and pixel dimensions for camera calibration.

### Layer 3: Dual Segmentation & Keypoint Pipeline (`PyTorch / ONNX`)
- **DeepLabV3+-ResNet50**: Performs pixel-level semantic classification for torso silhouette extraction (MAE ±4.93 kg standalone proof).
- **YOLOv8-Seg / YOLOv11**: Computes fast bounding polygon masks for instant preview (~50ms).
- **2D Animal Keypoint Detector**: Identifies 8 key anatomical landmarks (*Withers, Hook/Hip, Pin Bone, Shoulder, Knee, Stifle, Muzzle, Hoof*).

### Layer 4: Geometry Correction & Scale Engine (`Morphometry Core`)
- **Perspective Unwarper**: Estimates 3D camera yaw/pitch angles ($\theta_{\text{yaw}}, \phi_{\text{pitch}}$) from landmark spatial ratios and applies affine unwarping to correct angled photos.
- **Self-Calibrating Scale Engine**: Converts pixel metrics to physical centimeters ($\text{cm/px} = \frac{\text{Withers Height (cm)}}{\text{Mask Height (px)}}$).

### Layer 5: Multi-Model Regression Engine (`Scikit-Learn / PyTorch KAN`)
- **Primary Model (XGBoost Regressor)**: Fitted on BAIF field data, achieving 3.83% MAPE / 20.59 kg MAE.
- **SOTA Model (Kolmogorov-Arnold Network - KAN)**: Learns B-spline univariate edge activation functions for physical volumetric equations ($W \approx \alpha \cdot G^2 \cdot L$).
- **Schaeffer Baseline**: $W = \frac{G^2 \times L}{10838}$ for comparison.

### Layer 6: Explainable AI & Client Presentation (`React + Vite / Streamlit`)
- **XAI Heatmap Component**: Renders LIME/SHAP visual overlays highlighting the ribcage, abdomen, and hindquarters.
- **Model Proofs Card**: Displays validated benchmarks (N=15 BAIF cattle) with confidence intervals.

---

## 3. Deferred Work

### Keypoint Detector Upgrade (Innovation 2 Perspective Unwarper Dependency)
- **Status**: Deferred to post-roadmap implementation.
- **Rationale**: Pre-flight audit revealed that the current heuristic landmark detector (`src/features/landmarks.py`) re-anchors search windows relative to silhouette bounding boxes, experiencing coordinate drift of **41.38 pixels at 15° rotation** and **78.57 pixels at 30° rotation**.
- **Upgrade Requirement**: To reach full intended perspective unwarping accuracy without bounding box drift, Innovation 2's affine unwarper requires a trained deep keypoint neural network (e.g. HRNet, YOLOv8-Pose, or MobileNet-Pose) trained on annotated livestock anatomical landmarks.
- **Production Status**: `ENABLE_PERSPECTIVE_UNWARP` defaults to `false` in `configs/features.yaml` until the keypoint model upgrade is completed.

