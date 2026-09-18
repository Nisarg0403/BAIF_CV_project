# Frontend Audit & Endpoint Coverage Report

## 1. Frontend Stack Identification
- **Primary Hosted Stack**: React 18 + Vite SPA located in `frontend/` (`frontend/src/App.jsx`). When built (`npm run build`), the compiled bundle in `frontend/dist/` is served directly by the FastAPI backend (`backend/main.py`) at the root URL `/`.
- **Demo / Prototyping Stack**: Streamlit application located in `web/app.py` for desktop visualization and interactive demonstration of feature extraction pipeline.

## 2. Existing Frontend Files & UI Flow
- **React Frontend Files (`frontend/src/`)**:
  - `App.jsx`: Main application container, view router (Predict, History, Herd, Analytics, Settings), header, side panel, and prediction results render logic.
  - `App.css`, `index.css`: Design system styling, glassmorphism UI elements, dark/light theme variables.
  - `components/CattleCanvas.jsx`: Interactive canvas for rendering cattle silhouette contour, landmarks, and morphometric keypoints.
  - `components/ModelProofsCard.jsx`: Validation card showing benchmark metrics (DeepLabV3+ vs YOLOv8).
  - `components/LiveCameraModal.jsx`: WebRTC live camera modal for real-time acquisition.
- **Current UI Flow**:
  1. User enters Cattle Tag ID and selects base model engine (DeepLabV3+ or YOLOv8).
  2. Stepper allows acquiring 3 photos (Left Side Profile, Right Side Profile, Rear View).
  3. Clicking "Run AI Weight Estimation" calls `POST /api/predict`.
  4. Analytical result panel renders Estimated Weight (kg), confidence score meter, and morphometric measurements table (Body Length, Chest Girth, Withers Height, Stature Height).

## 3. Backend Endpoints Called by UI
- `GET /api/history`: Fetches past prediction logs.
- `POST /api/predict`: Standard single/multi-view image prediction.
- **Uncalled Innovation Endpoints**:
  - `POST /api/predict_video` (Innovation 1: Video Keyframe)
  - `POST /api/predict_dual_angle` (Innovation 3: Dual Angle Fusion)
  - `POST /api/predict_kan` (Innovation 5: KAN Regressor)
  - `POST /api/predict_ensemble` (Innovation 5: Ensemble Regressor)

## 4. Feature Flag Status
- **Current State**: Feature flags are backend-only (read from `configs/features.yaml` on server startup).
- **Frontend Awareness**: Zero feature flags are currently read, toggled, or passed as HTTP headers/form parameters by the frontend UI. All 6 innovations are currently inaccessible via the hosted UI interface.
