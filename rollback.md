# 🔄 Rollback & Recovery Guide (`rollback.md`)
**Project:** BAIF AI-Based Cattle Weight & Morphometry Estimation  
**Version:** 3.0 (SOTA Pure Smartphone Zero-Hardware Architecture)  
**Date:** September 19, 2026  

---

## 1. Feature Flag Master Toggles (`configs/features.yaml`)

All six experimental software innovations are strictly controlled via boolean flags in `configs/features.yaml`.  
To instantly disable any innovation and revert the system to baseline performance (MAE 20.59 kg, 3.83% MAPE), set the corresponding flag to `false`:

```yaml
features:
  ENABLE_VIDEO_KEYFRAME: false      # Innovation 1: Multi-frame Video Keyframe Selection
  ENABLE_PERSPECTIVE_UNWARP: false  # Innovation 2: 2D-to-3D Keypoint Perspective Unwarper
  ENABLE_DUAL_ANGLE: false          # Innovation 3: Dual-Angle Guided Capture Fusion (Side + 45° Rear)
  ENABLE_EXIF_CALIBRATION: false    # Innovation 4: Zero-Marker EXIF Self-Calibration
  ENABLE_KAN: false                 # Innovation 5: Kolmogorov-Arnold Regressor (KAN)
  ENABLE_XAI_CARDS: false           # Innovation 6: Real-Time Quality Feedback + XAI Visual Cards
```

---

## 2. Experimental Software Innovation Overview

1. **`ENABLE_VIDEO_KEYFRAME`**: Multi-frame video keyframe selection using Laplacian variance blur detection to pick the sharpest silhouette frame.
2. **`ENABLE_PERSPECTIVE_UNWARP`**: Affine perspective unwarper that estimates camera pitch/yaw from 8 keypoint landmarks and normalizes angled photos.
3. **`ENABLE_DUAL_ANGLE`**: Fuses side profile silhouette area with 45° rear view barrel width using cross-attention and Ramanujan's 3D ellipse girth formula.
4. **`ENABLE_EXIF_CALIBRATION`**: Cross-validates EXIF focal length scale factor against withers-height scale factor, flagging discrepancies > 10%.
5. **`ENABLE_KAN`**: Pure NumPy Kolmogorov-Arnold Network B-spline univariate edge regressor for non-linear physical volumetric fitting.
6. **`ENABLE_XAI_CARDS`**: SHAP feature attribution and JET colormap visual saliency overlays mapped onto anatomical cattle regions.

---

## 3. Emergency Instant Recovery & Rollback Procedure

To revert the entire codebase to baseline performance instantly:

1. **Reset Feature Flags**:
   Set all six flags under `features:` in `configs/features.yaml` to `false`.
2. **Restart Backend API**:
   Restart the FastAPI backend server (`python backend/main.py`).
3. **Verify Baseline Regression**:
   Run `python -m pytest tests/test_perspective_unwarper.py` and `python src/evaluation/benchmark_comparison.py` to confirm predictions match baseline snapshot `data/baseline/baseline_predictions_n15.json` exactly (**MAE 20.59 kg, 3.83% MAPE**).

---

## 4. Fallback Audit Log Locations

All experimental modules write automated fallback audit logs to: `logs/experimental/`

- **Innovation 1 (Video Keyframe)**: `logs/experimental/fallback_predict_video_<timestamp>.log`
- **Innovation 2 (Perspective Unwarp)**: `logs/experimental/fallback_unwarp_<timestamp>.log`
- **Innovation 3 (Dual-Angle Fusion)**: `logs/experimental/fallback_predict_dual_angle_<timestamp>.log`
- **Innovation 4 (EXIF Calibration)**: `logs/experimental/exif_calibration.log`
- **Innovation 5 (KAN Regressor)**: `logs/experimental/fallback_predict_kan_<timestamp>.log`, `logs/experimental/fallback_predict_ensemble_<timestamp>.log`
- **Innovation 6 (XAI Cards)**: `logs/experimental/xai_explainer.log`
- **Keypoint Audits**: `logs/experimental/keypoint_audit.json`, `logs/experimental/keypoint_rotation_analysis.json`

---

## 5. Known Limitations

### Innovation 2 Keypoint Drift Limitation
- **Pre-flight Audit Finding**: The baseline heuristic landmark detector (`src/features/landmarks.py`) uses bounding-box percentage windows to find peak contours.
- **Rotation Drift Measured**:
  - At **15° camera rotation**: Withers Point A drifts by **41.38 pixels**.
  - At **30° camera rotation**: Withers Point A drifts by **78.57 pixels**.
- **Impact**: On rotated/angled images, heuristic landmarks re-anchor relative to the new expanded bounding box rather than tracking true anatomical features.
- **Production Recommendation**: Keep `ENABLE_PERSPECTIVE_UNWARP` set to `false` in production until a deep neural keypoint regressor (e.g. HRNet, YOLO-Pose) is trained post-roadmap to replace the heuristic detector.

### Innovation 3 Dual-Angle Weight Computation & Attention Uniformity
- **Weight Computation Method**: `POST /api/predict_dual_angle` computes the final weight using the 3D Schaeffer volumetric formula \(W = \frac{G_{\text{fused}}^2 \times L}{10838.0}\), where \(G_{\text{fused}}\) is calculated via Ramanujan's ellipse perimeter formula from side height \(H_{\text{side}}\) and rear barrel width \(W_{\text{barrel}}\).
- **Cross-Attention Vector Usage**: The cross-attention weight vector \(\boldsymbol{\alpha}\) is multiplied into `fused_features` and returned in the API response metadata for explainability/transparency, but the final scalar weight relies on the 3D Ramanujan volumetric girth equation.
- **Softmax Temperature & Uniform Proximity**: Uses temperature \(T = 1.0\) and scale vector \(\mathbf{s} = [200, 200, 40000]\). Calculated attention weights (e.g. \([0.332, 0.368, 0.300]\)) remain within **~5% of uniform distribution** (\(1/3 \approx 0.333\)), presenting a tuning opportunity for future temperature scaling (\(T < 1.0\)) or learned projection matrices post-roadmap.
- **Production Path**: 3D Ramanujan-Schaeffer volumetric integration is the intended production path until a paired side+rear multi-view dataset is collected for retraining XGBoost.
