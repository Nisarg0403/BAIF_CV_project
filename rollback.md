# 🔄 Rollback & Recovery Guide (`rollback.md`)
**Project:** BAIF AI-Based Cattle Weight & Morphometry Estimation  
**Date:** September 19, 2026  

---

## 1. Feature Flag Master Toggles (`configs/features.yaml`)

All experimental software innovations are controlled via boolean flags in `configs/features.yaml`.  
To instantly disable any innovation and revert the system to baseline performance (MAE 20.59 kg, 3.83% MAPE), set the corresponding flag to `false`:

```yaml
features:
  ENABLE_VIDEO_KEYFRAME: false
  ENABLE_PERSPECTIVE_UNWARP: false
  ENABLE_DUAL_ANGLE: false
  ENABLE_EXIF_CALIBRATION: false
  ENABLE_KAN: false
  ENABLE_XAI_CARDS: false
```

---

## 2. Emergency Instant Recovery Procedure

1. **Global Reset to Baseline**:
   Edit `configs/features.yaml` and set all flags under `features:` to `false`.
2. **Restart API Service**:
   Restart the FastAPI backend server (`python backend/main.py`).
3. **Verify Baseline Behavior**:
   Run `python -m pytest tests/test_perspective_unwarper.py` to confirm that baseline outputs match the saved snapshot `data/baseline/baseline_predictions_n15.json` with zero error degradation.

---

## 3. Experimental Logs Location

All experimental modules write fallback audit logs to the following directory:
- Path: `logs/experimental/`
- Key Log Files:
  - `fallback_predict_video_<timestamp>.log`
  - `fallback_unwarp_<timestamp>.log`
  - `fallback_predict_kan_<timestamp>.log`
  - `fallback_predict_ensemble_<timestamp>.log`
  - `fallback_predict_dual_angle_<timestamp>.log`
  - `keypoint_audit.json`
  - `keypoint_rotation_analysis.json`

---

## 4. Known Limitations

### Innovation 2 Keypoint Drift Limitation
- **Pre-flight Audit Finding**: The baseline heuristic landmark detector (`src/features/landmarks.py`) uses bounding-box percentage windows to find peaks.
- **Rotation Drift Measured**:
  - At **15° camera rotation**: Withers Point A drifts by **41.38 pixels**.
  - At **30° camera rotation**: Withers Point A drifts by **78.57 pixels**.
- **Impact**: On rotated/angled images, heuristic landmarks re-anchor relative to the new expanded bounding box rather than tracking true anatomical features.
- **Production Recommendation**: Keep `ENABLE_PERSPECTIVE_UNWARP` set to `false` in production until a deep neural keypoint regressor (e.g. HRNet, YOLO-Pose) is trained post-roadmap to replace the heuristic detector.

### Innovation 3 Dual-Angle Weight Computation Limitation
- **Weight Computation Method**: `POST /api/predict_dual_angle` computes the final weight using the 3D Schaeffer volumetric formula \(W = \frac{G_{\text{fused}}^2 \times L}{10838.0}\), where \(G_{\text{fused}}\) is calculated via Ramanujan's ellipse perimeter formula from side height \(H_{\text{side}}\) and rear barrel width \(W_{\text{barrel}}\).
- **Cross-Attention Vector Usage**: The cross-attention weight vector \(\boldsymbol{\alpha}\) is multiplied into `fused_features` and returned in the API response metadata for explainability/transparency, but the final scalar weight relies on the 3D Ramanujan volumetric girth equation.
- **Softmax Temperature & Uniform Proximity**: Uses temperature \(T = 1.0\) and scale vector \(\mathbf{s} = [200, 200, 40000]\). Calculated attention weights (e.g. \([0.332, 0.368, 0.300]\)) remain within **~5% of uniform distribution** (\(1/3 \approx 0.333\)), presenting a tuning opportunity for future temperature scaling (\(T < 1.0\)) or learned projection matrices post-roadmap.
- **Production Path**: 3D Ramanujan-Schaeffer volumetric integration is the intended production path until a paired side+rear multi-view dataset is collected for retraining XGBoost.

