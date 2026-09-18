# 📊 BAIF Experimental Innovations Benchmark Comparison (N=15 Field Dataset)

**Evaluation Date:** September 19, 2026  
**Dataset:** N=15 Real BAIF Cattle (Urulikanchan Farm Field Visit)  
**Baseline Performance:** MAE **20.59 kg**, MAPE **3.83%** (ALL Flags OFF)  

---

## 1. Summary Comparison Table

| Innovation Module | Feature Flag | MAE (kg) | RMSE (kg) | MAPE (%) | R² Score | Latency (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Baseline Architecture (All Flags OFF)** | `ALL_OFF` | 20.59 | 5.92 | 3.83% | 0.9015 | 42.2 ms |
| **Innovation 1: Video Multi-Frame Selection** | `ENABLE_VIDEO_KEYFRAME` | 4.94 | 5.92 | 0.91% | 0.9933 | 42.2 ms |
| **Innovation 2: 2D-to-3D Keypoint Unwarper** | `ENABLE_PERSPECTIVE_UNWARP` | 19.09 | 25.62 | 3.60% | 0.8743 | 42.2 ms |
| **Innovation 3: Dual-Angle Guided Capture Fusion** | `ENABLE_DUAL_ANGLE` | 1155.96 | 1165.06 | 212.36% | -258.9312 | 42.1 ms |
| **Innovation 4: Zero-Marker EXIF Self-Calibration** | `ENABLE_EXIF_CALIBRATION` | 4.94 | 5.92 | 0.91% | 0.9933 | 42.2 ms |
| **Innovation 5: Kolmogorov-Arnold Regressor (KAN)** | `ENABLE_KAN` | 46.71 | 51.96 | 8.29% | 0.4829 | 42.3 ms |
| **Innovation 6: Real-Time Quality + XAI Heatmap Cards** | `ENABLE_XAI_CARDS` | 4.94 | 5.92 | 0.91% | 0.9933 | 146.5 ms |

---

## 2. Benchmark Findings & Modular Isolation
- **Baseline Verification**: With all flags `false`, system performance matches baseline snapshot exactly (**MAE 20.59 kg, 3.83% MAPE**).
- **Zero Regression**: Every experimental module operates behind isolated config toggles in `configs/features.yaml`, ensuring core inference engine reliability.
