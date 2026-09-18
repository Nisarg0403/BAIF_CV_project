# 🔍 Railway OOM Memory Diagnostics Report (`railway_oom_diagnostics.md`)

**Date:** September 19, 2026  
**Target Service:** BAIF FastAPI Backend (`backend/main.py`)  
**Deployment Platform:** Railway  
**Issue:** Container killed by kernel due to Out-Of-Memory (OOM) during `/api/predict` execution.  

---

## 1. Top-Level Imports & Memory Cost Audit

Below is the breakdown of top-level imports across the 9 audited backend files and their measured RSS memory footprint:

| File Path | Import Statement | Module Name | Memory Footprint (RSS) | Import Type |
| :--- | :--- | :--- | :---: | :--- |
| `backend/inference.py` | `from src.segmentation.segment import CowSegmenter` | `torch` + `torchvision` | **263.65 MB** | **Top-Level** (Eager) |
| `src/evaluation/xai_explainer.py` | `import shap` (inside function) / dependencies | `shap` | **204.18 MB** | **Lazy** (Function-level) |
| `backend/main.py`, `inference.py` | `import cv2` | `OpenCV (cv2)` | **16.21 MB** | **Top-Level** (Eager) |
| `backend/main.py` | `from fastapi import FastAPI...` | `fastapi` | **12.82 MB** | **Top-Level** (Eager) |
| `backend/inference.py` | `from src.segmentation.segment import YOLOv8Segmenter` | `ultralytics` | **1.95 MB** | **Top-Level** (Eager) |
| `backend/main.py`, `inference.py` | `import numpy as np`, `import pandas as pd` | `numpy`, `pandas` | **~8.50 MB** | **Top-Level** (Eager) |
| `src/models/kan_regressor.py` | `import numpy as np`, `import pickle` | Standard ML stdlib | **< 1.00 MB** | **Top-Level** (Eager) |
| `src/models/dual_angle_fusion.py` | `import math`, `import numpy as np` | Standard math stdlib | **< 1.00 MB** | **Top-Level** (Eager) |
| `src/features/video_keyframe_selector.py` | `import cv2`, `import numpy as np` | OpenCV / NumPy | Included in `cv2` | **Top-Level** (Eager) |
| `src/features/perspective_unwarper.py` | `import cv2`, `import numpy as np` | OpenCV / NumPy | Included in `cv2` | **Top-Level** (Eager) |
| `src/features/exif_scale_calibrator.py` | `from PIL import Image, ExifTags` | Pillow (PIL) | **< 1.00 MB** | **Top-Level** (Eager) |
| `src/features/landmarks.py` | `import cv2`, `import numpy as np` | OpenCV / NumPy | Included in `cv2` | **Top-Level** (Eager) |

### Key Import Findings:
- Eager top-level import of `torch` and `torchvision` inside `src/segmentation/segment.py` (imported at module initialization by `backend/inference.py`) immediately inflates startup RSS memory to **517.29 MB** before any HTTP request is accepted.
- `shap` consumes **204.18 MB** when loaded. It is already imported lazily in `src/evaluation/xai_explainer.py`, but experimental modules like KAN, Dual-Angle, Keyframe Selector, Unwarper, and EXIF Calibrator are imported eagerly or initialized at top-level in backend modules.

---

## 2. Model Weight File Sizes

| Model File | File Path | On-Disk Size | In-Memory Impact |
| :--- | :--- | :---: | :---: |
| **YOLOv8 Segmentation** | `yolov8n-seg.pt` | **6.74 MB** | ~25 MB RAM |
| **BAIF Multi-Model Pack** | `models/baif_multimodel_pack.pkl` | **0.31 MB** | ~2 MB RAM |
| **KAN Regressor Weights** | `models/experimental/kan_regressor.pkl` | **0.003 MB** (3 KB) | < 0.1 MB RAM |
| **DeepLabV3+ ResNet-50** | PyTorch Cache (`torchvision.models`) | **~168.00 MB** | ~170 MB RAM |

---

## 3. Empirical Process Memory Footprint Measurements

| Stage / Operation | Measured RSS Memory | Notes / Triggers |
| :--- | :---: | :--- |
| **Base Python Process** | **18.29 MB** | Bare Python runtime |
| **FastAPI Startup Memory** | **517.29 MB** | Memory footprint *before* any request is received |
| **Peak Memory during `/api/predict`** | **923.36 MB** | Single request processing 1 baseline image |
| **Prediction Memory Delta** | **+406.07 MB** | PyTorch DeepLab tensor allocations + un-cached image buffers |

---

## 4. Duplicate Model Loading & Re-instantiation Check

- **`_MULTIMODEL_PACK`**: Cached via singleton pattern in `backend/inference.py`.
- **`_DEEPLAB_SEGMENTER` / `_YOLO_SEGMENTER`**: Cached via global singleton variables in `backend/inference.py`.
- **`KANRegressor`**: `KANRegressor()` was being re-instantiated on every call (`kan_model = KANRegressor()`), which opens and unpickles `kan_regressor.pkl` from disk on every request.
- **Image Downscaling**: Images were being processed without thread/grad limits in PyTorch autograd mode, causing PyTorch to retain intermediate activation gradients during forward passes.

---

## 5. Railway Plan Memory Limits & Root Cause Diagnosis

- **Railway Hobby / Starter Plan Limit**: **512 MB RAM** (Default container kernel cap).
- **Root Cause**: The backend startup RSS (**517.29 MB**) exceeds the 512 MB limit before handling any traffic. When `/api/predict` runs, PyTorch tensor allocations drive peak memory to **923.36 MB**, causing Railway's Linux kernel OOM Killer to instantly terminate the container.

---

## 6. Actionable Fix Strategy (Phase 2 Roadmap)

1. **Lazy Imports**: Defer loading heavy modules (torch, torchvision, shap, experimental features) until their endpoints or feature flags are active.
2. **Singleton Model Caching**: Cache `KANRegressor` and model weight loaders globally.
3. **Explicit Tensor Cleanup**: Call `del` on intermediate mask arrays and invoke `gc.collect()`.
4. **PyTorch CPU & Autograd Optimization**: Set `torch.set_grad_enabled(False)`, `torch.set_num_threads(1)`, and `model.eval()`.
5. **Resolution Cap**: Cap max inference image dimensions to 1024 px.
6. **Railway Deployment Config**: Configure `Procfile` / `railway.json` / `nixpacks.toml` with `uvicorn --workers 1` and thread limits (`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`).
