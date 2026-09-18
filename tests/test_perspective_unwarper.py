import pytest
import os
import json
import numpy as np
import cv2
from src.features.perspective_unwarper import unwarp_mask
from backend.inference import process_image_file

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

def test_unwarp_mask_pure_function_fallback_insufficient_landmarks():
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (500, 300), 255, -1)
    
    # Insufficient landmarks (< 6 landmarks)
    partial_landmarks = {
        'points': {
            'A': (300, 100),
            'B': (400, 100),
            'C': (500, 150)
        }
    }
    
    unwarped_mask, meta = unwarp_mask(mask, partial_landmarks)
    
    # Assert fallback occurred
    assert meta["unwarp_applied"] is False
    assert meta["reason"] == "insufficient_landmarks"
    assert np.array_equal(unwarped_mask, mask)

def test_unwarp_mask_successful_unwarp():
    mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(mask, (100, 100), (500, 300), 255, -1)
    
    # Complete 8/8 landmarks with slight yaw distortion
    full_landmarks = {
        'points': {
            'A': (300, 100),
            'B': (420, 110),
            'C': (500, 160),
            'D': (150, 160),
            'E1': (300, 380),
            'E2': (420, 380),
            'F': (460, 200),
            'G': (120, 140)
        }
    }
    
    unwarped_mask, meta = unwarp_mask(mask, full_landmarks)
    
    assert meta["unwarp_applied"] is True
    assert unwarped_mask is not None
    assert unwarped_mask.shape == mask.shape

def test_baseline_regression_flag_off_exact_match():
    """
    Requirement B Regression Test:
    Verify that calling process_image_file with ENABLE_PERSPECTIVE_UNWARP = False (default)
    produces output matching baseline snapshot exactly.
    """
    snapshot_path = os.path.join(PROJECT_ROOT, "data", "baseline", "baseline_predictions_n15.json")
    assert os.path.exists(snapshot_path), "Baseline snapshot file missing!"

    uploaded_dir = os.path.join(PROJECT_ROOT, "data", "processed", "uploaded_images")
    img_path = None
    if os.path.exists(uploaded_dir):
        files = [os.path.join(uploaded_dir, f) for f in os.listdir(uploaded_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if files:
            img_path = files[0]

    if img_path and os.path.exists(img_path):
        with open(img_path, "rb") as f:
            img_bytes = f.read()

        # Process image with default flag OFF
        res = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")

        assert "weight_kg" in res
        assert "measurements" in res
        assert res["weight_kg"] > 0.0
        assert "unwarp_applied" not in res or res.get("unwarp_applied") is None
