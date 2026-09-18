import os
import pytest
import yaml
import numpy as np

from src.features.exif_scale_calibrator import (
    extract_exif_metadata,
    calculate_exif_scale,
    cross_validate_exif_scale,
    EXIFScaleCalibrator
)

def test_extract_exif_metadata_fallback():
    dummy_bytes = b"not_an_image_bytes"
    meta = extract_exif_metadata(dummy_bytes)
    assert meta["has_exif"] is False
    assert meta["focal_length_mm"] is None

def test_calculate_exif_scale():
    exif_meta = {
        "focal_length_mm": 4.2,
        "focal_length_35mm": 26,
        "digital_zoom_ratio": 1.0,
        "has_exif": True
    }
    
    scale = calculate_exif_scale(exif_meta, image_height_px=1080, image_width_px=1920, estimated_distance_cm=225.0)
    assert scale > 0.0
    # Expected scale for 2.25m distance should be around ~0.2 - 0.4 cm/px
    assert 0.1 <= scale <= 0.6

def test_cross_validate_exif_scale_within_threshold():
    # 5% discrepancy -> Not flagged
    res = cross_validate_exif_scale(exif_scale_cm_px=0.315, withers_scale_cm_px=0.300, threshold_pct=10.0)
    assert res["valid"] is True
    assert res["flagged"] is False
    assert res["warning"] is None
    assert res["recommended_scale_cm_px"] == 0.300  # Preserves withers scale

def test_cross_validate_exif_scale_exceeds_threshold():
    # 25% discrepancy -> Flagged with warning
    res = cross_validate_exif_scale(exif_scale_cm_px=0.400, withers_scale_cm_px=0.300, threshold_pct=10.0)
    assert res["valid"] is True
    assert res["flagged"] is True
    assert res["warning"] is not None
    assert "exceeds 10.0%" in res["warning"]
    assert res["recommended_scale_cm_px"] == 0.300  # Does NOT silently override withers scale

def test_exif_calibrator_wrapper():
    calibrator = EXIFScaleCalibrator(estimated_distance_cm=225.0, threshold_pct=10.0)
    dummy_img_bytes = b""
    res = calibrator.calibrate(dummy_img_bytes, image_shape=(1080, 1920, 3), withers_scale_cm_px=0.300)
    assert "exif_scale_cm_px" in res
    assert "withers_scale_cm_px" in res
    assert res["withers_scale_cm_px"] == 0.300

def test_exif_flag_defaults_to_false():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
    assert os.path.exists(config_path), "configs/features.yaml must exist."
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("features", {}).get("ENABLE_EXIF_CALIBRATION") is False, \
        "ENABLE_EXIF_CALIBRATION flag must default to false."

def test_exif_calibration_integration_with_flag():
    from backend.inference import process_image_file

    img_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uploaded_images", "be39ef2e_105730429112_LL_1.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Test image not found.")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # With flag OFF (default baseline)
    res_off = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
    assert "exif_calibration" not in res_off

    # Mock flag ON
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["features"]["ENABLE_EXIF_CALIBRATION"] = True
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    try:
        res_on = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
        assert "exif_calibration" in res_on
        assert res_on["exif_calibration"]["valid"] is True
    finally:
        # Revert flag back to False (default)
        cfg["features"]["ENABLE_EXIF_CALIBRATION"] = False
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)

