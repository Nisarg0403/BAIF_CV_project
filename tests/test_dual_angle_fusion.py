import os
import json
import pytest
import yaml
import numpy as np

from src.models.dual_angle_fusion import (
    compute_cross_attention_weights,
    fuse_dual_angle_morphometrics,
    DualAngleFusionModule
)

def test_cross_attention_weights():
    f_side = np.array([140.0, 125.0, 10500.0])
    f_rear = np.array([65.0, 125.0, 4875.0])
    
    weights = compute_cross_attention_weights(f_side, f_rear)
    assert weights.shape == (1, 3)
    assert np.isclose(np.sum(weights), 1.0)
    assert np.all(weights >= 0)

def test_fuse_dual_angle_morphometrics():
    side_len = 140.0
    side_h = 125.0
    side_area = 10500.0
    rear_w = 65.0

    res = fuse_dual_angle_morphometrics(
        side_length_cm=side_len,
        side_height_cm=side_h,
        side_area_cm2=side_area,
        rear_barrel_width_cm=rear_w
    )

    assert "fused_girth_cm" in res
    assert "volume_approx_cm3" in res
    assert "fused_weight_kg" in res
    assert res["fused_girth_cm"] > 0
    assert res["volume_approx_cm3"] > 0
    assert res["fused_weight_kg"] > 0

    # Verify Ramanujan girth is larger than max axis diameter and realistic for cattle (approx 180 - 250 cm)
    # semi-axes a = 62.5, b = 32.5. Girth should be > 2*pi*b (~204 cm) and < 2*pi*a (~392 cm)
    assert 200.0 <= res["fused_girth_cm"] <= 350.0

def test_rear_barrel_width_extraction():
    module = DualAngleFusionModule()
    
    # Create synthetic rear mask (500x500) with rectangular torso of width 200px
    rear_mask = np.zeros((500, 500), dtype=np.uint8)
    rear_mask[100:400, 150:350] = 255  # width = 200 px

    scale = 0.3  # cm/px
    barrel_w_cm = module.extract_rear_barrel_width(rear_mask, calibration_factor=scale)
    assert np.isclose(barrel_w_cm, 200 * 0.3)

def test_process_dual_views():
    module = DualAngleFusionModule()
    
    side_res = {
        "measurements": {
            "body_length_cm": 145.0,
            "withers_height_cm": 128.0,
            "silhouette_area_cm2": 11136.0,
            "raw_height_px": 400.0
        }
    }

    rear_mask = np.zeros((500, 500), dtype=np.uint8)
    rear_mask[100:400, 150:350] = 255

    fused_res = module.process_dual_views(side_res, rear_mask=rear_mask)
    assert fused_res["fused_weight_kg"] > 0
    assert fused_res["side_length_cm"] == 145.0
    assert fused_res["side_height_cm"] == 128.0

def test_dual_angle_flag_defaults_to_false():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
    assert os.path.exists(config_path), "configs/features.yaml must exist."
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("features", {}).get("ENABLE_DUAL_ANGLE") is False, \
        "ENABLE_DUAL_ANGLE flag must default to false."

def test_predict_dual_angle_endpoint_fallback():
    from fastapi.testclient import TestClient
    from backend.main import app
    import io

    client = TestClient(app)

    img_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uploaded_images", "be39ef2e_105730429112_LL_1.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Test image not found.")

    with open(img_path, "rb") as f:
        side_bytes = f.read()

    response = client.post(
        "/api/predict_dual_angle",
        files={"side_file": ("side.jpg", io.BytesIO(side_bytes), "image/jpeg")},
        data={"cattle_id": "TAG-TEST-DUAL", "model_engine": "deeplab"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["warning"] == "dual_angle_fallback"
    assert "prediction" in data
    assert data["prediction"]["weight_kg"] > 0


