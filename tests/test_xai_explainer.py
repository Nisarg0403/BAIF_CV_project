import os
import pytest
import yaml
import numpy as np
import cv2

from src.evaluation.xai_explainer import (
    compute_shap_attributions,
    generate_xai_heatmap_overlay,
    XAIExplainer
)

def test_compute_shap_attributions():
    feats = {
        "body_length_cm": 145.0,
        "chest_girth_cm": 170.0,
        "silhouette_area_cm2": 11000.0,
        "withers_height_cm": 128.0
    }
    
    res = compute_shap_attributions(feats)
    assert "shap_values" in res
    assert "base_value" in res
    assert "importance_ranking" in res
    assert len(res["importance_ranking"]) == 4
    assert res["base_value"] > 0

def test_generate_xai_heatmap_overlay():
    dummy_img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", dummy_img)
    img_bytes = buf.tobytes()

    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[100:300, 100:500] = 255

    shap_dict = {
        "shap_values": {
            "chest_girth_cm": 15.2,
            "silhouette_area_cm2": 8.4,
            "body_length_cm": 5.1,
            "withers_height_cm": 2.0
        }
    }

    b64_out = generate_xai_heatmap_overlay(img_bytes, mask, shap_dict)
    assert isinstance(b64_out, str)
    assert b64_out.startswith("data:image/jpeg;base64,")
    assert len(b64_out) > 100

def test_xai_explainer_class():
    explainer = XAIExplainer()
    dummy_img = np.ones((400, 600, 3), dtype=np.uint8) * 200
    _, buf = cv2.imencode(".jpg", dummy_img)
    img_bytes = buf.tobytes()

    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[100:300, 100:500] = 255

    meas = {
        "body_length_cm": 145.0,
        "chest_girth_cm": 170.0,
        "silhouette_area_cm2": 11000.0,
        "withers_height_cm": 128.0
    }

    out = explainer.explain_and_render(img_bytes, mask, meas)
    assert "shap_attributions" in out
    assert "xai_heatmap_b64" in out
    assert "card_summary" in out
    assert out["xai_heatmap_b64"].startswith("data:image/jpeg;base64,")

def test_xai_flag_defaults_to_false():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
    assert os.path.exists(config_path), "configs/features.yaml must exist."
    
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    assert cfg.get("features", {}).get("ENABLE_XAI_CARDS") is False, \
        "ENABLE_XAI_CARDS flag must default to false."

def test_xai_cards_integration_with_flag():
    from backend.inference import process_image_file

    img_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "uploaded_images", "be39ef2e_105730429112_LL_1.jpg")
    if not os.path.exists(img_path):
        pytest.skip("Test image not found.")

    with open(img_path, "rb") as f:
        img_bytes = f.read()

    # With flag OFF (default baseline)
    res_off = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
    assert "xai_heatmap_b64" not in res_off

    # Mock flag ON
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "features.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    cfg["features"]["ENABLE_XAI_CARDS"] = True
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    try:
        res_on = process_image_file(img_bytes, side_name="Left Side Profile", model_engine="deeplab")
        assert "xai_heatmap_b64" in res_on
        assert "xai_explanation" in res_on
        assert res_on["xai_heatmap_b64"].startswith("data:image/jpeg;base64,")
    finally:
        # Revert flag back to False (default)
        cfg["features"]["ENABLE_XAI_CARDS"] = False
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)

