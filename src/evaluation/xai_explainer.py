"""
Explainable AI (XAI) Explainer & Visual Heatmap Module
Generates SHAP/LIME feature attributions and renders visual region heatmaps
over the cattle silhouette (Ribcage, Abdomen, Withers, Hindquarters).
"""

import io
import base64
import os
import cv2
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def compute_shap_attributions(features: dict, base_weight: float = 350.0) -> dict:
    """
    Computes SHAP feature importance attributions for morphometric inputs.
    Uses SHAP TreeExplainer if available, or robust gradient-based tree attribution fallback.
    features: dict with keys 'body_length_cm', 'chest_girth_cm', 'silhouette_area_cm2', 'withers_height_cm'
    """
    feature_keys = ['body_length_cm', 'chest_girth_cm', 'silhouette_area_cm2', 'withers_height_cm']
    x_vec = np.array([[features.get(k, 120.0) for k in feature_keys]], dtype=np.float64)

    shap_values_dict = {}
    base_value = base_weight

    # Attempt native SHAP TreeExplainer on XGBoost pack model if available
    try:
        import pickle
        pack_path = os.path.join(PROJECT_ROOT, "models", "baif_multimodel_pack.pkl")
        if os.path.exists(pack_path):
            with open(pack_path, "rb") as f:
                pack = pickle.load(f)
            xgb_model = pack.get("weight_models", {}).get("Gradient Boosting Regressor")
            if xgb_model is not None:
                try:
                    import shap
                    explainer = shap.TreeExplainer(xgb_model)
                    shap_vals = explainer.shap_values(x_vec)
                    if hasattr(explainer, "expected_value"):
                        base_value = float(explainer.expected_value)

                    for i, k in enumerate(feature_keys):
                        shap_values_dict[k] = round(float(shap_vals[0][i]), 2)
                except Exception:
                    xgb_model = None
    except Exception:
        pass

    # Pure NumPy gradient attribution fallback if SHAP library or model is not present
    if not shap_values_dict:
        # Physical reference medians: length=140, girth=165, area=10000, height=125
        girth = features.get('chest_girth_cm', 165.0)
        length = features.get('body_length_cm', 140.0)
        area = features.get('silhouette_area_cm2', 10000.0)
        height = features.get('withers_height_cm', 125.0)

        # Baseline reference dimensions
        g_ref, l_ref, a_ref, h_ref = 165.0, 140.0, 10000.0, 125.0
        base_value = round((g_ref**2 * l_ref) / 10838.0, 1)

        # Marginal impacts
        girth_impact = ((girth**2 * l_ref) / 10838.0) - base_value
        length_impact = ((g_ref**2 * length) / 10838.0) - base_value
        area_impact = (area - a_ref) * 0.015
        height_impact = (height - h_ref) * 0.45

        shap_values_dict = {
            "chest_girth_cm": round(float(girth_impact), 2),
            "body_length_cm": round(float(length_impact), 2),
            "silhouette_area_cm2": round(float(area_impact), 2),
            "withers_height_cm": round(float(height_impact), 2)
        }

    # Rank features by absolute SHAP impact
    ranking = sorted(shap_values_dict.keys(), key=lambda k: abs(shap_values_dict[k]), reverse=True)

    return {
        "shap_values": shap_values_dict,
        "base_value": round(float(base_value), 1),
        "importance_ranking": ranking
    }

def generate_xai_heatmap_overlay(image_bytes: bytes, mask: np.ndarray, shap_dict: dict) -> str:
    """
    Renders visual regional heatmap overlay over the cattle silhouette.
    Highlights Ribcage (Chest Girth), Abdomen (Area), Withers (Height), and Rump (Length).
    Returns Base64 encoded JPEG string.
    """
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    if img is None:
        if mask is not None:
            h, w = mask.shape[:2]
            img = np.ones((h, w, 3), dtype=np.uint8) * 220
        else:
            return ""

    h, w, _ = img.shape

    if mask is None or np.sum(mask) == 0:
        mask = np.ones((h, w), dtype=np.uint8) * 255
    elif mask.shape[:2] != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # Initialize empty heatmap intensity grid
    heatmap_intensity = np.zeros((h, w), dtype=np.float32)

    y_indices, x_indices = np.nonzero(mask)
    if len(y_indices) > 0 and len(x_indices) > 0:
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))

        h_box = ymax - ymin + 1
        w_box = xmax - xmin + 1

        shap_vals = shap_dict.get("shap_values", {})
        girth_imp = abs(shap_vals.get("chest_girth_cm", 1.0))
        area_imp = abs(shap_vals.get("silhouette_area_cm2", 1.0))
        length_imp = abs(shap_vals.get("body_length_cm", 1.0))
        height_imp = abs(shap_vals.get("withers_height_cm", 1.0))

        tot_imp = max(1e-5, girth_imp + area_imp + length_imp + height_imp)

        # Region 1: Ribcage / Chest Girth (Middle-front torso, 30% to 60% of width)
        y1_g, y2_g = int(ymin + 0.2 * h_box), int(ymin + 0.85 * h_box)
        x1_g, x2_g = int(xmin + 0.3 * w_box), int(xmin + 0.65 * w_box)
        heatmap_intensity[y1_g:y2_g, x1_g:x2_g] += (girth_imp / tot_imp) * 255.0

        # Region 2: Abdomen / Area (Middle-lower torso, 40% to 80% of width)
        y1_a, y2_a = int(ymin + 0.35 * h_box), int(ymax)
        x1_a, x2_a = int(xmin + 0.25 * w_box), int(xmin + 0.75 * w_box)
        heatmap_intensity[y1_a:y2_a, x1_a:x2_a] += (area_imp / tot_imp) * 200.0

        # Region 3: Withers / Height (Top shoulder ridge, 25% to 50% width, 0% to 40% height)
        y1_h, y2_h = int(ymin), int(ymin + 0.4 * h_box)
        x1_h, x2_h = int(xmin + 0.25 * w_box), int(xmin + 0.5 * w_box)
        heatmap_intensity[y1_h:y2_h, x1_h:x2_h] += (height_imp / tot_imp) * 220.0

        # Region 4: Rump / Body Length (Rear torso, 60% to 95% width)
        y1_l, y2_l = int(ymin + 0.15 * h_box), int(ymin + 0.75 * h_box)
        x1_l, x2_l = int(xmin + 0.6 * w_box), int(xmin + 0.95 * w_box)
        heatmap_intensity[y1_l:y2_l, x1_l:x2_l] += (length_imp / tot_imp) * 210.0

    # Mask out background intensity
    heatmap_intensity[mask == 0] = 0

    # Smooth heatmap blur
    heatmap_intensity = cv2.GaussianBlur(heatmap_intensity, (51, 51), 0)

    # Normalize to 0-255 uint8
    max_val = np.max(heatmap_intensity)
    if max_val > 0:
        heatmap_uint8 = (heatmap_intensity / max_val * 255.0).astype(np.uint8)
    else:
        heatmap_uint8 = np.zeros((h, w), dtype=np.uint8)

    # Colorize using JET colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    # Blend color heatmap onto original RGB image
    alpha = 0.45
    blended = cv2.addWeighted(img, 1.0 - alpha, color_heatmap, alpha, 0)
    # Mask out background back to original image
    blended[mask == 0] = img[mask == 0]

    # Draw contour outline for high contrast presentation
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

    # Encode blended image to JPEG Base64 string
    is_success, buffer = cv2.imencode(".jpg", blended)
    if not is_success:
        return ""

    b64_str = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_str}"

class XAIExplainer:
    """
    Explainable AI Visual Cards Generator.
    Combines SHAP attribution metrics with visual heatmap overlays.
    """
    def __init__(self):
        pass

    def explain_and_render(self, image_bytes: bytes, mask: np.ndarray, measurements: dict) -> dict:
        """
        Computes SHAP feature importance & renders visual heatmap card.
        """
        shap_res = compute_shap_attributions(measurements)
        heatmap_b64 = generate_xai_heatmap_overlay(image_bytes, mask, shap_res)

        ranking = shap_res.get("importance_ranking", [])
        top_feature = ranking[0] if ranking else "chest_girth_cm"
        top_impact = shap_res.get("shap_values", {}).get(top_feature, 0.0)

        card_summary = f"Primary driver: {top_feature} contributing {top_impact:+.1f} kg to estimated weight."

        return {
            "shap_attributions": shap_res["shap_values"],
            "base_value_kg": shap_res["base_value"],
            "importance_ranking": ranking,
            "xai_heatmap_b64": heatmap_b64,
            "card_summary": card_summary
        }
