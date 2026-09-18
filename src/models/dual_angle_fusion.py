"""
Dual-Angle Guided Capture Fusion Module (Side + 45° Rear View)
Combines side profile silhouette morphometrics with rear view barrel width
using cross-attention weight maps and 3D volumetric ellipsoidal integration.
"""

import math
import numpy as np

def compute_cross_attention_weights(side_feats: np.ndarray, rear_feats: np.ndarray) -> np.ndarray:
    """
    Computes per-feature cross-attention weights between side features Q and rear features K.
    side_feats: shape (D,) or (1, D)
    rear_feats: shape (D,) or (1, D)
    Returns normalized attention weights alpha of shape (1, D).
    """
    q = np.atleast_2d(side_feats)
    k = np.atleast_2d(rear_feats)
    
    d_k = q.shape[1]
    # Channelwise cross-attention score
    scores = (q * k) / math.sqrt(d_k)
    # Softmax normalization over feature dimension
    exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
    return weights

def fuse_dual_angle_morphometrics(
    side_length_cm: float,
    side_height_cm: float,
    side_area_cm2: float,
    rear_barrel_width_cm: float,
    rear_height_cm: float = None,
    rear_area_cm2: float = None
) -> dict:
    """
    Pure geometric & cross-attention fusion of side and rear morphometrics.
    
    Returns a dictionary with:
    - fused_girth_cm: Chest girth calculated via Ramanujan's ellipse perimeter formula 
                      using side height (vertical axis) and rear barrel width (horizontal axis).
    - volume_approx_cm3: Ellipsoidal 3D volume approximation (pi * H/2 * W_barrel/2 * L).
    - fused_weight_kg: Schaeffer volumetric estimate based on fused girth and side length.
    - attention_weights: Normalized cross-attention weights.
    - features_fused: Cross-attention weighted feature vector.
    """
    if rear_height_cm is None:
        rear_height_cm = side_height_cm
    if rear_area_cm2 is None:
        rear_area_cm2 = rear_barrel_width_cm * rear_height_cm * 0.6

    # 1. Feature Vector Construction
    # Side vector: [Length, Height, Area]
    # Rear vector: [Barrel Width, Height, Area]
    f_side = np.array([side_length_cm, side_height_cm, side_area_cm2], dtype=np.float64)
    f_rear = np.array([rear_barrel_width_cm, rear_height_cm, rear_area_cm2], dtype=np.float64)

    # 2. Cross-Attention
    attn_weights = compute_cross_attention_weights(f_side, f_rear)
    # Fused feature vector: side features + attention-weighted rear features
    fused_vector = f_side + attn_weights[0] * f_rear

    # 3. 3D Volumetric Approximation (Ellipsoid Cylinder)
    # Semi-axes: a = side_height_cm / 2, b = rear_barrel_width_cm / 2
    a = side_height_cm / 2.0
    b = rear_barrel_width_cm / 2.0
    L = side_length_cm

    # 3D Volume: V = pi * a * b * L
    volume_approx_cm3 = math.pi * a * b * L

    # 4. Ramanujan's Ellipse Perimeter Formula for Chest Girth G
    # G = pi * [ 3(a+b) - sqrt((3a + b)(a + 3b)) ]
    h_param = ((a - b) ** 2) / ((a + b) ** 2) if (a + b) > 0 else 0.0
    girth_ramanujan = math.pi * (a + b) * (1.0 + (3.0 * h_param) / (10.0 + math.sqrt(4.0 - 3.0 * h_param)))

    # 5. Volumetric Weight Estimation via Schaeffer Formula: W = G^2 * L / 10838
    fused_weight_kg = (girth_ramanujan ** 2 * L) / 10838.0

    return {
        "fused_girth_cm": round(float(girth_ramanujan), 2),
        "volume_approx_cm3": round(float(volume_approx_cm3), 2),
        "fused_weight_kg": round(float(fused_weight_kg), 2),
        "rear_barrel_width_cm": round(float(rear_barrel_width_cm), 2),
        "side_length_cm": round(float(side_length_cm), 2),
        "side_height_cm": round(float(side_height_cm), 2),
        "cross_attention_weights": attn_weights.tolist(),
        "fused_features": fused_vector.tolist()
    }

class DualAngleFusionModule:
    """
    Dual-Angle Cross-Attention Fusion Module.
    Combines side profile and 45° rear view segmentation masks and morphometrics.
    """
    def __init__(self):
        pass

    def extract_rear_barrel_width(self, rear_mask: np.ndarray, calibration_factor: float = 1.0) -> float:
        """
        Extracts barrel width from rear silhouette mask.
        rear_mask: binary uint8 segmentation mask of rear view
        calibration_factor: cm per pixel scale factor
        """
        if rear_mask is None or np.sum(rear_mask) == 0:
            raise ValueError("Invalid or empty rear mask.")

        y_indices, x_indices = np.nonzero(rear_mask)
        ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
        
        # Measure maximum horizontal width across middle 60% of torso height
        h_torso = ymax - ymin
        y_mid_start = int(ymin + 0.2 * h_torso)
        y_mid_end = int(ymin + 0.8 * h_torso)

        max_width_px = 0
        for y in range(y_mid_start, y_mid_end):
            xs = np.where(rear_mask[y, :] > 0)[0]
            if len(xs) > 0:
                width_px = xs[-1] - xs[0] + 1
                if width_px > max_width_px:
                    max_width_px = width_px

        if max_width_px == 0:
            xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
            max_width_px = xmax - xmin + 1

        return float(max_width_px * calibration_factor)

    def process_dual_views(self, side_res: dict, rear_mask: np.ndarray = None, rear_barrel_width_cm: float = None) -> dict:
        """
        Fuses side profile prediction results with rear view data.
        """
        side_meas = side_res.get("measurements", {})
        side_len = side_meas.get("body_length_cm", 140.0)
        side_height = side_meas.get("withers_height_cm", 125.0)
        side_area = side_meas.get("silhouette_area_cm2", side_len * side_height * 0.6)

        if rear_barrel_width_cm is None and rear_mask is not None:
            # Estimate scale from side view height
            px_height = side_meas.get("raw_height_px", 400.0)
            cm_px_scale = side_height / px_height if px_height > 0 else 0.3
            rear_barrel_width_cm = self.extract_rear_barrel_width(rear_mask, calibration_factor=cm_px_scale)

        if rear_barrel_width_cm is None:
            raise ValueError("Rear barrel width could not be determined.")

        fusion_result = fuse_dual_angle_morphometrics(
            side_length_cm=side_len,
            side_height_cm=side_height,
            side_area_cm2=side_area,
            rear_barrel_width_cm=rear_barrel_width_cm
        )

        return fusion_result
