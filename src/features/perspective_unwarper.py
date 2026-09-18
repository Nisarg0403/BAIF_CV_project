import os
import cv2
import logging
import numpy as np
from datetime import datetime
from typing import Tuple, Dict, Any, Optional

# Setup logger for experimental module
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "experimental")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("perspective_unwarper")
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "perspective_unwarper.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

REQUIRED_LANDMARKS = ['A', 'B', 'C', 'D', 'E1', 'E2', 'F', 'G']

def unwarp_mask(mask: np.ndarray, landmarks: dict) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Pure Function: Restores 2D cattle silhouette mask to a 90° orthogonal side profile using 
    anatomical landmarks (A, B, C, D, E1, E2, F, G).

    Parameters:
        mask (np.ndarray): Binary silhouette mask (2D numpy array)
        landmarks (dict): Dictionary containing detected landmark points

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: (unwarped_mask, metadata)
    """
    metadata = {
        "unwarp_applied": False,
        "reason": None,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0
    }

    if mask is None or mask.size == 0:
        metadata["reason"] = "empty_mask"
        return mask, metadata

    pts = landmarks.get('points', {}) if isinstance(landmarks, dict) else {}
    valid_pts = {k: v for k, v in pts.items() if k in REQUIRED_LANDMARKS and v is not None and len(v) == 2}

    # Requirement C: Fallback when keypoints are unreliable (< 6 of 8 landmarks detected)
    if len(valid_pts) < 6:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_log = os.path.join(LOG_DIR, f"fallback_unwarp_{timestamp_str}.log")
        try:
            with open(fallback_log, "w", encoding="utf-8") as f_log:
                f_log.write(f"Unwarp fallback triggered: Only {len(valid_pts)}/8 landmarks detected.\n")
        except Exception:
            pass

        metadata["unwarp_applied"] = False
        metadata["reason"] = "insufficient_landmarks"
        logger.warning(f"Unwarp skipped: insufficient landmarks ({len(valid_pts)}/8). Returning original mask.")
        return mask.copy(), metadata

    try:
        h, w = mask.shape[:2]
        
        # Landmark coordinates
        A = np.array(valid_pts.get('A', (int(w*0.5), int(h*0.3))), dtype=np.float32)
        B = np.array(valid_pts.get('B', (int(w*0.7), int(h*0.3))), dtype=np.float32)
        C = np.array(valid_pts.get('C', (int(w*0.85), int(h*0.4))), dtype=np.float32)
        D = np.array(valid_pts.get('D', (int(w*0.3), int(h*0.4))), dtype=np.float32)

        # 1. Infer Yaw Angle theta from Withers-Hip to Hip-Pin ratio deviation
        dist_AB = float(np.linalg.norm(A - B))
        dist_BC = float(np.linalg.norm(B - C)) + 1e-6
        ratio_AB_BC = dist_AB / dist_BC
        
        # Standard orthogonal ratio is approx 1.6 - 1.8
        target_ratio = 1.7
        yaw_deg = float(np.clip((target_ratio - ratio_AB_BC) * 25.0, -30.0, 30.0))

        # 2. Infer Pitch Angle phi from Withers (A) to Hip (B) horizontal tilt
        dy = float(B[1] - A[1])
        dx = float(abs(B[0] - A[0])) + 1e-6
        pitch_deg = float(np.clip(np.degrees(np.arctan2(dy, dx)), -15.0, 15.0))

        metadata["yaw_deg"] = round(yaw_deg, 2)
        metadata["pitch_deg"] = round(pitch_deg, 2)

        # If camera angle is near-perfect (yaw < 3° and pitch < 3°), no unwarping needed
        if abs(yaw_deg) < 3.0 and abs(pitch_deg) < 3.0:
            metadata["unwarp_applied"] = True
            metadata["reason"] = "already_orthogonal"
            return mask.copy(), metadata

        # 3. Construct Affine Correction Transform Matrix
        src_tri = np.float32([A, B, D])
        
        # Target unwarped positions
        A_dst = A.copy()
        B_dst = np.array([A[0] + dist_AB * (1.0 + yaw_deg * 0.005), A[1]], dtype=np.float32)
        D_dst = np.array([D[0], D[1] - pitch_deg * 0.5], dtype=np.float32)
        
        dst_tri = np.float32([A_dst, B_dst, D_dst])

        M = cv2.getAffineTransform(src_tri, dst_tri)
        unwarped_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        metadata["unwarp_applied"] = True
        metadata["reason"] = "affine_unwarp_successful"
        logger.info(f"Affine unwarp applied: yaw={yaw_deg:.1f}°, pitch={pitch_deg:.1f}°")
        
        return unwarped_mask, metadata

    except Exception as e:
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        fallback_log = os.path.join(LOG_DIR, f"fallback_unwarp_{timestamp_str}.log")
        try:
            with open(fallback_log, "w", encoding="utf-8") as f_log:
                f_log.write(f"Perspective unwarp exception: {e}\n")
        except Exception:
            pass

        metadata["unwarp_applied"] = False
        metadata["reason"] = f"exception_{e}"
        logger.error(f"Perspective unwarp failed: {e}. Returning original mask.")
        return mask.copy(), metadata
