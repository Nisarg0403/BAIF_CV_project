"""
Zero-Marker EXIF Self-Calibration Module
Extends standard EXIF metadata extraction to compute camera focal length & sensor scale.
Cross-validates EXIF scale against withers-height scale factor and flags discrepancies > 10%.
"""

import io
import math
from PIL import Image, ExifTags

def extract_exif_metadata(image_bytes: bytes) -> dict:
    """
    Extracts raw EXIF tags from image bytes.
    Returns dictionary of parsed key camera parameters.
    """
    exif_data = {
        "focal_length_mm": None,
        "focal_length_35mm": None,
        "digital_zoom_ratio": 1.0,
        "make": None,
        "model": None,
        "has_exif": False
    }

    try:
        img = Image.open(io.BytesIO(image_bytes))
        raw_exif = img._getexif()
        if raw_exif is None:
            return exif_data

        exif_data["has_exif"] = True
        
        # Build tag name lookup
        tag_lookup = {ExifTags.TAGS.get(k, k): v for k, v in raw_exif.items()}

        focal = tag_lookup.get("FocalLength")
        if focal is not None:
            try:
                exif_data["focal_length_mm"] = float(focal)
            except (TypeError, ValueError):
                pass

        focal_35 = tag_lookup.get("FocalLengthIn35mmFilm")
        if focal_35 is not None:
            try:
                exif_data["focal_length_35mm"] = float(focal_35)
            except (TypeError, ValueError):
                pass

        zoom = tag_lookup.get("DigitalZoomRatio")
        if zoom is not None:
            try:
                exif_data["digital_zoom_ratio"] = float(zoom)
            except (TypeError, ValueError):
                pass

        exif_data["make"] = str(tag_lookup.get("Make", "")).strip() or None
        exif_data["model"] = str(tag_lookup.get("Model", "")).strip() or None

    except Exception:
        pass

    return exif_data

def calculate_exif_scale(
    exif_meta: dict,
    image_height_px: int,
    image_width_px: int,
    estimated_distance_cm: float = 225.0
) -> float:
    """
    Calculates physical scale factor (cm per pixel) from camera EXIF parameters.
    Default camera-to-cattle distance = 225 cm (~2.25m standard BAIF capture distance).
    """
    focal_mm = exif_meta.get("focal_length_mm")
    focal_35mm = exif_meta.get("focal_length_35mm")
    zoom = exif_meta.get("digital_zoom_ratio", 1.0) or 1.0

    if not focal_mm and not focal_35mm:
        # Fallback to standard smartphone focal length ~4.2mm
        focal_mm = 4.2

    # Estimate sensor height in mm
    # Standard 1/2.55" sensor height ~ 4.8 mm, or derived from 35mm film ratio (36mm / 24mm)
    if focal_35mm and focal_35mm > 0:
        crop_factor = focal_35mm / focal_mm if (focal_mm and focal_mm > 0) else 7.0
        sensor_height_mm = 24.0 / crop_factor
    else:
        sensor_height_mm = 4.8  # Default smartphone sensor height

    # Effective focal length adjusted for digital zoom
    effective_focal_mm = (focal_mm or 4.2) * zoom

    # Distance in mm
    distance_mm = estimated_distance_cm * 10.0

    # Field of view height at distance D: FOV_h_mm = (D_mm * sensor_h_mm) / f_mm
    fov_height_mm = (distance_mm * sensor_height_mm) / effective_focal_mm
    fov_height_cm = fov_height_mm / 10.0

    # cm per pixel
    exif_scale_cm_px = fov_height_cm / float(image_height_px) if image_height_px > 0 else 0.3
    return float(exif_scale_cm_px)

def cross_validate_exif_scale(
    exif_scale_cm_px: float,
    withers_scale_cm_px: float,
    threshold_pct: float = 10.0
) -> dict:
    """
    Cross-validates EXIF scale against withers-height scale factor.
    Flags discrepancy > threshold_pct (default 10%), but does NOT silently override.
    """
    if withers_scale_cm_px <= 0:
        return {
            "valid": False,
            "discrepancy_pct": 0.0,
            "exif_scale_cm_px": round(float(exif_scale_cm_px), 4),
            "withers_scale_cm_px": round(float(withers_scale_cm_px), 4),
            "flagged": False,
            "warning": "invalid_withers_scale",
            "recommended_scale_cm_px": round(float(exif_scale_cm_px), 4)
        }

    diff_abs = abs(exif_scale_cm_px - withers_scale_cm_px)
    discrepancy_pct = (diff_abs / withers_scale_cm_px) * 100.0
    flagged = discrepancy_pct > threshold_pct

    warning_msg = None
    if flagged:
        warning_msg = f"EXIF scale discrepancy ({discrepancy_pct:.1f}%) exceeds {threshold_pct:.1f}% threshold."

    return {
        "valid": True,
        "discrepancy_pct": round(float(discrepancy_pct), 2),
        "exif_scale_cm_px": round(float(exif_scale_cm_px), 4),
        "withers_scale_cm_px": round(float(withers_scale_cm_px), 4),
        "flagged": flagged,
        "warning": warning_msg,
        # Preserves withers_scale_cm_px as primary to avoid silent override
        "recommended_scale_cm_px": round(float(withers_scale_cm_px), 4)
    }

class EXIFScaleCalibrator:
    """
    Zero-Marker EXIF Self-Calibrator wrapper.
    """
    def __init__(self, estimated_distance_cm: float = 225.0, threshold_pct: float = 10.0):
        self.estimated_distance_cm = estimated_distance_cm
        self.threshold_pct = threshold_pct

    def calibrate(self, image_bytes: bytes, image_shape: tuple, withers_scale_cm_px: float) -> dict:
        """
        Executes EXIF extraction, scale calculation, and cross-validation against withers scale.
        """
        h_px, w_px = image_shape[:2]
        exif_meta = extract_exif_metadata(image_bytes)
        exif_scale = calculate_exif_scale(exif_meta, h_px, w_px, self.estimated_distance_cm)
        validation = cross_validate_exif_scale(exif_scale, withers_scale_cm_px, self.threshold_pct)
        validation["exif_metadata"] = exif_meta
        return validation
