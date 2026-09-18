import os
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

def create_dummy_jpeg():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 250), (255, 255, 255), -1)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()

def test_api_predict_baseline_flow():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"cattle_id": "TAG-TEST-001"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction" in data

def test_innovation_1_video_keyframe_route():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict_video",
        files={"video_file": ("test.mp4", img_bytes, "video/mp4")},
        data={"cattle_id": "TAG-TEST-V1"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_innovation_2_per_request_unwarp_header():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        headers={"X-Enable-Unwarp": "true"},
        data={"cattle_id": "TAG-TEST-V2"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_innovation_3_dual_angle_route():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict_dual_angle",
        files={
            "side_file": ("side.jpg", img_bytes, "image/jpeg"),
            "rear_file": ("rear.jpg", img_bytes, "image/jpeg")
        },
        data={"cattle_id": "TAG-TEST-V3"}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True

def test_innovation_4_exif_calibration_header():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        headers={"X-Enable-EXIF": "true"},
        data={"cattle_id": "TAG-TEST-V4"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True

def test_innovation_5_kan_and_ensemble_routes():
    img_bytes = create_dummy_jpeg()
    resp_kan = client.post(
        "/api/predict_kan",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"cattle_id": "TAG-TEST-V5K"}
    )
    assert resp_kan.status_code == 200
    assert resp_kan.json()["success"] is True

    resp_ens = client.post(
        "/api/predict_ensemble",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"cattle_id": "TAG-TEST-V5E", "alpha": "0.5"}
    )
    assert resp_ens.status_code == 200
    assert resp_ens.json()["success"] is True

def test_innovation_6_xai_header():
    img_bytes = create_dummy_jpeg()
    response = client.post(
        "/api/predict",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        headers={"X-Enable-XAI": "true"},
        data={"cattle_id": "TAG-TEST-V6"}
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
