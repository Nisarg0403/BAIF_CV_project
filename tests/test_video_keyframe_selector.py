import pytest
import os
import cv2
import tempfile
import numpy as np
from src.features.video_keyframe_selector import (
    select_best_keyframe_from_frames,
    select_best_keyframe
)

def test_select_best_keyframe_from_frames_single():
    # Test single frame selection
    dummy_frame = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(dummy_frame, (100, 100), (300, 300), (255, 255, 255), -1)
    result = select_best_keyframe_from_frames([dummy_frame])
    assert result is not None
    assert result.shape == dummy_frame.shape

def test_select_best_keyframe_sharpness_ranking():
    # Frame 1: Very blurry image
    blurry = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(blurry, (100, 100), (400, 300), (200, 200, 200), -1)
    blurry = cv2.GaussianBlur(blurry, (31, 31), 0)

    # Frame 2: Crisp, sharp image with high contrast edges
    sharp = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(sharp, (100, 100), (400, 300), (255, 255, 255), -1)
    # Draw high frequency grid lines
    for i in range(100, 400, 20):
        cv2.line(sharp, (i, 100), (i, 300), (0, 0, 0), 2)

    frames = [blurry, sharp]
    selected = select_best_keyframe_from_frames(frames)
    # Expect sharp frame to be selected
    assert np.array_equal(selected, sharp)

def test_select_best_keyframe_from_video_file():
    # Create temporary synthetic MP4 video file
    temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_video.close()
    video_path = temp_video.name

    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_path, fourcc, 30.0, (600, 400))

        for i in range(15):
            frame = np.zeros((400, 600, 3), dtype=np.uint8)
            cv2.rectangle(frame, (50 + i*2, 100), (350 + i*2, 300), (255, 255, 255), -1)
            out.write(frame)
        out.release()

        # Execute keyframe selector on synthetic video file
        keyframe = select_best_keyframe(video_path, max_frames=30)
        assert keyframe is not None
        assert keyframe.shape == (400, 600, 3)

    finally:
        if os.path.exists(video_path):
            os.remove(video_path)

def test_empty_frames_raises_exception():
    with pytest.raises(ValueError):
        select_best_keyframe_from_frames([])
