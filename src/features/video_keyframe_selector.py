import os
import cv2
import tempfile
import numpy as np
import logging
from typing import Union, List, Tuple

# Setup logger for experimental module
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "experimental")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("video_keyframe_selector")
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "video_keyframe_selector.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

def select_best_keyframe_from_frames(frames: List[np.ndarray]) -> np.ndarray:
    """
    Selects the single most steady, orthogonal, and sharp keyframe from a list of BGR numpy image frames.
    Uses Farneback optical flow motion analysis, Laplacian sharpness, and contour aspect ratio heuristics.
    """
    if not frames:
        raise ValueError("No frames provided for keyframe selection.")

    if len(frames) == 1:
        return frames[0]

    num_frames = len(frames)
    scores = []
    
    # Pre-calculate motion scores across consecutive frame pairs
    motion_scores = [0.0] * num_frames
    for i in range(1, num_frames):
        try:
            g1 = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            g2 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            flow = cv2.calcOpticalFlowFarneback(
                g1, g2, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            m_val = float(np.mean(mag) + np.std(mag))
            motion_scores[i] = m_val
        except Exception:
            motion_scores[i] = 0.0

    if num_frames > 1:
        motion_scores[0] = motion_scores[1]

    for i, frame in enumerate(frames):
        try:
            if frame is None or frame.size == 0:
                scores.append(-9999.0)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # 1. Image Sharpness Score (higher variance = sharper image)
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

            # 2. Optical Flow Motion Score (lower displacement = steadier frame)
            motion = motion_scores[i]

            # 3. Contour & Aspect Ratio Orthogonality Heuristic
            # Side profile cows have aspect ratio approx 1.2 to 1.9
            aspect_score = 1.0
            _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_c = max(contours, key=cv2.contourArea)
                x, y, w_box, h_box = cv2.boundingRect(largest_c)
                if h_box > 0:
                    ar = w_box / float(h_box)
                    if 1.2 <= ar <= 1.9:
                        aspect_score = 1.5
                    elif 1.0 <= ar <= 2.2:
                        aspect_score = 1.2

            # Composite Score: Reward Sharpness & Aspect Ratio, Penalize Motion
            composite_score = (sharpness * 0.5) - (motion * 10.0) + (aspect_score * 5.0)
            scores.append(composite_score)

        except Exception as e:
            logger.warning(f"Error scoring frame {i}: {e}")
            scores.append(-9999.0)

    best_index = int(np.argmax(scores))
    logger.info(f"Selected keyframe index {best_index}/{num_frames} with score {scores[best_index]:.2f}")
    return frames[best_index]

def select_best_keyframe(video_input: Union[str, bytes], max_frames: int = 90) -> np.ndarray:
    """
    Accepts video file path or raw video bytes (2-3 sec, ~30-90 frames),
    processes frames, and returns the top-ranked single frame as a standard BGR numpy array.
    """
    temp_file_path = None
    try:
        if isinstance(video_input, bytes):
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(video_input)
            temp_file.close()
            temp_file_path = temp_file.name
            video_path = temp_file_path
        else:
            video_path = video_input

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")

        frames = []
        count = 0
        while cap.isOpened() and count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            count += 1
        cap.release()

        if not frames:
            raise ValueError("No valid video frames could be decoded from input.")

        best_frame = select_best_keyframe_from_frames(frames)
        return best_frame

    except Exception as e:
        logger.error(f"Keyframe selection failed: {e}. Falling back to frame 0 or empty array.")
        raise e
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass
