# 🔬 Keypoint_Net Heuristic Behavior Under Rotation Analysis

- **Implementation File**: `src/features/landmarks.py` (`detect_anatomical_landmarks`)

### 1. Classification of the 8 Anatomical Landmarks
- **Point A_Withers**: (a) Hybrid: Min-Y peak within (b) 35%-60% X-window heuristic
- **Point B_Hip_Hook**: (a) Hybrid: Min-Y peak within (b) 62%-85% X-window heuristic
- **Point C_Pin_Bone**: (a) Hybrid: Max-X boundary within (b) 82%-97% X-window heuristic
- **Point D_Point_Shoulder**: (a) Hybrid: Min-X boundary within (b) 22%-40% X-window heuristic
- **Point E1_Front_Hoof**: (b) Geometric derivation: Bottom-most Y under Point A's X-coordinate
- **Point E2_Rear_Hoof**: (b) Geometric derivation: Bottom-most Y under Point B's X-coordinate
- **Point F_Flank**: (b) Pure Geometric derivation: Midpoint of (B_x + C_x)/2, y = B_y + 15% height
- **Point G_Head_Eye**: (b) Pure Geometric derivation: Center of head region X-span, y = ymin + 20% height

### 2. Rotation Sensitivity Test Results (0°, 15°, 30°)
- **Point A (Withers) Drift at 15°**: 41.38 pixels
- **Point A (Withers) Drift at 30°**: 78.57 pixels

### 3. Key Finding & Verdict
FINDING: The landmarks re-anchor relative to the NEW bounding box window rather than strictly following physical rigid-body rotation. Because Points A, B, C, D search inside fixed percentage windows of the current mask's bounding box [xmin, xmax, ymin, ymax], rotating the mask changes the bounding box bounds, causing the heuristic search windows to shift. Consequently, the landmark positions re-adapt to the new silhouette bounding box contour rather than remaining fixed to physical anatomical coordinates.
