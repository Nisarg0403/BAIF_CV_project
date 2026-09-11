import numpy as np
import cv2

def detect_anatomical_landmarks(mask, view_type='side'):
    """
    Detects anatomical landmark points A, B, C, D, E1, E2, F, G from a binary cattle silhouette mask.
    
    Points:
      A: Withers (top peak of front shoulder)
      B: Hip / Hook bone (top peak of hind hip)
      C: Pin bone (rearmost projection near tail base)
      D: Point of Shoulder (front shoulder joint, excluding head/neck)
      E1: Front Hoof (ground contact beneath front leg)
      E2: Rear Hoof (ground contact beneath rear leg)
      F: Flank reference point
      G: Head / Eye reference point
    """
    landmarks = {
        'view_type': view_type,
        'points': {},
        'distances': {}
    }
    
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return landmarks
        
    xmin, xmax = int(np.min(x_indices)), int(np.max(x_indices))
    ymin, ymax = int(np.min(y_indices)), int(np.max(y_indices))
    
    width_span = xmax - xmin
    height_span = ymax - ymin
    
    view_lower = view_type.lower()
    
    if 'rear' in view_lower or 'back' in view_lower:
        # Rear View Landmarks
        # Hip width top points (B_left, B_right) and Hoof baseline (E1, E2)
        h_start = int(ymin + 0.25 * height_span)
        h_end = int(ymin + 0.60 * height_span)
        
        row_widths = []
        for y in range(h_start, h_end):
            xs = np.where(mask[y, :] == 255)[0]
            if len(xs) > 0:
                row_widths.append((xs[-1] - xs[0] + 1, y, xs[0], xs[-1]))
                
        if len(row_widths) > 0:
            max_w, max_y, x_l, x_r = max(row_widths, key=lambda item: item[0])
            landmarks['points']['B_left'] = (int(x_l), int(max_y))
            landmarks['points']['B_right'] = (int(x_r), int(max_y))
            landmarks['distances']['rump_width_px'] = float(max_w)
        else:
            landmarks['points']['B_left'] = (xmin, int(ymin + 0.4 * height_span))
            landmarks['points']['B_right'] = (xmax, int(ymin + 0.4 * height_span))
            landmarks['distances']['rump_width_px'] = float(width_span * 0.7)
            
        landmarks['points']['E1'] = (int(xmin + 0.3 * width_span), ymax)
        landmarks['points']['E2'] = (int(xmin + 0.7 * width_span), ymax)
        return landmarks

    # Side Profile (Left or Right) Landmark Detection
    # Determine orientation: compare top spine height in left 20% vs right 20%
    left_top_ys = [np.min(np.where(mask[:, x] == 255)[0]) for x in range(xmin, int(xmin + 0.20 * width_span)) if len(np.where(mask[:, x] == 255)[0]) > 0]
    right_top_ys = [np.min(np.where(mask[:, x] == 255)[0]) for x in range(int(xmax - 0.20 * width_span), xmax) if len(np.where(mask[:, x] == 255)[0]) > 0]

    left_avg_top = np.mean(left_top_ys) if left_top_ys else ymin
    right_avg_top = np.mean(right_top_ys) if right_top_ys else ymin

    # Rump top spine is higher (lower Y index) than head/neck top spine
    facing_right = left_avg_top < right_avg_top

    if facing_right:
        # Head on right, Rump on left
        withers_x_region = (int(xmin + 0.40 * width_span), int(xmin + 0.65 * width_span))
        hip_x_region = (int(xmin + 0.15 * width_span), int(xmin + 0.38 * width_span))
        shoulder_x_region = (int(xmin + 0.60 * width_span), int(xmin + 0.78 * width_span))
        pin_x_region = (int(xmin + 0.03 * width_span), int(xmin + 0.18 * width_span))
        head_x_region = (int(xmin + 0.75 * width_span), xmax)
    else:
        # Head on left, Rump on right
        withers_x_region = (int(xmin + 0.35 * width_span), int(xmin + 0.60 * width_span))
        hip_x_region = (int(xmin + 0.62 * width_span), int(xmin + 0.85 * width_span))
        shoulder_x_region = (int(xmin + 0.22 * width_span), int(xmin + 0.40 * width_span))
        pin_x_region = (int(xmin + 0.82 * width_span), int(xmin + 0.97 * width_span))
        head_x_region = (xmin, int(xmin + 0.25 * width_span))

    # 1. Point A: Withers (top-most ridge of front shoulder)
    withers_pts = []
    for x in range(withers_x_region[0], withers_x_region[1]):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) > 0:
            withers_pts.append((x, int(ys[0])))
    pt_A = min(withers_pts, key=lambda p: p[1]) if withers_pts else (int((withers_x_region[0]+withers_x_region[1])/2), ymin)

    # 2. Point B: Hip / Hook bone (top-most ridge of hind hip)
    hip_pts = []
    for x in range(hip_x_region[0], hip_x_region[1]):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) > 0:
            hip_pts.append((x, int(ys[0])))
    pt_B = min(hip_pts, key=lambda p: p[1]) if hip_pts else (int((hip_x_region[0]+hip_x_region[1])/2), ymin)

    # 3. Point C: Pin bone (rearmost protrusion near tail base)
    pin_pts = []
    for x in range(pin_x_region[0], pin_x_region[1]):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) > 0:
            target_y = int(ys[0] + 0.35 * (ys[-1] - ys[0]))
            pin_pts.append((x, target_y))
    if pin_pts:
        pt_C = min(pin_pts, key=lambda p: p[0]) if facing_right else max(pin_pts, key=lambda p: p[0])
    else:
        pt_C = (xmin if facing_right else xmax, int(ymin + 0.35 * height_span))

    # 4. Point D: Point of Shoulder (front shoulder joint)
    shoulder_pts = []
    for x in range(shoulder_x_region[0], shoulder_x_region[1]):
        ys = np.where(mask[:, x] == 255)[0]
        if len(ys) > 0:
            target_y = int(ys[0] + 0.45 * (ys[-1] - ys[0]))
            shoulder_pts.append((x, target_y))
    if shoulder_pts:
        pt_D = max(shoulder_pts, key=lambda p: p[0]) if facing_right else min(shoulder_pts, key=lambda p: p[0])
    else:
        pt_D = (xmax if facing_right else xmin, int(ymin + 0.45 * height_span))

    # 5. Point E1: Front Hoof (ground contact under front shoulder)
    e1_ys = np.where(mask[:, pt_A[0]] == 255)[0]
    pt_E1 = (pt_A[0], int(e1_ys[-1])) if len(e1_ys) > 0 else (pt_A[0], ymax)

    # 6. Point E2: Rear Hoof (ground contact under hind leg)
    e2_ys = np.where(mask[:, pt_B[0]] == 255)[0]
    pt_E2 = (pt_B[0], int(e2_ys[-1])) if len(e2_ys) > 0 else (pt_B[0], ymax)

    # 7. Point F: Flank reference point
    f_x = int((pt_B[0] + pt_C[0]) / 2)
    f_y = int(pt_B[1] + 0.15 * height_span)
    pt_F = (f_x, f_y)

    # 8. Point G: Eye / Head reference
    g_x = int((head_x_region[0] + head_x_region[1]) / 2)
    g_y = int(ymin + 0.20 * height_span)
    pt_G = (g_x, g_y)

    landmarks['points'] = {
        'A': pt_A,
        'B': pt_B,
        'C': pt_C,
        'D': pt_D,
        'E1': pt_E1,
        'E2': pt_E2,
        'F': pt_F,
        'G': pt_G
    }

    # True anatomical measurement distances
    # Body Length (D to C)
    body_length_px = float(np.sqrt((pt_C[0] - pt_D[0])**2 + (pt_C[1] - pt_D[1])**2))
    withers_height_px = float(abs(pt_E1[1] - pt_A[1]))
    hip_height_px = float(abs(pt_E2[1] - pt_B[1]))

    landmarks['distances'] = {
        'body_length_px': body_length_px,
        'withers_height_px': withers_height_px,
        'hip_height_px': hip_height_px
    }

    return landmarks

def draw_landmark_overlay(img_rgb, landmarks):
    """
    Draws red circular dot markers for Points A, B, C, D, E1, E2, F, G and red dashed measurement lines on the RGB image preview.
    """
    overlay = img_rgb.copy()
    points = landmarks.get('points', {})
    
    if not points:
        return overlay

    # Color definition: Bright Red for points and lines
    color_red = (255, 0, 0)
    color_text = (255, 255, 255)
    
    # 1. Draw dashed measurement lines
    def draw_dashed_line(img, pt1, pt2, color, thickness=2, dash_len=10):
        dist = np.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
        if dist == 0:
            return
        dx = (pt2[0] - pt1[0]) / dist
        dy = (pt2[1] - pt1[1]) / dist
        
        curr_dist = 0
        draw = True
        while curr_dist < dist:
            next_dist = min(curr_dist + dash_len, dist)
            start_p = (int(pt1[0] + dx * curr_dist), int(pt1[1] + dy * curr_dist))
            end_p = (int(pt1[0] + dx * next_dist), int(pt1[1] + dy * next_dist))
            if draw:
                cv2.line(img, start_p, end_p, color, thickness, cv2.LINE_AA)
            draw = not draw
            curr_dist = next_dist

    # Draw lines: D -> C (Body Length), A -> E1 (Withers Height), B -> E2 (Hip Height), C -> F
    if 'D' in points and 'C' in points:
        draw_dashed_line(overlay, points['D'], points['C'], color_red, thickness=3)
    if 'A' in points and 'E1' in points:
        draw_dashed_line(overlay, points['A'], points['E1'], color_red, thickness=3)
    if 'B' in points and 'E2' in points:
        draw_dashed_line(overlay, points['B'], points['E2'], color_red, thickness=3)
    if 'C' in points and 'F' in points:
        draw_dashed_line(overlay, points['C'], points['F'], color_red, thickness=2)

    # 2. Draw Red Circular Dots and Labels for points A, B, C, D, E1, E2, F, G
    dot_radius = max(5, int(min(img_rgb.shape[0], img_rgb.shape[1]) * 0.012))
    
    for pt_name, pt_coords in points.items():
        if pt_coords is None:
            continue
        cx, cy = int(pt_coords[0]), int(pt_coords[1])
        # Outer red circle
        cv2.circle(overlay, (cx, cy), dot_radius, color_red, -1, cv2.LINE_AA)
        cv2.circle(overlay, (cx, cy), dot_radius + 2, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Label text
        cv2.putText(overlay, pt_name, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, pt_name, (cx + 8, cy - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_text, 2, cv2.LINE_AA)

    return overlay

def fuse_multiview_features(results_list):
    """
    Fuses morphometric and landmark features across multi-view image captures (Left Side, Right Side, Rear View).
    Returns a consolidated dictionary containing 3D fused length, height, width, girth, and volumetric weight.
    """
    side_lengths = []
    side_heights = []
    side_girths = []
    side_weights = []
    rear_width_cm = None
    
    for res_dict, side_label in results_list:
        label_lower = side_label.lower()
        feats = res_dict.get('feats', {})
        
        if 'rear' in label_lower or 'back' in label_lower:
            if 'girth' in feats:
                rear_width_cm = float(feats['girth'] * 0.26)
        else:
            if 'length' in feats: side_lengths.append(feats['length'])
            if 'height' in feats: side_heights.append(feats['height'])
            if 'girth' in feats: side_girths.append(feats['girth'])
            if 'weight' in res_dict: side_weights.append(res_dict['weight'])
            
    fused_length = float(np.median(side_lengths)) if len(side_lengths) > 0 else 148.0
    fused_height = float(np.median(side_heights)) if len(side_heights) > 0 else 142.0
    fused_girth = float(np.median(side_girths)) if len(side_girths) > 0 else 182.0
    fused_side_weight = float(np.mean(side_weights)) if len(side_weights) > 0 else 490.0
    
    if rear_width_cm is not None:
        a = (fused_height * 0.46) / 2.0
        b = rear_width_cm / 2.0
        # Ramanujan Ellipse Perimeter Formula
        girth_ellipse = float(np.pi * (3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))))
        effective_girth = float(0.6 * fused_girth + 0.4 * girth_ellipse)
    else:
        rear_width_cm = fused_height * 0.36
        effective_girth = fused_girth
        
    volumetric_weight = float((effective_girth**2 * fused_length) / 10838.0)
    
    # Combined fused weight
    fused_weight = float(0.7 * fused_side_weight + 0.3 * volumetric_weight)
    
    return {
        'fused_length': fused_length,
        'fused_height': fused_height,
        'rear_width': rear_width_cm,
        'fused_girth': effective_girth,
        'volumetric_weight': volumetric_weight,
        'fused_weight': fused_weight,
        'views_count': len(results_list)
    }

# Alias for backward compatibility
extract_anatomical_landmarks = detect_anatomical_landmarks

if __name__ == "__main__":
    dummy_img = np.zeros((400, 600, 3), dtype=np.uint8)
    dummy_mask = np.zeros((400, 600), dtype=np.uint8)
    cv2.rectangle(dummy_mask, (100, 100), (500, 300), 255, thickness=-1)
    
    lm = detect_anatomical_landmarks(dummy_mask, 'side')
    print("Landmarks:", lm['points'])
    print("Distances:", lm['distances'])
    
    over = draw_landmark_overlay(dummy_img, lm)
    print("Overlay generated shape:", over.shape)
