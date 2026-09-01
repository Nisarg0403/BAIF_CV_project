import numpy as np
import cv2

def extract_morphometric_features(mask):
    """
    Extracts morphometric features (length, height, area, girth surrogate) from a binary cow mask.
    mask: numpy array of shape HxW, values are 0 or 255.
    Returns: dict containing float values for 'length', 'height', 'area', 'girth'.
    """
    features = {
        'length': 0.0,
        'height': 0.0,
        'area': 0.0,
        'girth': 0.0
    }
    
    # Check if mask is empty
    y_indices, x_indices = np.nonzero(mask)
    if len(x_indices) == 0 or len(y_indices) == 0:
        return features
        
    # 1. Bounding box coordinates
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    
    length = float(xmax - xmin)
    height = float(ymax - ymin)
    area = float(np.sum(mask == 255))
    
    # 2. Refined Torso Girth Surrogate:
    # We take the middle region of the cow's torso horizontally (40% to 70% of length)
    # to measure the main body barrel (spine to belly line) without leg/shadow inflation.
    start_x = int(xmin + 0.40 * length)
    end_x = int(xmin + 0.70 * length)
    
    if end_x > start_x:
        torso_heights = []
        barrel_depths = []
        
        for x in range(start_x, end_x):
            col_y = np.where(mask[:, x] == 255)[0]
            if len(col_y) > 0:
                y_top = col_y[0]
                y_bot = col_y[-1]
                total_span = y_bot - y_top + 1
                
                # Check contiguous body barrel pixels (spine down to belly line)
                # Legs usually start below ymin + 0.55 * height
                barrel_y_limit = int(ymin + 0.60 * height)
                barrel_pixels = np.sum((mask[:, x] == 255) & (np.arange(mask.shape[0]) <= barrel_y_limit))
                barrel_depths.append(barrel_pixels)
                
                # Raw column height for comparison
                torso_heights.append(np.sum(mask[:, x] == 255))
        
        if len(barrel_depths) > 0:
            # Torso height measured from spine to belly line
            torso_height = float(np.mean(barrel_depths))
            raw_girth = float(np.mean(torso_heights))
            
            # Girth surrogate calculation:
            # Chest girth is strongly correlated with torso depth (barrel height) and length.
            # Using barrel depth prevents leg/ground shadow inflation while preserving relative scaling.
            girth = raw_girth if raw_girth < (height * 0.7) else (torso_height / 0.55)
        else:
            torso_height = height * 0.55
            girth = height * 0.6
    else:
        torso_height = height * 0.55
        girth = height * 0.6
        
    features['length'] = length
    features['height'] = height
    features['area'] = area
    features['girth'] = girth
    features['torso_height'] = torso_height
    features['leg_height_ratio'] = float(1.0 - (torso_height / height)) if height > 0 else 0.45
    
    return features

if __name__ == "__main__":
    # Test with a dummy mask
    dummy_mask = np.zeros((300, 300), dtype=np.uint8)
    # Draw a rectangle in the middle representing a cow
    cv2.rectangle(dummy_mask, (50, 100), (250, 200), 255, thickness=-1)
    
    feats = extract_morphometric_features(dummy_mask)
    print("Dummy mask features:")
    for k, v in feats.items():
        print(f"  {k}: {v}")
