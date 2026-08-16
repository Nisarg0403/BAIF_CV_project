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
    
    # 2. Torso Girth Surrogate:
    # We take the middle-third of the cow's body horizontally (between 40% and 70% of length)
    # to avoid the head/neck on the left/right and the tail on the other side.
    # In this region, we measure the vertical height of the mask columns.
    start_x = int(xmin + 0.40 * length)
    end_x = int(xmin + 0.70 * length)
    
    if end_x > start_x:
        torso_heights = []
        for x in range(start_x, end_x):
            # Column height is the count of non-zero pixels in column x
            col_height = np.sum(mask[:, x] == 255)
            torso_heights.append(col_height)
            
        girth = float(np.mean(torso_heights))
    else:
        girth = height * 0.6  # Default approximation
        
    features['length'] = length
    features['height'] = height
    features['area'] = area
    features['girth'] = girth
    
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
