import os
import argparse
import cv2
import pickle
import numpy as np

from src.segmentation.segment import CowSegmenter
from src.features.morphometry import extract_morphometric_features

def predict_cattle_weight(image_path, model_path, segmenter, output_dir=None):
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} does not exist.")
        return None
        
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return None
        
    print(f"Processing image: {os.path.basename(image_path)}")
    print(f"Image shape: {img.shape}")
    
    # 1. Segment cow
    mask = segmenter.segment(img)
    if np.sum(mask) == 0:
        print("Error: Could not detect/segment any cattle in the image.")
        return None
        
    # 2. Extract features
    feats = extract_morphometric_features(mask)
    
    # 3. Load model and predict
    if not os.path.exists(model_path):
        print(f"Error: Trained model file {model_path} not found.")
        return None
        
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
        
    X_input = np.array([[feats['length'], feats['height'], feats['area'], feats['girth']]])
    predicted_weight = float(model.predict(X_input)[0])
    
    # Print results
    print("\n==========================================")
    print("      WEIGHT ESTIMATION RESULT            ")
    print("==========================================")
    print(f"Estimated Body Weight: {predicted_weight:.2f} kg")
    print(f"Extracted Morphometric Features (Pixels):")
    print(f"  - Body Length: {feats['length']:.1f} px")
    print(f"  - Body Height: {feats['height']:.1f} px")
    print(f"  - Torso Girth Surrogate: {feats['girth']:.1f} px")
    print(f"  - Silhouette Area: {feats['area']:.1f} px")
    print("==========================================\n")
    
    # 4. Generate Visualization Overlay
    # Create semi-transparent mask overlay
    overlay = img.copy()
    overlay[mask == 255] = [0, 255, 0]  # Color the segmented cow green
    alpha = 0.35  # Transparency factor
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw standard bounding box around the cow silhouette
    y_indices, x_indices = np.nonzero(mask)
    xmin, xmax = np.min(x_indices), np.max(x_indices)
    ymin, ymax = np.min(y_indices), np.max(y_indices)
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
    
    # Add text label with prediction on the image
    label = f"Weight: {predicted_weight:.1f} kg"
    cv2.putText(img, label, (xmin, max(30, ymin - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Save output visualization
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_prediction.jpg")
        cv2.imwrite(out_path, img)
        print(f"Saved visualization overlay to: {out_path}")
        
    return predicted_weight

if __name__ == "__main__":
    # Command line args helper (can be run with default params)
    model_path = r"c:\Users\NISARG\BAIF\models\best_weight_regressor.pkl"
    cropped_dir = r"c:\Users\NISARG\BAIF\data\processed\bristol_cropped"
    output_dir = r"c:\Users\NISARG\BAIF\results"
    
    # Default to first cropped file if no args provided
    files = [f for f in os.listdir(cropped_dir) if f.endswith('.jpg')] if os.path.exists(cropped_dir) else []
    if len(files) > 0:
        sample_image = os.path.join(cropped_dir, files[0])
    else:
        sample_image = ""
        
    # Check if user specified a file
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = sample_image
        
    if image_path:
        segmenter = CowSegmenter()
        predict_cattle_weight(image_path, model_path, segmenter, output_dir)
    else:
        print("No image file specified and no default sample file found.")
