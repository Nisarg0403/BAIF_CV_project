import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import cv2
import pandas as pd
import numpy as np
import torch

from src.segmentation.segment import CowSegmenter
from src.features.morphometry import extract_morphometric_features

def process_baif_dataset(excel_path, images_dir, output_csv):
    """
    Reads BAIF_DATA.xlsx and BAIF_IMAGES directory, runs segmentation,
    calculates scale calibration from Height_at_wither_cm, and outputs calibrated physical features.
    """
    print(f"Loading metadata from {excel_path}...")
    if not os.path.exists(excel_path):
        raise FileNotFoundError(f"Excel file not found: {excel_path}")
        
    df_raw = pd.read_excel(excel_path)
    
    # Strip whitespace from column names
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    print(f"Columns in dataset loaded successfully ({len(df_raw.columns)} columns).")
    
    segmenter = CowSegmenter()
    
    processed_rows = []
    
    for idx, row in df_raw.iterrows():
        tag_id = str(row['Animal Tag']).strip()
        tape_weight = row['Tape_Weight_Kg']
        body_length = row['Body_Length_cm']
        chest_girth = row['Chest_Girth_cm']
        withers_height = row['Height_at_wither_cm']
        breed = row.get('Breed', 'Unknown')
        bcs = row.get('BCS', np.nan)
        
        print(f"\nProcessing Animal Tag: {tag_id} (Actual Weight: {tape_weight} kg, Withers Height: {withers_height} cm)")
        
        animal_folder = os.path.join(images_dir, tag_id)
        if not os.path.exists(animal_folder):
            print(f"  Warning: Folder not found for tag {tag_id} at {animal_folder}")
            continue
            
        # We look for Left Profile (_LL_1) and Right Profile (_LR_1)
        image_files = {
            'left': None,
            'right': None
        }
        for f in os.listdir(animal_folder):
            if f.lower().endswith('.jpg') or f.lower().endswith('.jpeg') or f.lower().endswith('.png'):
                if '_ll_' in f.lower() or 'left' in f.lower():
                    image_files['left'] = os.path.join(animal_folder, f)
                elif '_lr_' in f.lower() or 'right' in f.lower():
                    image_files['right'] = os.path.join(animal_folder, f)
                    
        for side, img_path in image_files.items():
            if img_path is None or not os.path.exists(img_path):
                print(f"  Side '{side}' image missing for Tag {tag_id}")
                continue
                
            print(f"  Extracting features for side '{side}': {os.path.basename(img_path)}")
            img = cv2.imread(img_path)
            if img is None:
                print(f"  Error reading image {img_path}")
                continue
                
            # Segment cow
            mask = segmenter.segment(img)
            if np.sum(mask) == 0:
                print(f"  Segmentation failed for {img_path}")
                continue
                
            # Extract raw morphometric features (pixels)
            raw_feats = extract_morphometric_features(mask)
            if raw_feats['height'] == 0:
                print(f"  Height extracted as 0 for {img_path}")
                continue
                
            # Scale Calibration (cm per pixel) using Height_at_wither_cm
            scale_cm_per_px = withers_height / raw_feats['height'] if raw_feats['height'] > 0 else 0.0
            
            calibrated_length_cm = raw_feats['length'] * scale_cm_per_px
            calibrated_height_cm = withers_height  # By definition
            calibrated_girth_cm = raw_feats['girth'] * scale_cm_per_px
            calibrated_area_cm2 = raw_feats['area'] * (scale_cm_per_px ** 2)
            
            row_data = {
                'animal_tag': tag_id,
                'side': side,
                'image_file': os.path.basename(img_path),
                'breed': breed,
                'bcs': bcs,
                'tape_weight_kg': tape_weight,
                'tape_length_cm': body_length,
                'tape_girth_cm': chest_girth,
                'tape_withers_height_cm': withers_height,
                'raw_length_px': raw_feats['length'],
                'raw_height_px': raw_feats['height'],
                'raw_girth_px': raw_feats['girth'],
                'raw_area_px': raw_feats['area'],
                'scale_cm_per_px': scale_cm_per_px,
                'cv_length_cm': calibrated_length_cm,
                'cv_height_cm': calibrated_height_cm,
                'cv_girth_cm': calibrated_girth_cm,
                'cv_area_cm2': calibrated_area_cm2
            }
            processed_rows.append(row_data)
            
    df_out = pd.DataFrame(processed_rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_out.to_csv(output_csv, index=False)
    print(f"\n==========================================")
    print(f"Successfully processed {len(df_out)} images for {df_out['animal_tag'].nunique()} unique cattle.")
    print(f"Saved feature dataset to: {output_csv}")
    print(f"==========================================\n")
    return df_out

if __name__ == "__main__":
    excel_path = r"c:\Users\NISARG\BAIF\data\BAIF_DATA.xlsx"
    images_dir = r"c:\Users\NISARG\BAIF\data\BAIF_IMAGES"
    output_csv = r"c:\Users\NISARG\BAIF\data\processed\baif_features.csv"
    process_baif_dataset(excel_path, images_dir, output_csv)
