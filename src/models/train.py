import os
import cv2
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

from src.segmentation.segment import CowSegmenter
from src.features.morphometry import extract_morphometric_features

def generate_dataset(cropped_dir, max_samples=150):
    """
    Processes a subset of cropped images, extracts features, and synthesizes realistic weights.
    """
    print(f"Extracting features from up to {max_samples} images...")
    files = [f for f in os.listdir(cropped_dir) if f.endswith('.jpg')]
    if len(files) == 0:
        raise ValueError("No cropped images found!")
        
    # Use a fixed random seed for reproducibility
    np.random.seed(42)
    selected_files = np.random.choice(files, min(len(files), max_samples), replace=False)
    
    segmenter = CowSegmenter()
    
    data = []
    for i, file in enumerate(selected_files):
        img_path = os.path.join(cropped_dir, file)
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        # 1. Segment cow
        mask = segmenter.segment(img)
        
        # 2. Extract features
        feats = extract_morphometric_features(mask)
        if feats['length'] == 0:
            continue
            
        # 3. Calibrate pixels to cm (approximate scale calibration)
        # Length in cm: pixel length * 0.4
        length_cm = feats['length'] * 0.4
        # Girth in cm: pixel girth * 0.7 (torso height surrogate)
        girth_cm = feats['girth'] * 0.7
        # Height in cm: pixel height * 0.5
        height_cm = feats['height'] * 0.5
        
        # 4. Synthesize target weight in kg using Schaeffer's Formula:
        # Weight (kg) = (Girth (cm)^2 * Length (cm)) / 10838
        # Add some Gaussian noise (std=12 kg) to represent biological and capture variations
        base_weight = (girth_cm ** 2 * length_cm) / 10838
        synthetic_weight = base_weight + np.random.normal(0, 12.0)
        # Keep weight in a realistic minimum range
        synthetic_weight = max(150.0, synthetic_weight)
        
        row = {
            'image_id': file,
            'length_px': feats['length'],
            'height_px': feats['height'],
            'area_px': feats['area'],
            'girth_px': feats['girth'],
            'length_cm': length_cm,
            'height_cm': height_cm,
            'girth_cm': girth_cm,
            'weight_kg': synthetic_weight
        }
        data.append(row)
        
        if (i + 1) % 30 == 0:
            print(f"Processed {i + 1}/{len(selected_files)} images...")
            
    df = pd.DataFrame(data)
    # Save the feature table
    os.makedirs(r"c:\Users\NISARG\BAIF\data\processed", exist_ok=True)
    df.to_csv(r"c:\Users\NISARG\BAIF\data\processed\bristol_features.csv", index=False)
    print(f"Saved extracted features and synthetic weights for {len(df)} samples.")
    return df

def train_and_evaluate(df):
    X = df[['length_px', 'height_px', 'area_px', 'girth_px']]
    y = df['weight_kg']
    
    # Train-test split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1000, epsilon=1.0),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    }
    
    results = []
    best_r2 = -float('inf')
    best_model = None
    best_model_name = ""
    
    for name, model in models.items():
        # Train
        model.fit(X_train, y_train)
        
        # Predict
        preds = model.predict(X_val)
        
        # Evaluate
        mae = mean_absolute_error(y_val, preds)
        rmse = np.sqrt(mean_squared_error(y_val, preds))
        r2 = r2_score(y_val, preds)
        corr = np.corrcoef(y_val, preds)[0, 1] if len(y_val) > 1 else 0.0
        
        results.append({
            'Model': name,
            'MAE (kg)': mae,
            'RMSE (kg)': rmse,
            'R2 Score': r2,
            'Correlation': corr
        })
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            best_model_name = name
            
    results_df = pd.DataFrame(results)
    print("\n--- Model Performance Comparison ---")
    print(results_df.to_string(index=False))
    
    # Save best model
    models_dir = r"c:\Users\NISARG\BAIF\models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "best_weight_regressor.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"\nBest model: {best_model_name} (R2={best_r2:.4f})")
    print(f"Saved best model to {best_model_path}")
    
    # Also save the scaler or metadata if needed, here the models train directly on px features

if __name__ == "__main__":
    cropped_dir = r"c:\Users\NISARG\BAIF\data\processed\bristol_cropped"
    try:
        df = generate_dataset(cropped_dir, max_samples=150)
        train_and_evaluate(df)
    except Exception as e:
        print("Error during training pipeline:", e)
