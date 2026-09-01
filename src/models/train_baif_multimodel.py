import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.multioutput import MultiOutputRegressor

def train_baif_multimodel_pack(features_csv, models_dir):
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Feature file not found: {features_csv}")
        
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} records for multi-model training...")
    
    # 1. Feature Engineering for Image-to-Measurement Translation
    # Input features extracted from mask silhouette:
    # Aspect Ratio, Bounding Box Length, Height, Area, Thickness
    df['aspect_ratio'] = df['raw_length_px'] / df['raw_height_px']
    df['normalized_area'] = df['raw_area_px'] / (df['raw_length_px'] * df['raw_height_px'])
    
    X_mask_features = df[['raw_length_px', 'raw_height_px', 'raw_area_px', 'aspect_ratio', 'normalized_area']].values
    
    # Ground-truth physical dimensions from BAIF dataset
    Y_physical = df[['tape_length_cm', 'tape_withers_height_cm', 'tape_girth_cm']].values
    # Note: Stature height is roughly Withers Height + 3 cm based on dataset averages
    
    # Target weight
    y_weight = df['tape_weight_kg'].values
    
    # 2. Train Physical Measurement Estimators (Translates Image Silhouette -> Physical CM)
    measurement_estimator = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=42))
    measurement_estimator.fit(X_mask_features, Y_physical)
    
    # Predict physical dimensions for training weight regressors
    predicted_physical = measurement_estimator.predict(X_mask_features)
    df['pred_length_cm'] = predicted_physical[:, 0]
    df['pred_withers_height_cm'] = predicted_physical[:, 1]
    df['pred_girth_cm'] = predicted_physical[:, 2]
    df['pred_stature_height_cm'] = df['pred_withers_height_cm'] + 3.2  # Real stature ratio offset
    df['pred_area_cm2'] = df['pred_length_cm'] * df['pred_withers_height_cm'] * df['normalized_area']
    
    # Features for Weight Regressors
    X_calibrated = df[['pred_length_cm', 'pred_girth_cm', 'pred_area_cm2', 'pred_withers_height_cm']].values
    
    # 3. Train Multiple Weight Regressors
    weight_models = {
        'Gradient Boosting Regressor': GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42),
        'Ridge Linear Regression': Ridge(alpha=1.0)
    }
    
    fitted_weight_models = {}
    for name, model in weight_models.items():
        model.fit(X_calibrated, y_weight)
        fitted_weight_models[name] = model
        print(f"Trained weight model: {name}")
        
    # 4. Save Multi-Model Pack
    pack = {
        'measurement_estimator': measurement_estimator,
        'weight_models': fitted_weight_models,
        'feature_names': ['raw_length_px', 'raw_height_px', 'raw_area_px', 'aspect_ratio', 'normalized_area']
    }
    
    os.makedirs(models_dir, exist_ok=True)
    pack_path = os.path.join(models_dir, "baif_multimodel_pack.pkl")
    with open(pack_path, 'wb') as f:
        pickle.dump(pack, f)
        
    print(f"\n==========================================")
    print(f"Successfully saved Multi-Model Pack to: {pack_path}")
    print(f"==========================================\n")
    return pack

if __name__ == "__main__":
    features_csv = r"c:\Users\NISARG\BAIF\data\processed\baif_features.csv"
    models_dir = r"c:\Users\NISARG\BAIF\models"
    train_baif_multimodel_pack(features_csv, models_dir)
