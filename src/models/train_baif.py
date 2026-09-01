import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_predictions(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    r2 = r2_score(y_true, y_pred)
    corr = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0.0
    
    # Accuracy bands
    abs_pct_err = np.abs((y_true - y_pred) / y_true) * 100
    within_5 = np.mean(abs_pct_err <= 5.0) * 100
    within_10 = np.mean(abs_pct_err <= 10.0) * 100
    within_15 = np.mean(abs_pct_err <= 15.0) * 100
    
    return {
        'MAE (kg)': mae,
        'RMSE (kg)': rmse,
        'MAPE (%)': mape,
        'R2 Score': r2,
        'Pearson r': corr,
        'Within 5% (%)': within_5,
        'Within 10% (%)': within_10,
        'Within 15% (%)': within_15
    }

def train_baif_models(features_csv, models_dir):
    if not os.path.exists(features_csv):
        raise FileNotFoundError(f"Feature dataset not found: {features_csv}")
        
    df = pd.read_csv(features_csv)
    print(f"Loaded {len(df)} feature records across {df['animal_tag'].nunique()} unique cattle.")
    
    # Define features and target
    y = df['tape_weight_kg'].values
    groups = df['animal_tag'].values
    
    # Feature sets
    X_manual = df[['tape_length_cm', 'tape_girth_cm', 'tape_withers_height_cm']].values
    X_cv = df[['cv_length_cm', 'cv_girth_cm', 'cv_area_cm2', 'tape_withers_height_cm']].values
    
    # Group K-Fold Cross Validation (5 folds for 15 animals)
    n_splits = min(5, df['animal_tag'].nunique())
    gkf = GroupKFold(n_splits=n_splits)
    
    # 1. Schaeffer's Formula Baseline
    schaeffer_preds = (df['tape_girth_cm']**2 * df['tape_length_cm']) / 10838.0
    schaeffer_metrics = evaluate_predictions(y, schaeffer_preds.values)
    
    # Model containers
    model_types = {
        'Upper-Bound Manual Gradient Boosting': (X_manual, GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)),
        'Primary CV-Extracted Gradient Boosting': (X_cv, GradientBoostingRegressor(n_estimators=50, max_depth=3, learning_rate=0.05, random_state=42)),
        'Primary CV-Extracted Random Forest': (X_cv, RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)),
        'Primary CV-Extracted Ridge Regression': (X_cv, Ridge(alpha=1.0))
    }
    
    cv_results = []
    
    # Add Schaeffer baseline
    schaeffer_metrics['Model'] = 'Baseline 1: Schaeffer Formula (Manual Tape)'
    cv_results.append(schaeffer_metrics)
    
    trained_models = {}
    
    for name, (X, model_obj) in model_types.items():
        oof_preds = np.zeros(len(df))
        
        for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups)):
            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            
            # Clone model
            from sklearn.base import clone
            fold_model = clone(model_obj)
            fold_model.fit(X_train, y_train)
            
            oof_preds[val_idx] = fold_model.predict(X_val)
            
        metrics = evaluate_predictions(y, oof_preds)
        metrics['Model'] = name
        cv_results.append(metrics)
        
        # Train final model on full dataset
        final_model = clone(model_obj)
        final_model.fit(X, y)
        trained_models[name] = final_model
        
    results_df = pd.DataFrame(cv_results)
    cols_order = ['Model', 'MAE (kg)', 'RMSE (kg)', 'MAPE (%)', 'R2 Score', 'Pearson r', 'Within 5% (%)', 'Within 10% (%)', 'Within 15% (%)']
    results_df = results_df[cols_order]
    
    print("\n=========================================================================================")
    print("                      BAIF REAL DATA MODEL BENCHMARK RESULTS (GROUP K-FOLD CV)            ")
    print("=========================================================================================")
    print(results_df.to_string(index=False))
    print("=========================================================================================\n")
    
    # Save the primary CV Gradient Boosting model as best regressor
    os.makedirs(models_dir, exist_ok=True)
    best_model_path = os.path.join(models_dir, "baif_best_weight_regressor.pkl")
    with open(best_model_path, 'wb') as f:
        pickle.dump(trained_models['Primary CV-Extracted Gradient Boosting'], f)
    print(f"Saved primary BAIF Gradient Boosting model to: {best_model_path}")
    
    # Also save the results summary to reports
    reports_dir = r"c:\Users\NISARG\BAIF\reports"
    os.makedirs(reports_dir, exist_ok=True)
    results_df.to_csv(os.path.join(reports_dir, "baif_model_benchmark_results.csv"), index=False)
    print(f"Saved benchmark metrics report to: {os.path.join(reports_dir, 'baif_model_benchmark_results.csv')}")
    
    return results_df

if __name__ == "__main__":
    features_csv = r"c:\Users\NISARG\BAIF\data\processed\baif_features.csv"
    models_dir = r"c:\Users\NISARG\BAIF\models"
    train_baif_models(features_csv, models_dir)
