import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    features_csv = r"c:\Users\NISARG\BAIF\data\processed\bristol_features.csv"
    model_pkl = r"c:\Users\NISARG\BAIF\models\best_weight_regressor.pkl"
    
    print("Loading features and model...")
    if not os.path.exists(features_csv):
        print(f"Features file {features_csv} not found! Please run train.py first.")
        return
    if not os.path.exists(model_pkl):
        print(f"Model file {model_pkl} not found! Please run train.py first.")
        return
        
    df = pd.read_csv(features_csv)
    with open(model_pkl, 'rb') as f:
        model = pickle.load(f)
        
    X = df[['length_px', 'height_px', 'area_px', 'girth_px']]
    y_actual = df['weight_kg']
    
    # Predict
    y_pred = model.predict(X)
    
    # Compute metrics
    mae = mean_absolute_error(y_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    r2 = r2_score(y_actual, y_pred)
    corr = np.corrcoef(y_actual, y_pred)[0, 1]
    
    print("\n==========================================")
    print("       FINAL EVALUATION REPORT            ")
    print("==========================================")
    print(f"Total samples evaluated: {len(df)}")
    print(f"Mean Absolute Error (MAE): {mae:.2f} kg")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} kg")
    print(f"Coefficient of Determination (R2): {r2:.4f}")
    print(f"Pearson Correlation Coefficient: {corr:.4f}")
    print("==========================================\n")
    
    # Plotting Predicted vs Actual
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="whitegrid")
    
    # Scatter plot
    sns.scatterplot(x=y_actual, y=y_pred, alpha=0.7, color='darkblue', edgecolor='w', s=80, label='Cattle Samples')
    
    # Reference diagonal line (y = x)
    min_val = min(y_actual.min(), y_pred.min()) - 10
    max_val = max(y_actual.max(), y_pred.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=2, label='Perfect Prediction (y=x)')
    
    plt.title('Cattle Body Weight Estimation: Predicted vs Actual Weight', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Actual Weight (kg)', fontsize=12)
    plt.ylabel('Predicted Weight (kg)', fontsize=12)
    
    # Annotate metrics on plot
    stats_text = (f"MAE: {mae:.2f} kg\n"
                  f"RMSE: {rmse:.2f} kg\n"
                  f"R²: {r2:.3f}\n"
                  f"Corr: {corr:.3f}")
    plt.gca().text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=11,
                   verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.legend(loc='lower right', fontsize=11)
    
    # Save plot
    results_dir = r"c:\Users\NISARG\BAIF\results"
    os.makedirs(results_dir, exist_ok=True)
    plot_path = os.path.join(results_dir, "predicted_vs_actual.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved evaluation plot to: {plot_path}")

if __name__ == "__main__":
    main()
