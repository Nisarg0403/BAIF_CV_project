import pytest
import numpy as np
import os
from src.models.kan_regressor import (
    b_spline_basis,
    KANRegressor,
    get_or_create_animal_split
)

def test_b_spline_basis_expansion():
    dummy_col = np.array([0.5, 1.2, -0.8, 2.1])
    basis = b_spline_basis(dummy_col)
    assert basis.shape == (4, 5)

def test_kan_regressor_fit_and_predict():
    # Synthetic dataset simulating cattle morphometry: Length, Girth, Area, Height -> Weight
    np.random.seed(42)
    lengths = np.random.uniform(140, 170, 20)
    girths = np.random.uniform(160, 190, 20)
    heights = np.random.uniform(120, 145, 20)
    areas = lengths * heights * 0.65
    
    # Schaeffer-like ground truth weights
    weights = (girths**2 * lengths) / 10838.0 + np.random.normal(0, 5, 20)
    
    X = np.column_stack([lengths, girths, areas, heights])
    y = weights

    regressor = KANRegressor(in_features=4)
    regressor.fit(X, y, alpha_reg=1e-3)

    assert regressor.is_trained
    preds = regressor.predict(X[:5])
    assert len(preds) == 5
    assert np.all(preds > 200) and np.all(preds < 800)

def test_kan_ensemble_predict():
    regressor = KANRegressor(in_features=4)
    X_single = np.array([[150.0, 180.0, 14000.0, 135.0]])
    xgb_pred = 550.0

    # Alpha = 1.0 should return pure XGBoost prediction
    res_pure = regressor.ensemble_predict(xgb_pred, X_single, alpha=1.0)
    assert res_pure == 550.0

    # Alpha = 0.5 should average XGB and KAN
    res_ensemble = regressor.ensemble_predict(xgb_pred, X_single, alpha=0.5)
    assert isinstance(res_ensemble, float)

def test_animal_level_split_integrity():
    cattle_tags = [
        "TAG-105730429112", "TAG-105730429112",  # Cow 1 (2 photos)
        "TAG-105730428596", "TAG-105730428596",  # Cow 2 (2 photos)
        "TAG-103324191170", "TAG-103324191170",  # Cow 3 (2 photos)
        "TAG-105730428574", "TAG-105730428574",  # Cow 4 (2 photos)
        "TAG-103324191452", "TAG-103324191452"   # Cow 5 (2 photos)
    ]
    train_tags, test_tags = get_or_create_animal_split(cattle_tags, test_ratio=0.2)
    
    # Assert zero overlap between train and test tags
    overlap = set(train_tags).intersection(set(test_tags))
    assert len(overlap) == 0
    assert len(train_tags) + len(test_tags) == 5
