import os
import json
import logging
import pickle
import numpy as np
from typing import Tuple, Optional, Dict

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "logs", "experimental")
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger("kan_regressor")
if not logger.handlers:
    fh = logging.FileHandler(os.path.join(LOG_DIR, "kan_regressor.log"))
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models", "experimental")
os.makedirs(MODEL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "kan_regressor.pkl")

SPLIT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "splits")
os.makedirs(SPLIT_DIR, exist_ok=True)
SPLIT_PATH = os.path.join(SPLIT_DIR, "baif_train_test_split.json")


def b_spline_basis(x: np.ndarray, degree: int = 3) -> np.ndarray:
    """
    Computes B-Spline Basis Polynomial Expansion [x, x^2, x^3, SiLU(x), SiLU(x)^2].
    """
    silu_x = x / (1.0 + np.exp(-np.clip(x, -10, 10)))
    basis = [x, x**2, x**3, silu_x, silu_x**2]
    return np.column_stack(basis)


class KANRegressor:
    """
    Kolmogorov-Arnold Network (KAN) Regressor for Livestock Weight Estimation.
    Maps physical features [Length, Girth, Silhouette Area, Withers Height] -> Mass (kg).
    Uses B-Spline non-linear feature transformation with Ridge regularization.
    """
    def __init__(self, in_features: int = 4):
        self.in_features = in_features
        self.weights = None
        self.intercept = 0.0
        self.mean = np.zeros(in_features)
        self.std = np.ones(in_features)
        self.is_trained = False

        if os.path.exists(MODEL_PATH):
            self.load_model(MODEL_PATH)

    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        X_norm = (X - self.mean) / (self.std + 1e-8)
        feature_chunks = []
        for j in range(X_norm.shape[1]):
            col = X_norm[:, j]
            chunk = b_spline_basis(col)
            feature_chunks.append(chunk)
        return np.column_stack(feature_chunks)

    def fit(self, X: np.ndarray, y: np.ndarray, alpha_reg: float = 1e-3):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-8
        
        Phi = self._transform_features(X)
        N, D = Phi.shape
        
        # Ridge regression solve on spline basis matrix: (Phi^T Phi + alpha I)^-1 Phi^T y
        Phi_bias = np.column_stack([np.ones(N), Phi])
        I = np.eye(D + 1)
        I[0, 0] = 0.0  # Do not regularize bias term
        
        w_sol = np.linalg.solve(Phi_bias.T @ Phi_bias + alpha_reg * I, Phi_bias.T @ y)
        self.intercept = float(w_sol[0])
        self.weights = w_sol[1:]
        
        self.is_trained = True
        self.save_model(MODEL_PATH)
        
        preds = self.predict(X)
        mse = float(np.mean((preds - y)**2))
        logger.info(f"B-Spline KAN Regressor trained on {N} samples. MSE Loss: {mse:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained or self.weights is None:
            # Fallback to analytical volumetric equation: W = (Girth^2 * Length) / 10838.0
            lengths = X[:, 0]
            girths = X[:, 1]
            return (girths**2 * lengths) / 10838.0

        Phi = self._transform_features(X)
        preds = self.intercept + Phi @ self.weights
        return np.array(preds, dtype=float)

    def ensemble_predict(self, xgb_pred: float, X_single: np.ndarray, alpha: float = 1.0) -> float:
        """
        Calculates weighted ensemble prediction: final_weight = alpha * XGB + (1 - alpha) * KAN.
        Default alpha = 1.0 maintains pure XGBoost baseline output when KAN is default/disabled.
        """
        if alpha >= 1.0:
            return float(round(xgb_pred, 1))

        kan_pred = float(self.predict(X_single)[0])
        if np.isnan(kan_pred):
            return float(round(xgb_pred, 1))

        final_weight = alpha * float(xgb_pred) + (1.0 - alpha) * kan_pred
        return float(round(final_weight, 1))

    def save_model(self, path: str):
        state = {
            "weights": self.weights,
            "intercept": self.intercept,
            "mean": self.mean,
            "std": self.std,
            "is_trained": self.is_trained
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load_model(self, path: str):
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.weights = state["weights"]
            self.intercept = state["intercept"]
            self.mean = state["mean"]
            self.std = state["std"]
            self.is_trained = state.get("is_trained", True)
        except Exception as e:
            logger.warning(f"Could not load KAN model state from {path}: {e}")
            self.is_trained = False


def get_or_create_animal_split(cattle_tags: list, test_ratio: float = 0.2) -> Tuple[list, list]:
    """
    Ensures Animal-Level Split Integrity.
    All images belonging to a single animal ID are placed exclusively in train OR test set.
    """
    if os.path.exists(SPLIT_PATH):
        try:
            with open(SPLIT_PATH, "r") as f:
                split_data = json.load(f)
                return split_data["train_tags"], split_data["test_tags"]
        except Exception as e:
            logger.warning(f"Could not read split file: {e}. Recreating split.")

    unique_tags = list(set(cattle_tags))
    unique_tags.sort()
    np.random.seed(42)
    np.random.shuffle(unique_tags)

    num_test = max(1, int(len(unique_tags) * test_ratio))
    test_tags = unique_tags[:num_test]
    train_tags = unique_tags[num_test:]

    split_data = {
        "train_tags": train_tags,
        "test_tags": test_tags,
        "split_type": "animal_level_split"
    }

    with open(SPLIT_PATH, "w") as f:
        json.dump(split_data, f, indent=2)

    logger.info(f"Persisted animal-level split: {len(train_tags)} train cows, {len(test_tags)} test cows.")
    return train_tags, test_tags
