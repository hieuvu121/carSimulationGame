"""Classifier Protocol + GestureClassifier (load model, predict()).

Owner: Siri (Classifier).
Specification: docs/interfaces.md §4.5.
"""
import joblib
import numpy as np
from pathlib import Path
from typing import Protocol

class Classifier(Protocol):
    """Protocol defining how the game interacts with the classifier."""
    def predict(self, features: np.ndarray) -> str:
        ...

class GestureClassifier:
    """Loads the trained SVM and makes real-time predictions."""
    
    def __init__(self, model_path: Path | str = "models/classifier.pkl"):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
        self.model = joblib.load(path)
        
    def predict(self, features: np.ndarray) -> str:
        """Predict the gesture for a single frame of landmarks."""
        # Ensure 2D array format required by scikit-learn
        if features.ndim == 1:
            features = features.reshape(1, -1)
            
        prediction = self.model.predict(features)
        return prediction[0]