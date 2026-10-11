"""One-time test-set evaluation, confusion matrix, per-condition tables.

Owner: Siri (Classifier).
Specification: docs/interfaces.md §4.5.
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from src.config import SPLITS_PATH
from src.data.dataset import load_dataset
from src.data.split import load_splits

def evaluate_model() -> None:
    """Load the reserved test data, evaluate the model, and print metrics."""
    print("Loading unseen test data...")
    splits = load_splits(SPLITS_PATH)
    dataset, _ = load_dataset(excluded_files=splits.excluded_files)
    
    # Filter for only the reserved test users
    test_mask = np.isin(dataset.groups, splits.test_users)
    X_test = dataset.X[test_mask]
    y_test = dataset.y[test_mask]
    
    print("Loading trained model...")
    model_path = Path("models/classifier.pkl")
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Please run train.py first.")
    
    model = joblib.load(model_path)
    
    print("Predicting test gestures...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n=== OVERALL TEST ACCURACY: {acc * 100:.2f}% ===")
    
    print("\n=== CONFUSION MATRIX ===")
    print(confusion_matrix(y_test, y_pred))
    
    print("\n=== CLASSIFICATION REPORT ===")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate_model()