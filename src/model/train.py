"""Build_pipeline(), tune (LOUO over train users), train_model(), save.

Owner: Siri (Classifier).
Specification: docs/interfaces.md §4.5.
"""
import joblib
import numpy as np
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneGroupOut, GridSearchCV

from src.config import SPLITS_PATH
from src.data.dataset import load_dataset
from src.data.split import load_splits

def build_pipeline() -> Pipeline:
    """Create a machine learning pipeline with scaling and an SVM."""
    return Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(probability=True, random_state=42))
    ])

def train_model() -> None:
    """Train and tune the SVM classifier, then save it to disk."""
    print("Loading dataset...")
    # Load dataset splits and excluded files
    splits = load_splits(SPLITS_PATH)
    dataset, _ = load_dataset(excluded_files=splits.excluded_files)
    
    # Filter for only the training users
    train_mask = np.isin(dataset.groups, splits.train_users)
    X_train = dataset.X[train_mask]
    y_train = dataset.y[train_mask]
    groups_train = dataset.groups[train_mask]
    
    print(f"Training on {len(X_train)} samples...")
    
    pipeline = build_pipeline()
    
    # Hyperparameter tuning setup
    param_grid = {
        'svm__C': [0.1, 1, 10],
        'svm__kernel': ['linear', 'rbf']
    }
    
    print("Tuning hyperparameters (Leave-One-Group-Out CV)...")
    logo = LeaveOneGroupOut()
    search = GridSearchCV(pipeline, param_grid, cv=logo, n_jobs=-1, verbose=1)
    
    # Fit the model using groups for cross-validation
    search.fit(X_train, y_train, groups=groups_train)
    
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV accuracy: {search.best_score_:.3f}")
    
    # Save the finalized model
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "classifier.pkl"
    
    joblib.dump(search.best_estimator_, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    train_model()