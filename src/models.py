"""
Model definitions and training for player behavior classification.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Dict, Any, Tuple
import joblib
import os


def create_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create all model instances with configured hyperparameters."""
    models = {}
    
    # Logistic Regression
    lr_cfg = config['models']['logistic_regression']
    models['Logistic Regression'] = LogisticRegression(
        max_iter=lr_cfg['max_iter'],
        C=lr_cfg['C'],
        solver=lr_cfg['solver'],
        random_state=config['random_seed'],
        n_jobs=-1
    )
    
    # SVM
    svm_cfg = config['models']['svm']
    models['SVM'] = SVC(
        kernel=svm_cfg['kernel'],
        C=svm_cfg['C'],
        gamma=svm_cfg['gamma'],
        probability=svm_cfg['probability'],
        random_state=config['random_seed']
    )
    
    # Random Forest
    rf_cfg = config['models']['random_forest']
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=rf_cfg['n_estimators'],
        max_depth=rf_cfg['max_depth'],
        min_samples_split=rf_cfg['min_samples_split'],
        min_samples_leaf=rf_cfg['min_samples_leaf'],
        n_jobs=rf_cfg['n_jobs'],
        random_state=config['random_seed']
    )
    
    # XGBoost
    xgb_cfg = config['models']['xgboost']
    models['XGBoost'] = XGBClassifier(
        n_estimators=xgb_cfg['n_estimators'],
        max_depth=xgb_cfg['max_depth'],
        learning_rate=xgb_cfg['learning_rate'],
        subsample=xgb_cfg['subsample'],
        colsample_bytree=xgb_cfg['colsample_bytree'],
        n_jobs=xgb_cfg['n_jobs'],
        eval_metric=xgb_cfg['eval_metric'],
        random_state=config['random_seed'],
        verbosity=0
    )
    
    return models


def train_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """Train a single model."""
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train: np.ndarray, y_train: np.ndarray, 
                     X_val: np.ndarray, y_val: np.ndarray,
                     config: Dict[str, Any]) -> Tuple[Dict, Dict]:
    """
    Train all models and return trained models + validation results.
    
    Returns:
        models: Dict of trained model instances
        results: Dict of validation metrics per model
    """
    from evaluation import compute_metrics
    
    models = create_models(config)
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model = train_model(model, X_train, y_train)
        
        # Validate
        y_pred = model.predict(X_val)
        metrics = compute_metrics(y_val, y_pred)
        results[name] = metrics
        
        print(f"  Val Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
    
    return models, results


def save_models(models: Dict[str, Any], config: Dict[str, Any]):
    """Save trained models to disk."""
    models_dir = config['paths']['models_dir']
    os.makedirs(models_dir, exist_ok=True)
    
    for name, model in models.items():
        filename = name.lower().replace(' ', '_') + '.joblib'
        path = os.path.join(models_dir, filename)
        joblib.dump(model, path)
        print(f"Saved {name} to {path}")


def load_models(config: Dict[str, Any]) -> Dict[str, Any]:
    """Load trained models from disk."""
    models_dir = config['paths']['models_dir']
    models = {}
    
    model_names = ['Logistic Regression', 'SVM', 'Random Forest', 'XGBoost']
    for name in model_names:
        filename = name.lower().replace(' ', '_') + '.joblib'
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            models[name] = joblib.load(path)
            print(f"Loaded {name} from {path}")
    
    return models


if __name__ == "__main__":
    import yaml
    from data_generation import generate_synthetic_telemetry
    from features import prepare_features_and_split
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = generate_synthetic_telemetry(config)
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)
    
    models, results = train_all_models(X_train, y_train, X_val, y_val, config)
    save_models(models, config)