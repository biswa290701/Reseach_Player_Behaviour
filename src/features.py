"""
Feature engineering and data splitting for player behavior classification.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Dict, Any, Tuple, List


# Single source of truth for feature names and order
FEATURE_NAMES = [
    'combat_frequency', 'damage_dealt', 'damage_taken', 'kill_count',
    'death_count', 'exploration_rate', 'distance_traveled', 'resource_collection',
    'ability_usage', 'risk_taking', 'defensive_actions', 'objective_focus',
    'social_interactions', 'session_duration'
]

CLASS_NAMES = ['Aggressive', 'Balanced', 'Defensive', 'Explorer']


def get_feature_names() -> List[str]:
    """Return list of feature names in order."""
    return FEATURE_NAMES.copy()


def get_class_names() -> List[str]:
    """Return list of class names in label encoding order."""
    return CLASS_NAMES.copy()


def prepare_features_and_split(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple:
    """
    Prepare features, encode labels, and split data with NO data leakage.
    
    Critical: Split FIRST, then fit scaler ONLY on training data.
    
    Args:
        df: DataFrame with features and 'behavior_class' column
        config: Configuration dictionary
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test, label_encoder
    """
    # Verify feature columns exist
    feature_cols = [c for c in df.columns if c != 'behavior_class']
    assert feature_cols == FEATURE_NAMES, f"Feature mismatch: {feature_cols} vs {FEATURE_NAMES}"
    
    X = df[feature_cols].values
    y = df['behavior_class'].values
    
    # ---- SPLIT FIRST (no leakage) ----
    test_size = config['features']['test_size']
    val_size = config['features']['val_size']
    random_seed = config['random_seed']
    
    # First split: train+val vs test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )
    
    # Second split: train vs val (from remaining data)
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=random_seed, stratify=y_temp
    )
    
    # ---- ENCODE LABELS (fit on train only) ----
    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)
    
    # Verify all classes present in training
    train_classes = set(label_encoder.classes_)
    all_classes = set(CLASS_NAMES)
    assert train_classes == all_classes, f"Missing classes in train: {all_classes - train_classes}"
    
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)
    
    # ---- SCALE FEATURES (fit on train only) ----
    if config['features']['scale_features']:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    
    return X_train, X_val, X_test, y_train_enc, y_val_enc, y_test_enc, label_encoder, scaler


def decode_predictions(y_pred_encoded: np.ndarray, label_encoder: LabelEncoder) -> np.ndarray:
    """Convert encoded predictions back to original class names."""
    return label_encoder.inverse_transform(y_pred_encoded)


if __name__ == "__main__":
    from data_generation import generate_synthetic_telemetry
    from utils.paths import load_config
    
    config = load_config()
    
    df = generate_synthetic_telemetry(config)
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Classes: {le.classes_}")
    print(f"Feature names: {get_feature_names()}")
    print(f"Class names: {get_class_names()}")