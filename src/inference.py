"""
Inference pipeline for player behavior classification.

Loads the trained Logistic Regression pipeline and provides a clean interface
for predicting player behavior from gameplay telemetry.
"""

import numpy as np
import joblib
from typing import Dict, Any, Tuple, List
from pathlib import Path

from features import FEATURE_NAMES, CLASS_NAMES
from utils.paths import get_project_root


class BehaviorPredictor:
    """
    Wrapper around the saved ML pipeline for real-time behavior prediction.
    
    The pipeline contains a fitted StandardScaler and LogisticRegression classifier.
    No refitting occurs - the pipeline is used exactly as saved.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the predictor by loading the saved model.
        
        Args:
            model_path: Path to the saved pipeline joblib file.
                       Defaults to models/logistic_regression_tuned.joblib
        """
        if model_path is None:
            model_path = get_project_root() / "models" / "logistic_regression_tuned.joblib"
        
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Load the complete pipeline (scaler + classifier)
        self.pipeline = joblib.load(self.model_path)
        self.feature_names = FEATURE_NAMES
        self.class_names = CLASS_NAMES
        
        # Verify pipeline structure
        assert 'scaler' in self.pipeline.named_steps
        assert 'clf' in self.pipeline.named_steps
        assert hasattr(self.pipeline.named_steps['clf'], 'predict')
        
        print(f"Loaded model from {model_path}")
        print(f"Pipeline steps: {list(self.pipeline.named_steps.keys())}")
        print(f"Expected features: {len(self.feature_names)}")
        print(f"Classes: {self.class_names}")
    
    def validate_telemetry(self, telemetry: Dict[str, Any]) -> None:
        """
        Validate that telemetry contains all required features with valid values.
        
        Args:
            telemetry: Dictionary of feature name -> value
            
        Raises:
            ValueError: If features are missing, extra, or out of range
        """
        # Check for missing features
        missing = set(self.feature_names) - set(telemetry.keys())
        if missing:
            raise ValueError(f"Missing required features: {sorted(missing)}")
        
        # Check for extra features
        extra = set(telemetry.keys()) - set(self.feature_names)
        if extra:
            raise ValueError(f"Unexpected features: {sorted(extra)}")
        
        # Validate each feature value
        for name in self.feature_names:
            value = telemetry[name]
            if not isinstance(value, (int, float, np.number)):
                raise ValueError(f"Feature '{name}' must be numeric, got {type(value)}")
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Feature '{name}' must be in [0, 1], got {value}")
    
    def predict_behavior(self, telemetry: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict player behavior from gameplay telemetry.
        
        Args:
            telemetry: Dictionary with 14 features matching FEATURE_NAMES
            
        Returns:
            Dictionary containing:
            - predicted_behavior: str (Aggressive, Balanced, Defensive, Explorer)
            - predicted_class: int (0-3)
            - probabilities: dict mapping class name -> probability (if available)
            - confidence: float (max probability)
        """
        # Validate input
        self.validate_telemetry(telemetry)
        
        # Construct feature array in correct order
        features = np.array([[telemetry[name] for name in self.feature_names]])
        
        # Predict using the pipeline (handles scaling internally)
        pred_encoded = self.pipeline.predict(features)[0]
        
        # Decode prediction
        predicted_behavior = self.class_names[pred_encoded]
        
        # Get probabilities if available
        probabilities = {}
        confidence = 1.0
        if hasattr(self.pipeline.named_steps['clf'], 'predict_proba'):
            proba = self.pipeline.predict_proba(features)[0]
            probabilities = {self.class_names[i]: float(proba[i]) for i in range(len(proba))}
            confidence = float(max(proba))
        
        return {
            'predicted_behavior': predicted_behavior,
            'predicted_class': int(pred_encoded),
            'probabilities': probabilities,
            'confidence': confidence
        }


def predict_behavior(telemetry: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function for single predictions.
    
    Creates a predictor, runs prediction, and returns result.
    For repeated predictions, instantiate BehaviorPredictor once and reuse.
    
    Args:
        telemetry: Dictionary with 14 features matching FEATURE_NAMES
        
    Returns:
        Dictionary with predicted_behavior, predicted_class, probabilities, confidence
    """
    predictor = BehaviorPredictor()
    return predictor.predict_behavior(telemetry)


if __name__ == "__main__":
    # Quick test with a sample profile
    predictor = BehaviorPredictor()
    
    sample = {
        'combat_frequency': 0.8, 'damage_dealt': 0.75, 'damage_taken': 0.6,
        'kill_count': 0.7, 'death_count': 0.3, 'exploration_rate': 0.3,
        'distance_traveled': 0.5, 'resource_collection': 0.4,
        'ability_usage': 0.65, 'risk_taking': 0.75, 'defensive_actions': 0.2,
        'objective_focus': 0.6, 'social_interactions': 0.3, 'session_duration': 0.5
    }
    
    result = predictor.predict_behavior(sample)
    print(f"\nTest prediction: {result}")