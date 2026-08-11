"""
Synthetic gameplay telemetry generation for player behavior classification.

Generates realistic, non-trivially-separable data with:
- 4 behavioral classes: Aggressive, Defensive, Explorer, Balanced
- 14 gameplay-derived features with correlations and noise
- Configurable class overlap and noise levels
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from features import FEATURE_NAMES, CLASS_NAMES
from utils.paths import get_project_root, load_config

# Order must match FEATURE_NAMES in features.py
BEHAVIOR_CLASSES = ['Aggressive', 'Defensive', 'Explorer', 'Balanced']


def generate_synthetic_telemetry(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Generate synthetic player telemetry data with realistic correlations and overlap.
    
    Features represent gameplay metrics:
    - combat_frequency: How often player engages in combat
    - damage_dealt: Average damage per encounter
    - damage_taken: Average damage received
    - kill_count: Enemies defeated per session
    - death_count: Times player died
    - exploration_rate: Map coverage percentage
    - distance_traveled: Total units moved
    - resource_collection: Resources gathered per minute
    - ability_usage: Special ability activations per minute
    - risk_taking: High-risk actions (low health fights, etc.)
    - defensive_actions: Blocking, dodging, shielding frequency
    - objective_focus: Primary objective completion rate
    - social_interactions: Co-op/trade/chat frequency
    - session_duration: Minutes per play session
    """
    np.random.seed(config['random_seed'])
    
    n_per_class = config['data_generation']['n_samples_per_class']
    n_features = config['data_generation']['n_features']
    noise_level = config['data_generation']['noise_level']
    class_overlap = config['data_generation']['class_overlap']
    
    # Base feature means for each class (designed to create overlap)
    # Each row is a class, each column is a feature
    class_means = np.array([
        # Aggressive: high combat, damage, kills, risk; low defense, exploration
        [0.85, 0.80, 0.65, 0.75, 0.40, 0.30, 0.55, 0.45, 0.70, 0.80, 0.20, 0.60, 0.35, 0.50],
        # Defensive: low combat, high defense, low risk; moderate exploration
        [0.25, 0.30, 0.25, 0.20, 0.15, 0.45, 0.40, 0.35, 0.30, 0.20, 0.85, 0.40, 0.30, 0.55],
        # Explorer: low combat, high exploration, distance, resources; moderate social
        [0.30, 0.35, 0.30, 0.25, 0.20, 0.85, 0.80, 0.75, 0.40, 0.35, 0.30, 0.50, 0.60, 0.65],
        # Balanced: moderate across all dimensions
        [0.55, 0.55, 0.50, 0.50, 0.30, 0.55, 0.55, 0.55, 0.50, 0.50, 0.50, 0.55, 0.50, 0.55],
    ])
    
    # Feature covariance matrix (creates realistic correlations)
    # Positive correlations: combat-kill-damage, exploration-distance-resources
    # Negative correlations: combat-exploration, risk-defense
    base_cov = np.eye(n_features) * 0.08
    
    # Add correlations
    corr_pairs = [
        (0, 1, 0.6),   # combat_freq <-> damage_dealt
        (0, 3, 0.55),  # combat_freq <-> kill_count
        (1, 3, 0.65),  # damage_dealt <-> kill_count
        (5, 6, 0.7),   # exploration_rate <-> distance_traveled
        (5, 7, 0.55),  # exploration_rate <-> resource_collection
        (6, 7, 0.6),   # distance <-> resources
        (9, 10, -0.5), # risk_taking <-> defensive_actions
        (0, 5, -0.4),  # combat <-> exploration
        (3, 4, 0.4),   # kills <-> deaths
        (8, 0, 0.45),  # ability_usage <-> combat
        (11, 0, 0.3),  # objective_focus <-> combat
        (12, 5, 0.35), # social <-> exploration
    ]
    
    for i, j, corr in corr_pairs:
        base_cov[i, j] = corr * 0.08
        base_cov[j, i] = corr * 0.08
    
    # Generate data for each class
    all_data = []
    all_labels = []
    
    for class_idx, class_name in enumerate(BEHAVIOR_CLASSES):
        # Add class-specific covariance variation
        class_cov = base_cov + np.eye(n_features) * class_overlap * 0.05
        
        # Generate samples
        samples = np.random.multivariate_normal(
            class_means[class_idx], 
            class_cov, 
            n_per_class
        )
        
        # Clip to valid range [0, 1] then add noise
        samples = np.clip(samples, 0, 1)
        noise = np.random.normal(0, noise_level * 0.1, samples.shape)
        samples = np.clip(samples + noise, 0, 1)
        
        all_data.append(samples)
        all_labels.extend([class_name] * n_per_class)
    
    # Combine and shuffle
    X = np.vstack(all_data)
    y = np.array(all_labels)
    
    shuffle_idx = np.random.permutation(len(y))
    X = X[shuffle_idx]
    y = y[shuffle_idx]
    
    # Feature names from single source of truth
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df['behavior_class'] = y
    
    return df


def save_data(df: pd.DataFrame, path: str):
    """Save generated data to CSV."""
    df.to_csv(path, index=False)
    print(f"Data saved to {path}")


if __name__ == "__main__":
    config = load_config()
    
    df = generate_synthetic_telemetry(config)
    output_path = get_project_root() / config['data_generation']['output_path']
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_data(df, str(output_path))
    print(f"Generated {len(df)} samples")
    print(df['behavior_class'].value_counts())