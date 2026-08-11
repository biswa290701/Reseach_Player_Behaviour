"""
Evaluation utilities for player behavior classification.
Includes metrics computation, confusion matrices, feature importance, and cross-validation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.inspection import permutation_importance
from typing import Dict, Any, List
import os

from features import CLASS_NAMES


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                    average: str = 'weighted') -> Dict[str, float]:
    """Compute standard classification metrics."""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average=average, zero_division=0),
        'recall': recall_score(y_true, y_pred, average=average, zero_division=0),
        'f1': f1_score(y_true, y_pred, average=average, zero_division=0)
    }


def evaluate_models(models: Dict[str, Any], X_test: np.ndarray, 
                    y_test: np.ndarray) -> Dict[str, Dict]:
    """Evaluate all models on test set."""
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = compute_metrics(y_test, y_pred)
    return results


def plot_confusion_matrices(models: Dict[str, Any], X_test: np.ndarray,
                            y_test: np.ndarray, config: Dict[str, Any]):
    """Plot confusion matrices for all models."""
    class_names = CLASS_NAMES
    n_models = len(models)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Normalize by row (true class)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    ax=axes[idx], cbar=idx == 3)
        axes[idx].set_title(f'{name}')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('True')
    
    plt.tight_layout()
    save_path = os.path.join(config['paths']['figures_dir'], 'confusion_matrices.png')
    plt.savefig(save_path, dpi=150)
    plt.show()


def plot_feature_importance(models: Dict[str, Any], X_test: np.ndarray,
                            y_test: np.ndarray, config: Dict[str, Any]):
    """Plot permutation feature importance for tree-based models."""
    from features import get_feature_names
    
    feature_names = get_feature_names()
    
    # Only compute for models with feature_importances_ or use permutation
    tree_models = {k: v for k, v in models.items() 
                   if k in ['Random Forest', 'XGBoost']}
    
    if not tree_models:
        print("No tree-based models available for feature importance")
        return
    
    n_models = len(tree_models)
    fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
    if n_models == 1:
        axes = [axes]
    
    for idx, (name, model) in enumerate(tree_models.items()):
        # Permutation importance
        result = permutation_importance(
            model, X_test, y_test, 
            n_repeats=10, random_state=config['random_seed'],
            n_jobs=-1
        )
        
        importances = result.importances_mean
        std = result.importances_std
        
        # Sort by importance
        sorted_idx = np.argsort(importances)[::-1]
        
        axes[idx].barh(range(len(importances)), importances[sorted_idx], 
                       xerr=std[sorted_idx], align='center')
        axes[idx].set_yticks(range(len(importances)))
        axes[idx].set_yticklabels([feature_names[i] for i in sorted_idx])
        axes[idx].set_xlabel('Permutation Importance')
        axes[idx].set_title(f'{name} - Feature Importance')
        axes[idx].invert_yaxis()
    
    plt.tight_layout()
    save_path = os.path.join(config['paths']['figures_dir'], 'feature_importance.png')
    plt.savefig(save_path, dpi=150)
    plt.show()


def cross_validate_models(models: Dict[str, Any], X: np.ndarray, 
                          y: np.ndarray, config: Dict[str, Any]) -> Dict[str, Dict]:
    """Perform stratified k-fold cross-validation for all models."""
    cv_config = config['cv']
    cv = StratifiedKFold(
        n_splits=cv_config['n_splits'],
        shuffle=cv_config['shuffle'],
        random_state=config['random_seed']
    )
    
    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    results = {}
    
    for name, model in models.items():
        print(f"Cross-validating {name}...")
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        
        results[name] = {
            'accuracy': cv_results['test_accuracy'],
            'precision': cv_results['test_precision_weighted'],
            'recall': cv_results['test_recall_weighted'],
            'f1': cv_results['test_f1_weighted']
        }
    
    return results


def print_classification_reports(models: Dict[str, Any], X_test: np.ndarray,
                                 y_test: np.ndarray):
    """Print detailed classification report for each model."""
    class_names = CLASS_NAMES
    
    for name, model in models.items():
        y_pred = model.predict(X_test)
        print(f"\n{'='*50}")
        print(f"{name} - Classification Report")
        print(f"{'='*50}")
        print(classification_report(y_test, y_pred, target_names=class_names, 
                                    zero_division=0))


def plot_cv_results(cv_results: Dict[str, Dict], config: Dict[str, Any]):
    """Plot cross-validation results comparison."""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    model_names = list(cv_results.keys())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics):
        means = [cv_results[m][metric].mean() for m in model_names]
        stds = [cv_results[m][metric].std() for m in model_names]
        
        bars = axes[idx].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7)
        axes[idx].set_title(f'{metric.capitalize()} (CV)')
        axes[idx].set_ylabel('Score')
        axes[idx].set_ylim(0, 1.05)
        axes[idx].tick_params(axis='x', rotation=15)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{mean:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    save_path = os.path.join(config['paths']['figures_dir'], 'cv_comparison.png')
    plt.savefig(save_path, dpi=150)
    plt.show()


if __name__ == "__main__":
    import yaml
    from data_generation import generate_synthetic_telemetry
    from features import prepare_features_and_split
    from models import train_all_models
    
    with open('../config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    df = generate_synthetic_telemetry(config)
    X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)
    
    models, _ = train_all_models(X_train, y_train, X_val, y_val, config)
    
    # Test evaluation
    test_results = evaluate_models(models, X_test, y_test)
    for name, metrics in test_results.items():
        print(f"{name}: {metrics}")
    
    # Cross-validation
    cv_results = cross_validate_models(models, X_train, y_train, config)
    plot_cv_results(cv_results, config)
    
    # Feature importance
    plot_feature_importance(models, X_test, y_test, config)