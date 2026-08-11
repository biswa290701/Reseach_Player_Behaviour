import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from utils.paths import get_project_root, load_config
from features import prepare_features_and_split
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

config = load_config()
root = get_project_root()
df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)

print('=== DATA SPLIT (CV uses ONLY training set) ===')
print('Train (for CV):', X_train.shape[0], 'samples')
print('Validation (held out):', X_val.shape[0], 'samples')
print('Test (held out):', X_test.shape[0], 'samples')
print()

# Define models with pipelines for proper CV preprocessing
models = {}

# Logistic Regression with StandardScaler in pipeline
models['Logistic Regression'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42))
])

# SVM with StandardScaler in pipeline
models['SVM'] = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42))
])

# Random Forest - no scaling needed
models['Random Forest'] = Pipeline([
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, 
                                    min_samples_leaf=2, n_jobs=-1, random_state=42))
])

# Scoring metrics
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score, average='weighted', zero_division=0),
    'recall': make_scorer(recall_score, average='weighted', zero_division=0),
    'f1': make_scorer(f1_score, average='weighted', zero_division=0)
}

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print('=== 5-FOLD STRATIFIED CROSS-VALIDATION ===')
print('Concept: The 6,500 training samples are split into 5 folds (1,300 each).')
print('Each fold serves as validation once while the other 4 folds (5,200 samples) train.')
print('This gives 5 performance estimates per model, revealing stability across data subsets.')
print()

all_results = {}

for name, model in models.items():
    print(f'Running CV for {name}...')
    cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    
    all_results[name] = {}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        scores = cv_results[f'test_{metric}']
        mean_score = scores.mean()
        std_score = scores.std()
        all_results[name][metric] = {'mean': mean_score, 'std': std_score, 'fold_scores': scores}
        
        print(f'  {metric}: {mean_score:.4f} +/- {std_score:.4f}')
        print(f'    Fold scores: {", ".join([f"{s:.4f}" for s in scores])}')
    print()

# Save detailed results
results_rows = []
for name, metrics in all_results.items():
    for metric, vals in metrics.items():
        for fold_idx, score in enumerate(vals['fold_scores']):
            results_rows.append({
                'Model': name,
                'Metric': metric,
                'Fold': fold_idx + 1,
                'Score': score
            })
        results_rows.append({
            'Model': name,
            'Metric': metric,
            'Fold': 'mean',
            'Score': vals['mean']
        })
        results_rows.append({
            'Model': name,
            'Metric': metric,
            'Fold': 'std',
            'Score': vals['std']
        })

results_df = pd.DataFrame(results_rows)
models_dir = root / 'models'
models_dir.mkdir(exist_ok=True)
results_df.to_csv(models_dir / 'cross_validation_results.csv', index=False)
print('Saved detailed results to models/cross_validation_results.csv')

# Summary table
summary_rows = []
for name in all_results.keys():
    row = {'Model': name}
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        row[f'{metric}_mean'] = all_results[name][metric]['mean']
        row[f'{metric}_std'] = all_results[name][metric]['std']
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
print()
print('=== SUMMARY TABLE (Mean +/- Std) ===')
print(summary_df.to_string(index=False))

# Compare with previous validation results
print()
print('=== COMPARISON WITH VALIDATION SET ===')
val_results = {
    'Logistic Regression': 0.7533,
    'SVM': 0.7533,
    'Random Forest': 0.7400
}
for name in all_results.keys():
    cv_acc = all_results[name]['accuracy']['mean']
    val_acc = val_results[name]
    diff = cv_acc - val_acc
    print(f'{name}: CV={cv_acc:.4f}, Val={val_acc:.4f}, Diff={diff:+.4f}')

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
metrics_list = ['accuracy', 'precision', 'recall', 'f1']

for idx, metric in enumerate(metrics_list):
    model_names = list(all_results.keys())
    means = [all_results[m][metric]['mean'] for m in model_names]
    stds = [all_results[m][metric]['std'] for m in model_names]
    
    bars = axes[idx].bar(model_names, means, yerr=stds, capsize=5, alpha=0.7, 
                          color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[idx].set_title(f'{metric.capitalize()} (5-fold CV)')
    axes[idx].set_ylabel('Score')
    axes[idx].set_ylim(0, 1.0)
    axes[idx].tick_params(axis='x', rotation=15)
    
    for bar, mean in zip(bars, means):
        axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                      f'{mean:.3f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
figures_dir = root / 'reports' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / 'cross_validation_comparison.png', dpi=150)
plt.close()
print()
print('Saved cross_validation_comparison.png to reports/figures/')

# Save summary table separately
summary_df.to_csv(models_dir / 'cv_summary.csv', index=False)
print('Saved cv_summary.csv to models/')