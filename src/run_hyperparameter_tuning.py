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
from sklearn.model_selection import StratifiedKFold, GridSearchCV

config = load_config()
root = get_project_root()
df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)

print('=== DATA SPLIT (Tuning uses ONLY training set) ===')
print('Train (for tuning CV):', X_train.shape[0], 'samples')
print('Validation (held out):', X_val.shape[0], 'samples')
print('Test (held out):', X_test.shape[0], 'samples')
print()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'f1_weighted'

# ============================================================
# LOGISTIC REGRESSION
# ============================================================
print('=== LOGISTIC REGRESSION TUNING ===')
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='lbfgs', max_iter=2000, random_state=42))
])

lr_param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l2']
}

lr_grid = GridSearchCV(
    lr_pipe, lr_param_grid, cv=cv, scoring=scoring,
    n_jobs=-1, refit=True, verbose=1, return_train_score=False
)
lr_grid.fit(X_train, y_train)

print(f'Best params: {lr_grid.best_params_}')
print(f'Best CV F1: {lr_grid.best_score_:.4f}')

# Get CV results for best params - compute accuracy from best estimator
lr_best = lr_grid.best_estimator_
# We need to manually get accuracy for best params
from sklearn.model_selection import cross_validate
lr_cv_results = cross_validate(lr_best, X_train, y_train, cv=cv, 
                                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], n_jobs=-1)
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    mean = lr_cv_results[f'test_{metric}'].mean()
    std = lr_cv_results[f'test_{metric}'].std()
    print(f'  {metric}: {mean:.4f} +/- {std:.4f}')

# Save tuned model
models_dir = root / 'models'
models_dir.mkdir(exist_ok=True)
joblib.dump(lr_best, models_dir / 'logistic_regression_tuned.joblib')
print('Saved logistic_regression_tuned.joblib')

# Save all CV results
lr_results_df = pd.DataFrame(lr_grid.cv_results_)
lr_results_df.to_csv(models_dir / 'lr_tuning_results.csv', index=False)
print('Saved lr_tuning_results.csv')

lr_n_combos = len(lr_grid.cv_results_['params'])
print(f'Total combinations tested: {lr_n_combos}')
print()

lr_best_metrics = {m: lr_cv_results[f'test_{m}'].mean() for m in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']}

# ============================================================
# SVM
# ============================================================
print('=== SVM TUNING ===')
svm_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(random_state=42))
])

svm_param_grid = {
    'clf__C': [0.1, 1, 10, 100],
    'clf__gamma': ['scale', 0.01, 0.1, 1],
    'clf__kernel': ['rbf']
}

svm_grid = GridSearchCV(
    svm_pipe, svm_param_grid, cv=cv, scoring=scoring,
    n_jobs=-1, refit=True, verbose=1, return_train_score=False
)
svm_grid.fit(X_train, y_train)

print(f'Best params: {svm_grid.best_params_}')
print(f'Best CV F1: {svm_grid.best_score_:.4f}')

svm_best = svm_grid.best_estimator_
svm_cv_results = cross_validate(svm_best, X_train, y_train, cv=cv,
                                 scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], n_jobs=-1)
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    mean = svm_cv_results[f'test_{metric}'].mean()
    std = svm_cv_results[f'test_{metric}'].std()
    print(f'  {metric}: {mean:.4f} +/- {std:.4f}')

joblib.dump(svm_best, models_dir / 'svm_tuned.joblib')
print('Saved svm_tuned.joblib')

svm_results_df = pd.DataFrame(svm_grid.cv_results_)
svm_results_df.to_csv(models_dir / 'svm_tuning_results.csv', index=False)
print('Saved svm_tuning_results.csv')

svm_n_combos = len(svm_grid.cv_results_['params'])
print(f'Total combinations tested: {svm_n_combos}')
print()

svm_best_metrics = {m: svm_cv_results[f'test_{m}'].mean() for m in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']}

# ============================================================
# RANDOM FOREST
# ============================================================
print('=== RANDOM FOREST TUNING ===')
rf_pipe = Pipeline([
    ('clf', RandomForestClassifier(random_state=42, n_jobs=-1))
])

rf_param_grid = {
    'clf__n_estimators': [100, 300],
    'clf__max_depth': [None, 10, 20],
    'clf__min_samples_split': [2, 5],
    'clf__min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(
    rf_pipe, rf_param_grid, cv=cv, scoring=scoring,
    n_jobs=-1, refit=True, verbose=1, return_train_score=False
)
rf_grid.fit(X_train, y_train)

print(f'Best params: {rf_grid.best_params_}')
print(f'Best CV F1: {rf_grid.best_score_:.4f}')

rf_best = rf_grid.best_estimator_
rf_cv_results = cross_validate(rf_best, X_train, y_train, cv=cv,
                                scoring=['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'], n_jobs=-1)
for metric in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']:
    mean = rf_cv_results[f'test_{metric}'].mean()
    std = rf_cv_results[f'test_{metric}'].std()
    print(f'  {metric}: {mean:.4f} +/- {std:.4f}')

joblib.dump(rf_best, models_dir / 'random_forest_tuned.joblib')
print('Saved random_forest_tuned.joblib')

rf_results_df = pd.DataFrame(rf_grid.cv_results_)
rf_results_df.to_csv(models_dir / 'rf_tuning_results.csv', index=False)
print('Saved rf_tuning_results.csv')

rf_n_combos = len(rf_grid.cv_results_['params'])
print(f'Total combinations tested: {rf_n_combos}')
print()

rf_best_metrics = {m: rf_cv_results[f'test_{m}'].mean() for m in ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']}

# ============================================================
# COMPARISON: BASELINE vs TUNED
# ============================================================
print('=== BASELINE vs TUNED COMPARISON ===')

# Baseline CV results from previous run
baseline = {
    'Logistic Regression': {'accuracy': 0.7532, 'f1': 0.7512},
    'SVM': {'accuracy': 0.7429, 'f1': 0.7416},
    'Random Forest': {'accuracy': 0.7363, 'f1': 0.7331}
}

tuned = {
    'Logistic Regression': {'accuracy': lr_best_metrics['accuracy'], 'f1': lr_best_metrics['f1_weighted']},
    'SVM': {'accuracy': svm_best_metrics['accuracy'], 'f1': svm_best_metrics['f1_weighted']},
    'Random Forest': {'accuracy': rf_best_metrics['accuracy'], 'f1': rf_best_metrics['f1_weighted']}
}

comparison_rows = []
for name in baseline.keys():
    b_acc = baseline[name]['accuracy']
    t_acc = tuned[name]['accuracy']
    b_f1 = baseline[name]['f1']
    t_f1 = tuned[name]['f1']
    comparison_rows.append({
        'Model': name,
        'Baseline_CV_Accuracy': b_acc,
        'Tuned_CV_Accuracy': t_acc,
        'Accuracy_Improvement': t_acc - b_acc,
        'Baseline_CV_F1': b_f1,
        'Tuned_CV_F1': t_f1,
        'F1_Improvement': t_f1 - b_f1
    })

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.to_string(index=False))
comp_df.to_csv(models_dir / 'tuning_comparison.csv', index=False)
print('Saved tuning_comparison.csv')

# Save best hyperparameters summary
best_params_rows = []
for name, grid in [('Logistic Regression', lr_grid), ('SVM', svm_grid), ('Random Forest', rf_grid)]:
    row = {'Model': name}
    for k, v in grid.best_params_.items():
        row[k] = v
    best_params_rows.append(row)

best_params_df = pd.DataFrame(best_params_rows)
best_params_df.to_csv(models_dir / 'best_hyperparameters.csv', index=False)
print('Saved best_hyperparameters.csv')
print(best_params_df.to_string(index=False))

# Visualization
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

models_list = list(baseline.keys())
x = np.arange(len(models_list))
width = 0.35

# Accuracy
axes[0].bar(x - width/2, [baseline[m]['accuracy'] for m in models_list], width, label='Baseline', alpha=0.8)
axes[0].bar(x + width/2, [tuned[m]['accuracy'] for m in models_list], width, label='Tuned', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(models_list, rotation=15)
axes[0].set_ylabel('Accuracy')
axes[0].set_title('CV Accuracy: Baseline vs Tuned')
axes[0].legend()
axes[0].set_ylim(0.7, 0.8)

# F1
axes[1].bar(x - width/2, [baseline[m]['f1'] for m in models_list], width, label='Baseline', alpha=0.8)
axes[1].bar(x + width/2, [tuned[m]['f1'] for m in models_list], width, label='Tuned', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(models_list, rotation=15)
axes[1].set_ylabel('Weighted F1')
axes[1].set_title('CV F1: Baseline vs Tuned')
axes[1].legend()
axes[1].set_ylim(0.7, 0.8)

plt.tight_layout()
figures_dir = root / 'reports' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / 'tuning_comparison.png', dpi=150)
plt.close()
print('Saved tuning_comparison.png')

print()
print('=== SUMMARY ===')
print(f'Logistic Regression combinations: {lr_n_combos}')
print(f'SVM combinations: {svm_n_combos}')
print(f'Random Forest combinations: {rf_n_combos}')