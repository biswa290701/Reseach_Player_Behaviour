import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from utils.paths import get_project_root, load_config
from features import prepare_features_and_split
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

config = load_config()
root = get_project_root()
df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)

print('=== DATA SPLIT ===')
print('Train:', X_train.shape[0])
print('Validation:', X_val.shape[0])
print('Test (held out until now):', X_test.shape[0])
print('Classes:', le.classes_)
print()

# Load tuned models
models_dir = root / 'models'
lr = joblib.load(models_dir / 'logistic_regression_tuned.joblib')
svm = joblib.load(models_dir / 'svm_tuned.joblib')
rf = joblib.load(models_dir / 'random_forest_tuned.joblib')

models = {
    'Logistic Regression': lr,
    'SVM': svm,
    'Random Forest': rf
}

class_names = ['Aggressive', 'Balanced', 'Defensive', 'Explorer']

print('=== FINAL TEST EVALUATION ===')
test_results = {}

for name, model in models.items():
    print(f'\n--- {name} ---')
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    test_results[name] = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1
    }
    
    print(f'Test Accuracy:  {acc:.4f}')
    print(f'Test Precision: {prec:.4f}')
    print(f'Test Recall:    {rec:.4f}')
    print(f'Test F1:        {f1:.4f}')
    
    # Per-class metrics
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0)
    print('\nPer-class:')
    for i, cls in enumerate(class_names):
        print(f'  {cls}: Prec={prec_pc[i]:.4f}, Rec={rec_pc[i]:.4f}, F1={f1_pc[i]:.4f}')
    
    # Classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

# Confusion matrices
print('\n=== CONFUSION MATRICES ===')
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(f'{name}')
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
plt.tight_layout()
figures_dir = root / 'reports' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / 'final_test_confusion_matrices.png', dpi=150)
plt.close()
print('Saved final_test_confusion_matrices.png')

# CV results from previous runs
cv_results = {
    'Logistic Regression': {'accuracy': 0.7537, 'f1': 0.7515},
    'SVM': {'accuracy': 0.7552, 'f1': 0.7532},
    'Random Forest': {'accuracy': 0.7354, 'f1': 0.7313}
}

# Final comparison table
print('\n=== FINAL MODEL COMPARISON ===')
comparison_rows = []
for name in models.keys():
    cv_acc = cv_results[name]['accuracy']
    cv_f1 = cv_results[name]['f1']
    test_acc = test_results[name]['accuracy']
    test_f1 = test_results[name]['f1']
    test_prec = test_results[name]['precision']
    test_rec = test_results[name]['recall']
    
    comparison_rows.append({
        'Model': name,
        'CV_Accuracy': cv_acc,
        'Test_Accuracy': test_acc,
        'Acc_Diff': test_acc - cv_acc,
        'CV_F1': cv_f1,
        'Test_F1': test_f1,
        'F1_Diff': test_f1 - cv_f1,
        'Test_Precision': test_prec,
        'Test_Recall': test_rec
    })

comp_df = pd.DataFrame(comparison_rows)
print(comp_df.to_string(index=False))
comp_df.to_csv(models_dir / 'final_model_comparison.csv', index=False)
print('\nSaved final_model_comparison.csv')

# Detailed per-class results
per_class_rows = []
for name, model in models.items():
    y_pred = model.predict(X_test)
    prec_pc = precision_score(y_test, y_pred, average=None, zero_division=0)
    rec_pc = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_test, y_pred, average=None, zero_division=0)
    for i, cls in enumerate(class_names):
        per_class_rows.append({
            'Model': name,
            'Class': cls,
            'Precision': prec_pc[i],
            'Recall': rec_pc[i],
            'F1': f1_pc[i]
        })

per_class_df = pd.DataFrame(per_class_rows)
per_class_df.to_csv(models_dir / 'final_test_per_class.csv', index=False)
print('Saved final_test_per_class.csv')

# Summary
print('\n=== SUMMARY ===')
best_f1 = max(test_results.items(), key=lambda x: x[1]['f1'])
best_acc = max(test_results.items(), key=lambda x: x[1]['accuracy'])
print(f'Highest Test F1: {best_f1[0]} ({best_f1[1]["f1"]:.4f})')
print(f'Highest Test Accuracy: {best_acc[0]} ({best_acc[1]["accuracy"]:.4f})')

print('\nCV vs Test Differences:')
for row in comparison_rows:
    print(f'  {row["Model"]}: Acc_diff={row["Acc_Diff"]:+.4f}, F1_diff={row["F1_Diff"]:+.4f}')