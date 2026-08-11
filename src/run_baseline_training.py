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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

config = load_config()
root = get_project_root()
df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)

print('=== DATA SPLIT ===')
print('Train:', X_train.shape[0])
print('Validation:', X_val.shape[0])
print('Test:', X_test.shape[0])
print('Classes:', le.classes_)

models = {}
models['Logistic Regression'] = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', random_state=42)
models['Random Forest'] = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_split=5, min_samples_leaf=2, n_jobs=-1, random_state=42)

print('Training...')
for name, model in models.items():
    print('  Training', name)
    model.fit(X_train, y_train)

print('Evaluating...')
val_results = {}
for name, model in models.items():
    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)
    val_results[name] = {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}
    print(name + ': Acc=' + str(round(acc,4)) + ', Prec=' + str(round(prec,4)) + ', Rec=' + str(round(rec,4)) + ', F1=' + str(round(f1,4)))

print()
print('=== CLASSIFICATION REPORTS ===')
class_names = ['Aggressive', 'Balanced', 'Defensive', 'Explorer']
for name, model in models.items():
    y_pred = model.predict(X_val)
    print()
    print(name)
    print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

print()
print('=== CONFUSION MATRICES ===')
import matplotlib.pyplot as plt
import seaborn as sns
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for idx, (name, model) in enumerate(models.items()):
    y_pred = model.predict(X_val)
    cm = confusion_matrix(y_val, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=axes[idx])
    axes[idx].set_title(name)
    axes[idx].set_xlabel('Predicted')
    axes[idx].set_ylabel('True')
plt.tight_layout()
figures_dir = root / 'reports' / 'figures'
figures_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(figures_dir / 'confusion_matrices.png', dpi=150)
plt.close()
print('Saved confusion_matrices.png')

models_dir = root / 'models'
models_dir.mkdir(exist_ok=True)
for name, model in models.items():
    filename = name.lower().replace(' ', '_') + '.joblib'
    joblib.dump(model, models_dir / filename)
    print('Saved', name)

# Comparison table
comp_df = pd.DataFrame([{'Model': k, 'Val_Accuracy': v['accuracy'], 'Val_F1': v['f1']} for k, v in val_results.items()])
print()
print('=== COMPARISON ===')
print(comp_df.to_string(index=False))
comp_df.to_csv(models_dir / 'model_comparison.csv', index=False)
print('Saved model_comparison.csv')

# RF feature importance
print()
print('=== RF FEATURE IMPORTANCE ===')
from features import get_feature_names
rf = models['Random Forest']
feature_names = get_feature_names()
importances = rf.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
for i, idx in enumerate(sorted_idx):
    print(str(i+1) + '. ' + feature_names[idx] + ': ' + str(round(importances[idx], 4)))

plt.figure(figsize=(10, 6))
plt.barh(range(len(importances)), importances[sorted_idx], align='center')
plt.yticks(range(len(importances)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Feature Importance (Gini)')
plt.title('Random Forest - Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(figures_dir / 'rf_feature_importance.png', dpi=150)
plt.close()
print('Saved rf_feature_importance.png')