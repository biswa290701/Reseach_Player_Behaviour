import sys
sys.path.insert(0, 'src')
import pandas as pd
import numpy as np
from utils.paths import get_project_root, load_config
from features import prepare_features_and_split
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

config = load_config()
root = get_project_root()
df = pd.read_csv(root / 'data' / 'raw' / 'synthetic_telemetry.csv')

X_train, X_val, X_test, y_train, y_val, y_test, le, scaler = prepare_features_and_split(df, config)

print('Training SVM...')
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

y_pred = svm.predict(X_val)
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, average='weighted', zero_division=0)
rec = recall_score(y_val, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

print('SVM: Acc=' + str(round(acc,4)) + ', Prec=' + str(round(prec,4)) + ', Rec=' + str(round(rec,4)) + ', F1=' + str(round(f1,4)))

print()
print('=== CLASSIFICATION REPORT ===')
class_names = ['Aggressive', 'Balanced', 'Defensive', 'Explorer']
print(classification_report(y_val, y_pred, target_names=class_names, zero_division=0))

# Save model
models_dir = root / 'models'
models_dir.mkdir(exist_ok=True)
joblib.dump(svm, models_dir / 'svm.joblib')
print('Saved SVM')

# Load existing comparison and add SVM
comp_df = pd.read_csv(models_dir / 'model_comparison.csv')
new_row = pd.DataFrame({'Model': ['SVM'], 'Val_Accuracy': [acc], 'Val_F1': [f1]})
comp_df = pd.concat([comp_df, new_row], ignore_index=True)
comp_df.to_csv(models_dir / 'model_comparison.csv', index=False)
print('Updated model_comparison.csv')
print(comp_df.to_string(index=False))