import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

# === Path to full dataset ===
full_data_path = r"C:\Users\somas\PycharmProjects\ADHD_PREDICTION_MODEL\data\synthetic_adhd_full_dataset.csv"

# === Load dataset ===
df = pd.read_csv(full_data_path)

# === Check label column ===
label_col = 'adhd_label'
if label_col not in df.columns:
    raise ValueError(f"Label column '{label_col}' not found in dataset.")

# === Prepare features and target ===
X = df.drop(columns=[label_col])
y = df[label_col]

# === Train-test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Feature scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train XGBoost model ===
model = XGBClassifier(
    n_estimators=150,
    max_depth=4,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42
)
model.fit(X_train_scaled, y_train)

# === Evaluate ===
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# === Save model and scaler ===
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/adhd_xgb_model.pkl')
joblib.dump(scaler, 'models/adhd_scaler.pkl')
print("\n✅ Model and scaler saved in 'models/' folder.")
