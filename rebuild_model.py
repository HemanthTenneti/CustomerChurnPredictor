"""Rebuild model.pkl and scaler.pkl from the local Dataset/churn.csv."""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

CSV_PATH = os.path.join("Dataset", "churn.csv")
MODELS_DIR = "models"

os.makedirs(MODELS_DIR, exist_ok=True)

# ── Load and preprocess ────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df = df.drop(columns=["customerID"])
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── Scale ──────────────────────────────────────────────────────────────────
print("Fitting scaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ── Train ──────────────────────────────────────────────────────────────────
print("Training Logistic Regression...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

acc = model.score(X_test_scaled, y_test)
print(f"Test accuracy: {acc:.4f}")

# ── Save ───────────────────────────────────────────────────────────────────
joblib.dump([model, X_test_scaled], os.path.join(MODELS_DIR, "model.pkl"))
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

print(f"Saved: {MODELS_DIR}/model.pkl, {MODELS_DIR}/scaler.pkl")
print("Done!")
