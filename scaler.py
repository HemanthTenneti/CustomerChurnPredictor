"""Shared scaler builder — avoids circular imports between app.py and agent/tools.py."""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


_SCALER_CACHE_PATH = "models/scaler.pkl"


def build_scaler() -> StandardScaler:
    """Replay preprocessing from the notebook to fit an identical scaler.

    Uses cached scaler if available to avoid slow Kaggle download on every startup.
    """
    # Try to load from cache first
    if os.path.exists(_SCALER_CACHE_PATH):
        return joblib.load(_SCALER_CACHE_PATH)

    # Fallback: Build from Kaggle (slow)
    print("Building scaler from Kaggle dataset (first run, may be slow)...")
    import kagglehub

    dataset_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
    csv_file = next(f for f in os.listdir(dataset_path) if f.endswith(".csv"))
    df = pd.read_csv(os.path.join(dataset_path, csv_file))

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df = df.drop(columns=["customerID"])
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)

    # Cache for future runs
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, _SCALER_CACHE_PATH)

    return scaler
