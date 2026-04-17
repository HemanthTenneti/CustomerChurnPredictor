"""Agent tools — LangChain @tool wrappers for ML prediction, RAG retrieval, and risk analysis."""

from langchain_core.tools import tool
import numpy as np

# Lazy imports to avoid loading model at module level
_model = None
_scaler = None
_FEATURE_COLS = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "gender_Male",
    "Partner_Yes",
    "Dependents_Yes",
    "PhoneService_Yes",
    "MultipleLines_No phone service",
    "MultipleLines_Yes",
    "InternetService_Fiber optic",
    "InternetService_No",
    "OnlineSecurity_No internet service",
    "OnlineSecurity_Yes",
    "OnlineBackup_No internet service",
    "OnlineBackup_Yes",
    "DeviceProtection_No internet service",
    "DeviceProtection_Yes",
    "TechSupport_No internet service",
    "TechSupport_Yes",
    "StreamingTV_No internet service",
    "StreamingTV_Yes",
    "StreamingMovies_No internet service",
    "StreamingMovies_Yes",
    "Contract_One year",
    "Contract_Two year",
    "PaperlessBilling_Yes",
    "PaymentMethod_Credit card (automatic)",
    "PaymentMethod_Electronic check",
    "PaymentMethod_Mailed check",
]


def _get_model_and_scaler():
    """Load the model and scaler on first use."""
    global _model, _scaler
    if _model is None:
        import joblib
        from scaler import build_scaler

        _model = joblib.load("models/model.pkl")[0]
        _scaler = build_scaler()
    return _model, _scaler


def _encode(customer_features: dict) -> np.ndarray:
    """Encode raw customer features into the model's expected vector."""
    _, scaler = _get_model_and_scaler()

    row = {col: 0 for col in _FEATURE_COLS}
    row["SeniorCitizen"] = 1 if customer_features.get("senior") == "Yes" else 0
    row["tenure"] = float(customer_features.get("tenure", 0))
    row["MonthlyCharges"] = float(customer_features.get("monthly", 0))
    row["TotalCharges"] = float(customer_features.get("total", 0))

    if customer_features.get("gender") == "Male":
        row["gender_Male"] = 1
    if customer_features.get("partner") == "Yes":
        row["Partner_Yes"] = 1
    if customer_features.get("dependents") == "Yes":
        row["Dependents_Yes"] = 1
    if customer_features.get("phone") == "Yes":
        row["PhoneService_Yes"] = 1

    multilines = customer_features.get("multilines", "No")
    if multilines == "No phone service":
        row["MultipleLines_No phone service"] = 1
    elif multilines == "Yes":
        row["MultipleLines_Yes"] = 1

    internet = customer_features.get("internet", "No")
    if internet == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif internet == "No":
        row["InternetService_No"] = 1

    if customer_features.get("online_sec") == "No internet service":
        row["OnlineSecurity_No internet service"] = 1
    elif customer_features.get("online_sec") == "Yes":
        row["OnlineSecurity_Yes"] = 1

    if customer_features.get("online_bkp") == "No internet service":
        row["OnlineBackup_No internet service"] = 1
    elif customer_features.get("online_bkp") == "Yes":
        row["OnlineBackup_Yes"] = 1

    if customer_features.get("device_prot") == "No internet service":
        row["DeviceProtection_No internet service"] = 1
    elif customer_features.get("device_prot") == "Yes":
        row["DeviceProtection_Yes"] = 1

    if customer_features.get("tech_sup") == "No internet service":
        row["TechSupport_No internet service"] = 1
    elif customer_features.get("tech_sup") == "Yes":
        row["TechSupport_Yes"] = 1

    if customer_features.get("streaming_tv") == "No internet service":
        row["StreamingTV_No internet service"] = 1
    elif customer_features.get("streaming_tv") == "Yes":
        row["StreamingTV_Yes"] = 1

    if customer_features.get("streaming_movies") == "No internet service":
        row["StreamingMovies_No internet service"] = 1
    elif customer_features.get("streaming_movies") == "Yes":
        row["StreamingMovies_Yes"] = 1

    contract = customer_features.get("contract", "Month-to-month")
    if contract == "One year":
        row["Contract_One year"] = 1
    elif contract == "Two year":
        row["Contract_Two year"] = 1

    if customer_features.get("paperless") == "Yes":
        row["PaperlessBilling_Yes"] = 1

    payment = customer_features.get("payment", "")
    if payment == "Credit card (automatic)":
        row["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1

    raw = np.array([list(row.values())], dtype=float)
    return scaler.transform(raw)


@tool
def predict_churn_tool(customer_features: dict) -> dict:
    """Predict churn probability and risk level for a customer.

    Args:
        customer_features: Dictionary with keys matching the Gradio form fields.

    Returns:
        Dictionary with probability, prediction, and risk_level.
    """
    model, _ = _get_model_and_scaler()
    vec = _encode(customer_features)
    prob = float(model.predict_proba(vec)[0][1])
    pred = bool(model.predict(vec)[0])

    if prob >= 0.7:
        risk = "High"
    elif prob >= 0.4:
        risk = "Medium"
    else:
        risk = "Low"

    return {"probability": prob, "prediction": pred, "risk_level": risk}


_retriever = None


def _get_retriever():
    """Lazy-load RAG retriever on first use (avoids blocking on embeddings download)."""
    global _retriever
    if _retriever is None:
        from rag.retriever import RetentionRetriever

        _retriever = RetentionRetriever()
    return _retriever


@tool
def retrieve_retention_strategies_tool(query: str) -> str:
    """Retrieve relevant retention strategies from the knowledge base.

    Args:
        query: Search query describing the customer's risk profile.

    Returns:
        Top 3 relevant chunks joined as a single string.
    """
    retriever = _get_retriever()
    chunks = retriever.retrieve(query, k=3)
    return "\n\n---\n\n".join(chunks)


@tool
def identify_top_risk_factors_tool(customer_features: dict) -> list[str]:
    """Identify the top risk factors for a customer based on deterministic rules.

    Args:
        customer_features: Dictionary with customer attribute keys.

    Returns:
        List of up to 3 risk factor descriptions.
    """
    factors = []

    if customer_features.get("contract") == "Month-to-month":
        factors.append("month-to-month contract")
    tenure = float(customer_features.get("tenure", 0))
    if tenure < 12:
        factors.append("new customer (short tenure)")
    if customer_features.get("internet") == "Fiber optic":
        factors.append("fiber optic subscriber (premium pricing)")
    if customer_features.get("payment") == "Electronic check":
        factors.append("electronic check payment")
    monthly = float(customer_features.get("monthly", 0))
    if monthly > 70:
        factors.append("high monthly charges")
    online_sec = customer_features.get("online_sec", "No")
    tech_sup = customer_features.get("tech_sup", "No")
    if online_sec == "No" and tech_sup == "No":
        factors.append("no value-add services")
    if customer_features.get("senior") == "Yes":
        factors.append("senior citizen")

    return factors[:3]
