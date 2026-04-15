"""
Customer Churn Predictor — Gradio App with AI Agent Analysis.

Tab 1 (AI Agent Analysis): LangGraph agent pipeline with RAG + Groq LLM.
Tab 2 (Quick Predict): Logistic Regression prediction with risk gauge.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import math
import joblib
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
import gradio as gr

from dotenv import load_dotenv

load_dotenv()

# ── Load the saved model ────────────────────────────────────────────────────
model = joblib.load("models/model.pkl")[0]

# ── Rebuild the StandardScaler ──────────────────────────────────────────────
from scaler import build_scaler

scaler = build_scaler()

# The 30 columns from one-hot encoding the Telco data
FEATURE_COLS = [
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

EXAMPLE_PROFILES = [
    {
        "label": "⚡ High-Risk Customer",
        "senior": "No",
        "tenure": 2,
        "monthly": 90.0,
        "total": 180.0,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "phone": "Yes",
        "multilines": "No",
        "internet": "Fiber optic",
        "online_sec": "No",
        "online_bkp": "No",
        "device_prot": "No",
        "tech_sup": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "Yes",
        "contract": "Month-to-month",
        "paperless": "Yes",
        "payment": "Electronic check",
    },
    {
        "label": "🛡️ Loyal Customer",
        "senior": "No",
        "tenure": 60,
        "monthly": 55.0,
        "total": 3300.0,
        "gender": "Male",
        "partner": "Yes",
        "dependents": "Yes",
        "phone": "Yes",
        "multilines": "Yes",
        "internet": "DSL",
        "online_sec": "Yes",
        "online_bkp": "Yes",
        "device_prot": "Yes",
        "tech_sup": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Two year",
        "paperless": "No",
        "payment": "Bank transfer (automatic)",
    },
    {
        "label": "👤 New Senior",
        "senior": "Yes",
        "tenure": 5,
        "monthly": 75.0,
        "total": 375.0,
        "gender": "Female",
        "partner": "No",
        "dependents": "No",
        "phone": "Yes",
        "multilines": "No",
        "internet": "Fiber optic",
        "online_sec": "No",
        "online_bkp": "No",
        "device_prot": "No",
        "tech_sup": "No",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "contract": "Month-to-month",
        "paperless": "Yes",
        "payment": "Mailed check",
    },
]


# ── Encoding ───────────────────────────────────────────────────────────────
def encode_input(
    senior,
    tenure,
    monthly,
    total,
    gender,
    partner,
    dependents,
    phone,
    multilines,
    internet,
    online_sec,
    online_bkp,
    device_prot,
    tech_sup,
    streaming_tv,
    streaming_movies,
    contract,
    paperless,
    payment,
):
    row = {col: 0 for col in FEATURE_COLS}
    row["SeniorCitizen"] = 1 if senior == "Yes" else 0
    row["tenure"] = float(tenure)
    row["MonthlyCharges"] = float(monthly)
    row["TotalCharges"] = float(total)
    if gender == "Male":
        row["gender_Male"] = 1
    if partner == "Yes":
        row["Partner_Yes"] = 1
    if dependents == "Yes":
        row["Dependents_Yes"] = 1
    if phone == "Yes":
        row["PhoneService_Yes"] = 1
    if multilines == "No phone service":
        row["MultipleLines_No phone service"] = 1
    elif multilines == "Yes":
        row["MultipleLines_Yes"] = 1
    if internet == "Fiber optic":
        row["InternetService_Fiber optic"] = 1
    elif internet == "No":
        row["InternetService_No"] = 1
    if online_sec == "No internet service":
        row["OnlineSecurity_No internet service"] = 1
    elif online_sec == "Yes":
        row["OnlineSecurity_Yes"] = 1
    if online_bkp == "No internet service":
        row["OnlineBackup_No internet service"] = 1
    elif online_bkp == "Yes":
        row["OnlineBackup_Yes"] = 1
    if device_prot == "No internet service":
        row["DeviceProtection_No internet service"] = 1
    elif device_prot == "Yes":
        row["DeviceProtection_Yes"] = 1
    if tech_sup == "No internet service":
        row["TechSupport_No internet service"] = 1
    elif tech_sup == "Yes":
        row["TechSupport_Yes"] = 1
    if streaming_tv == "No internet service":
        row["StreamingTV_No internet service"] = 1
    elif streaming_tv == "Yes":
        row["StreamingTV_Yes"] = 1
    if streaming_movies == "No internet service":
        row["StreamingMovies_No internet service"] = 1
    elif streaming_movies == "Yes":
        row["StreamingMovies_Yes"] = 1
    if contract == "One year":
        row["Contract_One year"] = 1
    elif contract == "Two year":
        row["Contract_Two year"] = 1
    if paperless == "Yes":
        row["PaperlessBilling_Yes"] = 1
    if payment == "Credit card (automatic)":
        row["PaymentMethod_Credit card (automatic)"] = 1
    elif payment == "Electronic check":
        row["PaymentMethod_Electronic check"] = 1
    elif payment == "Mailed check":
        row["PaymentMethod_Mailed check"] = 1
    raw = np.array([list(row.values())], dtype=float)
    return scaler.transform(raw)


# ── Gauge chart ─────────────────────────────────────────────────────────────
RISK_COLORS = {"High": "#ef4444", "Medium": "#eab308", "Low": "#22c55e"}


def make_gauge(prob: float) -> plt.Figure:
    if prob >= 0.7:
        fill_color = "#ef4444"
    elif prob >= 0.4:
        fill_color = "#eab308"
    else:
        fill_color = "#22c55e"

    fig, ax = plt.subplots(figsize=(4.0, 2.5), facecolor="#0f1923")
    ax.set_facecolor("#0f1923")

    track = Wedge(
        (0, 0),
        1.0,
        0,
        180,
        width=0.32,
        facecolor="#1b2b3d",
        edgecolor="#263a50",
        lw=1.0,
    )
    ax.add_patch(track)

    end_angle = 180 - prob * 180
    if prob > 0.001:
        filled = Wedge(
            (0, 0),
            1.0,
            end_angle,
            180,
            width=0.32,
            facecolor=fill_color,
            edgecolor="none",
            alpha=0.85,
        )
        ax.add_patch(filled)

    angle_rad = math.radians(end_angle)
    nx = 0.74 * math.cos(angle_rad)
    ny = 0.74 * math.sin(angle_rad)
    ax.annotate(
        "",
        xy=(nx, ny),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#94a3b8", lw=1.8, mutation_scale=12),
    )
    ax.plot(0, 0, "o", color="#94a3b8", markersize=4.5, zorder=5)

    for pos, label in [(-1.05, "0%"), (0, "50%"), (1.05, "100%")]:
        y = 1.12 if label == "50%" else -0.10
        ax.text(
            pos,
            y,
            label,
            color="#475569",
            fontsize=7,
            ha="center",
            fontfamily="monospace",
        )

    ax.text(
        0,
        -0.34,
        f"{prob * 100:.1f}%",
        ha="center",
        va="center",
        fontsize=20,
        fontweight="bold",
        color=fill_color,
        fontfamily="monospace",
    )

    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.52, 1.22)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


# ── Tab 1: Quick Predict ────────────────────────────────────────────────────
def predict(*args):
    vec = encode_input(*args)
    prob = float(model.predict_proba(vec)[0][1])
    pred = model.predict(vec)[0]

    label = "CHURN" if pred == 1 else "NO CHURN"
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
    risk_color = RISK_COLORS[risk]
    label_color = "#ef4444" if pred == 1 else "#22c55e"

    result_md = f"""<div style="text-align:center;padding:16px 0 12px;">
      <span style="font-size:1.5rem;font-weight:700;color:{label_color};
            letter-spacing:.06em;">{label}</span><br>
      <span style="color:#94a3b8;font-size:0.82rem;margin-top:6px;display:inline-block;">
        Risk: <b style="color:{risk_color}">{risk}</b>&nbsp;&nbsp;·&nbsp;&nbsp;
        Probability: <b style="color:#e2e8f0">{prob * 100:.1f}%</b>
      </span>
    </div>"""
    return result_md, make_gauge(prob)


def fill_example(idx: int):
    p = EXAMPLE_PROFILES[idx]
    return (
        p["senior"],
        p["tenure"],
        p["monthly"],
        p["total"],
        p["gender"],
        p["partner"],
        p["dependents"],
        p["phone"],
        p["multilines"],
        p["internet"],
        p["online_sec"],
        p["online_bkp"],
        p["device_prot"],
        p["tech_sup"],
        p["streaming_tv"],
        p["streaming_movies"],
        p["contract"],
        p["paperless"],
        p["payment"],
    )


# ── Tab 2: AI Agent Analysis ────────────────────────────────────────────────
def _no_key_error(msg: str = ""):
    return (
        f'<div style="background:#1a1020;border:1px solid #7c3aed33;border-radius:10px;'
        f'padding:24px;font-family:system-ui;color:#c084fc;text-align:center;">'
        f"<b>⚠ Agent Unavailable</b><br><br>"
        f"{'Error: ' + msg + '<br><br>' if msg else ''}"
        f"Set your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
        f"Quick Predict still works without it.</div>"
    ), ""


def run_agent_with_rag(
    senior,
    tenure,
    monthly,
    total,
    gender,
    partner,
    dependents,
    phone,
    multilines,
    internet,
    online_sec,
    online_bkp,
    device_prot,
    tech_sup,
    streaming_tv,
    streaming_movies,
    contract,
    paperless,
    payment,
):
    features = {
        "senior": senior,
        "tenure": tenure,
        "monthly": monthly,
        "total": total,
        "gender": gender,
        "partner": partner,
        "dependents": dependents,
        "phone": phone,
        "multilines": multilines,
        "internet": internet,
        "online_sec": online_sec,
        "online_bkp": online_bkp,
        "device_prot": device_prot,
        "tech_sup": tech_sup,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract": contract,
        "paperless": paperless,
        "payment": payment,
    }

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return _no_key_error()

    try:
        from agent.churn_agent import get_agent

        agent = get_agent()
        state = agent.run(features)
    except Exception as exc:
        return _no_key_error(str(exc))

    if state.get("error"):
        return _no_key_error(state["error"])

    prob = state["churn_probability"]
    risk = state["risk_level"]
    rc = RISK_COLORS.get(risk, "#94a3b8")

    # ── Risk factors as tags ────────────────────────────────────────────
    factors_html = ""
    for f in state.get("risk_factors", []):
        factors_html += f'<span style="display:inline-block;background:{rc}18;color:{rc};border:1px solid {rc}44;border-radius:20px;padding:4px 14px;margin:3px 4px;font-size:0.8rem;">{f}</span>'

    # ── Recommendations as numbered cards ────────────────────────────────
    recs = state.get("recommendations", [])
    recs_html = ""
    for i, r in enumerate(recs, 1):
        recs_html += f"""
        <div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;">
          <div style="min-width:32px;height:32px;border-radius:8px;background:#0ea5e918;
                      color:#38bdf8;display:flex;align-items:center;justify-content:center;
                      font-weight:700;font-size:0.85rem;border:1px solid #0ea5e933;">{i}</div>
          <div style="flex:1;color:#cbd5e1;font-size:0.88rem;line-height:1.55;padding-top:4px;">{r}</div>
        </div>"""

    output = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif;">

      <!-- ═══ Risk Score Hero ═══ -->
      <div style="text-align:center;padding:24px 0 18px;border-bottom:1px solid #263a50;margin-bottom:20px;">
        <div style="font-size:3rem;font-weight:800;color:{rc};letter-spacing:-.02em;">{prob:.1%}</div>
        <div style="margin-top:6px;">
          <span style="background:{rc}20;color:{rc};border:1px solid {rc}55;border-radius:6px;
                padding:5px 18px;font-size:0.9rem;font-weight:700;letter-spacing:.04em;">{risk.upper()} RISK</span>
        </div>
      </div>

      <!-- ═══ Risk Factors ═══ -->
      <div style="margin-bottom:24px;">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:.1em;
                    margin-bottom:10px;font-weight:600;">Identified Risk Factors</div>
        <div style="line-height:2.2;">{factors_html}</div>
      </div>

      <!-- ═══ Explanation ═══ -->
      <div style="background:#162231;border:1px solid #263a50;border-radius:10px;
                  padding:20px 22px;margin-bottom:24px;">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:.1em;
                    margin-bottom:10px;font-weight:600;">Why This Customer Is at Risk</div>
        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.7;">
          {state.get("explanation", "N/A")}
        </div>
      </div>

      <!-- ═══ Recommendations ═══ -->
      <div style="margin-bottom:24px;">
        <div style="color:#64748b;font-size:0.7rem;text-transform:uppercase;letter-spacing:.1em;
                    margin-bottom:14px;font-weight:600;">Recommended Retention Actions</div>
        {recs_html}
      </div>

      <!-- ═══ Executive Summary ═══ -->
      <div style="background:#0c1a2e;border-left:3px solid #0ea5e9;border-radius:0 8px 8px 0;
                  padding:14px 20px;margin-bottom:8px;">
        <div style="color:#64748b;font-size:0.65rem;text-transform:uppercase;letter-spacing:.1em;
                    margin-bottom:6px;font-weight:600;">Executive Summary</div>
        <div style="color:#e2e8f0;font-size:0.9rem;line-height:1.6;font-style:italic;">
          {state.get("executive_summary", "N/A")}
        </div>
      </div>

    </div>
    """
    return output, state.get("retrieved_context", "No context retrieved.")


# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════

css = """
/* ── Reset & Base ───────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, gradio-app, .wrap {
    background: #0f1923 !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #e2e8f0 !important;
}

/* ── Header ────────────────────────────────────────────────────────────── */
#app-header {
    text-align: center;
    padding: 28px 0 18px;
    margin-bottom: 16px;
    border-bottom: 1px solid #263a50;
}
#app-header h1 {
    margin: 0; font-size: 1.5rem; font-weight: 700;
    color: #f1f5f9; letter-spacing: -.01em;
}
#app-header p {
    margin: 8px 0 0; font-size: 0.78rem; color: #64748b;
}

/* ── Panels / Cards ────────────────────────────────────────────────────── */
.gr-group, .gr-box, .block {
    background: #162231 !important;
    border: 1px solid #263a50 !important;
    border-radius: 10px !important;
}

/* ── Top Tabs ──────────────────────────────────────────────────────────── */
.tab-nav button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.8rem !important;
    color: #64748b !important; background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 24px !important; border-radius: 0 !important;
    text-transform: uppercase; letter-spacing: .08em; font-weight: 600;
    transition: color .15s, border-color .15s;
}
.tab-nav button.selected {
    color: #e2e8f0 !important;
    border-bottom-color: #0ea5e9 !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #94a3b8 !important;
    background: transparent !important;
}
.tabitem { padding: 12px 0 0 !important; }

/* ── Inner Tabs (Demographics/Account/Services) ────────────────────────── */
.tabitem .tab-nav button {
    font-size: 0.72rem !important;
    padding: 6px 14px !important;
    letter-spacing: .06em;
}
.tabitem .tab-nav button.selected {
    border-bottom-color: #38bdf8 !important;
    color: #cbd5e1 !important;
}

/* ── Field Labels ──────────────────────────────────────────────────────── */
label, label span,
.label-wrap, .label-wrap span,
.block > label, .block > label > span,
.block label, .block label span,
.form label, .form label span,
span.svelte-1gfkn6j, span.svelte-1b6s6vi,
[class*="label"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.72rem !important;
    color: #94a3b8 !important;
    text-transform: uppercase; letter-spacing: .05em;
    font-weight: 500 !important;
}

/* ── Inputs / Dropdowns ────────────────────────────────────────────────── */
input[type="number"], input[type="text"], textarea, select,
.gr-input input, .gr-dropdown select {
    background: #0f1923 !important; border: 1px solid #263a50 !important;
    border-radius: 8px !important; color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.85rem !important;
    text-transform: none !important; letter-spacing: 0 !important;
    transition: border-color .15s;
}
input:focus, select:focus, textarea:focus {
    border-color: #0ea5e9 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,.1) !important;
}
ul[role="listbox"], ul[role="listbox"] li,
.multiselect span, input.svelte-1gfkn6j,
[data-testid="dropdown"] input,
[data-testid="dropdown"] span {
    text-transform: none !important; letter-spacing: 0 !important;
    font-size: 0.85rem !important; color: #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Agent Button ──────────────────────────────────────────────────────── */
.agent-btn {
    background: linear-gradient(135deg, #0c4a6e, #0e3a5c) !important;
    color: #7dd3fc !important;
    border: 1px solid #0ea5e944 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.85rem !important; font-weight: 700 !important;
    border-radius: 10px !important; padding: 12px 0 !important;
    letter-spacing: .04em; width: 100% !important;
    transition: all .2s; cursor: pointer;
}
.agent-btn:hover {
    background: linear-gradient(135deg, #0e5a82, #0f4a6e) !important;
    border-color: #0ea5e966 !important;
    color: #bae6fd !important;
    box-shadow: 0 4px 20px rgba(14,165,233,.15);
}

/* ── Predict Button ────────────────────────────────────────────────────── */
.predict-btn {
    background: #162231 !important; color: #94a3b8 !important;
    border: 1px solid #263a50 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    border-radius: 8px !important; padding: 10px 0 !important;
    letter-spacing: .04em; width: 100% !important;
    transition: all .15s;
}
.predict-btn:hover {
    background: #1b2b3d !important; border-color: #334155 !important;
    color: #cbd5e1 !important;
}

/* ── Example Buttons ───────────────────────────────────────────────────── */
.ex-btn {
    background: #162231 !important; color: #94a3b8 !important;
    border: 1px solid #263a50 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.75rem !important; border-radius: 8px !important;
    padding: 8px 10px !important; flex: 1 !important;
    transition: all .15s; font-weight: 500;
}
.ex-btn:hover {
    color: #e2e8f0 !important; border-color: #0ea5e944 !important;
    background: #1b2b3d !important;
}

/* ── Agent Output Container ────────────────────────────────────────────── */
.agent-output {
    background: #0f1923 !important;
    border: 1px solid #263a50 !important;
    border-radius: 10px !important;
    padding: 0 !important;
    min-height: 400px;
    overflow-y: auto;
}

/* ── Result Box ────────────────────────────────────────────────────────── */
#result-box {
    background: #162231 !important; border: 1px solid #263a50 !important;
    border-radius: 10px !important; min-height: 78px;
}

/* ── Gauge ─────────────────────────────────────────────────────────────── */
#gauge-plot { background: transparent !important; border: none !important; }

/* ── Section Label ─────────────────────────────────────────────────────── */
.slabel {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.68rem; color: #64748b;
    text-transform: uppercase; letter-spacing: .1em; font-weight: 600;
    padding: 4px 0 8px; border-bottom: 1px solid #1e3048; margin-bottom: 10px;
}

/* ── Accordion ─────────────────────────────────────────────────────────── */
.accordan-header, .accordion .label-wrap {
    color: #64748b !important;
    font-size: 0.72rem !important;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #263a50; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
"""


# ── Shared input builder ────────────────────────────────────────────────────
def _build_input_tabs():
    """Create the three-tab input layout. Returns list of components."""
    with gr.Tabs():
        with gr.Tab("Demographics"):
            with gr.Group():
                with gr.Row():
                    senior = gr.Dropdown(
                        ["No", "Yes"], value="No", label="Senior Citizen"
                    )
                    gender = gr.Dropdown(
                        ["Female", "Male"], value="Female", label="Gender"
                    )
                with gr.Row():
                    partner = gr.Dropdown(["No", "Yes"], value="No", label="Partner")
                    dependents = gr.Dropdown(
                        ["No", "Yes"], value="No", label="Dependents"
                    )

        with gr.Tab("Account & Billing"):
            with gr.Group():
                with gr.Row():
                    tenure = gr.Number(
                        value=12,
                        label="Tenure (months)",
                        minimum=0,
                        maximum=72,
                        precision=0,
                    )
                    monthly = gr.Number(value=65.0, label="Monthly Charges ($)")
                total = gr.Number(value=780.0, label="Total Charges ($)")
                with gr.Row():
                    contract = gr.Dropdown(
                        ["Month-to-month", "One year", "Two year"],
                        value="Month-to-month",
                        label="Contract",
                    )
                    paperless = gr.Dropdown(
                        ["No", "Yes"], value="Yes", label="Paperless Billing"
                    )
                payment = gr.Dropdown(
                    [
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                        "Electronic check",
                        "Mailed check",
                    ],
                    value="Electronic check",
                    label="Payment Method",
                )

        with gr.Tab("Services"):
            with gr.Group():
                with gr.Row():
                    phone = gr.Dropdown(
                        ["No", "Yes"], value="Yes", label="Phone Service"
                    )
                    multilines = gr.Dropdown(
                        ["No", "No phone service", "Yes"],
                        value="No",
                        label="Multiple Lines",
                    )
                internet = gr.Dropdown(
                    ["DSL", "Fiber optic", "No"],
                    value="Fiber optic",
                    label="Internet Service",
                )
                with gr.Row():
                    online_sec = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Online Security",
                    )
                    online_bkp = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Online Backup",
                    )
                with gr.Row():
                    device_prot = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Device Protection",
                    )
                    tech_sup = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Tech Support",
                    )
                with gr.Row():
                    streaming_tv = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Streaming TV",
                    )
                    streaming_movies = gr.Dropdown(
                        ["No", "No internet service", "Yes"],
                        value="No",
                        label="Streaming Movies",
                    )

    return [
        senior,
        tenure,
        monthly,
        total,
        gender,
        partner,
        dependents,
        phone,
        multilines,
        internet,
        online_sec,
        online_bkp,
        device_prot,
        tech_sup,
        streaming_tv,
        streaming_movies,
        contract,
        paperless,
        payment,
    ]


# ══════════════════════════════════════════════════════════════════════════════
# LAYOUT — Agent Tab First
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(
    theme=gr.themes.Base(), css=css, title="Customer Churn Predictor"
) as demo:
    gr.HTML("""
    <div id="app-header">
      <h1>Customer Churn Predictor</h1>
      <p>Agentic AI &nbsp;·&nbsp; LangGraph + RAG &nbsp;·&nbsp; Groq LLM &nbsp;·&nbsp; Telco Dataset</p>
    </div>
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: AI Agent Analysis (PRIMARY — loads first)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("🤖 AI Agent Analysis"):
        with gr.Row():
            with gr.Column(scale=5, min_width=330):
                tab2_inputs = _build_input_tabs()
                agent_btn = gr.Button(
                    "🤖  Run Agent Analysis", elem_classes=["agent-btn"]
                )

            with gr.Column(scale=5, min_width=380):
                gr.HTML('<div class="slabel">Analysis Report</div>')
                agent_output = gr.HTML(
                    value='<div style="text-align:center;color:#475569;padding:80px 20px;'
                    'font-family:system-ui;font-size:0.85rem;">'
                    "Select a customer profile and click <b>Run Agent Analysis</b> "
                    "to generate a full AI-powered retention report.</div>",
                    elem_classes=["agent-output"],
                )
                with gr.Accordion("📄 Retrieved Knowledge (RAG)", open=False):
                    rag_context = gr.Textbox(
                        label="Source Chunks Used by Agent",
                        lines=6,
                        interactive=False,
                    )

        gr.HTML(
            '<div class="slabel" style="margin-top:16px">Quick-Start Profiles</div>'
        )
        with gr.Row():
            for i, ep in enumerate(EXAMPLE_PROFILES):
                btn2 = gr.Button(ep["label"], elem_classes=["ex-btn"])
                btn2.click(
                    fn=lambda idx=i: fill_example(idx), inputs=[], outputs=tab2_inputs
                )

        agent_btn.click(
            fn=run_agent_with_rag,
            inputs=tab2_inputs,
            outputs=[agent_output, rag_context],
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: Quick Predict (SECONDARY)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("📊 Quick Predict"):
        with gr.Row():
            with gr.Column(scale=5, min_width=330):
                tab1_inputs = _build_input_tabs()

            with gr.Column(scale=4, min_width=260):
                gr.HTML('<div class="slabel">Churn Risk Gauge</div>')
                gauge_plot = gr.Plot(show_label=False, elem_id="gauge-plot")
                result_box = gr.Markdown(
                    value='<div style="text-align:center;color:#475569;padding:20px 0;'
                    'font-size:0.82rem;">run prediction to see result</div>',
                    elem_id="result-box",
                )
                predict_btn = gr.Button(
                    "▶  Predict Churn", elem_classes=["predict-btn"]
                )

        gr.HTML(
            '<div class="slabel" style="margin-top:16px">Quick-Start Profiles</div>'
        )
        with gr.Row():
            for i, ep in enumerate(EXAMPLE_PROFILES):
                btn = gr.Button(ep["label"], elem_classes=["ex-btn"])
                btn.click(
                    fn=lambda idx=i: fill_example(idx), inputs=[], outputs=tab1_inputs
                )

        predict_btn.click(
            fn=predict, inputs=tab1_inputs, outputs=[result_box, gauge_plot]
        )


if __name__ == "__main__":
    demo.launch()
