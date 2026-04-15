"""
Customer Churn Predictor — Gradio App with AI Agent Analysis.

Tab 1 (Quick Predict): Logistic Regression prediction with risk gauge.
Tab 2 (AI Agent Analysis): LangGraph agent pipeline with RAG + Groq LLM.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import math
import joblib
import numpy as np
import pandas as pd
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

# Three pre-built demo profiles
EXAMPLE_PROFILES = [
    {
        "label": "High-Risk Customer",
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
        "label": "Loyal Customer",
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
        "label": "New Senior",
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
def make_gauge(prob: float) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.0, 2.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    track = Wedge(
        (0, 0),
        1.0,
        0,
        180,
        width=0.32,
        facecolor="#161b27",
        edgecolor="#1e2732",
        lw=1.0,
    )
    ax.add_patch(track)

    if prob < 0.4:
        fill_color = "#4ade80"
    elif prob < 0.7:
        fill_color = "#fbbf24"
    else:
        fill_color = "#f87171"

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
        arrowprops=dict(arrowstyle="-|>", color="#cbd5e1", lw=1.8, mutation_scale=12),
    )
    ax.plot(0, 0, "o", color="#cbd5e1", markersize=4.5, zorder=5)

    ax.text(
        -1.05,
        -0.10,
        "0%",
        color="#6b7a99",
        fontsize=7,
        ha="center",
        fontfamily="monospace",
    )
    ax.text(
        0, 1.12, "50%", color="#6b7a99", fontsize=7, ha="center", fontfamily="monospace"
    )
    ax.text(
        1.05,
        -0.10,
        "100%",
        color="#6b7a99",
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
    risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"}[risk]
    label_color = "#f87171" if pred == 1 else "#4ade80"

    result_md = f"""<div style="font-family:'JetBrains Mono',monospace;text-align:center;padding:14px 0 10px;">
  <span style="font-size:1.45rem;font-weight:700;color:{label_color};letter-spacing:.06em;">{label}</span><br>
  <span style="color:#8899bb;font-size:0.78rem;margin-top:4px;display:inline-block;">
    Risk: <b style="color:{risk_color}">{risk}</b>&nbsp;&nbsp;·&nbsp;&nbsp;Probability: <b style="color:#c8d3e8">{prob * 100:.1f}%</b>
  </span>
</div>"""
    gauge = make_gauge(prob)
    return result_md, gauge


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
def run_agent_analysis(
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
    """Run the full LangGraph agent pipeline and return formatted Markdown."""
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key or api_key == "your_groq_api_key_here":
        return (
            '<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            "<b>⚠ Agent analysis unavailable</b><br><br>"
            "Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            "The Quick Predict tab still works without an API key.</div>"
        )

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

    try:
        from agent.churn_agent import get_agent

        agent = get_agent()
        state = agent.run(features)
    except Exception as exc:
        return (
            f'<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            f'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            f"<b>⚠ Agent analysis unavailable</b><br><br>"
            f"Error: {exc}<br><br>"
            f"Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            f"The Quick Predict tab still works without an API key.</div>"
        )

    if state.get("error"):
        return (
            f'<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            f'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            f"<b>⚠ Agent analysis unavailable</b><br><br>"
            f"{state['error']}<br><br>"
            f"Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            f"The Quick Predict tab still works without an API key.</div>"
        )

    prob = state["churn_probability"]
    risk = state["risk_level"]
    risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"}.get(
        risk, "#cbd5e8"
    )

    risk_factors_md = ""
    for f in state.get("risk_factors", []):
        risk_factors_md += f"  • {f}\n"

    recs_md = ""
    for i, r in enumerate(state.get("recommendations", []), 1):
        recs_md += f"{i}. {r}\n"

    retrieved = state.get("retrieved_context", "No context retrieved.")
    # Truncate for display
    if len(retrieved) > 600:
        retrieved = retrieved[:600] + "..."

    output = f"""## Risk Assessment

<div style="text-align:center;margin-bottom:12px;">
  <span style="font-size:2rem;font-weight:800;color:{risk_color};">{prob:.1%}</span>
  &nbsp;&nbsp;
  <span style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}55;
        border-radius:4px;padding:3px 12px;font-size:0.85rem;font-weight:700;">{risk} RISK</span>
</div>

**Top Risk Factors:**
{risk_factors_md}

---

## Why This Customer Is at Risk

{state.get("explanation", "N/A")}

---

## Recommended Retention Actions

{recs_md}

---

**Executive Summary:** *{state.get("executive_summary", "N/A")}*
"""
    return output


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
    """Run agent analysis and also return the retrieved RAG context."""
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
        return (
            '<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            "<b>⚠ Agent analysis unavailable</b><br><br>"
            "Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            "The Quick Predict tab still works without an API key.</div>"
        ), ""

    try:
        from agent.churn_agent import get_agent

        agent = get_agent()
        state = agent.run(features)
    except Exception as exc:
        return (
            f'<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            f'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            f"<b>⚠ Agent analysis unavailable</b><br><br>"
            f"Error: {exc}<br><br>"
            f"Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            f"The Quick Predict tab still works without an API key.</div>"
        ), ""

    if state.get("error"):
        return (
            f'<div style="background:#1c1218;border:1px solid #5c2233;border-radius:8px;'
            f'padding:18px 22px;font-family:monospace;color:#f87171;text-align:center;">'
            f"<b>⚠ Agent analysis unavailable</b><br><br>"
            f"{state['error']}<br><br>"
            f"Check your <code>GROQ_API_KEY</code> in <code>.env</code>.<br>"
            f"The Quick Predict tab still works without an API key.</div>"
        ), ""

    prob = state["churn_probability"]
    risk = state["risk_level"]
    risk_color = {"High": "#f87171", "Medium": "#fbbf24", "Low": "#4ade80"}.get(
        risk, "#cbd5e8"
    )

    risk_factors_md = ""
    for f in state.get("risk_factors", []):
        risk_factors_md += f"  • {f}\n"

    recs_md = ""
    for i, r in enumerate(state.get("recommendations", []), 1):
        recs_md += f"{i}. {r}\n"

    output = f"""## Risk Assessment

<div style="text-align:center;margin-bottom:12px;">
  <span style="font-size:2rem;font-weight:800;color:{risk_color};">{prob:.1%}</span>
  &nbsp;&nbsp;
  <span style="background:{risk_color}22;color:{risk_color};border:1px solid {risk_color}55;
        border-radius:4px;padding:3px 12px;font-size:0.85rem;font-weight:700;">{risk} RISK</span>
</div>

**Top Risk Factors:**
{risk_factors_md}

---

## Why This Customer Is at Risk

{state.get("explanation", "N/A")}

---

## Recommended Retention Actions

{recs_md}

---

**Executive Summary:** *{state.get("executive_summary", "N/A")}*
"""
    return output, state.get("retrieved_context", "No context retrieved.")


# ── Styling ─────────────────────────────────────────────────────────────────
MONO = "'JetBrains Mono','Fira Code','Cascadia Code','Courier New',monospace"

css = f"""
*, *::before, *::after {{ box-sizing: border-box; }}

body, .gradio-container, gradio-app, .wrap {{
    background: #0d1117 !important;
    font-family: {MONO} !important;
    color: #c8d3e8 !important;
}}

/* ── header ── */
#app-header {{
    text-align: center;
    padding: 20px 0 10px;
    margin-bottom: 12px;
    border-bottom: 1px solid #21293a;
}}
#app-header h1 {{
    margin: 0; font-size: 1.4rem; font-weight: 700;
    color: #dce6f5; letter-spacing: .03em; font-family: {MONO};
}}
#app-header p {{
    margin: 5px 0 0; font-size: 0.72rem;
    color: #6b7a99; font-family: {MONO};
}}

/* ── panels ── */
.gr-group, .gr-box, .block {{
    background: #131922 !important;
    border: 1px solid #21293a !important;
    border-radius: 8px !important;
}}

/* ── tabs ── */
.tab-nav button {{
    font-family: {MONO} !important; font-size: 0.72rem !important;
    color: #5a6e8a !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 7px 18px !important; border-radius: 0 !important;
    text-transform: uppercase; letter-spacing: .06em;
    transition: color .15s, border-color .15s;
}}
.tab-nav button.selected {{
    color: #c8d3e8 !important;
    border-bottom-color: #7aa2cc !important;
    background: transparent !important;
}}
.tab-nav button:hover {{
    color: #a0b4cc !important;
    background: transparent !important;
}}
.tabitem {{ padding: 10px 0 0 !important; }}

/* ── field labels ── */
label, label span,
.label-wrap, .label-wrap span,
.block > label, .block > label > span,
.block label, .block label span,
.form label, .form label span,
span.svelte-1gfkn6j, span.svelte-1b6s6vi,
[class*="label"] {{
    font-family: {MONO} !important; font-size: 0.7rem !important;
    color: #7a8fa8 !important; text-transform: uppercase; letter-spacing: .05em;
    font-weight: 400 !important;
}}

/* ── inputs / dropdowns ── */
input[type="number"], input[type="text"], textarea, select,
.gr-input input, .gr-dropdown select {{
    background: #161d2a !important; border: 1px solid #21293a !important;
    border-radius: 5px !important; color: #c8d3e8 !important;
    font-family: {MONO} !important; font-size: 0.8rem !important;
    text-transform: none !important; letter-spacing: 0 !important;
}}
input:focus, select:focus, textarea:focus {{
    border-color: #3a5270 !important;
    outline: none !important; box-shadow: 0 0 0 2px rgba(120,162,204,.12) !important;
}}

ul[role="listbox"], ul[role="listbox"] li,
.multiselect span, input.svelte-1gfkn6j,
[data-testid="dropdown"] input,
[data-testid="dropdown"] span {{
    text-transform: none !important; letter-spacing: 0 !important;
    font-size: 0.82rem !important; color: #c8d3e8 !important;
    font-family: {MONO} !important;
}}

/* ── predict button ── */
.predict-btn {{
    background: #17263d !important; color: #93c5fd !important;
    border: 1px solid #2a4060 !important; font-family: {MONO} !important;
    font-size: 0.8rem !important; font-weight: 700 !important;
    border-radius: 6px !important; padding: 10px 0 !important;
    letter-spacing: .05em; width: 100% !important;
    transition: background .15s, border-color .15s;
}}
.predict-btn:hover {{
    background: #1c304d !important; border-color: #3a5a80 !important;
    color: #bae0ff !important;
}}

/* ── agent button ── */
.agent-btn {{
    background: #1a2e1a !important; color: #86efac !important;
    border: 1px solid #2d5a2d !important; font-family: {MONO} !important;
    font-size: 0.8rem !important; font-weight: 700 !important;
    border-radius: 6px !important; padding: 10px 0 !important;
    letter-spacing: .05em; width: 100% !important;
    transition: background .15s, border-color .15s;
}}
.agent-btn:hover {{
    background: #1e3d1e !important; border-color: #3d7a3d !important;
    color: #bbf7d0 !important;
}}

/* ── example buttons ── */
.ex-btn {{
    background: #131922 !important; color: #7a8fa8 !important;
    border: 1px solid #21293a !important; font-family: {MONO} !important;
    font-size: 0.72rem !important; border-radius: 5px !important;
    padding: 6px 8px !important; flex: 1 !important;
    transition: color .15s, border-color .15s, background .15s;
}}
.ex-btn:hover {{
    color: #c8d3e8 !important; border-color: #3a5270 !important;
    background: #161d2a !important;
}}

/* ── result box ── */
#result-box {{
    background: #131922 !important; border: 1px solid #21293a !important;
    border-radius: 7px !important; min-height: 78px;
}}

/* ── gauge ── */
#gauge-plot {{ background: transparent !important; border: none !important; }}

/* ── section label ── */
.slabel {{
    font-family: {MONO}; font-size: 0.65rem; color: #6b7a99;
    text-transform: uppercase; letter-spacing: .1em;
    padding: 2px 0 6px; border-bottom: 1px solid #1e2838; margin-bottom: 8px;
}}

/* ── agent output markdown ── */
.agent-output {{
    background: #131922 !important;
    border: 1px solid #21293a !important;
    border-radius: 7px !important;
    padding: 16px 20px !important;
}}
.agent-output h2 {{
    color: #93c5fd !important; font-size: 0.95rem !important;
    font-weight: 700 !important; margin: 16px 0 8px !important;
    border-bottom: 1px solid #21293a !important; padding-bottom: 4px !important;
}}
.agent-output h2:first-child {{ margin-top: 0 !important; }}
.agent-output strong {{ color: #a0b4cc !important; }}
.agent-output li {{ color: #c8d3e8 !important; }}
.agent-output p {{ color: #c8d3e8 !important; }}

::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: #21293a; border-radius: 4px; }}
"""


# ── Shared input builder ────────────────────────────────────────────────────
def _build_input_tabs():
    """Create the three-tab input layout. Returns (components_list, component_refs)."""
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

    inputs = [
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
    return inputs


# ── Layout ──────────────────────────────────────────────────────────────────
with gr.Blocks(
    theme=gr.themes.Base(), css=css, title="Customer Churn Predictor"
) as demo:
    gr.HTML("""
    <div id="app-header">
      <h1>Customer Churn Predictor</h1>
      <p>Logistic Regression &nbsp;·&nbsp; LangGraph Agent &nbsp;·&nbsp; Telco Dataset</p>
    </div>
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: Quick Predict
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("Quick Predict"):
        with gr.Row():
            with gr.Column(scale=5, min_width=330):
                tab1_inputs = _build_input_tabs()

            with gr.Column(scale=4, min_width=260):
                gr.HTML('<div class="slabel">Churn Risk Gauge</div>')
                gauge_plot = gr.Plot(show_label=False, elem_id="gauge-plot")
                result_box = gr.Markdown(
                    value='<div style="font-family:monospace;text-align:center;color:#5a6e8a;padding:20px 0;font-size:.78rem;">run prediction to see result</div>',
                    elem_id="result-box",
                )
                predict_btn = gr.Button(
                    "▶  Predict Churn", elem_classes=["predict-btn"]
                )

        gr.HTML('<div class="slabel" style="margin-top:14px">Example Profiles</div>')
        with gr.Row():
            for i, ep in enumerate(EXAMPLE_PROFILES):
                btn = gr.Button(ep["label"], elem_classes=["ex-btn"])
                btn.click(
                    fn=lambda idx=i: fill_example(idx),
                    inputs=[],
                    outputs=tab1_inputs,
                )

        predict_btn.click(
            fn=predict, inputs=tab1_inputs, outputs=[result_box, gauge_plot]
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: AI Agent Analysis
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("AI Agent Analysis"):
        with gr.Row():
            with gr.Column(scale=5, min_width=330):
                tab2_inputs = _build_input_tabs()
                agent_btn = gr.Button(
                    "🤖  Run Agent Analysis", elem_classes=["agent-btn"]
                )

            with gr.Column(scale=4, min_width=260):
                gr.HTML('<div class="slabel">Agent Analysis Output</div>')
                agent_output = gr.Markdown(
                    value='<div style="font-family:monospace;text-align:center;color:#5a6e8a;padding:20px 0;font-size:.78rem;">click Run Agent Analysis to see results</div>',
                    elem_classes=["agent-output"],
                )
                with gr.Accordion("Retrieved Knowledge (RAG)", open=False):
                    rag_context = gr.Textbox(
                        label="Retrieved Chunks",
                        lines=8,
                        interactive=False,
                    )

        gr.HTML('<div class="slabel" style="margin-top:14px">Example Profiles</div>')
        with gr.Row():
            for i, ep in enumerate(EXAMPLE_PROFILES):
                btn2 = gr.Button(ep["label"], elem_classes=["ex-btn"])
                btn2.click(
                    fn=lambda idx=i: fill_example(idx),
                    inputs=[],
                    outputs=tab2_inputs,
                )

        agent_btn.click(
            fn=run_agent_with_rag,
            inputs=tab2_inputs,
            outputs=[agent_output, rag_context],
        )


if __name__ == "__main__":
    demo.launch()
