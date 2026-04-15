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

model = joblib.load("models/model.pkl")[0]

from scaler import build_scaler

scaler = build_scaler()

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
    return scaler.transform(np.array([list(row.values())], dtype=float))


RISK_COLORS = {"High": "#e11d48", "Medium": "#d97706", "Low": "#059669"}


# ── SVG Icons (Lucide-style, no external deps) ──────────────────────────────
SVG = {
    "agent": '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg>',
    "predict": '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-3 3"/></svg>',
    "risk": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>',
    "brain": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/><path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/><path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/><path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/><path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/><path d="M3.477 10.896a4 4 0 0 1 .585-.396"/><path d="M19.938 10.5a4 4 0 0 1 .585.396"/><path d="M6 18a4 4 0 0 1-1.967-.516"/><path d="M19.967 17.484A4 4 0 0 1 18 18"/></svg>',
    "lightbulb": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M15 14c.2-1 .7-1.7 1.5-2.5 1-.9 1.5-2.2 1.5-3.5A6 6 0 0 0 6 8c0 1 .2 2.2 1.5 3.5.7.7 1.3 1.5 1.5 2.5"/><path d="M9 18h6"/><path d="M10 22h4"/></svg>',
    "target": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "book": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>',
    "zap": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 13c0 5-3.5 7.5-7.66 8.95a1 1 0 0 1-.67-.01C7.5 20.5 4 18 4 13V6a1 1 0 0 1 1-1c2 0 4.5-1.2 6.24-2.72a1.17 1.17 0 0 1 1.52 0C14.51 3.81 17 5 19 5a1 1 0 0 1 1 1z"/></svg>',
    "user": '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>',
}


def make_gauge(prob: float) -> plt.Figure:
    fc = "#e11d48" if prob >= 0.7 else "#d97706" if prob >= 0.4 else "#059669"
    fig, ax = plt.subplots(figsize=(4.0, 2.5), facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.add_patch(
        Wedge(
            (0, 0),
            1.0,
            0,
            180,
            width=0.32,
            facecolor="#f1f5f9",
            edgecolor="#e2e8f0",
            lw=1.0,
        )
    )
    ea = 180 - prob * 180
    if prob > 0.001:
        ax.add_patch(
            Wedge(
                (0, 0),
                1.0,
                ea,
                180,
                width=0.32,
                facecolor=fc,
                edgecolor="none",
                alpha=0.85,
            )
        )
    ar = math.radians(ea)
    ax.annotate(
        "",
        xy=(0.74 * math.cos(ar), 0.74 * math.sin(ar)),
        xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#64748b", lw=1.8, mutation_scale=12),
    )
    ax.plot(0, 0, "o", color="#64748b", markersize=4.5, zorder=5)
    for p, l in [(-1.05, "0%"), (0, "50%"), (1.05, "100%")]:
        ax.text(
            p,
            1.12 if l == "50%" else -0.10,
            l,
            color="#94a3b8",
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
        color=fc,
        fontfamily="monospace",
    )
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-0.52, 1.22)
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    return fig


def predict(*args):
    vec = encode_input(*args)
    prob = float(model.predict_proba(vec)[0][1])
    pred = model.predict(vec)[0]
    label = "CHURN" if pred == 1 else "NO CHURN"
    risk = "High" if prob >= 0.7 else ("Medium" if prob >= 0.4 else "Low")
    rc = RISK_COLORS[risk]
    lc = "#e11d48" if pred == 1 else "#059669"
    md = (
        f'<div style="text-align:center;padding:16px 0 12px;">'
        f'<span style="font-size:1.5rem;font-weight:700;color:{lc};letter-spacing:.06em;">{label}</span><br>'
        f'<span style="color:#000000;font-size:0.82rem;margin-top:6px;display:inline-block;">'
        f'Risk: <b style="color:{rc}">{risk}</b> &middot; Probability: <b style="color:{rc}">{prob * 100:.1f}%</b></span></div>'
    )
    return md, make_gauge(prob)


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


def _no_key_error(msg=""):
    return (
        f'<div style="background:#faf5ff;border:1px solid #d8b4fe;border-radius:12px;'
        f'padding:32px;font-family:system-ui;color:#7c3aed;text-align:center;">'
        f'<div style="font-size:1.2rem;font-weight:700;margin-bottom:12px;">Agent Unavailable</div>'
        f'<div style="font-size:0.85rem;color:#8b5cf6;">'
        f"{'Error: ' + msg + '<br><br>' if msg else ''}"
        f'Set your <code style="background:#f3e8ff;padding:2px 8px;border-radius:4px;">GROQ_API_KEY</code> in <code style="background:#f3e8ff;padding:2px 8px;border-radius:4px;">.env</code></div></div>'
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

        state = get_agent().run(features)
    except Exception as exc:
        return _no_key_error(str(exc))
    if state.get("error"):
        return _no_key_error(state["error"])

    prob = state["churn_probability"]
    risk = state["risk_level"]
    rc = RISK_COLORS.get(risk, "#64748b")

    factors_html = ""
    for f in state.get("risk_factors", []):
        factors_html += (
            f'<span style="display:inline-flex;align-items:center;gap:5px;'
            f"background:{rc}10;color:{rc};border:1px solid {rc}30;"
            f"border-radius:20px;padding:5px 14px;margin:3px 4px;"
            f'font-size:0.78rem;font-weight:600;">'
            f"{SVG['zap']}{f}</span>"
        )

    recs = state.get("recommendations", [])
    recs_html = ""
    for i, r in enumerate(recs, 1):
        colors = ["#6d28d9", "#6d28d9", "#6d28d9"]
        c = colors[(i - 1) % 3]
        recs_html += (
            f'<div style="display:flex;gap:14px;align-items:flex-start;margin-bottom:16px;">'
            f'<div style="min-width:34px;height:34px;border-radius:10px;background:{c}10;'
            f"color:{c};display:flex;align-items:center;justify-content:center;"
            f"font-weight:700;font-size:0.85rem;border:1px solid {c}25;"
            f'">{i}</div>'
            f'<div style="flex:1;color:#000000;font-size:0.88rem;line-height:1.6;padding-top:5px;">{r}</div></div>'
        )

    output = f"""
    <div style="font-family:'Inter',system-ui,-apple-system,sans-serif;">

      <!-- Risk Score -->
      <div style="text-align:center;padding:32px 0 24px;">
        <div style="font-size:3.2rem;font-weight:800;color:{rc};letter-spacing:-.03em;">{prob:.1%}</div>
        <div style="margin-top:8px;">
          <span style="background:{rc}12;color:{rc};border:1px solid {rc}35;
                border-radius:8px;padding:6px 20px;font-size:0.88rem;
                font-weight:700;letter-spacing:.06em;">{risk.upper()} RISK</span>
        </div>
      </div>

      <div style="height:1px;background:#e2e8f0;margin:0 0 28px;"></div>

      <!-- Risk Factors -->
      <div style="margin-bottom:28px;">
        <div style="display:flex;align-items:center;gap:8px;color:#000000;font-size:0.7rem;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;font-weight:700;">
          <span style="color:#6d28d9;display:flex;">{SVG["risk"]}</span> Identified Risk Factors
        </div>
        <div style="line-height:2.4;">{factors_html}</div>
      </div>

      <!-- Explanation -->
      <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:12px;
                  padding:24px;margin-bottom:28px;">
        <div style="display:flex;align-items:center;gap:8px;color:#000000;font-size:0.7rem;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:12px;font-weight:700;">
          <span style="color:#6d28d9;display:flex;">{SVG["brain"]}</span> Why This Customer Is at Risk
        </div>
        <div style="color:#000000;font-size:0.92rem;line-height:1.75;">
          {state.get("explanation", "N/A")}
        </div>
      </div>

      <!-- Recommendations -->
      <div style="margin-bottom:28px;">
        <div style="display:flex;align-items:center;gap:8px;color:#000000;font-size:0.7rem;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:16px;font-weight:700;">
          <span style="color:#6d28d9;display:flex;">{SVG["lightbulb"]}</span> Recommended Retention Actions
        </div>
        {recs_html}
      </div>

      <!-- Executive Summary -->
      <div style="background:#faf5ff;border-left:3px solid #7c3aed;border-radius:0 10px 10px 0;
                  padding:18px 24px;margin-bottom:8px;">
        <div style="display:flex;align-items:center;gap:8px;color:#000000;font-size:0.65rem;
                    text-transform:uppercase;letter-spacing:.1em;margin-bottom:8px;font-weight:700;">
          <span style="color:#6d28d9;display:flex;">{SVG["target"]}</span> Executive Summary
        </div>
        <div style="color:#000000;font-size:0.9rem;line-height:1.65;font-style:italic;">
          {state.get("executive_summary", "N/A")}
        </div>
      </div>

    </div>
    """
    return output, state.get("retrieved_context", "No context retrieved.")


# ══════════════════════════════════════════════════════════════════════════════
# STYLING — Clean white theme
# ══════════════════════════════════════════════════════════════════════════════

css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

*, *::before, *::after { box-sizing: border-box; }

body, .gradio-container, gradio-app, .wrap {
    background: #ffffff !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    color: #000000 !important;
    --block-title-text-color: #000000 !important;
    --block-label-text-color: #000000 !important;
}

/* ══════════════════════════════════════════════════════════════════════════════
   CRITICAL: Fix Gradio label colors to BLACK - must be early in CSS
   ══════════════════════════════════════════════════════════════════════════════ */
:root, body, .gradio-container {
    --block-title-text-color: #000000 !important;
    --block-label-text-color: #000000 !important;
    --block-title-text-weight: 700 !important;
    --block-label-text-weight: 700 !important;
}

body span[data-testid="block-info"] { color: #000000 !important; }
body .svelte-jdcl7l { color: #000000 !important; font-weight: 700 !important; }
span[data-testid="block-info"] { color: #000000 !important; }
.svelte-jdcl7l { color: #000000 !important; font-weight: 700 !important; }
[class*="svelte"][data-testid="block-info"] { color: #000000 !important; }

/* ── Header ────────────────────────────────────────────────────────────── */
#app-header {
    text-align: center; padding: 30px 0 20px; margin-bottom: 20px;
    border-bottom: 1px solid #e2e8f0;
}
#app-header h1 {
    margin: 0; font-size: 1.6rem; font-weight: 800; color: #000000;
    letter-spacing: -.02em;
}
#app-header p {
    margin: 10px 0 0; font-size: 0.78rem; color: #000000;
    font-weight: 500; letter-spacing: .04em;
}

/* ── Panels ────────────────────────────────────────────────────────────── */
.gr-group, .gr-box, .block {
    background: #f8fafc !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
}

/* ── Top Tabs ──────────────────────────────────────────────────────────── */
.tab-nav button {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    color: #666666 !important; background: transparent !important;
    border: none !important; border-bottom: 2px solid transparent !important;
    padding: 12px 28px !important; border-radius: 0 !important;
    text-transform: uppercase; letter-spacing: .08em;
    transition: color .2s, border-color .2s;
}
.tab-nav button.selected {
    color: #000000 !important;
    border-bottom-color: #6366f1 !important;
    background: transparent !important;
}
.tab-nav button:hover {
    color: #000000 !important; background: transparent !important;
}
.tabitem { padding: 14px 0 0 !important; }

/* ── Inner Tabs ────────────────────────────────────────────────────────── */
.tabitem .tab-nav button {
    font-size: 0.74rem !important; padding: 7px 16px !important;
    letter-spacing: .06em;
}
.tabitem .tab-nav button.selected {
    border-bottom-color: #7c3aed !important; color: #6366f1 !important;
}

/* ── Labels ────────────────────────────────────────────────────────────── */
label, label span, .label-wrap, .label-wrap span,
.block > label, .block > label > span,
.block label, .block label span,
.form label, .form label span,
span.svelte-1gfkn6j, span.svelte-1b6s6vi,
[class*="label"] {
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.72rem !important;
    color: #000000 !important;
    text-transform: uppercase; letter-spacing: .05em;
    font-weight: 700 !important;
}

/* ── Inputs ────────────────────────────────────────────────────────────── */
input[type="number"], input[type="text"], textarea, select,
.gr-input input, .gr-dropdown select {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    color: #000000 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.85rem !important;
    text-transform: none !important; letter-spacing: 0 !important;
    transition: border-color .2s, box-shadow .2s;
}
input:focus, select:focus, textarea:focus {
    border-color: #6366f1 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.1) !important;
}
ul[role="listbox"], ul[role="listbox"] li,
.multiselect span, input.svelte-1gfkn6j,
[data-testid="dropdown"] input, [data-testid="dropdown"] span {
    text-transform: none !important; letter-spacing: 0 !important;
    font-size: 0.85rem !important; color: #000000 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
}

/* ── Agent Button ──────────────────────────────────────────────────────── */
.agent-btn {
    background: #6366f1 !important;
    color: #fff !important;
    border: none !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.88rem !important; font-weight: 700 !important;
    border-radius: 12px !important; padding: 14px 0 !important;
    letter-spacing: .03em; width: 100% !important;
    transition: background .2s;
}
.agent-btn:hover {
    background: #4f46e5 !important;
}

/* ── Predict Button ────────────────────────────────────────────────────── */
.predict-btn {
    background: #f8fafc !important; color: #000000 !important;
    border: 1px solid #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.82rem !important; font-weight: 600 !important;
    border-radius: 10px !important; padding: 12px 0 !important;
    letter-spacing: .04em; width: 100% !important;
    transition: all .2s;
}
.predict-btn:hover {
    background: #f1f5f9 !important; border-color: #cbd5e1 !important;
    color: #000000 !important;
}

/* ── Example Buttons ───────────────────────────────────────────────────── */
.ex-btn {
    background: #f8fafc !important; color: #000000 !important;
    border: 1px solid #e2e8f0 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.72rem !important; border-radius: 10px !important;
    padding: 7px 10px !important; flex: 1 !important;
    transition: all .15s; font-weight: 600;
}
.ex-btn:hover {
    color: #000000 !important; border-color: #cbd5e1 !important;
    background: #f1f5f9 !important;
}

/* ── Agent Output ──────────────────────────────────────────────────────── */
.agent-output {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    padding: 12px 32px 32px 32px !important;
    min-height: 420px;
    overflow-y: auto;
}

/* ── Result Box ────────────────────────────────────────────────────────── */
#result-box {
    background: #f8fafc !important; border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important; min-height: 78px;
}

/* ── Gauge ─────────────────────────────────────────────────────────────── */
#gauge-plot { background: transparent !important; border: none !important; }

/* ── Section Labels ────────────────────────────────────────────────────── */
.slabel {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.65rem; color: #000000;
    text-transform: uppercase; letter-spacing: .12em; font-weight: 700;
    padding: 4px 0 8px; border-bottom: 1px solid #e2e8f0; margin-bottom: 10px;
}

/* ── Scrollbar ─────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #e2e8f0; border-radius: 5px; }
::-webkit-scrollbar-thumb:hover { background: #cbd5e1; }

/* ── Gradio built-in overrides ─────────────────────────────────────────── */
footer { display: none !important; }

/* ── AGGRESSIVE LABEL & TEXT FIXES FOR GRADIO ──────────────────────────── */
/* Target Gradio's Svelte-generated classes for form labels */
span.has-info.svelte-jdcl7l { 
    color: #000000 !important; 
    font-weight: 700 !important;
}

span.svelte-jdcl7l { 
    color: #000000 !important; 
    font-weight: 700 !important;
}

span[data-testid="block-info"] { 
    color: #000000 !important; 
    font-weight: 700 !important;
}

/* Dropdown & select elements */
input[role="listbox"] { 
    color: #000000 !important; 
}

/* Dropdown options */
[role="option"] { 
    color: #000000 !important; 
}
li[role="option"] { 
    color: #000000 !important; 
}

/* Gradio-specific input text colors */
.gr-input input,
.gr-dropdown select,
.gr-textbox textarea { 
    color: #000000 !important; 
}

/* Tab button text */
button[role="tab"] { 
    color: #000000 !important; 
}
button[role="tab"][aria-selected="true"] { 
    color: #000000 !important; 
    font-weight: 700 !important;
}

/* Accordion headers */
.gr-accordion-header { 
    color: #000000 !important; 
}

/* Radio & checkbox labels */
.gr-radio-group label,
.gr-checkbox-group label { 
    color: #000000 !important; 
}

/* Svelte scope classes (Gradio form elements) */
.svelte-jdcl7l,
.svelte-1xfsv4t,
.svelte-11gaq1 { 
    color: #000000 !important; 
    font-weight: 700 !important;
}

/* FINAL OVERRIDE - use [attr] selectors which have higher specificity */
[data-testid="block-info"] { 
    color: #000000 !important;
    font-weight: 700 !important;
}

/* Force all text to black at the end */
* { 
    --text-color-primary: #000000 !important;
}

/* END OF CSS */
"""

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM GRADIO THEME — Override text colors to black/dark gray
# ══════════════════════════════════════════════════════════════════════════════
theme = gr.themes.Base(
    primary_hue="indigo",
    secondary_hue="indigo",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="md",
    text_size="md",
    font=[
        gr.themes.GoogleFont("Inter"),
        "ui-sans-serif",
        "system-ui",
        "sans-serif",
    ],
).set(
    # PRIMARY TEXT (Labels, headings) - CRITICAL: Set to pure black
    block_label_text_color="#000000",
    block_label_text_color_dark="#000000",
    block_label_text_weight="700",
    block_title_text_color="#000000",
    block_title_text_color_dark="#000000",
    block_title_text_weight="700",
    body_text_color="#000000",
    body_text_color_dark="#000000",
    body_text_weight="400",
    # BORDERS & BACKGROUNDS (Keep light for contrast)
    border_color_primary="#e2e8f0",
    border_color_primary_dark="#334155",
    background_fill_primary="#ffffff",
    background_fill_primary_dark="#1e293b",
    block_background_fill="#f8fafc",
    block_background_fill_dark="#0f172a",
)


# ── Shared input builder ────────────────────────────────────────────────────
def _build_input_tabs():
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
# LAYOUT
# ══════════════════════════════════════════════════════════════════════════════
with gr.Blocks(theme=theme, css=css, title="Customer Churn Predictor") as demo:
    # ── Load fix-labels script via external file ──────────────────────────
    gr.HTML(
        '<script src="file=/Users/hemanth10etii/Coding/CustomerChurnPredictor/fix-labels.js"></script>'
    )

    # ── Header ────────────────────────────────────────────────────────────
    gr.HTML(f"""
    <div id="app-header">
      <div style="width:48px;height:48px;margin:0 auto 14px;border-radius:14px;
                  background:#6366f1;display:flex;align-items:center;justify-content:center;">
        {SVG["agent"].replace('width="18"', 'width="26"').replace('height="18"', 'height="26"').replace('stroke="currentColor"', 'stroke="#fff"')}
      </div>
      <h1>Customer Churn Predictor</h1>
      <p>Agentic AI &nbsp;&middot;&nbsp; LangGraph + RAG &nbsp;&middot;&nbsp; Groq LLM &nbsp;&middot;&nbsp; Telco Dataset</p>
    </div>
    """)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1: AI Agent Analysis (PRIMARY)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("AI Agent Analysis"):
        with gr.Row():
            # ── Left: Inputs + Button + Profiles (always visible) ───────
            with gr.Column(scale=4, min_width=320):
                tab2_inputs = _build_input_tabs()

                agent_btn = gr.Button("Run Agent Analysis", elem_classes=["agent-btn"])

                # Profiles RIGHT under the button
                gr.HTML('<div class="slabel">Quick-Start Profiles</div>')
                with gr.Row():
                    for i, ep in enumerate(EXAMPLE_PROFILES):
                        b = gr.Button(ep["label"], elem_classes=["ex-btn"])
                        b.click(
                            fn=lambda idx=i: fill_example(idx),
                            inputs=[],
                            outputs=tab2_inputs,
                        )

            # ── Right: Analysis Output ──────────────────────────────────
            with gr.Column(scale=6, min_width=420):
                gr.HTML('<div class="slabel">Analysis Report</div>')
                agent_output = gr.HTML(
                    value='<div style="text-align:center;color:#000000;padding:100px 24px;'
                    'font-family:system-ui;font-size:0.88rem;line-height:1.8;">'
                    '<div style="font-size:2rem;margin-bottom:12px;opacity:.3;">'
                    + SVG["agent"]
                    .replace('stroke="currentColor"', 'stroke="#000000"')
                    .replace('width="18"', 'width="36"')
                    .replace('height="18"', 'height="36"')
                    + "</div>"
                    "Select a customer profile and click<br>"
                    '<b style="color:#6366f1;">Run Agent Analysis</b> to generate<br>'
                    "a full AI-powered retention report.</div>",
                    elem_classes=["agent-output"],
                )
                with gr.Accordion("Retrieved Knowledge (RAG)", open=False):
                    rag_context = gr.Textbox(
                        label="Source Chunks", lines=5, interactive=False
                    )

        agent_btn.click(
            fn=run_agent_with_rag,
            inputs=tab2_inputs,
            outputs=[agent_output, rag_context],
        )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2: Quick Predict (SECONDARY)
    # ═══════════════════════════════════════════════════════════════════════
    with gr.Tab("Quick Predict"):
        with gr.Row():
            with gr.Column(scale=5, min_width=320):
                tab1_inputs = _build_input_tabs()
                predict_btn = gr.Button("Predict Churn", elem_classes=["predict-btn"])

                gr.HTML('<div class="slabel">Quick-Start Profiles</div>')
                with gr.Row():
                    for i, ep in enumerate(EXAMPLE_PROFILES):
                        b = gr.Button(ep["label"], elem_classes=["ex-btn"])
                        b.click(
                            fn=lambda idx=i: fill_example(idx),
                            inputs=[],
                            outputs=tab1_inputs,
                        )

            with gr.Column(scale=4, min_width=280):
                gr.HTML('<div class="slabel">Churn Risk Gauge</div>')
                gauge_plot = gr.Plot(show_label=False, elem_id="gauge-plot")
                result_box = gr.Markdown(
                    value='<div style="text-align:center;color:#000000;padding:20px 0;'
                    'font-size:0.82rem;">run prediction to see result</div>',
                    elem_id="result-box",
                )

        predict_btn.click(
            fn=predict, inputs=tab1_inputs, outputs=[result_box, gauge_plot]
        )


if __name__ == "__main__":
    demo.launch()
