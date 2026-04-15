# Customer Churn Predictor

**GenAI-powered Agentic Customer Retention System for Telecom Churn Prediction**

---

## Architecture

```
User → Gradio UI → [Quick Predict | AI Agent Analysis]
                          ↓                ↓
                    ML Model (LogReg)   LangGraph Agent (5 nodes)
                          ↓                ↓
                    Risk Gauge       [ChromaDB (RAG) + Groq LLM]
                          ↓                ↓
                    Instant Result    Explanation + Recommendations
```

---

## Tech Stack

| Component | Technology | Purpose |
|---|---|---|
| ML Model | Logistic Regression (scikit-learn) | Churn probability prediction |
| Feature Scaling | StandardScaler | Normalization for LogReg |
| Web UI | Gradio | Two-tab interactive dashboard |
| LLM | Groq `llama3-70b-8192` (via langchain-groq) | Explanation, recommendations, summary |
| Agent Framework | LangGraph | 5-node state machine orchestration |
| Vector Store | ChromaDB | Local persistent embedding storage |
| Embeddings | `all-MiniLM-L6-v2` (Sentence Transformers) | 384-dim local embeddings |
| Dataset | IBM Telco Customer Churn (Kaggle) | 7,043 customers, 20 features |

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/HemanthTenneti/CustomerChurnPredictor
cd CustomerChurnPredictor

# 2. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your GROQ_API_KEY from https://console.groq.com

# 5. Build the RAG vector store
python rag/ingest.py

# 6. Launch the app
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it in any browser.

> **Note:** The Quick Predict tab works with zero API key — ML runs fully offline. The AI Agent Analysis tab requires a valid `GROQ_API_KEY`.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `GROQ_API_KEY` | Yes (for AI tab) | API key from [Groq Console](https://console.groq.com) |
| `LANGCHAIN_TRACING_V2` | No | Enable LangSmith tracing (`true`/`false`) |
| `LANGCHAIN_API_KEY` | No | LangSmith API key for trace visualization |

---

## Project Structure

```
CustomerChurnPredictor/
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
├── app.py                        # Gradio app (two tabs)
├── agent/
│   ├── __init__.py
│   ├── churn_agent.py            # LangGraph 5-node agent
│   ├── tools.py                  # @tool: predict, retrieve, risk factors
│   └── prompts.py                # All LLM prompt templates
├── rag/
│   ├── __init__.py
│   ├── ingest.py                 # Build ChromaDB vector store
│   ├── retriever.py              # RAG query interface
│   └── knowledge_base/
│       └── retention_strategies.md
├── models/
│   └── model.pkl                 # Saved (LogisticRegression, X_test_scaled)
├── Dataset/
│   └── churn.csv
├── Report/
│   └── End_Sem_Report/
│       ├── main.tex
│       └── sections/
└── CustomerChurnPredictor.ipynb  # Training notebook
```

---

## Agent Pipeline Walkthrough

The LangGraph agent executes five sequential nodes:

1. **`predict`** — Encodes customer features and runs the Logistic Regression model. Outputs churn probability, binary prediction, and risk level (Low/Medium/High).

2. **`retrieve_context`** — Queries ChromaDB with a formatted string combining the risk level and identified risk factors. Returns the top 3 most relevant chunks from the retention strategies knowledge base.

3. **`explain`** — Sends the customer profile, probability, and retrieved context to Groq's LLM. The LLM produces a 3–4 sentence plain-English explanation of why this customer is at churn risk.

4. **`recommend`** — Sends the explanation and context to the LLM with a retention specialist persona. Produces exactly 3 numbered, personalized, actionable recommendations.

5. **`format_output`** — Calls the LLM one final time for a single executive-summary sentence (under 25 words) capturing the retention action priority.

Every node has error handling. If any step fails (bad API key, timeout, etc.), a user-friendly error message is returned and the Gradio app stays running.

---

## Model Benchmarking

| Model | Accuracy | Precision | F1-Score | ROC-AUC |
|---|---|---|---|---|
| **Logistic Regression** | **0.8070** | **0.6584** | **0.6092** | **0.8416** |
| Linear Regression | 0.7963 | 0.6426 | 0.5773 | 0.8301 |
| Random Forest | 0.7864 | 0.6237 | 0.5501 | 0.8251 |
| XGBoost | 0.7850 | 0.6079 | 0.5690 | 0.8214 |
| Decision Tree | 0.7559 | 0.5387 | 0.5486 | 0.7607 |

Selected model: **Logistic Regression** (highest F1-Score and ROC-AUC).

---

<!-- Add screenshots here -->
