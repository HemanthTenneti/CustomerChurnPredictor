# Customer Churn Predictor

## Executive Summary

Customer churn—the rate at which subscribers cancel their service—costs telecoms far more than retention does. This project builds an end-to-end machine learning pipeline on IBM's **Telco Customer Churn** dataset (7,043 customers, 20 features) that predicts whether an individual customer is likely to leave, and surfaces that prediction through an interactive **Gradio web app**.

Five models were benchmarked: Linear Regression, Logistic Regression, Decision Tree, Random Forest, and XGBoost. **Logistic Regression achieved the highest F1-Score** and was selected as the production model. The trained model is saved to `models/model.pkl` and loaded at app startup; the StandardScaler is reconstructed from the raw dataset each time the app launches (it was not serialised separately).

The Gradio UI lets users adjust 20 customer attributes across three tabs—Demographics, Account & Billing, and Services—and returns an instant churn probability with a colour-coded risk gauge.

---

## Folder Structure

```
project/
├── app.py                          # Gradio web application
├── CustomerChurnPrediction.ipynb   # Training notebook (EDA → models → export)
├── README.md
├── requirements.txt
├── dataset/                        # Empty at rest; data downloaded via kagglehub at runtime
└── models/
    └── model.pkl                   # Saved tuple: (LogisticRegression, X_test_scaled)
```

---

## Models & Libraries

### Machine Learning Models Benchmarked

| Model                    | Notes                                                                                         |
| ------------------------ | --------------------------------------------------------------------------------------------- |
| `LinearRegression`       | Baseline regressor, thresholded at 0.5                                                        |
| `LogisticRegression`     | **Selected** — best F1-Score; trained on scaled features (`max_iter=1000`, `random_state=42`) |
| `DecisionTreeClassifier` | `max_depth=10`, `random_state=42`; trained on unscaled features                               |
| `RandomForestClassifier` | `n_estimators=100`, `random_state=42`; trained on unscaled features                           |
| `XGBClassifier`          | `eval_metric='logloss'`, `random_state=42`; trained on unscaled features                      |

Selection criterion: highest **F1-Score** on the held-out 20% test split.

### Preprocessing Pipeline

1. **TotalCharges** — coerced to numeric; 11 blank strings filled with column median.
2. **customerID** — dropped (non-predictive identifier).
3. **Churn** — mapped `Yes → 1`, `No → 0`.
4. **Categorical columns** — one-hot encoded with `pd.get_dummies(drop_first=True)`, yielding 30 binary/numeric feature columns.
5. **Train / test split** — 80 / 20, stratified on `Churn`, `random_state=42`.
6. **StandardScaler** — fit on the training set only; applied to both splits before training Logistic/Linear Regression.

### Key Libraries

| Library                  | Role                                                   |
| ------------------------ | ------------------------------------------------------ |
| `pandas`                 | Data loading, cleaning, `get_dummies` encoding         |
| `numpy`                  | Numerical operations                                   |
| `scikit-learn`           | `StandardScaler`, all classifiers, metrics             |
| `xgboost`                | `XGBClassifier`                                        |
| `joblib`                 | Model serialisation / deserialisation                  |
| `kagglehub`              | Dataset download at runtime                            |
| `matplotlib` / `seaborn` | EDA charts, confusion matrices, ROC curves, risk gauge |
| `gradio`                 | Interactive web UI                                     |

---

## Usage

### Running the App

```bash
# Activate your virtual environment first (see Installation below)
python app.py
```

Gradio will print a local URL (typically `http://127.0.0.1:7860`). Open it in any browser.

> **Note:** On first launch the app downloads the Telco dataset from Kaggle via `kagglehub` to rebuild the StandardScaler. Subsequent launches use the cached download. A Kaggle account is **not** required—`kagglehub` handles anonymous access for public datasets.

### Using the UI

1. **Input Tabs** (left panel)
   - **Demographics** — gender, senior citizen status, partner, dependents, tenure.
   - **Account & Billing** — contract type, paperless billing, payment method, monthly & total charges.
   - **Services** — phone, multiple lines, internet service, online security, online backup, device protection, tech support, streaming TV & movies.
2. Click **Predict Churn** to get an instant result.
3. The **right panel** shows:
   - A half-circle risk gauge (green < 30 %, amber 30–60 %, red > 60 %).
   - A markdown summary with the exact probability and a plain-English risk label.
4. Use the **Example Profiles** buttons at the bottom to pre-fill the form with representative customers (High-Risk, Loyal, New Senior).

---

## Installation

### Prerequisites

- Python 3.10 or later
- `pip`

### Steps

```bash
# 1. Clone or download the project
git clone https://github.com/HemanthTenneti/CustomerChurnPredictor
cd project

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
#    macOS / Linux:
source .venv/bin/activate
#    Windows (PowerShell):
# .venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
python app.py
```

The training notebook has an additional `!pip install kagglehub` cell that handles the dependency inline when running in Jupyter.

### Re-training (optional)

Open `CustomerChurnPrediction.ipynb` in Jupyter and run all cells. The final cell exports the best model (by F1-Score) to `models/model.pkl`, overwriting the existing file.
