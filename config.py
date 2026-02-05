import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base project directory (relative to this file)
BASE_DIR = Path(__file__).parent

# Data paths 
DATA_DIR = BASE_DIR / "data"
CHURN_DATA_PATH = DATA_DIR / "churn.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VALIDATION_DATA_DIR = DATA_DIR / "validation"

# ML paths
ML_DIR = BASE_DIR / "ml"
MODELS_DIR = ML_DIR / "models"
PREPROCESSING_DIR = ML_DIR / "preprocessing"
EVALUATION_DIR = ML_DIR / "evaluation"

# UI paths
UI_DIR = BASE_DIR / "ui"
COMPONENTS_DIR = UI_DIR / "components"
STATIC_DIR = UI_DIR / "static"

# Output paths
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
REPORTS_DIR = OUTPUTS_DIR / "reports"

# Documentation paths
DOCS_DIR = BASE_DIR / "docs"


# Ensure directories exist
def create_directories():
    directories = [
        PROCESSED_DATA_DIR,
        VALIDATION_DATA_DIR,
        MODELS_DIR,
        PREPROCESSING_DIR,
        EVALUATION_DIR,
        COMPONENTS_DIR,
        STATIC_DIR,
        PLOTS_DIR,
        REPORTS_DIR,
        DOCS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# Environment variables with defaults
APP_CONFIG = {
    "APP_NAME": os.getenv("APP_NAME", "Customer Churn Predictor"),
    "DEBUG": os.getenv("DEBUG", "False").lower() == "true",
    "HOST": os.getenv("HOST", "0.0.0.0"),
    "PORT": int(os.getenv("PORT", 8501)),
}

if __name__ == "__main__":
    # Create directories when run directly
    create_directories()
    print("Project directories created successfully")
    print("Configuration loaded successfully")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data path: {CHURN_DATA_PATH}")
