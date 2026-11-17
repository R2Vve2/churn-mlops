"""Single source of truth for feature definitions, paths and defaults."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("CHURN_DATA_DIR", ROOT / "data"))
MODEL_DIR = Path(os.environ.get("CHURN_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.environ.get("CHURN_REPORT_DIR", ROOT / "reports"))
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///" + (ROOT / "mlflow.db").as_posix())
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "churn-prediction")

TARGET = "churn"
ID_COLUMN = "customer_id"
RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "num_services",
]
CATEGORICAL_FEATURES = [
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "tech_support",
    "online_security",
    "senior_citizen",
    "partner",
    "dependents",
    "streaming",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

CATEGORY_LEVELS: dict[str, list[str]] = {
    "contract": ["Month-to-month", "One year", "Two year"],
    "internet_service": ["DSL", "Fiber optic", "No"],
    "payment_method": ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    "paperless_billing": ["Yes", "No"],
    "tech_support": ["Yes", "No"],
    "online_security": ["Yes", "No"],
    "senior_citizen": ["Yes", "No"],
    "partner": ["Yes", "No"],
    "dependents": ["Yes", "No"],
    "streaming": ["Yes", "No"],
}

# Business assumptions used for cost-sensitive threshold selection.
COST_FALSE_NEGATIVE = 500.0  # lost customer lifetime value we failed to act on
COST_FALSE_POSITIVE = 50.0  # retention offer given to a customer who would have stayed
