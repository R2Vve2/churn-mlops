"""Request/response contracts. Pydantic enforces the same ranges and category levels as the
training-time schema, so bad inputs are rejected with a 422 before they reach the model."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

Contract = Literal["Month-to-month", "One year", "Two year"]
Internet = Literal["DSL", "Fiber optic", "No"]
Payment = Literal["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
YesNo = Literal["Yes", "No"]


class Customer(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer_id": "C0001",
                "tenure_months": 3,
                "monthly_charges": 89.5,
                "total_charges": 268.5,
                "num_support_tickets": 3,
                "num_services": 3,
                "contract": "Month-to-month",
                "internet_service": "Fiber optic",
                "payment_method": "Electronic check",
                "paperless_billing": "Yes",
                "tech_support": "No",
                "online_security": "No",
                "senior_citizen": "No",
                "partner": "No",
                "dependents": "No",
                "streaming": "Yes",
            }
        }
    )

    customer_id: str | None = None
    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float | None = Field(default=None, ge=0, le=60_000, description="May be null for brand-new customers")
    num_support_tickets: int = Field(..., ge=0, le=100)
    num_services: int = Field(..., ge=0, le=10)
    contract: Contract
    internet_service: Internet
    payment_method: Payment
    paperless_billing: YesNo
    tech_support: YesNo
    online_security: YesNo
    senior_citizen: YesNo
    partner: YesNo
    dependents: YesNo
    streaming: YesNo


class Prediction(BaseModel):
    customer_id: str | None
    churn_probability: float
    churn_predicted: bool
    risk_tier: Literal["low", "medium", "high"]


class PredictionResponse(Prediction):
    threshold: float
    model_version: str


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(..., min_length=1, max_length=5000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n: int
    predicted_churn_rate: float
    threshold: float
    model_version: str


class ModelInfo(BaseModel):
    version: str
    algorithm: str
    trained_at: str
    git_sha: str | None
    threshold: float
    n_train: int
    test_metrics: dict[str, float]
    cv_results: dict[str, dict[str, float]]
    top_features: dict[str, float]


class DriftResponse(BaseModel):
    n_rows: int
    overall_status: Literal["ok", "warn", "alert"]
    prediction_psi: float | None
    features: list[dict[str, Any]]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    algorithm: str
    predictions_served: int
