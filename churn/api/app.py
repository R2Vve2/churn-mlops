"""FastAPI model-serving service.

    uvicorn churn.api.app:app --reload     # docs at http://localhost:8000/docs

The model is loaded once at start-up from the file registry (``CHURN_MODEL_DIR``). Every
prediction is counted for the Prometheus-style ``/metrics`` endpoint, and
``/monitoring/drift`` compares a batch of incoming customers against the training
distribution snapshot stored with the model.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse

from churn import __version__
from churn.api.schemas import (
    BatchRequest,
    BatchResponse,
    Customer,
    DriftResponse,
    HealthResponse,
    ModelInfo,
    Prediction,
    PredictionResponse,
)
from churn.config import FEATURES
from churn.monitoring.drift import compute_drift
from churn.registry import ModelRegistry

log = logging.getLogger("churn.api")


def risk_tier(p: float) -> str:
    return "high" if p >= 0.6 else "medium" if p >= 0.3 else "low"


class Metrics:
    """Minimal in-process counters exposed in Prometheus text format."""

    def __init__(self) -> None:
        self.predictions_total = 0
        self.churn_predicted_total = 0
        self.requests_total: dict[str, int] = {}
        self.latency_sum = 0.0
        self.latency_count = 0

    def observe(self, endpoint: str, n: int, n_churn: int, seconds: float) -> None:
        self.predictions_total += n
        self.churn_predicted_total += n_churn
        self.requests_total[endpoint] = self.requests_total.get(endpoint, 0) + 1
        self.latency_sum += seconds
        self.latency_count += 1

    def render(self, model_version: str) -> str:
        lines = [
            "# TYPE churn_predictions_total counter",
            f"churn_predictions_total {self.predictions_total}",
            "# TYPE churn_predicted_positive_total counter",
            f"churn_predicted_positive_total {self.churn_predicted_total}",
            "# TYPE churn_request_latency_seconds summary",
            f"churn_request_latency_seconds_sum {self.latency_sum:.6f}",
            f"churn_request_latency_seconds_count {self.latency_count}",
            "# TYPE churn_model_info gauge",
            f'churn_model_info{{version="{model_version}"}} 1',
        ]
        lines.append("# TYPE churn_requests_total counter")
        lines.extend(f'churn_requests_total{{endpoint="{e}"}} {c}' for e, c in sorted(self.requests_total.items()))
        return "\n".join(lines) + "\n"


def create_app(model_dir: str | Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        registry = ModelRegistry(model_dir)
        app.state.pipeline, app.state.meta = registry.load()
        app.state.metrics = Metrics()
        log.info("loaded model %s (%s)", app.state.meta["version"], app.state.meta["algorithm"])
        yield

    app = FastAPI(title="Churn Prediction API", version=__version__, lifespan=lifespan)

    def to_frame(customers: list[Customer]) -> pd.DataFrame:
        return pd.DataFrame([c.model_dump() for c in customers])[FEATURES]

    def predict_proba(customers: list[Customer]) -> np.ndarray:
        return app.state.pipeline.predict_proba(to_frame(customers))[:, 1]

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_version=app.state.meta["version"], algorithm=app.state.meta["algorithm"],
                              predictions_served=app.state.metrics.predictions_total)

    @app.get("/model/info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        m = app.state.meta
        return ModelInfo(
            version=m["version"], algorithm=m["algorithm"], trained_at=m["trained_at"], git_sha=m.get("git_sha"),
            threshold=m["threshold"], n_train=m["n_train"],
            test_metrics={k: v for k, v in m["test_metrics"].items() if isinstance(v, int | float)},
            cv_results=m["cv_results"], top_features=dict(list(m["feature_importance"].items())[:10]),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(customer: Customer) -> PredictionResponse:
        t0 = time.perf_counter()
        p = float(predict_proba([customer])[0])
        thr = app.state.meta["threshold"]
        churn = p >= thr
        app.state.metrics.observe("/predict", 1, int(churn), time.perf_counter() - t0)
        return PredictionResponse(customer_id=customer.customer_id, churn_probability=round(p, 4), churn_predicted=bool(churn),
                                  risk_tier=risk_tier(p), threshold=thr, model_version=app.state.meta["version"])

    @app.post("/predict/batch", response_model=BatchResponse)
    def predict_batch(req: BatchRequest) -> BatchResponse:
        t0 = time.perf_counter()
        probs = predict_proba(req.customers)
        thr = app.state.meta["threshold"]
        preds = [
            Prediction(customer_id=c.customer_id, churn_probability=round(float(p), 4), churn_predicted=bool(p >= thr), risk_tier=risk_tier(float(p)))
            for c, p in zip(req.customers, probs, strict=True)
        ]
        n_churn = sum(pr.churn_predicted for pr in preds)
        app.state.metrics.observe("/predict/batch", len(preds), n_churn, time.perf_counter() - t0)
        return BatchResponse(predictions=preds, n=len(preds), predicted_churn_rate=round(n_churn / len(preds), 4), threshold=thr,
                             model_version=app.state.meta["version"])

    @app.post("/monitoring/drift", response_model=DriftResponse)
    def drift(req: BatchRequest) -> DriftResponse:
        if len(req.customers) < 50:
            raise HTTPException(status_code=400, detail="drift detection needs at least 50 rows for stable PSI estimates")
        frame = to_frame(req.customers)
        report = compute_drift(app.state.meta["drift_reference"], frame, app.state.pipeline.predict_proba(frame)[:, 1])
        return DriftResponse(**report.to_dict())

    @app.get("/metrics", response_class=PlainTextResponse)
    def metrics() -> str:
        return app.state.metrics.render(app.state.meta["version"])

    return app


app = create_app()
