# Churn MLOps — end-to-end customer churn prediction system

[![CI](https://github.com/R2Vve2/churn-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/R2Vve2/churn-mlops/actions/workflows/ci.yml)

A complete, production-style machine learning system, not a notebook: validated data → model selection with
experiment tracking → cost-sensitive decision threshold → versioned model registry → FastAPI serving →
drift monitoring → CI/CD with a quality gate → Docker.

```
 data/raw/customers.csv ─► schema validation ─► stratified split ─► 5-fold CV over 3 candidate models ─► MLflow
                                                                        │
                        model card + plots ◄── permutation importance ◄─┤ best by CV ROC-AUC
                        drift reference (PSI) ◄─────────────────────────┤ threshold tuned on out-of-fold probabilities
                        models/<version>/ (registry) ◄──────────────────┘ evaluated ONCE on the hold-out set
                                   │
                        FastAPI  /predict  /predict/batch  /model/info  /monitoring/drift  /metrics
```

## Why this project

| Hiring-market skill | Where it shows up here |
|---|---|
| ML fundamentals done right | Stratified CV, selection by CV not test, out-of-fold threshold tuning, calibration (Brier), PR-AUC for imbalance |
| Business-aware ML | Threshold chosen to minimise expected cost (missed churner = 500, wasted offer = 50), compared to a do-nothing baseline |
| Experiment tracking | MLflow (SQLite backend): nested runs per candidate, params, metrics, plots, model card and model artifact |
| Feature engineering & leakage safety | All preprocessing lives inside the sklearn `Pipeline`, so training and serving transform identically |
| Model serving | FastAPI with strict pydantic contracts (ranges + category levels), single and batch endpoints, OpenAPI docs |
| Monitoring / observability | PSI drift detection per feature and for the prediction distribution; Prometheus-format `/metrics` |
| Explainability & governance | Permutation importance, auto-generated model card, data fingerprint + git SHA in metadata |
| MLOps / DevOps | File-based model registry with promotion pointer, multi-stage Docker (non-root), GitHub Actions: lint → tests → retrain with ROC-AUC ≥ 0.80 gate → container smoke test |
| Testing | 23 tests covering data generation, schema validation, preprocessing, drift, training, registry and the API |

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt

python -m churn.training.train           # ~30 s: trains 3 models, logs to MLflow, registers the best
mlflow ui --backend-store-uri sqlite:///mlflow.db       # http://localhost:5000 to browse experiments
uvicorn churn.api.app:app --reload       # http://localhost:8000/docs
```

Score a customer:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "customer_id": "C0001", "tenure_months": 3, "monthly_charges": 89.5, "total_charges": 268.5,
  "num_support_tickets": 3, "num_services": 3, "contract": "Month-to-month", "internet_service": "Fiber optic",
  "payment_method": "Electronic check", "paperless_billing": "Yes", "tech_support": "No", "online_security": "No",
  "senior_citizen": "No", "partner": "No", "dependents": "No", "streaming": "Yes"}'
```

```json
{"customer_id":"C0001","churn_probability":0.93,"churn_predicted":true,"risk_tier":"high","threshold":0.145,"model_version":"v20261117_053140"}
```

### Docker

```bash
docker compose up --build     # API on :8000, MLflow UI on :5000
```

## Data

The repository ships a **synthetic telecom dataset generator** (`churn/data/generate.py`) with the schema of the
classic IBM Telco churn dataset and an explicit causal structure (month-to-month contracts, fibre, support tickets
and high charges raise churn; tenure, long contracts and tech support lower it). It is seeded, licence-free and
includes realistic dirt (missing `total_charges` for new customers). To train on real data with the same columns:

```bash
python -m churn.training.train --data path/to/your.csv
```

Every dataset passes `churn/data/schema.py` (column presence, numeric ranges, allowed categories, missing values)
before training, and the same rules are enforced on inference requests by the pydantic models.

## Results (synthetic data, 7,000 rows, seed 42)

| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| **logistic_regression** (selected) | **0.834 ± 0.009** | 0.710 |
| hist_gradient_boosting | 0.819 ± 0.010 | 0.686 |
| random_forest | 0.818 ± 0.011 | 0.678 |

Hold-out set (1,400 rows, evaluated once): ROC-AUC **0.838**, PR-AUC 0.695. At the cost-optimal threshold (0.145)
recall is 0.977 and precision 0.444: because a missed churner costs ten times a wasted retention offer, the
optimal policy flags aggressively. Logistic regression winning is expected here: the generator's churn
propensity is itself a logistic function, so the linear model is the correctly specified one. On real data the
boosted trees usually win, and the pipeline would select them automatically.

See `reports/model_card.md` and `reports/figures/` (ROC, PR, calibration, confusion matrix, feature importance).

## Project layout

| Path | Purpose |
|---|---|
| `churn/config.py` | Feature lists, category levels, paths, cost assumptions (single source of truth) |
| `churn/data/` | Synthetic generator, loader, schema validation |
| `churn/features/preprocessing.py` | `ColumnTransformer`: median imputation + scaling, one-hot with fixed levels |
| `churn/training/train.py` | CV model selection, threshold tuning, evaluation, MLflow logging, model card |
| `churn/training/evaluate.py` | Metrics, cost-based threshold search, plots |
| `churn/monitoring/drift.py` | PSI reference snapshot + drift report |
| `churn/registry.py` | Versioned file registry with `latest.json` promotion pointer |
| `churn/api/` | FastAPI service, pydantic schemas, Prometheus-style metrics |
| `tests/` | Pytest suite (runs in ~6 s, trains a small model once per session) |
| `.github/workflows/ci.yml` | Lint, tests, retrain + quality gate, Docker build + smoke test |

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + served model version |
| `GET` | `/model/info` | Algorithm, metrics, threshold, top features |
| `POST` | `/predict` | Score one customer → probability, decision, risk tier |
| `POST` | `/predict/batch` | Score up to 5,000 customers |
| `POST` | `/monitoring/drift` | PSI of a batch vs. the training distribution (≥ 50 rows) |
| `GET` | `/metrics` | Prometheus text format counters and latency |

## Roadmap

- Scheduled retraining job triggered by a drift `alert`
- Shadow deployment / champion-challenger comparison in the registry
- SHAP per-prediction explanations in the API response
- Feature store integration and online/offline consistency tests
