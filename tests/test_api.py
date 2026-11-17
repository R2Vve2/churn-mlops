import pytest
from fastapi.testclient import TestClient

from churn.api.app import create_app
from churn.api.schemas import Customer

EXAMPLE = Customer.model_config["json_schema_extra"]["example"]


@pytest.fixture(scope="module")
def client(trained):
    with TestClient(create_app(model_dir=trained.model_dir)) as c:
        yield c


def test_health_and_model_info(client, trained):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["model_version"] == trained.version
    info = client.get("/model/info").json()
    assert info["algorithm"] == trained.best_model
    assert "roc_auc" in info["test_metrics"] and len(info["top_features"]) == 10


def test_predict_high_risk_customer(client):
    r = client.post("/predict", json=EXAMPLE)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["churn_probability"] <= 1
    assert body["risk_tier"] in {"low", "medium", "high"}
    assert body["customer_id"] == "C0001"


def test_loyal_customer_scores_lower_than_risky_one(client):
    loyal = {**EXAMPLE, "customer_id": "L1", "tenure_months": 70, "contract": "Two year", "num_support_tickets": 0,
             "tech_support": "Yes", "online_security": "Yes", "payment_method": "Credit card", "monthly_charges": 40.0, "total_charges": 2800.0}
    p_loyal = client.post("/predict", json=loyal).json()["churn_probability"]
    p_risky = client.post("/predict", json=EXAMPLE).json()["churn_probability"]
    assert p_loyal < p_risky


def test_null_total_charges_allowed(client):
    r = client.post("/predict", json={**EXAMPLE, "total_charges": None})
    assert r.status_code == 200


def test_validation_rejects_bad_input(client):
    assert client.post("/predict", json={**EXAMPLE, "contract": "Lifetime"}).status_code == 422
    assert client.post("/predict", json={**EXAMPLE, "tenure_months": -1}).status_code == 422
    assert client.post("/predict", json={k: v for k, v in EXAMPLE.items() if k != "contract"}).status_code == 422


def test_batch_and_metrics(client):
    customers = [{**EXAMPLE, "customer_id": f"B{i}", "tenure_months": i % 60} for i in range(25)]
    r = client.post("/predict/batch", json={"customers": customers})
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 25 and len(body["predictions"]) == 25
    assert 0 <= body["predicted_churn_rate"] <= 1
    text = client.get("/metrics").text
    assert "churn_predictions_total" in text and 'endpoint="/predict/batch"' in text


def test_drift_endpoint(client, small_df):
    import json

    rows = json.loads(small_df.drop(columns=["churn"]).head(200).to_json(orient="records"))  # NaN -> null
    r = client.post("/monitoring/drift", json={"customers": rows})
    assert r.status_code == 200
    body = r.json()
    assert body["overall_status"] in {"ok", "warn"} and body["n_rows"] == 200
    assert client.post("/monitoring/drift", json={"customers": rows[:10]}).status_code == 400
