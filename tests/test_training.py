import json

import pandas as pd

from churn.config import FEATURES
from churn.registry import ModelRegistry
from churn.training.evaluate import choose_threshold, compute_metrics


def test_training_beats_baseline_and_writes_artifacts(trained):
    assert trained.test_metrics["roc_auc"] >= 0.75
    assert trained.test_metrics["pr_auc"] >= 0.45
    assert 0.05 <= trained.threshold <= 0.95
    assert set(trained.cv_results) == {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert (trained.model_dir / trained.version / "model.joblib").exists()
    assert (trained.report_dir / "model_card.md").exists()
    assert (trained.report_dir / "figures" / "roc_curve.png").exists()
    metrics = json.loads((trained.report_dir / "metrics.json").read_text())
    assert metrics["version"] == trained.version


def test_registry_round_trip(trained, small_df):
    registry = ModelRegistry(trained.model_dir)
    assert registry.latest_version() == trained.version
    assert trained.version in registry.list_versions()
    pipeline, meta = registry.load()
    probs = pipeline.predict_proba(small_df[FEATURES].head(20))[:, 1]
    assert probs.shape == (20,) and ((probs >= 0) & (probs <= 1)).all()
    assert meta["features"] == FEATURES
    assert "drift_reference" in meta and "tenure_months" in meta["drift_reference"]
    assert sum(meta["feature_importance"].values()) > 0


def test_cost_threshold_prefers_recall_when_fn_expensive():
    import numpy as np

    rng = np.random.default_rng(0)
    y = (rng.uniform(size=2000) < 0.3).astype(int)
    proba = np.clip(y * 0.4 + rng.uniform(size=2000) * 0.6, 0, 1)
    t = choose_threshold(y, proba, cost_fn=500, cost_fp=50)
    assert t["cost_optimal"] < t["f1_optimal"] + 1e-9  # expensive misses push the cut-off down
    m = compute_metrics(y, proba, t["cost_optimal"])
    assert m["recall"] > 0.8


def test_model_card_mentions_selected_algorithm(trained):
    card = (trained.report_dir / "model_card.md").read_text(encoding="utf-8")
    assert trained.best_model in card and "Hold-out performance" in card
    assert isinstance(pd.read_csv(trained.model_dir.parent / "customers.csv"), pd.DataFrame)
