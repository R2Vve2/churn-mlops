import numpy as np
import pandas as pd

from churn.config import FEATURES
from churn.features.preprocessing import build_preprocessor, feature_names
from churn.monitoring.drift import build_reference, compute_drift, psi


def test_preprocessor_output_shape_and_names(small_df):
    prep = build_preprocessor()
    X = prep.fit_transform(small_df[FEATURES])
    names = feature_names(prep)
    assert X.shape == (len(small_df), len(names))
    assert "tenure_months" in names and "contract_Two year" in names
    assert not np.isnan(X).any()  # total_charges NaNs were imputed


def test_preprocessor_handles_unknown_category(small_df):
    prep = build_preprocessor().fit(small_df[FEATURES])
    row = small_df[FEATURES].head(1).copy()
    row["contract"] = "Weird"
    out = prep.transform(row)
    assert out.shape[1] == len(feature_names(prep))  # no crash, unknown level encoded as all zeros


def test_psi_zero_for_identical_and_large_for_shift():
    assert psi([0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]) == 0.0
    assert psi([0.25, 0.25, 0.25, 0.25], [0.7, 0.1, 0.1, 0.1]) > 0.25


def test_drift_report_detects_shift(small_df):
    ref = build_reference(small_df[FEATURES], proba=np.random.default_rng(0).uniform(size=len(small_df)))
    same = compute_drift(ref, small_df[FEATURES].sample(600, random_state=1))
    assert same.overall_status == "ok"
    shifted = small_df[FEATURES].sample(600, random_state=1).copy()
    shifted["monthly_charges"] = shifted["monthly_charges"] * 2.5
    shifted["contract"] = "Two year"
    report = compute_drift(ref, shifted)
    status = {f.feature: f.status for f in report.features}
    assert status["monthly_charges"] == "alert" and status["contract"] == "alert"
    assert report.overall_status == "alert"
    assert isinstance(report.to_dict()["features"], list)


def test_drift_with_prediction_distribution(small_df):
    rng = np.random.default_rng(1)
    ref = build_reference(small_df[FEATURES], proba=rng.beta(2, 5, size=len(small_df)))
    report = compute_drift(ref, small_df[FEATURES], proba=rng.beta(5, 2, size=len(small_df)))
    assert report.prediction_psi is not None and report.prediction_psi > 0.25
    assert pd.notna(report.prediction_psi)
