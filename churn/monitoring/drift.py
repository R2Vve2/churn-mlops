"""Data drift monitoring with the Population Stability Index (PSI).

At training time we snapshot the distribution of every feature (``build_reference``). In
production, batches of incoming requests are compared against that snapshot
(``compute_drift``). PSI < 0.10 is stable, 0.10-0.25 warrants investigation, > 0.25 is a
significant shift and a retraining trigger.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from churn.config import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES

PSI_WARN = 0.10
PSI_ALERT = 0.25
_EPS = 1e-4


def psi(expected: np.ndarray, actual: np.ndarray) -> float:
    e = np.clip(np.asarray(expected, dtype=float), _EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), _EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def _numeric_reference(series: pd.Series, bins: int = 10) -> dict[str, Any]:
    values = series.dropna().to_numpy(dtype=float)
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    return {"type": "numeric", "edges": edges.tolist(), "props": (counts / max(counts.sum(), 1)).tolist(),
            "mean": float(values.mean()), "std": float(values.std())}


def _categorical_reference(series: pd.Series, levels: list[str]) -> dict[str, Any]:
    counts = series.astype(str).value_counts()
    props = np.array([counts.get(lvl, 0) for lvl in levels], dtype=float)
    return {"type": "categorical", "levels": levels, "props": (props / max(props.sum(), 1)).tolist()}


def build_reference(df: pd.DataFrame, proba: np.ndarray | None = None) -> dict[str, Any]:
    ref: dict[str, Any] = {c: _numeric_reference(df[c]) for c in NUMERIC_FEATURES}
    ref.update({c: _categorical_reference(df[c], CATEGORY_LEVELS[c]) for c in CATEGORICAL_FEATURES})
    if proba is not None:
        ref["__prediction__"] = _numeric_reference(pd.Series(proba))
    ref["__n__"] = int(len(df))
    return ref


def _actual_props(ref: dict[str, Any], series: pd.Series) -> np.ndarray:
    if ref["type"] == "numeric":
        values = series.dropna().to_numpy(dtype=float)
        counts, _ = np.histogram(values, bins=np.asarray(ref["edges"]))
    else:
        counts_s = series.astype(str).value_counts()
        counts = np.array([counts_s.get(lvl, 0) for lvl in ref["levels"]], dtype=float)
    return counts / max(counts.sum(), 1)


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    status: str  # ok | warn | alert


@dataclass
class DriftReport:
    n_rows: int
    features: list[FeatureDrift]
    prediction_psi: float | None
    overall_status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _status(value: float) -> str:
    return "alert" if value >= PSI_ALERT else "warn" if value >= PSI_WARN else "ok"


def compute_drift(reference: dict[str, Any], df: pd.DataFrame, proba: np.ndarray | None = None) -> DriftReport:
    feats: list[FeatureDrift] = []
    for col in NUMERIC_FEATURES + CATEGORICAL_FEATURES:
        if col not in df.columns or col not in reference:
            continue
        value = psi(reference[col]["props"], _actual_props(reference[col], df[col]))
        feats.append(FeatureDrift(col, round(value, 4), _status(value)))
    pred_psi = None
    if proba is not None and "__prediction__" in reference:
        pred_psi = round(psi(reference["__prediction__"]["props"], _actual_props(reference["__prediction__"], pd.Series(proba))), 4)
    worst = max([f.psi for f in feats] + ([pred_psi] if pred_psi is not None else []), default=0.0)
    return DriftReport(n_rows=int(len(df)), features=feats, prediction_psi=pred_psi, overall_status=_status(worst))
