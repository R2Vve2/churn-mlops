"""Data validation: fail fast on schema or range violations before training or scoring.

Implemented with plain pandas so the rules are explicit and dependency-free; the same
checks are applied to training data and (via the API models) to inference requests.
"""

from __future__ import annotations

import pandas as pd

from churn.config import CATEGORICAL_FEATURES, CATEGORY_LEVELS, FEATURES, NUMERIC_FEATURES, TARGET

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "tenure_months": (0, 120),
    "monthly_charges": (0, 500),
    "total_charges": (0, 60_000),
    "num_support_tickets": (0, 100),
    "num_services": (0, 10),
}


class SchemaError(ValueError):
    pass


def validate_dataframe(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Return the validated frame (columns ordered) or raise ``SchemaError`` listing every problem."""
    problems: list[str] = []
    expected = FEATURES + ([TARGET] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    for col in NUMERIC_FEATURES:
        series = pd.to_numeric(df[col], errors="coerce")
        bad_type = series.isna() & df[col].notna()
        if bad_type.any():
            problems.append(f"{col}: {int(bad_type.sum())} non-numeric values")
        lo, hi = NUMERIC_RANGES[col]
        out_of_range = (series < lo) | (series > hi)
        if out_of_range.any():
            problems.append(f"{col}: {int(out_of_range.sum())} values outside [{lo}, {hi}]")
        if col != "total_charges" and series.isna().any():
            problems.append(f"{col}: {int(series.isna().sum())} missing values (only total_charges may be missing)")

    for col in CATEGORICAL_FEATURES:
        allowed = set(CATEGORY_LEVELS[col])
        unknown = set(df[col].dropna().astype(str).unique()) - allowed
        if unknown:
            problems.append(f"{col}: unknown categories {sorted(unknown)[:5]}")
        if df[col].isna().any():
            problems.append(f"{col}: {int(df[col].isna().sum())} missing values")

    if require_target:
        if not set(df[TARGET].dropna().unique()) <= {0, 1}:
            problems.append(f"{TARGET}: must be 0/1")
        if df[TARGET].isna().any():
            problems.append(f"{TARGET}: missing values")

    if problems:
        raise SchemaError("; ".join(problems))
    cols = [c for c in df.columns if c not in expected] + expected
    return df[cols].copy()
