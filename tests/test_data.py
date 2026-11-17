import pandas as pd
import pytest

from churn.config import FEATURES, TARGET
from churn.data.generate import generate_customers
from churn.data.schema import SchemaError, validate_dataframe


def test_generator_shape_and_base_rate(small_df):
    assert set(FEATURES + [TARGET, "customer_id"]) <= set(small_df.columns)
    assert len(small_df) == 1500
    assert 0.18 <= small_df[TARGET].mean() <= 0.36


def test_generator_is_reproducible():
    a, b = generate_customers(200, seed=3), generate_customers(200, seed=3)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(generate_customers(200, seed=4))


def test_causal_structure_holds(small_df):
    by_contract = small_df.groupby("contract")[TARGET].mean()
    assert by_contract["Month-to-month"] > by_contract["Two year"]
    assert small_df[small_df.tenure_months < 6][TARGET].mean() > small_df[small_df.tenure_months > 48][TARGET].mean()


def test_validation_passes_and_orders_columns(small_df):
    out = validate_dataframe(small_df)
    assert list(out.columns)[-1] == TARGET


def test_validation_catches_bad_category(small_df):
    bad = small_df.copy()
    bad.loc[0, "contract"] = "Lifetime"
    with pytest.raises(SchemaError, match="contract"):
        validate_dataframe(bad)


def test_validation_catches_out_of_range_and_missing(small_df):
    bad = small_df.copy()
    bad.loc[0, "tenure_months"] = 999
    bad.loc[1, "monthly_charges"] = None
    with pytest.raises(SchemaError) as exc:
        validate_dataframe(bad)
    assert "tenure_months" in str(exc.value) and "monthly_charges" in str(exc.value)


def test_validation_missing_column():
    with pytest.raises(SchemaError, match="missing columns"):
        validate_dataframe(pd.DataFrame({"tenure_months": [1]}))
