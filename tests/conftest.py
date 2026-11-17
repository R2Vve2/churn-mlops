import os

import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from churn.data.generate import generate_customers  # noqa: E402
from churn.training.train import train  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    return generate_customers(n=1500, seed=7)


@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    """Train once per test session on a small dataset (no MLflow) and share the artifacts."""
    root = tmp_path_factory.mktemp("train")
    data_path = root / "customers.csv"
    generate_customers(n=1500, seed=7).to_csv(data_path, index=False)
    return train(data_path=data_path, model_dir=root / "models", report_dir=root / "reports", quick=True, log_to_mlflow=False)
