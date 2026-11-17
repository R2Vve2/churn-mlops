from __future__ import annotations

from pathlib import Path

import pandas as pd

from churn.config import DATA_DIR
from churn.data.generate import generate_customers
from churn.data.schema import validate_dataframe

DEFAULT_PATH = DATA_DIR / "raw" / "customers.csv"


def load_dataset(path: str | Path | None = None, n: int = 7000, seed: int = 42) -> pd.DataFrame:
    """Load the CSV at ``path`` (or the default location); generate and cache it if absent."""
    path = Path(path) if path else DEFAULT_PATH
    if path.exists():
        df = pd.read_csv(path)
    else:
        df = generate_customers(n=n, seed=seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
    return validate_dataframe(df)
