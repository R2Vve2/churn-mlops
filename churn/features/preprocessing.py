"""Feature preprocessing as a scikit-learn ``ColumnTransformer``.

Keeping preprocessing inside the model pipeline guarantees the exact same transformations
at training and inference time (no train/serve skew), and lets the whole thing be saved as
one artifact.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.config import CATEGORICAL_FEATURES, CATEGORY_LEVELS, NUMERIC_FEATURES


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = OneHotEncoder(
        categories=[CATEGORY_LEVELS[c] for c in CATEGORICAL_FEATURES],
        handle_unknown="ignore",  # an unseen level becomes an all-zero row instead of crashing
        sparse_output=False,
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("cat", categorical, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    )


def feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return [str(n) for n in preprocessor.get_feature_names_out()]
