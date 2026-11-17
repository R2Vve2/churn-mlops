"""Train, compare and register churn models.

    python -m churn.training.train                 # full run, logs to MLflow, promotes best model
    python -m churn.training.train --quick         # small/fast settings (used by tests)
    python -m churn.training.train --data my.csv   # train on your own data with the same schema

Steps: load + validate data -> stratified hold-out split -> for each candidate model run
stratified k-fold CV on the training split -> select the best by CV ROC-AUC (never by the
test set) -> tune the decision threshold on out-of-fold probabilities for business cost ->
evaluate once on the hold-out -> permutation importance -> drift reference snapshot ->
plots + model card -> save to the file registry and MLflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from churn import __version__
from churn.config import (
    COST_FALSE_NEGATIVE,
    COST_FALSE_POSITIVE,
    FEATURES,
    MLFLOW_EXPERIMENT,
    MLFLOW_TRACKING_URI,
    MODEL_DIR,
    RANDOM_STATE,
    REPORT_DIR,
    TARGET,
)
from churn.data.load import load_dataset
from churn.features.preprocessing import build_preprocessor
from churn.monitoring.drift import build_reference
from churn.registry import ModelRegistry
from churn.training.evaluate import choose_threshold, compute_metrics, save_plots

log = logging.getLogger("churn.train")


def candidate_models(quick: bool = False) -> dict[str, Any]:
    trees = 60 if quick else 300
    return {
        "logistic_regression": LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=trees, min_samples_leaf=4, max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.06, max_iter=trees, max_leaf_nodes=15, l2_regularization=0.5, random_state=RANDOM_STATE
        ),
    }


def make_pipeline(model: Any) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001 - git may be absent in containers
        return None


def _data_fingerprint(df: pd.DataFrame) -> str:
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes()).hexdigest()[:16]


@dataclass
class TrainResult:
    version: str
    best_model: str
    cv_results: dict[str, dict[str, float]]
    test_metrics: dict[str, float]
    threshold: float
    model_dir: Path
    report_dir: Path


def train(
    data_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    report_dir: str | Path | None = None,
    quick: bool = False,
    log_to_mlflow: bool = True,
    n_rows: int = 7000,
) -> TrainResult:
    t0 = time.perf_counter()
    model_dir = Path(model_dir or MODEL_DIR)
    report_dir = Path(report_dir or REPORT_DIR)
    df = load_dataset(data_path, n=n_rows)
    X, y = df[FEATURES], df[TARGET].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=RANDOM_STATE)
    log.info("data: %d rows, churn rate %.3f, train/test %d/%d", len(df), y.mean(), len(X_train), len(X_test))

    mlflow = None
    if log_to_mlflow:
        import mlflow as _mlflow

        mlflow = _mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)

    parent_ctx = mlflow.start_run(run_name="model-selection") if mlflow else _NullContext()
    with parent_ctx:
        if mlflow:
            mlflow.log_params({"n_rows": len(df), "cv_folds": cv.get_n_splits(), "quick": quick, "data_fingerprint": _data_fingerprint(df)})

        # 1. Cross-validate every candidate on the training split.
        cv_results: dict[str, dict[str, float]] = {}
        for name, model in candidate_models(quick).items():
            pipe = make_pipeline(model)
            scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=["roc_auc", "average_precision"], n_jobs=1)
            cv_results[name] = {
                "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
                "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
                "cv_pr_auc_mean": float(scores["test_average_precision"].mean()),
                "fit_seconds": float(scores["fit_time"].mean()),
            }
            log.info("%-24s CV ROC-AUC %.4f +/- %.4f  PR-AUC %.4f", name, *[cv_results[name][k] for k in ("cv_roc_auc_mean", "cv_roc_auc_std", "cv_pr_auc_mean")])
            if mlflow:
                with mlflow.start_run(run_name=name, nested=True):
                    mlflow.log_params({"model": name, **{k: str(v) for k, v in model.get_params().items() if k in {"C", "n_estimators", "max_iter", "learning_rate", "max_leaf_nodes", "min_samples_leaf"}}})
                    mlflow.log_metrics(cv_results[name])

        # 2. Select by CV, never by the hold-out set.
        best_name = max(cv_results, key=lambda n: cv_results[n]["cv_roc_auc_mean"])
        best = make_pipeline(candidate_models(quick)[best_name])

        # 3. Threshold from out-of-fold probabilities (unbiased), then fit on the full training split.
        oof = cross_val_predict(best, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        thresholds = choose_threshold(y_train, oof)
        threshold = thresholds["cost_optimal"]
        best.fit(X_train, y_train)

        # 4. Single, final evaluation on the hold-out set.
        proba_test = best.predict_proba(X_test)[:, 1]
        test_metrics = compute_metrics(y_test, proba_test, threshold)
        baseline_cost = float((y_test == 1).sum() * COST_FALSE_NEGATIVE)  # "do nothing" policy
        test_metrics["baseline_do_nothing_cost"] = baseline_cost
        test_metrics["cost_saving_vs_do_nothing"] = baseline_cost - test_metrics["expected_cost"]
        log.info("best=%s  test ROC-AUC %.4f  PR-AUC %.4f  F1@%.2f %.3f  recall %.3f  precision %.3f",
                 best_name, test_metrics["roc_auc"], test_metrics["pr_auc"], threshold, test_metrics["f1"], test_metrics["recall"], test_metrics["precision"])

        # 5. Explainability: permutation importance on the hold-out set (model-agnostic).
        perm = permutation_importance(best, X_test, y_test, scoring="roc_auc", n_repeats=3 if quick else 8, random_state=RANDOM_STATE, n_jobs=1)
        importances = {f: float(v) for f, v in sorted(zip(FEATURES, perm.importances_mean, strict=True), key=lambda kv: -kv[1])}

        # 6. Artifacts: plots, model card, drift reference, registry entry.
        plots = save_plots(y_test, proba_test, threshold, importances, report_dir / "figures")
        reference = build_reference(X_train, best.predict_proba(X_train)[:, 1])
        registry = ModelRegistry(model_dir)
        version = registry.new_version()
        metadata = {
            "version": version,
            "algorithm": best_name,
            "trained_at": datetime.now(UTC).isoformat(),
            "package_version": __version__,
            "git_sha": _git_sha(),
            "data_fingerprint": _data_fingerprint(df),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "features": FEATURES,
            "threshold": threshold,
            "thresholds": thresholds,
            "cost_assumptions": {"false_negative": COST_FALSE_NEGATIVE, "false_positive": COST_FALSE_POSITIVE},
            "cv_results": cv_results,
            "test_metrics": test_metrics,
            "feature_importance": importances,
            "drift_reference": reference,
        }
        registry.save(best, metadata, version=version)
        card = _model_card(metadata)
        (report_dir / "model_card.md").write_text(card, encoding="utf-8")
        (report_dir / "metrics.json").write_text(json.dumps({"version": version, "cv_results": cv_results, "test_metrics": test_metrics, "threshold": threshold}, indent=2), encoding="utf-8")

        if mlflow:
            mlflow.log_params({"best_model": best_name, "threshold": threshold})
            mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items() if isinstance(v, int | float)})
            mlflow.log_metrics({f"cv_{best_name}_roc_auc": cv_results[best_name]["cv_roc_auc_mean"]})
            for p in plots:
                mlflow.log_artifact(str(p), artifact_path="figures")
            mlflow.log_artifact(str(report_dir / "model_card.md"))
            mlflow.log_artifact(str(model_dir / version / "metadata.json"))
            mlflow.sklearn.log_model(best, name="model", input_example=X_train.head(3), serialization_format="cloudpickle")
            mlflow.set_tag("registry_version", version)

    log.info("done in %.1fs -> %s/%s", time.perf_counter() - t0, model_dir, version)
    return TrainResult(version, best_name, cv_results, test_metrics, threshold, model_dir, report_dir)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _model_card(m: dict[str, Any]) -> str:
    tm = m["test_metrics"]
    rows = "\n".join(f"| {name} | {r['cv_roc_auc_mean']:.4f} ± {r['cv_roc_auc_std']:.4f} | {r['cv_pr_auc_mean']:.4f} |" for name, r in m["cv_results"].items())
    top = "\n".join(f"| {f} | {v:.4f} |" for f, v in list(m["feature_importance"].items())[:10])
    return f"""# Model card: customer churn classifier `{m['version']}`

**Algorithm:** {m['algorithm']} (scikit-learn pipeline with imputation, scaling and one-hot encoding)
**Trained:** {m['trained_at']} · git `{m['git_sha']}` · data fingerprint `{m['data_fingerprint']}`
**Training rows:** {m['n_train']} · **Hold-out rows:** {m['n_test']}

## Intended use
Rank active customers by probability of churning in the next period so the retention team can target offers.
Not intended for automated account decisions without human review.

## Model selection (stratified cross-validation on the training split)
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
{rows}

## Hold-out performance (evaluated once)
| Metric | Value |
|---|---|
| ROC-AUC | {tm['roc_auc']:.4f} |
| PR-AUC | {tm['pr_auc']:.4f} |
| Brier score | {tm['brier']:.4f} |
| Decision threshold | {m['threshold']:.2f} (cost-optimal; F1-optimal would be {m['thresholds']['f1_optimal']:.2f}) |
| Precision / Recall / F1 | {tm['precision']:.3f} / {tm['recall']:.3f} / {tm['f1']:.3f} |
| Expected cost (FN={m['cost_assumptions']['false_negative']:.0f}, FP={m['cost_assumptions']['false_positive']:.0f}) | {tm['expected_cost']:.0f} vs {tm['baseline_do_nothing_cost']:.0f} for doing nothing |

## Top features (permutation importance, drop in ROC-AUC)
| Feature | Importance |
|---|---|
{top}

## Monitoring
A PSI drift reference for every feature and for the predicted probability is stored with the model
(`metadata.json -> drift_reference`) and served by `POST /monitoring/drift`.

## Limitations
Trained on synthetic data with a known generating process; real-world performance will differ. Threshold
assumes the cost ratio above; re-tune when retention economics change.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=None, help="CSV with the expected schema (default: synthetic dataset)")
    parser.add_argument("--rows", type=int, default=7000, help="rows to generate when no --data is given")
    parser.add_argument("--quick", action="store_true", help="fewer trees/folds for fast runs")
    parser.add_argument("--no-mlflow", action="store_true", help="skip MLflow logging")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    result = train(args.data, args.model_dir, args.report_dir, quick=args.quick, log_to_mlflow=not args.no_mlflow, n_rows=args.rows)
    print(json.dumps({"version": result.version, "best_model": result.best_model, "threshold": result.threshold,
                      "test_roc_auc": round(result.test_metrics["roc_auc"], 4), "test_pr_auc": round(result.test_metrics["pr_auc"], 4)}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
