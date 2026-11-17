# Model card: customer churn classifier `v20261117_053140`

**Algorithm:** logistic_regression (scikit-learn pipeline with imputation, scaling and one-hot encoding)
**Trained:** 2026-11-17T05:31:40.634341+00:00 · git `None` · data fingerprint `212e440e2eefa9e6`
**Training rows:** 5600 · **Hold-out rows:** 1400

## Intended use
Rank active customers by probability of churning in the next period so the retention team can target offers.
Not intended for automated account decisions without human review.

## Model selection (stratified cross-validation on the training split)
| Model | CV ROC-AUC | CV PR-AUC |
|---|---|---|
| logistic_regression | 0.8340 ± 0.0092 | 0.7103 |
| random_forest | 0.8182 ± 0.0106 | 0.6779 |
| hist_gradient_boosting | 0.8188 ± 0.0104 | 0.6858 |

## Hold-out performance (evaluated once)
| Metric | Value |
|---|---|
| ROC-AUC | 0.8380 |
| PR-AUC | 0.6949 |
| Brier score | 0.1662 |
| Decision threshold | 0.14 (cost-optimal; F1-optimal would be 0.53) |
| Precision / Recall / F1 | 0.444 / 0.977 / 0.611 |
| Expected cost (FN=500, FP=50) | 34900 vs 240500 for doing nothing |

## Top features (permutation importance, drop in ROC-AUC)
| Feature | Importance |
|---|---|
| contract | 0.1241 |
| tenure_months | 0.0444 |
| monthly_charges | 0.0286 |
| internet_service | 0.0213 |
| num_support_tickets | 0.0118 |
| tech_support | 0.0115 |
| payment_method | 0.0076 |
| partner | 0.0030 |
| total_charges | 0.0008 |
| paperless_billing | 0.0006 |

## Monitoring
A PSI drift reference for every feature and for the predicted probability is stored with the model
(`metadata.json -> drift_reference`) and served by `POST /monitoring/drift`.

## Limitations
Trained on synthetic data with a known generating process; real-world performance will differ. Threshold
assumes the cost ratio above; re-tune when retention economics change.
