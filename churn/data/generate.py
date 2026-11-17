"""Synthetic telecom churn data with a known causal structure.

The generator mirrors the schema of the well-known IBM Telco churn dataset so the pipeline
can be pointed at the real file unchanged (``--data path/to/telco.csv``), while remaining
fully reproducible and licence-free. Churn probability is a logistic function of the
features (month-to-month contracts, fibre, high charges and support tickets raise it;
tenure, tech support and long contracts lower it) plus noise, giving a base rate near 26%.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.config import CATEGORY_LEVELS, ID_COLUMN, RANDOM_STATE, TARGET


def generate_customers(n: int = 7000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    contract = rng.choice(CATEGORY_LEVELS["contract"], size=n, p=[0.55, 0.21, 0.24])
    internet = rng.choice(CATEGORY_LEVELS["internet_service"], size=n, p=[0.34, 0.44, 0.22])
    payment = rng.choice(CATEGORY_LEVELS["payment_method"], size=n, p=[0.34, 0.23, 0.22, 0.21])
    yes_no = lambda p: rng.choice(["Yes", "No"], size=n, p=[p, 1 - p])  # noqa: E731
    paperless = yes_no(0.59)
    senior = yes_no(0.16)
    partner = yes_no(0.48)
    dependents = yes_no(0.30)

    # Tenure depends on contract type: long contracts -> longer tenure.
    tenure = np.where(
        contract == "Month-to-month",
        rng.gamma(shape=1.6, scale=12, size=n),
        np.where(contract == "One year", rng.gamma(3.0, 14, n), rng.gamma(5.0, 12, n)),
    )
    tenure = np.clip(np.round(tenure), 1, 72).astype(int)

    has_internet = internet != "No"
    tech_support = np.where(has_internet, yes_no(0.37), "No")
    online_security = np.where(has_internet, yes_no(0.36), "No")
    streaming = np.where(has_internet, yes_no(0.49), "No")

    num_services = (
        1
        + has_internet.astype(int)
        + (tech_support == "Yes").astype(int)
        + (online_security == "Yes").astype(int)
        + (streaming == "Yes").astype(int)
    )
    base_charge = np.where(internet == "Fiber optic", 70, np.where(internet == "DSL", 45, 20))
    monthly = base_charge + 8 * (num_services - 1) + rng.normal(0, 6, n)
    monthly = np.clip(monthly, 18.0, 120.0).round(2)
    total = (monthly * tenure * rng.uniform(0.92, 1.0, n)).round(2)

    tickets = rng.poisson(lam=np.where(internet == "Fiber optic", 1.4, 0.7), size=n)

    # Latent churn propensity (log-odds).
    logit = (
        -1.1
        + 1.5 * (contract == "Month-to-month")
        - 0.9 * (contract == "Two year")
        + 0.7 * (internet == "Fiber optic")
        - 0.6 * (internet == "No")
        - 0.035 * tenure
        + 0.012 * (monthly - 60)
        + 0.35 * tickets
        - 0.6 * (tech_support == "Yes")
        - 0.4 * (online_security == "Yes")
        + 0.5 * (payment == "Electronic check")
        + 0.3 * (senior == "Yes")
        - 0.25 * (partner == "Yes")
        - 0.2 * (dependents == "Yes")
        + 0.15 * (paperless == "Yes")
        + rng.normal(0, 0.6, n)
    )
    churn = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            ID_COLUMN: [f"C{seed % 100:02d}{i:06d}" for i in range(n)],
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_tickets": tickets,
            "num_services": num_services,
            "contract": contract,
            "internet_service": internet,
            "payment_method": payment,
            "paperless_billing": paperless,
            "tech_support": tech_support,
            "online_security": online_security,
            "senior_citizen": senior,
            "partner": partner,
            "dependents": dependents,
            "streaming": streaming,
            TARGET: churn,
        }
    )
    # Realistic dirt: a few missing total_charges for brand-new customers.
    new_customers = df.index[df["tenure_months"] <= 1]
    df.loc[rng.choice(new_customers, size=min(8, len(new_customers)), replace=False), "total_charges"] = np.nan
    return df


if __name__ == "__main__":
    import argparse

    from churn.config import DATA_DIR

    parser = argparse.ArgumentParser(description="Generate the synthetic churn dataset.")
    parser.add_argument("--n", type=int, default=7000)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--out", default=str(DATA_DIR / "raw" / "customers.csv"))
    args = parser.parse_args()
    out = generate_customers(args.n, args.seed)
    from pathlib import Path

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"wrote {len(out)} rows to {args.out} (churn rate {out[TARGET].mean():.3f})")
