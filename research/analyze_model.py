"""Auditable diagnostics for the frozen, validation-selected Ridge model.

This script only analyzes the already defined model and test predictions. It
does not tune parameters or replace the model using test-period information.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, mean_absolute_error

from train_forecast import DATA, RESULTS, build_dataset, make_model


BACKTEST = RESULTS / "backtest_2025_to_2026_01.csv"
FEATURE_GROUPS = {
    "current_new_home": ["new_mom_pct"],
    "past_new_home": ["new_lag1", "new_lag2", "new_lag3", "new_lag6", "new_mean3", "new_mean6"],
    "second_hand": [
        "second_mom_pct", "second_lag1", "second_lag2", "second_lag3",
        "second_lag6", "second_mean3", "second_mean6", "national_second_mean",
    ],
    "national_new_home": ["national_new_mean", "share_cities_rising"],
    "calendar": ["calendar_sin", "calendar_cos"],
    "city_identity": ["city"],
}


def paired_month_bootstrap(backtest: pd.DataFrame, repetitions: int = 20000) -> dict:
    """Resample months, retaining all 70 correlated city errors together."""
    monthly_gain = backtest.groupby("target_month")["ridge_gain_over_last_month_pp"].mean().to_numpy()
    rng = np.random.default_rng(202603)
    resampled = rng.choice(monthly_gain, size=(repetitions, len(monthly_gain)), replace=True).mean(axis=1)
    # Circular blocks preserve some dependence between adjacent months.
    block_length = 3
    block_starts = rng.integers(0, len(monthly_gain), size=(repetitions, 5))
    offsets = np.arange(block_length)
    block_indices = (block_starts[:, :, None] + offsets) % len(monthly_gain)
    block_resampled = monthly_gain[block_indices.reshape(repetitions, -1)[:, :len(monthly_gain)]].mean(axis=1)
    return {
        "number_of_month_clusters": len(monthly_gain),
        "ridge_mae_gain_over_last_month_pp": round(float(monthly_gain.mean()), 4),
        "independent_month_bootstrap_95pct_interval_pp": [
            round(float(x), 4) for x in np.quantile(resampled, [0.025, 0.975])
        ],
        "circular_three_month_block_bootstrap_95pct_interval_pp": [
            round(float(x), 4) for x in np.quantile(block_resampled, [0.025, 0.975])
        ],
        "bootstrap_repetitions": repetitions,
        "interpretation": "Exploratory uncertainty across 13 months; neither interval guarantees future gains.",
    }


def validation_permutation_importance() -> pd.DataFrame:
    """Perturb feature groups only on 2024 validation data."""
    df, numeric = build_dataset()
    known = df[df["target_change_pct"].notna()]
    train = known[known["target_month"] <= pd.Period("2023-12")]
    validation = known[known["target_month"].between(pd.Period("2024-01"), pd.Period("2024-12"))]
    model = make_model("ridge", numeric)
    features = numeric + ["city"]
    model.fit(train[features], train["target_change_pct"])
    x = validation[features].copy()
    actual = validation["target_change_pct"].to_numpy()
    reference_mae = mean_absolute_error(actual, model.predict(x))
    rng = np.random.default_rng(202604)
    records = []
    for group_name, columns in FEATURE_GROUPS.items():
        deltas = []
        for _ in range(30):
            reordered = rng.permutation(len(x))
            altered = x.copy()
            altered.loc[:, columns] = x.iloc[reordered][columns].to_numpy()
            altered_mae = mean_absolute_error(actual, model.predict(altered))
            deltas.append(altered_mae - reference_mae)
        records.append({
            "feature_group": group_name,
            "validation_reference_mae_pp": round(float(reference_mae), 4),
            "mean_mae_increase_pp": round(float(np.mean(deltas)), 4),
            "std_mae_increase_pp": round(float(np.std(deltas)), 4),
            "permutations": len(deltas),
        })
    return pd.DataFrame(records).sort_values("mean_mae_increase_pp", ascending=False)


def main() -> None:
    source = pd.read_csv(DATA)
    if source.duplicated(["city", "period"]).any():
        raise ValueError("Duplicate city-month rows in the source panel")
    data_quality = {
        "city_month_rows": len(source),
        "distinct_cities": int(source["city"].nunique()),
        "distinct_months": int(source["period"].nunique()),
        "missing_cells": int(source.isna().sum().sum()),
        "new_home_change_range_pct": [float(source["new_mom_pct"].min()), float(source["new_mom_pct"].max())],
        "second_hand_change_range_pct": [float(source["second_mom_pct"].min()), float(source["second_mom_pct"].max())],
        "source_urls": int(source["source_url"].nunique()),
    }
    backtest = pd.read_csv(BACKTEST)
    if len(backtest) != 910 or backtest.groupby("target_month")["city"].nunique().ne(70).any():
        raise ValueError("Unexpected test panel; run train_forecast.py first")
    for method in ("last_month", "ridge", "gradient_boosting"):
        backtest[f"ae_{method}"] = (
            backtest["target_change_pct"] - backtest[f"pred_{method}"]
        ).abs()
    backtest["ridge_gain_over_last_month_pp"] = backtest["ae_last_month"] - backtest["ae_ridge"]

    monthly = backtest.groupby("target_month", as_index=False).agg(
        actual_mean_pct=("target_change_pct", "mean"),
        last_month_mae_pp=("ae_last_month", "mean"),
        ridge_mae_pp=("ae_ridge", "mean"),
        gradient_boosting_mae_pp=("ae_gradient_boosting", "mean"),
        ridge_gain_over_last_month_pp=("ridge_gain_over_last_month_pp", "mean"),
    )
    monthly.round(4).to_csv(RESULTS / "monthly_error.csv", index=False, encoding="utf-8-sig")

    worst = backtest.nlargest(12, "ae_ridge")[[
        "target_month", "city", "target_change_pct", "pred_ridge", "pred_last_month",
        "ae_ridge", "target_source_url",
    ]]
    worst.round(4).to_csv(RESULTS / "largest_errors.csv", index=False, encoding="utf-8-sig")

    importance = validation_permutation_importance()
    importance.to_csv(RESULTS / "validation_permutation_importance.csv", index=False, encoding="utf-8-sig")
    sensitivity = {}
    for name, part in {
        "2025_only": backtest[backtest["target_month"].str.startswith("2025-")],
        "2026_january_only": backtest[backtest["target_month"] == "2026-01"],
    }.items():
        sensitivity[name] = {
            "rows": len(part),
            "last_month_mae_pp": round(float(part["ae_last_month"].mean()), 4),
            "ridge_mae_pp": round(float(part["ae_ridge"].mean()), 4),
        }
    counts = backtest["target_change_pct"].apply(
        lambda value: "rising" if value > 0 else "falling" if value < 0 else "unchanged"
    ).value_counts()
    sign_breakdown = {}
    for name, selected in {
        "falling": backtest["target_change_pct"] < 0,
        "unchanged": backtest["target_change_pct"] == 0,
        "rising": backtest["target_change_pct"] > 0,
    }.items():
        part = backtest[selected]
        sign_breakdown[name] = {
            "rows": len(part),
            "last_month_mae_pp": round(float(part["ae_last_month"].mean()), 4),
            "ridge_mae_pp": round(float(part["ae_ridge"].mean()), 4),
        }
    nonzero = backtest[backtest["target_change_pct"] != 0]
    balanced_direction_accuracy = {
        method: round(float(balanced_accuracy_score(
            nonzero["target_change_pct"] > 0,
            nonzero[f"pred_{method}"] > 0,
        )), 4)
        for method in ("last_month", "ridge", "gradient_boosting")
    }
    diagnostics = {
        "data_quality": data_quality,
        "paired_month_bootstrap": paired_month_bootstrap(backtest),
        "sensitivity_to_2026_january": sensitivity,
        "test_target_direction_counts": {
            key: int(counts.get(key, 0)) for key in ("falling", "unchanged", "rising")
        },
        "test_error_by_actual_direction": sign_breakdown,
        "balanced_rise_vs_fall_accuracy_excluding_unchanged": balanced_direction_accuracy,
        "worst_month_for_ridge": monthly.loc[monthly["ridge_mae_pp"].idxmax(), "target_month"],
        "best_monthly_gain": monthly.loc[monthly["ridge_gain_over_last_month_pp"].idxmax(), "target_month"],
        "worst_monthly_gain": monthly.loc[monthly["ridge_gain_over_last_month_pp"].idxmin(), "target_month"],
        "notes": [
            "All feature importance calculations use 2024 validation data only.",
            "Correlated features can share importance; a low value does not imply a feature has no causal effect.",
            "Test diagnostics are exploratory and are not used to change the selected model.",
        ],
    }
    (RESULTS / "diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(diagnostics, ensure_ascii=False, indent=2))
    print("Validation feature groups:")
    print(importance.to_string(index=False))


if __name__ == "__main__":
    main()
