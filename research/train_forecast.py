"""One-month-ahead city price-index forecast with an as-of February 2026 cutoff."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, median_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "official_nbs_70city.csv"
RESULTS = ROOT / "results"
SHANDONG = {"济南": "Jinan", "青岛": "Qingdao", "烟台": "Yantai", "济宁": "Jining"}
LAGS = (1, 2, 3, 6)


def build_dataset() -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(DATA).sort_values(["city", "period"]).reset_index(drop=True)
    df["month"] = pd.PeriodIndex(df["period"], freq="M")
    if df["month"].min() != pd.Period("2021-05") or df["month"].max() != pd.Period("2026-01"):
        raise ValueError("Unexpected source date range")
    if df.groupby("period")["city"].nunique().ne(70).any():
        raise ValueError("Every source month must have exactly 70 cities")
    df["national_new_mean"] = df.groupby("month")["new_mom_pct"].transform("mean")
    df["national_second_mean"] = df.groupby("month")["second_mom_pct"].transform("mean")
    df["share_cities_rising"] = df.groupby("month")["new_mom_pct"].transform(lambda x: (x > 0).mean())
    month_number = df["month"].dt.month
    df["calendar_sin"] = np.sin(2 * np.pi * month_number / 12)
    df["calendar_cos"] = np.cos(2 * np.pi * month_number / 12)

    group = df.groupby("city", sort=False)
    for lag in LAGS:
        df[f"new_lag{lag}"] = group["new_mom_pct"].shift(lag)
        df[f"second_lag{lag}"] = group["second_mom_pct"].shift(lag)
    for window in (3, 6):
        df[f"new_mean{window}"] = group["new_mom_pct"].transform(lambda x: x.rolling(window).mean())
        df[f"second_mean{window}"] = group["second_mom_pct"].transform(lambda x: x.rolling(window).mean())
    df["target_change_pct"] = group["new_mom_pct"].shift(-1)
    df["target_source_url"] = group["source_url"].shift(-1)
    df["target_month"] = df["month"] + 1

    numeric = [
        "new_mom_pct", "second_mom_pct", "national_new_mean",
        "national_second_mean", "share_cities_rising", "calendar_sin", "calendar_cos",
    ]
    numeric += [f"new_lag{x}" for x in LAGS] + [f"second_lag{x}" for x in LAGS]
    numeric += [f"new_mean{x}" for x in (3, 6)] + [f"second_mean{x}" for x in (3, 6)]
    df = df.dropna(subset=numeric).copy()
    # Lag windows are valid only because the source panel is complete and monthly.
    return df, numeric


def make_model(kind: str, numeric: list[str]) -> Pipeline:
    try:
        categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # scikit-learn before 1.2
        categorical = OneHotEncoder(handle_unknown="ignore", sparse=False)
    if kind == "ridge":
        prep = ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("city", categorical, ["city"]),
        ])
        regressor = Ridge(alpha=25.0)
    elif kind == "gradient_boosting":
        prep = ColumnTransformer([
            ("num", "passthrough", numeric),
            ("city", categorical, ["city"]),
        ])
        regressor = HistGradientBoostingRegressor(
            max_iter=120, max_leaf_nodes=15, learning_rate=0.05,
            min_samples_leaf=30, l2_regularization=0.5,
            early_stopping=False, random_state=42,
        )
    else:
        raise ValueError(kind)
    return Pipeline([("preprocess", prep), ("regressor", regressor)])


def scores(actual: np.ndarray, predicted: np.ndarray) -> dict:
    return {
        "mae_percentage_points": round(float(mean_absolute_error(actual, predicted)), 4),
        "median_ae_percentage_points": round(float(median_absolute_error(actual, predicted)), 4),
        "direction_accuracy": round(float(np.mean(np.sign(actual) == np.sign(predicted))), 4),
        "bias_percentage_points": round(float(np.mean(predicted - actual)), 4),
    }


def evaluate(train: pd.DataFrame, holdout: pd.DataFrame, numeric: list[str]) -> tuple[dict, pd.DataFrame]:
    actual = holdout["target_change_pct"].to_numpy()
    city_mean = train.groupby("city")["target_change_pct"].mean()
    predictions = {
        "zero_change": np.zeros(len(holdout)),
        "last_month": holdout["new_mom_pct"].to_numpy(),
        "city_mean": holdout["city"].map(city_mean).fillna(train["target_change_pct"].mean()).to_numpy(),
    }
    for kind in ("ridge", "gradient_boosting"):
        print(f"Training {kind}: {len(train)} rows; evaluating {len(holdout)} rows", flush=True)
        model = make_model(kind, numeric)
        model.fit(train[numeric + ["city"]], train["target_change_pct"])
        predictions[kind] = model.predict(holdout[numeric + ["city"]])
    metrics = {name: scores(actual, prediction) for name, prediction in predictions.items()}
    output = holdout[["period", "target_month", "city", "target_change_pct", "source_url", "target_source_url"]].copy()
    output = output.rename(columns={"source_url": "feature_source_url"})
    output["target_month"] = output["target_month"].astype(str)
    for name, prediction in predictions.items():
        output[f"pred_{name}"] = prediction
    return metrics, output


def save_shandong_plot(backtest: pd.DataFrame, selected: str) -> None:
    rows = backtest[backtest["city"].isin(SHANDONG)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharey=True)
    for ax, (city, english) in zip(axes.flat, SHANDONG.items()):
        part = rows[rows["city"] == city].sort_values("target_month")
        ax.plot(part["target_month"], part["target_change_pct"], label="Actual", color="#1b6b81")
        ax.plot(part["target_month"], part[f"pred_{selected}"], label="Forecast", color="#d2773b", linestyle="--")
        ax.axhline(0, color="#a9b5b9", linewidth=0.8)
        ax.set_title(english)
        ax.tick_params(axis="x", rotation=75)
        ax.set_xticks(part["target_month"].iloc[::3])
        ax.grid(alpha=0.2)
    axes[0, 0].legend()
    fig.suptitle("2025–Jan 2026 new-home price index: one-month-ahead forecasts")
    fig.supylabel("Month-on-month change (percentage points)")
    fig.tight_layout()
    fig.savefig(RESULTS / "shandong_backtest.png", dpi=160)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        raise FileNotFoundError("Run `python download_official_data.py` first")
    RESULTS.mkdir(exist_ok=True)
    df, numeric = build_dataset()
    known = df[df["target_change_pct"].notna()].copy()
    train = known[known["target_month"] <= pd.Period("2023-12")]
    validation = known[known["target_month"].between(pd.Period("2024-01"), pd.Period("2024-12"))]
    train_val = known[known["target_month"] <= pd.Period("2024-12")]
    test = known[known["target_month"].between(pd.Period("2025-01"), pd.Period("2026-01"))]
    if [len(train), len(validation), len(test)] != [1750, 840, 910]:
        raise ValueError(f"Unexpected split sizes: {len(train)}, {len(validation)}, {len(test)}")

    validation_metrics, _ = evaluate(train, validation, numeric)
    candidates = ["ridge", "gradient_boosting"]
    selected = min(candidates, key=lambda name: validation_metrics[name]["mae_percentage_points"])
    test_metrics, backtest = evaluate(train_val, test, numeric)
    backtest.to_csv(RESULTS / "backtest_2025_to_2026_01.csv", index=False, encoding="utf-8-sig")
    save_shandong_plot(backtest, selected)

    latest = df[df["month"] == pd.Period("2026-01")].copy()
    model = make_model(selected, numeric)
    model.fit(known[numeric + ["city"]], known["target_change_pct"])
    if selected == "ridge":
        names = model.named_steps["preprocess"].get_feature_names_out()
        coefficients = pd.DataFrame({
            "feature": names,
            "coefficient": model.named_steps["regressor"].coef_,
        })
        coefficients["absolute_weight"] = coefficients["coefficient"].abs()
        coefficients.sort_values("absolute_weight", ascending=False).to_csv(
            RESULTS / "ridge_coefficients.csv", index=False, encoding="utf-8-sig"
        )
    forecast = latest[["city", "period", "source_url"]].copy()
    forecast["forecast_month"] = "2026-02"
    forecast["predicted_change_pct"] = model.predict(latest[numeric + ["city"]])
    errors = (backtest["target_change_pct"] - backtest[f"pred_{selected}"]).abs()
    empirical_error = float(np.quantile(errors, 0.8))
    forecast["historical_80pct_abs_error_pp"] = empirical_error
    forecast.to_csv(RESULTS / "forecast_2026_02.csv", index=False, encoding="utf-8-sig")

    shandong_metrics = {}
    for city in SHANDONG:
        part = backtest[backtest["city"] == city]
        shandong_metrics[city] = scores(
            part["target_change_pct"].to_numpy(), part[f"pred_{selected}"].to_numpy()
        )
    new_only = [name for name in numeric if not name.startswith("second") and name != "national_second_mean"]
    ablation = make_model(selected, new_only)
    ablation.fit(train_val[new_only + ["city"]], train_val["target_change_pct"])
    ablation_prediction = ablation.predict(test[new_only + ["city"]])
    ablation_metrics = scores(test["target_change_pct"].to_numpy(), ablation_prediction)
    report = {
        "as_of_date": "2026-02-28",
        "latest_available_index_month": "2026-01",
        "latest_original_release": "2026-02-13",
        "data_months": 57,
        "cities_per_month": 70,
        "model_target": "next-month new-home price-index MoM percentage-point change",
        "split_rows": {"train": len(train), "validation": len(validation), "final_train": len(train_val), "test": len(test)},
        "validation": validation_metrics,
        "selected_on_validation": selected,
        "test": test_metrics,
        "shandong_test": shandong_metrics,
        "ablation_without_secondhand_features": ablation_metrics,
        "empirical_80pct_absolute_error_pp": round(empirical_error, 4),
        "notes": [
            "The official statistic is a city-level price index, not a price per square meter or an individual sale price.",
            "No observation first published after 2026-02-28 is used.",
            "The February 2026 forecast is made from the January 2026 release and is not scored here.",
        ],
    }
    (RESULTS / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print("Shandong February forecast:", flush=True)
    print(forecast[forecast["city"].isin(SHANDONG)][["city", "predicted_change_pct"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
