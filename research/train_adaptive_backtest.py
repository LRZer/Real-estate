"""Past-only rolling-origin experiments for a changing housing market.

This extension was designed after the old test outcomes were inspected. All
2025-Jan2026 comparisons are therefore retrospective, not a new blind test.
Policies are selected on 2024 rolling validation only and written to disk
before the retrospective evaluation. No February 2026 actual is loaded.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

# Keep tiny monthly fits efficient without depending on a BLAS introspection API.
for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch import nn

from train_forecast import build_dataset
from train_graph_temporal import GraphTemporalNet, SEEDS, TOP_K, load_panel, make_examples


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "results" / "adaptive"
POLICIES = ("expanding", "recent_24", "half_life_12")
FAMILIES = ("ridge", "temporal_net", "residual_graph_net")
EPOCHS = 60
HALF_LIFE = 12
torch.set_num_threads(1)


def save_json(name: str, value: dict) -> None:
    (OUTPUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def weights_for(months: list[str], policy: str) -> np.ndarray:
    latest = pd.Period(months[-1], freq="M").ordinal
    ages = np.array([latest - pd.Period(m, freq="M").ordinal for m in months])
    if policy == "half_life_12":
        weights = np.power(0.5, ages / HALF_LIFE)
    elif policy in ("expanding", "recent_24"):
        weights = np.ones(len(months))
    else:
        raise ValueError(policy)
    # Mean-one normalization keeps the Ridge penalty comparable across policies.
    return weights / weights.mean()


def fit_indices(examples: dict, target: str, policy: str) -> np.ndarray:
    indices = np.array([i for i, month in enumerate(examples["targets"][:-1]) if month < target])
    if policy == "recent_24":
        indices = indices[-24:]
    if len(indices) < 24 or examples["targets"][indices[-1]] >= target:
        raise ValueError("Insufficient history or future label in fit set")
    last_expected = str(pd.Period(target, freq="M") - 1)
    if examples["targets"][indices[-1]] != last_expected:
        raise ValueError("Training cutoff must equal the latest published input month")
    return indices


def fit_ridge(rows: pd.DataFrame, numeric: list[str], months: list[str],
              weights: np.ndarray, holdout: pd.DataFrame) -> np.ndarray:
    train = rows[rows["target_month"].astype(str).isin(months)]
    row_weights = train["target_month"].astype(str).map(dict(zip(months, weights))).to_numpy()
    scaler = StandardScaler().fit(train[numeric], sample_weight=row_weights)
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(train[["city"]])
    x = np.column_stack([scaler.transform(train[numeric]), encoder.transform(train[["city"]])])
    future_x = np.column_stack([scaler.transform(holdout[numeric]), encoder.transform(holdout[["city"]])])
    model = Ridge(alpha=25.0).fit(x, train["target_change_pct"], sample_weight=row_weights)
    return model.predict(future_x)


def weighted_graph(panel: np.ndarray, periods: list[str], months: list[str],
                   weights: np.ndarray, enabled: bool) -> np.ndarray:
    cities = panel.shape[1]
    if not enabled:
        return np.zeros((cities, cities), dtype=np.float32)
    values = panel[[periods.index(m) for m in months], :, 0].astype(np.float64)
    # Remove the contemporaneous common market movement before estimating edges.
    values -= values.mean(axis=1, keepdims=True)
    probabilities = weights / weights.sum()
    center = values - np.sum(values * probabilities[:, None], axis=0)
    covariance = (center * probabilities[:, None]).T @ center
    std = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    denominator = np.outer(std, std)
    corr = np.divide(covariance, denominator, out=np.zeros_like(covariance), where=denominator > 1e-12)
    corr = np.maximum(corr, 0.0)
    np.fill_diagonal(corr, 0.0)
    adjacency = np.zeros_like(corr)
    for c in range(cities):
        candidates = np.delete(np.arange(cities), c)
        nearest = candidates[np.argsort(corr[c, candidates])[-TOP_K:]]
        if corr[c, nearest].sum() > 0:
            adjacency[c, nearest] = corr[c, nearest] / corr[c, nearest].sum()
    return adjacency.astype(np.float32)


def fit_neural(examples: dict, panel: np.ndarray, periods: list[str], indices: np.ndarray,
               months: list[str], weights: np.ndarray, target_indices: np.ndarray,
               graph: bool) -> np.ndarray:
    training = examples["x"][indices]
    probabilities = weights / weights.sum()
    mean = np.sum(training.mean(axis=(1, 2)) * probabilities[:, None], axis=0)
    variance = np.sum(((training - mean) ** 2).mean(axis=(1, 2)) * probabilities[:, None], axis=0)
    std = np.sqrt(variance).clip(min=0.05)
    x = torch.from_numpy(((examples["x"] - mean) / std).astype(np.float32))
    last = torch.from_numpy(examples["x"][:, :, -1, 0])
    context = torch.from_numpy(examples["context"])
    calendar = torch.from_numpy(examples["calendar"])
    y = torch.from_numpy(examples["y"])
    sample_weights = torch.from_numpy(probabilities.astype(np.float32))
    adjacency = weighted_graph(panel, periods, months, weights, graph)
    forecasts = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        model = GraphTemporalNet(panel.shape[1], adjacency)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
        # Fixed budget: target-month outcomes never drive early stopping.
        for _ in range(EPOCHS):
            model.train()
            optimizer.zero_grad()
            prediction = model(x[indices], last[indices], context[indices], calendar[indices])
            city_losses = nn.functional.smooth_l1_loss(prediction, y[indices], beta=0.25, reduction="none")
            loss = (city_losses.mean(dim=1) * sample_weights).sum()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            forecasts.append(model(x[target_indices], last[target_indices],
                                   context[target_indices], calendar[target_indices]).numpy())
    return np.mean(forecasts, axis=0)


def fit_predict(family: str, policy: str, target: str, targets: list[str],
                examples: dict, panel: np.ndarray, periods: list[str], cities: list[str],
                rows: pd.DataFrame, numeric: list[str], audit: list[dict], stage: str) -> np.ndarray:
    indices = fit_indices(examples, target, policy)
    months = [examples["targets"][i] for i in indices]
    weights = weights_for(months, policy)
    target_indices = np.array([examples["targets"].index(m) for m in targets])
    # Both the label history and graph end before the first forecasted month.
    assert max(months) < min(targets)
    audit.append({
        "stage": stage, "family": family, "policy": policy,
        "first_forecast_month": targets[0], "last_forecast_month": targets[-1],
        "train_first_label_month": months[0], "train_last_label_month": months[-1],
        "train_months": len(months), "train_city_rows": len(months) * 70,
        "effective_weighted_months": float(weights.sum() ** 2 / np.sum(weights ** 2)),
        "oldest_to_newest_weight_ratio": float(weights[0] / weights[-1]),
        "graph_last_month": months[-1] if family == "residual_graph_net" else "",
        "target_month_used_for_fit": False,
    })
    if family == "ridge":
        holdout = rows[rows["target_month"].astype(str).isin(targets)].copy()
        holdout["target_string"] = holdout["target_month"].astype(str)
        holdout = holdout.sort_values(["target_string", "city"])
        expected = [(m, c) for m in targets for c in cities]
        if list(zip(holdout["target_string"], holdout["city"])) != expected:
            raise ValueError("Ridge and neural city/month alignment differs")
        return fit_ridge(rows, numeric, months, weights, holdout).reshape(len(targets), 70)
    return fit_neural(examples, panel, periods, indices, months, weights,
                      target_indices, graph=family == "residual_graph_net")


def metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    error = np.abs(actual - prediction)
    mask = actual != 0
    rising = actual > 0
    falling = actual < 0
    recalls = [np.mean(np.sign(prediction[a]) == np.sign(actual[a])) for a in (rising, falling) if a.any()]
    binary_recalls = [np.mean((prediction[a] > 0) == (actual[a] > 0))
                      for a in (rising, falling) if a.any()]
    return {
        "mae_pp": float(error.mean()), "bias_pp": float((prediction - actual).mean()),
        "rising_mae_pp": float(error[rising].mean()) if rising.any() else None,
        "falling_mae_pp": float(error[falling].mean()) if falling.any() else None,
        "balanced_binary_rise_accuracy_excluding_actual_zero": float(np.mean(binary_recalls)) if mask.any() else None,
        "balanced_threeway_sign_accuracy_on_nonzero_actual": float(np.mean(recalls)) if mask.any() else None,
    }


def block_interval(actual: np.ndarray, baseline: np.ndarray, forecast: np.ndarray) -> dict:
    gains = np.mean(np.abs(actual - baseline) - np.abs(actual - forecast), axis=1)
    rng = np.random.default_rng(2026)
    n = len(gains)
    starts = rng.integers(0, n, size=(20000, (n + 2) // 3))
    indices = ((starts[:, :, None] + np.arange(3)) % n).reshape(20000, -1)[:, :n]
    bounds = np.quantile(gains[indices].mean(axis=1), [0.025, 0.975])
    return {"mean_monthly_mae_gain_pp": float(gains.mean()),
            "circular_3month_block_95pct_interval_pp": bounds.tolist(),
            "months_beating_baseline": int((gains > 0).sum())}


def drift_diagnostics() -> None:
    source = pd.read_csv(ROOT / "data" / "official_nbs_70city.csv")
    source["year"] = source["period"].str[:4]
    yearly = source.groupby("year").agg(
        months=("period", "nunique"), new_mom_mean_pct=("new_mom_pct", "mean"),
        second_mom_mean_pct=("second_mom_pct", "mean"),
        falling_city_month_share=("new_mom_pct", lambda x: (x < 0).mean()),
    )
    yearly.to_csv(OUTPUT / "yearly_market_regimes.csv", encoding="utf-8-sig")
    monthly = source.groupby("period").agg(
        new_mean_pct=("new_mom_pct", "mean"), second_mean_pct=("second_mom_pct", "mean"),
        new_falling_share=("new_mom_pct", lambda x: (x < 0).mean()),
    )
    monthly.to_csv(OUTPUT / "monthly_market_regimes.csv", encoding="utf-8-sig")
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for field, label in (("new_mean_pct", "New homes"), ("second_mean_pct", "Second-hand homes")):
        axes[0].plot(monthly.index, monthly[field], label=label)
    axes[0].axhline(0, color="grey", linewidth=0.8)
    axes[0].set_ylabel("70-city unweighted mean (%)")
    axes[0].legend()
    axes[1].plot(monthly.index, monthly["new_falling_share"] * 100, color="#a65037")
    axes[1].set_ylabel("New-home declining city share (%)")
    axes[1].set_xticks(monthly.index[::6])
    axes[1].tick_params(axis="x", rotation=45)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle("Observed market shifts; monthly data through Jan 2026")
    fig.tight_layout()
    fig.savefig(OUTPUT / "market_regimes.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    drift_diagnostics()
    panel, periods, cities = load_panel()
    examples = make_examples(panel, periods)
    rows, numeric = build_dataset()
    validation = [m for m in examples["targets"] if "2024-01" <= m <= "2024-12"]
    test = [m for m in examples["targets"] if "2025-01" <= m <= "2026-01"]
    actual = examples["y"]
    val_indices = [examples["targets"].index(m) for m in validation]
    test_indices = [examples["targets"].index(m) for m in test]
    audit, validation_scores, validation_predictions = [], {}, {}
    for family in FAMILIES:
        validation_scores[family] = {}
        for policy in POLICIES:
            forecasts = []
            for month in validation:
                pred = fit_predict(family, policy, month, [month], examples, panel, periods,
                                   cities, rows, numeric, audit, "validation")
                forecasts.append(pred[0])
                if month.endswith(("03", "06", "09", "12")):
                    print(f"Validation {family}/{policy}: through {month}", flush=True)
            prediction = np.asarray(forecasts)
            validation_predictions[f"{family}__{policy}"] = prediction
            validation_scores[family][policy] = metrics(actual[val_indices], prediction)
            print(f"Validation score {family}/{policy}: {validation_scores[family][policy]['mae_pp']:.6f}", flush=True)
    selected = {family: min(POLICIES, key=lambda p: validation_scores[family][p]["mae_pp"])
                for family in FAMILIES}
    selected_family = min(FAMILIES, key=lambda f: validation_scores[f][selected[f]]["mae_pp"])
    selection = {
        "protocol": "monthly past-only rolling-origin validation",
        "validation_period": "2024-01..2024-12", "neural_fixed_epochs": EPOCHS,
        "policies": list(POLICIES), "half_life_months": HALF_LIFE,
        "validation_scores": validation_scores, "selected_policy_by_family": selected,
        "selected_family": selected_family,
        "selection_note": "Policies and model family chosen using 2024 only; written before 2025 evaluation.",
    }
    save_json("selection_2024.json", selection)
    print(f"Frozen policy selection: {selected}; selected family: {selected_family}", flush=True)

    forecasts = {"last_month": examples["x"][test_indices, :, -1, 0]}
    old = pd.read_csv(ROOT / "results" / "backtest_2025_to_2026_01.csv")
    forecasts["original_frozen_ridge"] = old.pivot(index="target_month", columns="city", values="pred_ridge").loc[test, cities].to_numpy()
    for family in FAMILIES:
        policy = selected[family]
        # Same selected policy and training budget, but no monthly refitting.
        forecasts[f"frozen_{family}"] = fit_predict(
            family, policy, test[0], test, examples, panel, periods, cities,
            rows, numeric, audit, "frozen_control")
        predictions = []
        for month in test:
            pred = fit_predict(family, policy, month, [month], examples, panel, periods,
                               cities, rows, numeric, audit, "retrospective_test")
            predictions.append(pred[0])
            if month.endswith(("03", "06", "09", "12", "01")):
                print(f"Rolling retrospective {family}/{policy}: through {month}", flush=True)
        forecasts[f"rolling_{family}"] = np.asarray(predictions)

    # Expanding Ridge is a low-cost control that isolates the benefit of recency.
    if selected["ridge"] == "expanding":
        forecasts["rolling_expanding_ridge"] = forecasts["rolling_ridge"]
    else:
        forecasts["rolling_expanding_ridge"] = np.asarray([
            fit_predict("ridge", "expanding", month, [month], examples, panel, periods,
                        cities, rows, numeric, audit, "expanding_control")[0] for month in test
        ])

    records, val_records = [], []
    source_links = pd.read_csv(ROOT / "data" / "official_nbs_70city.csv").set_index(["period", "city"])["source_url"].to_dict()
    for k, month in enumerate(test):
        input_month = str(pd.Period(month, freq="M") - 1)
        for c, city in enumerate(cities):
            record = {"target_month": month, "input_month": input_month, "city": city,
                      "actual_change_pct": float(actual[test_indices[k], c]),
                      "input_source_url": source_links[(input_month, city)],
                      "target_source_url": source_links[(month, city)]}
            record.update({f"pred_{name}": float(pred[k, c]) for name, pred in forecasts.items()})
            records.append(record)
    for k, month in enumerate(validation):
        for c, city in enumerate(cities):
            record = {"target_month": month, "city": city, "actual_change_pct": float(actual[val_indices[k], c])}
            record.update({f"pred_{name}": float(pred[k, c]) for name, pred in validation_predictions.items()})
            val_records.append(record)
    pd.DataFrame(records).to_csv(OUTPUT / "rolling_backtest_exploratory.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(val_records).to_csv(OUTPUT / "rolling_validation_2024.csv", index=False, encoding="utf-8-sig")

    all_metrics = {name: metrics(actual[test_indices], pred) for name, pred in forecasts.items()}
    intervals = {name: block_interval(actual[test_indices], forecasts["last_month"], pred)
                 for name, pred in forecasts.items() if name != "last_month"}
    summer = [i for i, month in enumerate(test) if "2025-06" <= month <= "2025-08"]
    segments = {
        "2025_Jun_Aug": {name: metrics(actual[test_indices][summer], pred[summer]) for name, pred in forecasts.items()},
        "2025_only": {name: metrics(actual[test_indices][:-1], pred[:-1]) for name, pred in forecasts.items()},
        "2026_Jan_only": {name: metrics(actual[test_indices][-1:], pred[-1:]) for name, pred in forecasts.items()},
    }
    monthly_records = []
    for k, month in enumerate(test):
        item = {"target_month": month}
        item.update({f"mae_{name}": float(np.abs(actual[test_indices[k]] - pred[k]).mean())
                     for name, pred in forecasts.items()})
        monthly_records.append(item)
    pd.DataFrame(monthly_records).to_csv(OUTPUT / "monthly_errors.csv", index=False, encoding="utf-8-sig")

    final = pd.DataFrame({"city": cities, "input_month": "2026-01", "forecast_month": "2026-02",
                          "input_source_url": [source_links[("2026-01", city)] for city in cities]})
    for family in FAMILIES:
        prediction = fit_predict(family, selected[family], "2026-02", ["2026-02"], examples,
                                 panel, periods, cities, rows, numeric, audit, "as_of_forecast")
        final[f"pred_{family}"] = prediction[0]
    final["selected_prediction_pct"] = final[f"pred_{selected_family}"]
    final.to_csv(OUTPUT / "forecast_2026_02.csv", index=False, encoding="utf-8-sig")
    audit_frame = pd.DataFrame(audit)
    if (audit_frame["train_last_label_month"] >= audit_frame["first_forecast_month"]).any():
        raise AssertionError("Temporal leakage in recorded fit cutoffs")
    audit_frame.to_csv(OUTPUT / "training_cutoff_audit.csv", index=False, encoding="utf-8-sig")
    report = {
        **selection, "as_of_date": "2026-02-28", "latest_actual_month": "2026-01",
        "retrospective_period": "2025-01..2026-01", "retrospective_scores": all_metrics,
        "gain_vs_last_month_baseline": intervals, "segments": segments,
        "methodological_note": "The old test outcomes were already inspected before this extension. These are exploratory retrospective results, not a new blind test.",
        "monthly_update_rule": "At target month m, fit only labels through m-1 after their official release; use the resulting actual at the next origin only.",
        "graph_rule": "Weighted correlations of market-demeaned cities; same past label window/weights as fitting, recomputed monthly.",
        "market_mean_note": "70-city unweighted descriptive means, not an official national price index or annual price growth.",
    }
    save_json("adaptive_results.json", report)
    print(json.dumps({"selected": selected_family, "policies": selected, "scores": all_metrics,
                      "summer": segments["2025_Jun_Aug"]}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
