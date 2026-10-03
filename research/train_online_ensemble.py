"""Past-error-weighted experts inspired by online ensembling research.

This is a small independent implementation, not OneNet or its RL algorithm.
It reads the previously saved rolling-origin predictions, selects its memory
and temperature on 2024, and freezes that rule before retrospective 2025 work.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from train_adaptive_backtest import OUTPUT, ROOT, block_interval, metrics, save_json


EXPERTS = ("ridge", "temporal_net", "residual_graph_net", "last_month")
MEMORIES = (0.5, 0.8, 0.95)
TEMPERATURES = (0.02, 0.05, 0.1)


def walk_experts(predictions: np.ndarray, actual: np.ndarray, memory: float,
                 temperature: float, initial_risk: np.ndarray | None = None) -> tuple:
    """Predict first, then update expert risks after the current label arrives."""
    risk = np.zeros(predictions.shape[-1]) if initial_risk is None else initial_risk.copy()
    combined, audit = [], []
    for t in range(len(predictions)):
        logits = -(risk - risk.min()) / temperature
        weights = np.exp(logits) / np.exp(logits).sum()
        # Weight decisions precede access to this month's actual values.
        combined.append(predictions[t] @ weights)
        item = {"month_index": t}
        item.update({f"pre_forecast_weight_{name}": float(weights[e]) for e, name in enumerate(EXPERTS)})
        item.update({f"past_risk_{name}_pp": float(risk[e]) for e, name in enumerate(EXPERTS)})
        error = np.mean(np.abs(predictions[t] - actual[t, :, None]), axis=0)
        risk = memory * risk + (1 - memory) * error
        item.update({f"released_month_mae_{name}_pp": float(error[e]) for e, name in enumerate(EXPERTS)})
        audit.append(item)
    return np.asarray(combined), risk, audit


def matrix(frame: pd.DataFrame, months: list[str], cities: list[str], field: str) -> np.ndarray:
    return frame.pivot(index="target_month", columns="city", values=field).loc[months, cities].to_numpy()


def save_comparison_plot(months: list[str], actual: np.ndarray, ensemble: np.ndarray,
                         test: pd.DataFrame, cities: list[str], weight_audit: list[dict]) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for field, label, color in (
        ("pred_last_month", "Last-month baseline", "#888888"),
        ("pred_original_frozen_ridge", "Frozen Ridge", "#b54e42"),
        ("pred_rolling_ridge", "Monthly Ridge", "#1b6b81"),
        ("pred_rolling_temporal_net", "Monthly temporal net", "#7568a9"),
    ):
        prediction = matrix(test, months, cities, field)
        axes[0].plot(months, np.abs(actual - prediction).mean(axis=1), label=label, color=color)
    axes[0].plot(months, np.abs(actual - ensemble).mean(axis=1), label="Online ensemble (exploratory)",
                 color="#cc8b21", linestyle="--")
    axes[0].set_ylabel("Monthly MAE (percentage points)")
    axes[0].legend(fontsize=9, ncol=2)
    weights = [[item[f"pre_forecast_weight_{name}"] for item in weight_audit] for name in EXPERTS]
    axes[1].stackplot(months, *weights, labels=["Ridge", "Temporal net", "Graph net", "Last month"],
                      colors=["#1b6b81", "#7568a9", "#60a38e", "#b7bec2"], alpha=0.85)
    axes[1].set_ylabel("Expert weights before each forecast")
    axes[1].set_ylim(0, 1)
    axes[1].legend(fontsize=9, ncol=4, loc="lower left")
    axes[1].tick_params(axis="x", rotation=45)
    axes[0].grid(alpha=0.2)
    fig.suptitle("Adaptation to a changing market: retrospective rolling-origin comparisons")
    fig.tight_layout()
    fig.savefig(OUTPUT / "adaptive_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    selection = json.loads((OUTPUT / "selection_2024.json").read_text(encoding="utf-8"))
    validation = pd.read_csv(OUTPUT / "rolling_validation_2024.csv")
    test = pd.read_csv(OUTPUT / "rolling_backtest_exploratory.csv")
    source = pd.read_csv(ROOT / "data" / "official_nbs_70city.csv")
    cities = sorted(source.city.unique())
    val_months = sorted(validation.target_month.unique())
    test_months = sorted(test.target_month.unique())
    val_actual = matrix(validation, val_months, cities, "actual_change_pct")
    test_actual = matrix(test, test_months, cities, "actual_change_pct")
    validation_parts = [matrix(validation, val_months, cities, f"pred_{family}__{selection['selected_policy_by_family'][family]}")
                        for family in EXPERTS[:-1]]
    prior_months = [str(pd.Period(m, freq="M") - 1) for m in val_months]
    last_month_validation = source.pivot(index="period", columns="city", values="new_mom_pct").loc[prior_months, cities].to_numpy()
    validation_parts.append(last_month_validation)
    val_experts = np.stack(validation_parts, axis=-1)
    test_experts = np.stack([matrix(test, test_months, cities, f"pred_rolling_{family}")
                             for family in EXPERTS[:-1]] +
                            [matrix(test, test_months, cities, "pred_last_month")], axis=-1)
    candidates, states = {}, {}
    for memory in MEMORIES:
        for temperature in TEMPERATURES:
            name = f"memory_{memory}_temperature_{temperature}"
            prediction, risk, audit = walk_experts(val_experts, val_actual, memory, temperature)
            candidates[name] = {"memory": memory, "temperature_pp": temperature,
                                "validation_mae_pp": float(np.abs(prediction - val_actual).mean())}
            states[name] = (prediction, risk, audit)
    selected = min(candidates, key=lambda name: candidates[name]["validation_mae_pp"])
    rule = candidates[selected]
    save_json("ensemble_selection_2024.json", {
        "experts": list(EXPERTS), "grid": candidates, "selected_rule": rule,
        "uniform_validation_mae_pp": float(np.abs(val_experts.mean(axis=-1) - val_actual).mean()),
        "selection_data": "2024 only; saved before retrospective evaluation",
    })
    val_prediction, end_validation_risk, val_audit = states[selected]
    forecast, final_risk, test_audit = walk_experts(
        test_experts, test_actual, rule["memory"], rule["temperature_pp"], end_validation_risk)
    uniform = test_experts.mean(axis=-1)
    for records, months, filename in ((val_audit, val_months, "ensemble_weights_2024.csv"),
                                       (test_audit, test_months, "ensemble_weights_2025_to_2026_01.csv")):
        for item, month in zip(records, months):
            item["forecast_month"] = month
            item["past_risk_known_through"] = str(pd.Period(month, freq="M") - 1)
            item["released_month_used_only_after_prediction"] = True
        pd.DataFrame(records).to_csv(OUTPUT / filename, index=False, encoding="utf-8-sig")
    records = []
    for i, month in enumerate(test_months):
        for c, city in enumerate(cities):
            records.append({"target_month": month, "city": city, "actual_change_pct": float(test_actual[i, c]),
                            "pred_online_ensemble": float(forecast[i, c]), "pred_uniform_ensemble": float(uniform[i, c])})
    pd.DataFrame(records).to_csv(OUTPUT / "ensemble_backtest_exploratory.csv", index=False, encoding="utf-8-sig")

    # Metamorphic check: changing y_t and all future labels cannot change yhat_t.
    altered = test_actual.copy()
    changed_from = 5
    altered[changed_from:] += 100
    changed, _, changed_audit = walk_experts(test_experts, altered, rule["memory"],
                                            rule["temperature_pp"], end_validation_risk)
    assert np.array_equal(changed[:changed_from + 1], forecast[:changed_from + 1])
    assert changed_audit[changed_from]["pre_forecast_weight_ridge"] == test_audit[changed_from]["pre_forecast_weight_ridge"]

    final = pd.read_csv(OUTPUT / "forecast_2026_02.csv").set_index("city").loc[cities].copy()
    latest = source[source.period == "2026-01"].set_index("city").loc[cities, "new_mom_pct"].to_numpy()
    final_experts = np.column_stack([final[f"pred_{family}"].to_numpy() for family in EXPERTS[:-1]] + [latest])
    logits = -(final_risk - final_risk.min()) / rule["temperature_pp"]
    weights = np.exp(logits) / np.exp(logits).sum()
    final["pred_online_ensemble"] = final_experts @ weights
    final.to_csv(OUTPUT / "ensemble_forecast_2026_02.csv", encoding="utf-8-sig")
    summer = [i for i, m in enumerate(test_months) if "2025-06" <= m <= "2025-08"]
    ridge_mae = selection["validation_scores"]["ridge"][selection["selected_policy_by_family"]["ridge"]]["mae_pp"]
    report = {
        "algorithm": "shared-city EWMA expert risk -> softmax weights -> weighted prediction",
        "research_relation": "Inspired by OneNet (NeurIPS 2023) online ensembling; not a reproduction and no RL implementation.",
        "selected_rule": rule, "validation": metrics(val_actual, val_prediction),
        "beats_selected_single_model_on_validation": rule["validation_mae_pp"] < ridge_mae,
        "retrospective": {"online_ensemble": metrics(test_actual, forecast), "uniform_ensemble": metrics(test_actual, uniform)},
        "summer_2025": metrics(test_actual[summer], forecast[summer]),
        "gain_vs_last_month_baseline": block_interval(test_actual, test_experts[:, :, -1], forecast),
        "as_of_2026_02_expert_weights": dict(zip(EXPERTS, weights.tolist())),
        "future_label_invariance_check": True,
        "methodological_note": "All 2025-Jan2026 outcomes were inspected before this extension. The comparisons are retrospective, not a new blind test; the small validation grid also adds selection uncertainty.",
    }
    save_json("online_ensemble_results.json", report)
    save_comparison_plot(test_months, test_actual, forecast, test, cities, test_audit)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
