"""Verify temporal cutoffs and invariance to unavailable future outcomes."""

from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from train_adaptive_backtest import OUTPUT, fit_indices, fit_predict, weights_for
from train_forecast import build_dataset
from train_graph_temporal import load_panel, make_examples
from train_online_ensemble import walk_experts


def main() -> None:
    audit = pd.read_csv(OUTPUT / "training_cutoff_audit.csv")
    assert (audit.train_last_label_month < audit.first_forecast_month).all()
    assert not audit.target_month_used_for_fit.any()
    for item in audit.itertuples():
        expected = str(pd.Period(item.first_forecast_month, freq="M") - 1)
        assert item.train_last_label_month == expected
        if item.policy == "recent_24":
            assert item.train_months == 24
        if item.family == "residual_graph_net":
            assert item.graph_last_month == item.train_last_label_month
    selection = json.loads((OUTPUT / "selection_2024.json").read_text(encoding="utf-8"))
    for family, policy in selection["selected_policy_by_family"].items():
        candidates = selection["validation_scores"][family]
        assert policy == min(candidates, key=lambda p: candidates[p]["mae_pp"])
    predictions = pd.read_csv(OUTPUT / "rolling_backtest_exploratory.csv")
    assert len(predictions) == 910
    assert predictions.groupby("target_month").city.nunique().eq(70).all()
    assert not predictions.duplicated(["target_month", "city"]).any()
    assert np.isfinite(predictions.select_dtypes(include=np.number)).all().all()
    # The expanding Ridge implementation must reproduce the original frozen fit.
    np.testing.assert_allclose(predictions.pred_frozen_ridge,
                               predictions.pred_original_frozen_ridge, atol=1e-10, rtol=0)

    panel, periods, cities = load_panel()
    examples = make_examples(panel, periods)
    rows, numeric = build_dataset()
    changed_examples = copy.deepcopy(examples)
    first = examples["targets"].index("2025-01")
    changed_examples["y"][first:] += 100
    changed_examples["x"][first + 1:] += 100
    changed_examples["context"][first + 1:] += 100
    changed_rows = rows.copy()
    changed_rows.loc[changed_rows.target_month >= pd.Period("2025-01"), "target_change_pct"] += 100
    changed_panel = panel.copy()
    changed_panel[periods.index("2025-01"):] += 100
    checks = {}
    for family in ("ridge", "residual_graph_net"):
        original = fit_predict(family, "half_life_12", "2025-01", ["2025-01"], examples,
                               panel, periods, cities, rows, numeric, [], "verification")
        changed = fit_predict(family, "half_life_12", "2025-01", ["2025-01"], changed_examples,
                              changed_panel, periods, cities, changed_rows, numeric, [], "verification")
        assert np.array_equal(original, changed), family
        checks[f"{family}_future_data_invariance"] = True
    indices = fit_indices(examples, "2025-01", "half_life_12")
    months = [examples["targets"][i] for i in indices]
    weights = weights_for(months, "half_life_12")
    assert np.isclose(weights[-13] / weights[-1], 0.5)

    # A released label can affect later weights, never its own forecast weights.
    toy_predictions = np.arange(5 * 70 * 4).reshape(5, 70, 4) / 1000
    toy_actual = np.zeros((5, 70))
    altered_actual = toy_actual.copy()
    altered_actual[2:] = 100
    original, _, _ = walk_experts(toy_predictions, toy_actual, 0.5, 0.02)
    changed, _, _ = walk_experts(toy_predictions, altered_actual, 0.5, 0.02)
    assert np.array_equal(original[:3], changed[:3])
    checks.update({"ensemble_predict_before_label_update": True,
                   "past_only_cutoff_rows_checked": len(audit), "city_month_predictions_checked": len(predictions),
                   "fixed_ridge_reproduces_original": True, "half_life_weight_check": True})
    (OUTPUT / "protocol_verification.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == "__main__":
    main()
