"""Meaningful checks for temporal leakage, causal state and saved predictions."""
from __future__ import annotations
import copy
import json
import numpy as np
import pandas as pd
from research_models import ROOT, load_research_data, model_fit, past_indices, prepare_tensors
from run_research_experiments import OUT, CASES, bias_correction, online_mix


def main():
    data = load_research_data()
    target = "2025-01"
    idx = data["targets"].index(target)
    changed = copy.deepcopy(data)
    changed["y"][idx:] += 100
    for name in ("seq", "features", "global_features"):
        changed[name][idx+1:] += 100
    checks = {"same_city_month_alignment": data["seq"].shape == (51, 70, 7, 2),
              "same_19_numeric_features": data["features"].shape == (51, 70, 19)}
    for name in ("ridge_25", "hgb_7", "nlinear", "fair_tcn", "revin_tcn", "decomp_ridge", "decomp_revin", "multitask_0.05_weighted"):
        original = model_fit(data, CASES[name], target, seeds=(13,))
        alternative = model_fit(changed, CASES[name], target, seeds=(13,))
        for key in ("prediction", "probabilities", "market_prediction"):
            if key in original:
                np.testing.assert_array_equal(original[key], alternative[key])
                assert np.isfinite(original[key]).all()
        assert int(original["parameter_count"]) <= 5000
        checks[name+"_future_invariance"] = True
        print(name, "passed", int(original["parameter_count"]), flush=True)
    past = past_indices(data, target)
    _, stats = prepare_tensors(data, past)
    _, changed_stats = prepare_tensors(changed, past)
    for key in stats:
        np.testing.assert_array_equal(stats[key], changed_stats[key])
    checks["past_only_normalizers"] = True
    toy = np.arange(5*70).reshape(5, 70)/1000
    y = np.zeros_like(toy)
    altered_y = y.copy(); altered_y[2:] = 100
    pred, _, _ = bias_correction(toy, y, CASES["bias_0.5_1.0"])
    altered_pred, _, _ = bias_correction(toy, altered_y, CASES["bias_0.5_1.0"])
    np.testing.assert_array_equal(pred[:3], altered_pred[:3])
    experts = np.stack([toy, toy+0.2, toy-0.1], axis=-1)
    pred, _, _ = online_mix(experts, y)
    altered_pred, _, _ = online_mix(experts, altered_y)
    np.testing.assert_array_equal(pred[:3], altered_pred[:3])
    checks["bias_predict_before_update"] = True
    checks["ensemble_predict_before_update"] = True
    adaptive = pd.read_csv(ROOT / "results/adaptive/rolling_backtest_exploratory.csv")
    reference = adaptive.query("target_month == '2025-01'").set_index("city").loc[data["cities"], "pred_rolling_ridge"].to_numpy()
    ridge = model_fit(data, CASES["ridge_25"], target)["prediction"]
    np.testing.assert_allclose(ridge, reference, atol=1e-10, rtol=0)
    checks["ridge_reproduces_previous_implementation"] = True
    audit_path = OUT / "training_cutoff_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        assert (audit.last_train_target < audit.forecast_month).all()
        assert not audit.train_future_target_used.any()
        expected = audit.forecast_month.map(lambda m: str(pd.Period(m)-1))
        assert audit.last_train_target.eq(expected).all()
        checks["fit_cutoffs_checked"] = len(audit)
        for filename, count in (("validation_predictions_2024.csv", 840), ("retrospective_predictions.csv", 910), ("forecast_2026_02.csv", 70)):
            frame = pd.read_csv(OUT / filename)
            assert len(frame) == count
            assert not frame.duplicated(["target_month", "city"]).any()
            assert np.isfinite(frame.select_dtypes(include=np.number)).all().all()
        checks["saved_city_month_predictions_checked"] = 1820
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "protocol_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
