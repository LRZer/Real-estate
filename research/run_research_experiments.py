"""Execute the predeclared small-panel research plan; no post-cutoff actuals."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import os
from pathlib import Path

for thread_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(thread_variable, "1")

import numpy as np
import pandas as pd
import torch
import sklearn

from research_models import ROOT, SEEDS, FINAL_SEEDS, load_research_data, model_fit, past_indices
from train_adaptive_backtest import metrics, block_interval

OUT = ROOT / "results" / "research"
CACHE = OUT / "cache"
VALIDATION = list(pd.period_range("2024-01", "2024-12", freq="M").astype(str))
RETROSPECTIVE = list(pd.period_range("2025-01", "2026-01", freq="M").astype(str))
CASES = {
    **{f"ridge_{a}": {"kind": "ridge", "alpha": a, "group": "baseline"} for a in (5, 25, 100)},
    **{f"hgb_{n}": {"kind": "hgb", "leaves": n, "group": "baseline"} for n in (15, 7)},
    "nlinear": {"kind": "nlinear", "group": "baseline"},
    "legacy_tcn": {"kind": "legacy", "group": "reference"},
    "fair_tcn": {"kind": "fair", "group": "fair_input"},
    "revin_tcn": {"kind": "revin", "group": "normalization"},
    **{f"bias_{rho}_{strength}": {"kind": "bias", "rho": rho, "strength": strength, "group": "calibration"}
       for rho in (0.5, 0.8) for strength in (0.5, 1.0)},
    "decomp_ridge": {"kind": "decomp_ridge", "group": "decomposition"},
    "decomp_tcn": {"kind": "decomp", "group": "decomposition"},
    "decomp_revin": {"kind": "decomp_revin", "group": "decomposition"},
    **{f"multitask_{eta}_{'weighted' if weighted else 'plain'}":
       {"kind": "multitask", "eta": eta, "weighted": weighted, "group": "direction"}
       for eta in (0.01, 0.05) for weighted in (False, True)},
}
NEURAL = [name for name, case in CASES.items() if case["kind"] not in ("ridge", "hgb", "decomp_ridge", "bias")]
NONLINEAR_CANDIDATES = [n for n in NEURAL if n not in ("nlinear", "legacy_tcn")]


def write_json(name: str, payload: dict) -> None:
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def fingerprint() -> str:
    digest = hashlib.sha256()
    for path in (ROOT / "research_models.py", ROOT / "run_research_experiments.py", ROOT / "data/official_nbs_70city.csv"):
        digest.update(path.read_bytes())
    return digest.hexdigest()


def cached_fit(data: dict, name: str, month: str, seeds=SEEDS) -> dict:
    key = hashlib.sha256((fingerprint()+name+month+str(tuple(seeds))).encode()).hexdigest()[:16]
    path = CACHE / f"{name}_{month}_{key}.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            return {k: saved[k] for k in saved.files}
    result = model_fit(data, CASES[name], month, seeds)
    if name in NEURAL and int(result["parameter_count"]) > 5000:
        raise ValueError("The predeclared neural parameter budget was exceeded")
    np.savez_compressed(path, **result)
    return result


def score(data: dict, months: list[str], result: dict) -> dict:
    indices = [data["targets"].index(m) for m in months]
    actual = data["y"][indices]
    report = metrics(actual, result["prediction"])
    monthly = np.abs(actual-result["prediction"]).mean(axis=1)
    report["monthly_mae_pp"] = dict(zip(months, monthly.tolist()))
    report["first_half_mae_pp"] = float(monthly[:len(monthly)//2].mean())
    report["second_half_mae_pp"] = float(monthly[len(monthly)//2:].mean())
    seed_pred = result["seed_predictions"]
    report["seed_mae_pp"] = np.abs(seed_pred-actual[:, None]).mean(axis=(0, 2)).tolist()
    report["seed_mae_std_pp"] = float(np.std(report["seed_mae_pp"]))
    if "market_prediction" in result:
        predicted_market = result["market_prediction"]
        actual_market = actual.mean(axis=1)
        market_error = predicted_market-actual_market
        residual_error = result["prediction"]-predicted_market[:, None] - (actual-actual_market[:, None])
        report["market_mae_pp"] = float(np.abs(market_error).mean())
        report["residual_mae_pp"] = float(np.abs(residual_error).mean())
    if "probabilities" in result:
        prediction = result["probabilities"].argmax(axis=-1)-1
        classes = np.sign(actual)
        recalls = [float(np.mean(prediction[classes==c]==c)) for c in (-1, 0, 1)]
        rise = prediction > 0
        true_rise = actual > 0
        report["classifier_rising_precision"] = float(true_rise[rise].mean()) if rise.any() else 0.0
        report["classifier_rising_recall"] = float(rise[true_rise].mean())
        report["classifier_macro_recall"] = float(np.mean(recalls))
        report["classifier_balanced_binary_rise_accuracy"] = float(np.mean([
            np.mean(rise[actual>0]), np.mean(~rise[actual<0])]))
        report["classifier_regression_direction_disagreement"] = float(np.mean(
            prediction != np.where(result["prediction"] > 0.05, 1, np.where(result["prediction"] < -0.05, -1, 0))))
    return report


def roll(data: dict, name: str, months: list[str], seeds=SEEDS, audit: list | None = None) -> dict:
    collected: dict[str, list] = {}
    for month in months:
        result = cached_fit(data, name, month, seeds)
        for key, value in result.items():
            if key != "parameter_count":
                collected.setdefault(key, []).append(value)
        if audit is not None:
            fit = past_indices(data, month)
            audit.append({"case": name, "forecast_month": month,
                          "first_train_target": data["targets"][fit[0]],
                          "last_train_target": data["targets"][fit[-1]],
                          "train_months": len(fit), "train_city_rows": len(fit)*70,
                          "seeds": ",".join(map(str, seeds)) if name in NEURAL else "deterministic",
                          "train_future_target_used": False,
                          "parameter_count": int(result["parameter_count"]),
                          "stage": "validation" if month in VALIDATION else "retrospective"})
        if month.endswith(("03", "06", "09", "12", "01")):
            print(f"{name} through {month}", flush=True)
    return {key: np.asarray(value) for key, value in collected.items()}


def bias_correction(base: np.ndarray, actual: np.ndarray, case: dict,
                    initial: float = 0.0) -> tuple[np.ndarray, float, list]:
    state = initial
    predictions, records = [], []
    for forecast, outcome in zip(base, actual):
        predictions.append(forecast-case["strength"]*state)
        record = {"past_bias_pp": state, "correction_pp": -case["strength"]*state}
        # Track base, not corrected, prediction error, always after forecasting.
        current_error = float((forecast-outcome).mean())
        state = case["rho"]*state+(1-case["rho"])*current_error
        record.update(released_base_error_pp=current_error, updated_bias_pp=state)
        records.append(record)
    return np.asarray(predictions), state, records


def as_result(prediction: np.ndarray) -> dict:
    return {"prediction": prediction, "seed_predictions": prediction[:, None]}


def save_predictions(filename: str, data: dict, months: list[str], results: dict) -> None:
    records = []
    for i, month in enumerate(months):
        input_month = str(pd.Period(month, freq="M")-1)
        for c, city in enumerate(data["cities"]):
            record = {"target_month": month, "input_month": input_month, "city": city,
                      "input_source_url": data["source_links"][(input_month, city)]}
            target = data["targets"].index(month)
            if target < len(data["y"]):
                record.update(actual_change_pct=float(data["y"][target, c]),
                              target_source_url=data["source_links"][(month, city)])
            record.update({f"pred_{name}": float(result["prediction"][i, c]) for name, result in results.items()})
            for name, result in results.items():
                if "probabilities" in result:
                    record.update({f"prob_{name}_{label}": float(result["probabilities"][i, c, j])
                                   for j, label in enumerate(("fall", "flat", "rise"))})
            records.append(record)
    pd.DataFrame(records).to_csv(OUT / filename, index=False, encoding="utf-8-sig")


def gate(candidate: dict, baseline: dict) -> dict:
    gains = 1-candidate["mae_pp"]/baseline["mae_pp"]
    months = sum(candidate["monthly_mae_pp"][m] < baseline["monthly_mae_pp"][m] for m in VALIDATION)
    halves = [candidate[k]/baseline[k]-1 for k in ("first_half_mae_pp", "second_half_mae_pp")]
    return {"relative_mae_gain": gains, "months_improved": months,
            "half_period_error_changes": halves,
            "passes_predeclared_price_gate": bool(gains >= 0.03 and months >= 8 and max(halves) <= 0.05)}


def validate(data: dict) -> tuple[dict, dict, list, dict]:
    results, scores, audit = {}, {}, []
    for name, case in CASES.items():
        if case["kind"] == "bias":
            indices = [data["targets"].index(m) for m in VALIDATION]
            prediction, end_state, records = bias_correction(results["ridge_25"]["prediction"], data["y"][indices], case)
            result = as_result(prediction)
            for row, month in zip(records, VALIDATION):
                row.update(target_month=month, state_known_through=str(pd.Period(month)-1))
            pd.DataFrame(records).to_csv(OUT / f"{name}_validation_state.csv", index=False, encoding="utf-8-sig")
        else:
            result = roll(data, name, VALIDATION, audit=audit)
        results[name] = result
        scores[name] = score(data, VALIDATION, result)
        print(f"VALIDATION {name}: {scores[name]['mae_pp']:.6f}", flush=True)
        write_json("validation_progress.json", scores)
    # The final neural family is selected on three seeds, then checked on five.
    screening_results = results.copy()
    save_predictions("validation_screening_three_seeds.csv", data, VALIDATION, screening_results)
    write_json("validation_screening_results.json", scores)
    best_neural = min(NONLINEAR_CANDIDATES, key=lambda n: scores[n]["mae_pp"])
    direction_cases = [n for n in CASES if CASES[n]["kind"] == "multitask"]
    best_direction = max(direction_cases, key=lambda n: scores[n]["classifier_balanced_binary_rise_accuracy"])
    for name in sorted({best_neural, best_direction}):
        scores[name]["three_seed_screening_mae_pp"] = scores[name]["mae_pp"]
        result = roll(data, name, VALIDATION, FINAL_SEEDS, audit)
        extra = score(data, VALIDATION, result)
        extra["three_seed_screening_mae_pp"] = scores[name]["three_seed_screening_mae_pp"]
        scores[name] = extra
        results[name] = result
    save_predictions("validation_predictions_2024.csv", data, VALIDATION, results)
    write_json("validation_results.json", scores)
    return results, scores, audit, screening_results


def online_mix(experts: np.ndarray, actual: np.ndarray, initial=None) -> tuple:
    risk = np.zeros(experts.shape[-1]) if initial is None else initial.copy()
    predictions, records = [], []
    for i in range(len(experts)):
        logits = -(risk-risk.min())/0.02
        weights = np.exp(logits)/np.exp(logits).sum()
        predictions.append(experts[i]@weights)
        records.append({"weights_before_forecast": weights.tolist(), "past_risk_pp": risk.tolist()})
        risk = 0.5*risk+0.5*np.abs(experts[i]-actual[i, :, None]).mean(axis=0)
    return np.asarray(predictions), risk, records


def select(data: dict, results: dict, scores: dict) -> dict:
    simple = [n for n, c in CASES.items() if c["group"] == "baseline"]
    baseline = min(simple, key=lambda n: scores[n]["mae_pp"])
    neural = min(NONLINEAR_CANDIDATES, key=lambda n: scores[n].get("three_seed_screening_mae_pp", scores[n]["mae_pp"]))
    direction = max([n for n in CASES if CASES[n]["kind"] == "multitask"],
                    key=lambda n: scores[n]["classifier_balanced_binary_rise_accuracy"])
    gates = {n: gate(s, scores[baseline]) for n, s in scores.items()}
    eligible = [n for n in CASES if n not in simple and gates[n]["passes_predeclared_price_gate"]]
    primary = min(eligible, key=lambda n: scores[n]["mae_pp"]) if eligible else baseline
    direction_base = scores["decomp_tcn"]
    direction_gates = {}
    for name in [n for n in CASES if CASES[n]["kind"] == "multitask"]:
        s = scores[name]
        direction_gates[name] = {
            "relative_overall_mae_change": s["mae_pp"]/direction_base["mae_pp"]-1,
            "balanced_accuracy_gain": s["classifier_balanced_binary_rise_accuracy"]-direction_base["balanced_binary_rise_accuracy_excluding_actual_zero"],
            "relative_rising_mae_gain": 1-s["rising_mae_pp"]/direction_base["rising_mae_pp"],
        }
        g = direction_gates[name]
        g["passes_direction_gate"] = bool(g["relative_overall_mae_change"] <= 0.02 and
            g["balanced_accuracy_gain"] >= 0.05 and g["relative_rising_mae_gain"] >= 0.1)
    # Every selected family is locked before any new retrospective evaluation.
    representatives = {baseline, primary, neural, direction, "ridge_25", "legacy_tcn", "fair_tcn", "revin_tcn", "decomp_tcn"}
    for group in ("calibration", "decomposition"):
        options = [n for n, c in CASES.items() if c["group"] == group]
        representatives.add(min(options, key=lambda n: scores[n]["mae_pp"]))
    selected = {"best_simple_baseline": baseline, "best_neural_screened": neural,
                "direction_representative": direction, "primary": primary,
                "price_gates": gates, "direction_gates": direction_gates,
                "retrospective_representatives": sorted(representatives),
                "final_five_seed_cases": sorted({neural, direction}),
                "selection_data": "2024 only; retrospective period already previously inspected"}
    write_json("selection_before_retrospective.json", selected)
    return selected


def quarterly_check(data: dict, val: dict, scores: dict) -> list:
    actual = data["y"][[data["targets"].index(m) for m in VALIDATION]]
    records = []
    for start in (3, 6, 9):
        chosen = min(val, key=lambda n: np.abs(val[n]["prediction"][:start]-actual[:start]).mean())
        records.append({"past_validation_end": VALIDATION[start-1], "next_quarter_start": VALIDATION[start],
                        "chosen": chosen, "next_quarter_mae_pp": float(np.abs(val[chosen]["prediction"][start:start+3]-actual[start:start+3]).mean()),
                        "ridge25_same_quarter_mae_pp": float(np.abs(val["ridge_25"]["prediction"][start:start+3]-actual[start:start+3]).mean())})
    write_json("quarterly_past_only_selection.json", {"quarters": records,
        "note": "Uses original three-seed predictions for all 20 configurations. Each quarter chooses using preceding quarters only; five-seed confirmation is excluded."})
    return records


def finish(data: dict, val: dict, val_scores: dict, audit: list, screening_results: dict) -> None:
    selection = select(data, val, val_scores)
    quarterly_check(data, screening_results, val_scores)
    print(f"LOCKED PRIMARY: {selection['primary']}; neural: {selection['best_neural_screened']}", flush=True)
    val_actual = data["y"][[data["targets"].index(m) for m in VALIDATION]]
    actual = data["y"][[data["targets"].index(m) for m in RETROSPECTIVE]]
    test = {}
    representatives = selection["retrospective_representatives"]
    # Bias corrections always use the fixed Ridge-25 base and its causal state.
    base_test = roll(data, "ridge_25", RETROSPECTIVE, audit=audit)
    for name in representatives:
        if name == "ridge_25":
            result = base_test
        elif CASES[name]["kind"] == "bias":
            _, state, _ = bias_correction(val["ridge_25"]["prediction"], val_actual, CASES[name])
            prediction, _, records = bias_correction(base_test["prediction"], actual, CASES[name], state)
            result = as_result(prediction)
            for row, month in zip(records, RETROSPECTIVE):
                row.update(target_month=month, state_known_through=str(pd.Period(month)-1))
            pd.DataFrame(records).to_csv(OUT / f"{name}_retrospective_state.csv", index=False, encoding="utf-8-sig")
        else:
            seeds = FINAL_SEEDS if name in selection["final_five_seed_cases"] else SEEDS
            result = roll(data, name, RETROSPECTIVE, seeds, audit)
        test[name] = result
    last = data["seq"][[data["targets"].index(m) for m in RETROSPECTIVE], :, -1, 0]
    test["last_month"] = as_result(last)
    best_base, neural = selection["best_simple_baseline"], selection["best_neural_screened"]
    val_last = data["seq"][[data["targets"].index(m) for m in VALIDATION], :, -1, 0]
    val_experts = np.stack([val[best_base]["prediction"], val[neural]["prediction"], val_last], axis=-1)
    test_experts = np.stack([test[best_base]["prediction"], test[neural]["prediction"], last], axis=-1)
    val_online, risk, _ = online_mix(val_experts, val_actual)
    test_online, final_risk, mix_records = online_mix(test_experts, actual, risk)
    test["online_ensemble"] = as_result(test_online)
    test["uniform_ensemble"] = as_result(test_experts.mean(axis=-1))
    write_json("ensemble_weights.json", {"experts": [best_base, neural, "last_month"],
        "months": dict(zip(RETROSPECTIVE, mix_records)), "final_risk_pp": final_risk.tolist()})
    save_predictions("retrospective_predictions.csv", data, RETROSPECTIVE, test)
    test_scores = {n: score(data, RETROSPECTIVE, r) for n, r in test.items()}
    write_json("retrospective_results.json", test_scores)
    write_json("ensemble_validation.json", {"online": metrics(val_actual, val_online),
        "uniform": metrics(val_actual, val_experts.mean(axis=-1)), "post_selection_experts": [best_base, neural],
        "note": "Experts selected on full-year validation; this score has additional selection uncertainty."})

    forecasts = {}
    for name in representatives:
        if CASES[name]["kind"] == "bias":
            _, state, _ = bias_correction(val["ridge_25"]["prediction"], val_actual, CASES[name])
            _, state, _ = bias_correction(base_test["prediction"], actual, CASES[name], state)
            checkpoint = OUT / "models" / name if name == selection["primary"] else None
            base = model_fit(data, CASES["ridge_25"], "2026-02", checkpoint_dir=checkpoint)["prediction"]
            if checkpoint is not None:
                (checkpoint / "bias_state.json").write_text(json.dumps({"past_bias_pp": state,
                    "strength": CASES[name]["strength"], "rho": CASES[name]["rho"],
                    "state_known_through": "2026-01"}, indent=2), encoding="utf-8")
            result = as_result((base-CASES[name]["strength"]*state)[None])
        else:
            seeds = FINAL_SEEDS if name in selection["final_five_seed_cases"] else SEEDS
            checkpoint = OUT / "models" / name if name in {selection["primary"], neural} else None
            fit_result = model_fit(data, CASES[name], "2026-02", seeds, checkpoint)
            result = {"prediction": fit_result["prediction"][None]}
            if "probabilities" in fit_result:
                result["probabilities"] = fit_result["probabilities"][None]
        forecasts[name] = result
        audit.append({"case": name, "forecast_month": "2026-02", "first_train_target": "2021-12",
                      "last_train_target": "2026-01", "train_months": 50, "train_city_rows": 3500,
                      "train_future_target_used": False, "stage": "final_forecast"})
    feb_experts = np.stack([forecasts[best_base]["prediction"][0], forecasts[neural]["prediction"][0],
                            data["seq"][-1, :, -1, 0]], axis=-1)
    logits = -(final_risk-final_risk.min())/0.02
    final_weights = np.exp(logits)/np.exp(logits).sum()
    forecasts["online_ensemble"] = as_result((feb_experts@final_weights)[None])
    forecasts["uniform_ensemble"] = as_result(feb_experts.mean(axis=-1)[None])
    save_predictions("forecast_2026_02.csv", data, ["2026-02"], forecasts)
    model_metadata = {n: {"config": CASES[n], "seeds": list(FINAL_SEEDS if n in selection["final_five_seed_cases"] else SEEDS),
                          "fit_through": "2026-01", "forecast_month": "2026-02"}
                      for n in {selection["primary"], neural}}
    write_json("model_metadata.json", model_metadata)
    pd.DataFrame(audit).to_csv(OUT / "training_cutoff_audit.csv", index=False, encoding="utf-8-sig")
    gains = {n: block_interval(actual, last, r["prediction"]) for n, r in test.items() if n != "last_month"}
    write_json("gain_intervals.json", gains)
    write_json("run_summary.json", {"status": "experiments_completed", "information_cutoff": "2026-02-28",
        "actual_data_end": "2026-01", "primary": selection["primary"], "best_neural": neural,
        "case_count": len(CASES)+2, "evaluation_status": "retrospective exploratory; not a fresh blind test",
        "validation": val_scores, "retrospective": test_scores, "selection": selection})
    print(json.dumps({"primary": selection["primary"], "neural": neural,
                      "scores": {n: round(s["mae_pp"], 5) for n, s in test_scores.items()}}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("validation", "all"), default="all")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(exist_ok=True)
    data = load_research_data()
    source_hash = hashlib.sha256((ROOT / "data/official_nbs_70city.csv").read_bytes()).hexdigest().upper()
    manifest = json.loads((ROOT / "data_manifest.json").read_text(encoding="utf-8"))
    assert source_hash == manifest["cleaned_data_sha256"]
    write_json("locked_experiment_registry.json", {"configurations": CASES, "ensemble_configurations": 2,
        "data_sha256": source_hash, "code_and_data_fingerprint": fingerprint(),
        "seeds_screening": SEEDS, "seeds_confirmation": FINAL_SEEDS, "neural_epochs": 60,
        "nonlinear_candidates": NONLINEAR_CANDIDATES,
        "validation_months": VALIDATION, "retrospective_months": RETROSPECTIVE,
        "max_neural_parameters": 5000, "declared_before_new_runs": True,
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "pandas": pd.__version__, "sklearn": sklearn.__version__, "torch": torch.__version__}})
    val, val_scores, audit, screening_results = validate(data)
    if args.phase == "all":
        finish(data, val, val_scores, audit, screening_results)


if __name__ == "__main__":
    main()
