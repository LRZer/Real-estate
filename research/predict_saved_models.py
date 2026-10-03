"""Predict February 2026 from saved weights, without training again."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from research_models import ROOT, ContextNet, load_research_data


def predict_saved(data: dict, name: str, metadata: dict) -> np.ndarray:
    if "cities" in metadata:
        assert data["cities"] == metadata["cities"], "City order differs from checkpoint"
        assert data["numeric"] == metadata["numeric_features"], "Feature order differs from checkpoint"
    directory = ROOT / "results/research/models" / name
    kind = metadata["config"]["kind"]
    x = data["features"][-1]
    if kind == "hgb":
        return joblib.load(directory / "hgb.joblib").predict(np.column_stack([x, np.eye(70)]))
    if kind in ("ridge", "bias", "decomp_ridge"):
        filename = "decomp_ridge.npz" if kind == "decomp_ridge" else "ridge.npz"
        with np.load(directory / filename) as s:
            features = np.column_stack([(x-s["feature_mean"])/s["feature_scale"], np.eye(70)])
            pred = features@s["coefficient"]+s["intercept"]
            if kind == "decomp_ridge":
                g = (data["global_features"][-1]-s["global_mean"])/s["global_scale"]
                pred = pred-pred.mean()+g@s["global_coefficient"]+s["global_intercept"]
        if kind == "bias":
            state = json.loads((directory / "bias_state.json").read_text(encoding="utf-8"))
            pred -= state["strength"]*state["past_bias_pp"]
        return pred
    with np.load(directory / "preprocessing.npz") as s:
        seq = data["seq"][-1:]
        batch = {"seq": torch.from_numpy(seq),
                 "normalized": torch.from_numpy(((seq-s["sequence_mean"])/s["sequence_std"]).astype(np.float32)),
                 "features": torch.from_numpy(((x-s["feature_mean"])/s["feature_scale"])[None].astype(np.float32)),
                 "global": torch.from_numpy(((data["global_features"][-1:]-s["global_mean"])/s["global_scale"]).astype(np.float32)),
                 "last": torch.from_numpy(seq[:, :, -1, 0])}
    predictions = []
    for seed in metadata["seeds"]:
        model = ContextNet(kind)
        model.load_state_dict(torch.load(directory / f"seed_{seed}.pt", map_location="cpu", weights_only=True))
        model.eval()
        with torch.no_grad():
            predictions.append(model(batch)["prediction"][0].numpy())
    return np.mean(predictions, axis=0)


def main():
    output = ROOT / "results/research"
    data = load_research_data()
    metadata = json.loads((output / "model_metadata.json").read_text(encoding="utf-8"))
    reference = pd.read_csv(output / "forecast_2026_02.csv").set_index("city").loc[data["cities"]]
    records = pd.DataFrame({"city": data["cities"], "forecast_month": "2026-02"})
    checks = {}
    for name, meta in metadata.items():
        prediction = predict_saved(data, name, meta)
        error = float(np.max(np.abs(prediction-reference[f"pred_{name}"])))
        assert error < 2e-7, (name, error)
        records[f"pred_{name}"] = prediction
        checks[name] = {"loaded_without_retraining": True, "maximum_difference_pp": error}
    records.to_csv(output / "reproduced_forecast_2026_02.csv", index=False, encoding="utf-8-sig")
    (output / "checkpoint_verification.json").write_text(json.dumps(checks, indent=2), encoding="utf-8")
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
