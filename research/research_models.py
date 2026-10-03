"""Small, explicitly specified forecasting models for the locked research plan."""

from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(name, "1")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from train_forecast import ROOT, build_dataset
from train_graph_temporal import GraphTemporalNet, load_panel

torch.set_num_threads(1)
SEEDS = (13, 31, 47)
FINAL_SEEDS = (13, 31, 47, 61, 79)


def load_research_data() -> dict:
    panel, periods, cities = load_panel()
    rows, numeric = build_dataset()
    rows["target_string"] = rows.target_month.astype(str)
    targets = sorted(rows.target_string.unique())
    if targets != list(pd.period_range("2021-12", "2026-02", freq="M").astype(str)):
        raise ValueError("Unexpected target months")
    arranged = rows.set_index(["target_string", "city"]).loc[
        pd.MultiIndex.from_product([targets, cities])]
    features = arranged[numeric].to_numpy().reshape(51, 70, len(numeric))
    y = arranged.target_change_pct.to_numpy().reshape(51, 70)[:-1]
    seq = np.asarray([panel[i-6:i+1].transpose(1, 0, 2) for i in range(6, len(periods))])
    national_seq = seq.mean(axis=1)
    extras = np.column_stack([features[:, 0, numeric.index(f)]
                             for f in ("share_cities_rising", "calendar_sin", "calendar_cos")])
    global_features = np.column_stack([national_seq.reshape(51, -1), extras])
    return {"seq": seq.astype(np.float32), "features": features, "y": y,
            "global_features": global_features, "targets": targets, "cities": cities,
            "numeric": numeric, "panel": panel, "periods": periods,
            "source_links": pd.read_csv(ROOT / "data/official_nbs_70city.csv").set_index(["period", "city"]).source_url.to_dict()}


def past_indices(data: dict, target: str) -> np.ndarray:
    indices = np.array([i for i, m in enumerate(data["targets"][:-1]) if m < target])
    assert len(indices) >= 25
    assert data["targets"][indices[-1]] == str(pd.Period(target, freq="M") - 1)
    return indices


def prepare_tensors(data: dict, fit: np.ndarray, legacy: bool = False) -> tuple[dict, dict]:
    seq = data["seq"][:, :, 1:] if legacy else data["seq"]
    mean = seq[fit].mean(axis=(0, 1, 2))
    std = seq[fit].std(axis=(0, 1, 2)).clip(min=0.05)
    scaler = StandardScaler().fit(data["features"][fit].reshape(-1, 19))
    global_scaler = StandardScaler().fit(data["global_features"][fit])
    features = scaler.transform(data["features"].reshape(-1, 19)).reshape(51, 70, 19)
    tensors = {
        "seq": torch.from_numpy(seq), "normalized": torch.from_numpy(((seq-mean)/std).astype(np.float32)),
        "features": torch.from_numpy(features.astype(np.float32)),
        "global": torch.from_numpy(global_scaler.transform(data["global_features"]).astype(np.float32)),
        "last": torch.from_numpy(data["seq"][:, :, -1, 0]),
        "y": torch.from_numpy(data["y"].astype(np.float32)),
        "legacy_context": torch.from_numpy(data["seq"][:, :, -1].mean(axis=1)),
        "legacy_calendar": torch.from_numpy(data["features"][:, 0, [5, 6]].astype(np.float32)),
    }
    stats = {"sequence_mean": mean, "sequence_std": std,
             "feature_mean": scaler.mean_, "feature_scale": scaler.scale_,
             "global_mean": global_scaler.mean_, "global_scale": global_scaler.scale_}
    return tensors, stats


class ContextNet(nn.Module):
    def __init__(self, kind: str, cities: int = 70):
        super().__init__()
        self.kind = kind
        self.local_norm = kind in ("revin", "decomp_revin")
        self.decompose = kind in ("decomp", "decomp_revin", "multitask")
        self.multitask = kind == "multitask"
        self.embedding = nn.Embedding(cities, 4)
        if kind == "nlinear":
            self.output = nn.Linear(14 + 19 + 4, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        else:
            self.encoder = nn.Sequential(nn.Conv1d(2, 12, 3, padding=1), nn.ReLU(),
                                         nn.Conv1d(12, 12, 3, padding=1), nn.ReLU())
            self.trunk = nn.Sequential(nn.Linear(12 + 19 + 4, 24), nn.ReLU(), nn.Dropout(0.15))
            self.output = nn.Linear(24, 1)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)
        if self.decompose:
            self.market = nn.Sequential(nn.Linear(17, 12), nn.ReLU(), nn.Linear(12, 1))
            nn.init.zeros_(self.market[-1].weight)
            nn.init.zeros_(self.market[-1].bias)
        if self.multitask:
            self.direction = nn.Linear(24, 3)

    def forward(self, batch: dict) -> dict:
        raw, features = batch["seq"], batch["features"]
        n, cities, window, channels = raw.shape
        city_id = self.embedding.weight.unsqueeze(0).expand(n, -1, -1)
        scale = torch.ones_like(batch["last"])
        z = batch["normalized"]
        if self.local_norm:
            mean = raw.mean(dim=2, keepdim=True)
            std = raw.std(dim=2, unbiased=False, keepdim=True).clamp(min=0.05)
            z = (raw-mean)/std
            scale = std[:, :, 0, 0]
        if self.kind == "nlinear":
            hidden = torch.cat([(z-z[:, :, -1:]).reshape(n, cities, -1), features, city_id], dim=-1)
        else:
            encoded = self.encoder(z.permute(0, 1, 3, 2).reshape(n*cities, channels, window))
            encoded = encoded.mean(dim=-1).reshape(n, cities, 12)
            hidden = self.trunk(torch.cat([encoded, features, city_id], dim=-1))
        delta = self.output(hidden).squeeze(-1) * scale
        result = {}
        if self.decompose:
            previous_market = batch["last"].mean(dim=1)
            market = previous_market + self.market(batch["global"]).squeeze(-1)
            residual = batch["last"] - previous_market[:, None] + delta
            residual = residual - residual.mean(dim=1, keepdim=True)
            result.update(prediction=market[:, None] + residual, market=market, residual=residual)
        else:
            result["prediction"] = batch["last"] + delta
        if self.multitask:
            result["logits"] = self.direction(hidden)
        return result


def get_batch(tensors: dict, indices: np.ndarray) -> dict:
    return {name: value[indices] for name, value in tensors.items() if name != "y"}


def neural_fit(data: dict, config: dict, target: str, seeds=SEEDS,
               checkpoint_dir: Path | None = None) -> dict:
    fit = past_indices(data, target)
    forecast_idx = np.array([data["targets"].index(target)])
    kind = config["kind"]
    legacy = kind == "legacy"
    tensors, stats = prepare_tensors(data, fit, legacy)
    predictions, markets, probabilities = [], [], []
    y = tensors["y"][fit]
    labels = torch.sign(y).long() + 1
    class_weight = None
    if kind == "multitask" and config["weighted"]:
        counts = torch.bincount(labels.flatten(), minlength=3).float()
        freq = counts / counts.sum()
        class_weight = freq.clamp(min=1e-6).rsqrt().clamp(max=3)
        class_weight = class_weight / class_weight[labels].mean()
    for seed in seeds:
        torch.manual_seed(seed)
        if legacy:
            model = GraphTemporalNet(70, np.zeros((70, 70), dtype=np.float32))
        else:
            model = ContextNet(kind)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
        training = get_batch(tensors, fit)
        for _ in range(60):
            model.train()
            optimizer.zero_grad()
            if legacy:
                prediction = model(training["normalized"], training["last"],
                                   training["legacy_context"], training["legacy_calendar"])
                output = {"prediction": prediction}
            else:
                output = model(training)
            loss = nn.functional.smooth_l1_loss(output["prediction"], y, beta=0.25)
            if "market" in output:
                loss = loss + 0.25 * nn.functional.smooth_l1_loss(output["market"], y.mean(dim=1), beta=0.25)
            if "logits" in output:
                loss = loss + config["eta"] * nn.functional.cross_entropy(
                    output["logits"].reshape(-1, 3), labels.flatten(), weight=class_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
        model.eval()
        future = get_batch(tensors, forecast_idx)
        with torch.no_grad():
            if legacy:
                output = {"prediction": model(future["normalized"], future["last"],
                                                future["legacy_context"], future["legacy_calendar"])}
            else:
                output = model(future)
        predictions.append(output["prediction"][0].numpy())
        if "market" in output:
            markets.append(output["market"][0].numpy())
        if "logits" in output:
            probabilities.append(output["logits"][0].softmax(dim=-1).numpy())
        if checkpoint_dir is not None:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_dir / f"seed_{seed}.pt")
            np.savez_compressed(checkpoint_dir / "preprocessing.npz", **stats)
    result = {"prediction": np.mean(predictions, axis=0), "seed_predictions": np.asarray(predictions),
              "parameter_count": np.array(sum(p.numel() for p in model.parameters()))}
    if markets:
        result["market_prediction"] = np.mean(markets)
    if probabilities:
        result["probabilities"] = np.mean(probabilities, axis=0)
    return result


def classical_fit(data: dict, config: dict, target: str, checkpoint_dir: Path | None = None) -> dict:
    fit = past_indices(data, target)
    i = data["targets"].index(target)
    features = data["features"]
    scaler = StandardScaler().fit(features[fit].reshape(-1, 19))
    training = np.column_stack([scaler.transform(features[fit].reshape(-1, 19)), np.tile(np.eye(70), (len(fit), 1))])
    future = np.column_stack([scaler.transform(features[i]), np.eye(70)])
    y = data["y"][fit]
    if config["kind"] == "hgb":
        training = np.column_stack([features[fit].reshape(-1, 19), np.tile(np.eye(70), (len(fit), 1))])
        future = np.column_stack([features[i], np.eye(70)])
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=config["leaves"],
            learning_rate=0.05, min_samples_leaf=30, l2_regularization=0.5, early_stopping=False, random_state=42)
        prediction = model.fit(training, y.flatten()).predict(future)
    elif config["kind"] == "decomp_ridge":
        global_scaler = StandardScaler().fit(data["global_features"][fit])
        g = y.mean(axis=1)
        global_model = Ridge(alpha=25).fit(global_scaler.transform(data["global_features"][fit]), g)
        ghat = global_model.predict(global_scaler.transform(data["global_features"][[i]]))[0]
        model = Ridge(alpha=25).fit(training, (y-g[:, None]).flatten())
        residual = model.predict(future)
        prediction = ghat + residual-residual.mean()
    else:
        model = Ridge(alpha=config["alpha"]).fit(training, y.flatten())
        prediction = model.predict(future)
    if checkpoint_dir is not None and config["kind"] == "ridge":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_dir / "ridge.npz", coefficient=model.coef_, intercept=model.intercept_,
                            feature_mean=scaler.mean_, feature_scale=scaler.scale_)
    if checkpoint_dir is not None and config["kind"] == "hgb":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, checkpoint_dir / "hgb.joblib")
    if checkpoint_dir is not None and config["kind"] == "decomp_ridge":
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(checkpoint_dir / "decomp_ridge.npz",
                            coefficient=model.coef_, intercept=model.intercept_,
                            feature_mean=scaler.mean_, feature_scale=scaler.scale_,
                            global_coefficient=global_model.coef_, global_intercept=global_model.intercept_,
                            global_mean=global_scaler.mean_, global_scale=global_scaler.scale_)
    return {"prediction": prediction, "seed_predictions": prediction[None], "parameter_count": np.array(0)}


def model_fit(data: dict, config: dict, target: str, seeds=SEEDS,
              checkpoint_dir: Path | None = None) -> dict:
    if config["kind"] in ("ridge", "hgb", "decomp_ridge"):
        return classical_fit(data, config, target, checkpoint_dir)
    return neural_fit(data, config, target, seeds, checkpoint_dir)
