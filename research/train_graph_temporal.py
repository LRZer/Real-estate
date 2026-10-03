"""Small graph-temporal research extension for the 70-city monthly panel.

Inspired by AGCRN (NeurIPS 2020) and MTGNN (KDD 2020), but this is a new,
much smaller architecture, not a reproduction of either paper. The graph is
estimated from training-period city co-movements only. Since the original test
period was inspected before this extension, its new-model results are
retrospective exploratory comparisons, not a fresh blind test.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "data" / "official_nbs_70city.csv"
RESULTS = ROOT / "results"
WINDOW = 6
SEEDS = (13, 31, 47)
MAX_EPOCHS = 240
PATIENCE = 35
TOP_K = 5
torch.set_num_threads(1)


def load_panel() -> tuple[np.ndarray, list[str], list[str]]:
    rows = pd.read_csv(SOURCE)
    periods = sorted(rows["period"].unique())
    cities = sorted(rows["city"].unique())
    if (len(periods) != 57 or periods[0] != "2021-05" or periods[-1] != "2026-01"
            or len(cities) != 70 or len(rows) != 3990):
        raise ValueError("Unexpected 70-city monthly panel")
    channels = []
    for field in ("new_mom_pct", "second_mom_pct"):
        matrix = rows.pivot(index="period", columns="city", values=field).loc[periods, cities]
        if matrix.isna().any().any():
            raise ValueError(f"Incomplete {field} panel")
        channels.append(matrix.to_numpy(dtype=np.float32))
    return np.stack(channels, axis=-1), periods, cities


def make_examples(panel: np.ndarray, periods: list[str]) -> dict:
    inputs, labels, contexts, calendars, target_periods = [], [], [], [], []
    for current in range(WINDOW, len(periods)):
        block = panel[current - WINDOW + 1:current + 1].transpose(1, 0, 2)
        inputs.append(block)
        contexts.append(panel[current].mean(axis=0))
        month_number = int(periods[current].split("-")[1])
        calendars.append([np.sin(2 * np.pi * month_number / 12), np.cos(2 * np.pi * month_number / 12)])
        if current + 1 < len(periods):
            labels.append(panel[current + 1, :, 0])
            target_periods.append(periods[current + 1])
        else:
            target_periods.append("2026-02")
    return {
        "x": np.asarray(inputs, dtype=np.float32),
        "y": np.asarray(labels, dtype=np.float32),
        "context": np.asarray(contexts, dtype=np.float32),
        "calendar": np.asarray(calendars, dtype=np.float32),
        "targets": target_periods,
    }


def correlation_graph(panel: np.ndarray, periods: list[str], last_known: str,
                      cities: list[str], residualize: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    """Training-only positive-correlation top-k graph, with no self edges."""
    end = periods.index(last_known) + 1
    values = panel[:end, :, 0].astype(np.float64)
    if residualize:
        values = values - values.mean(axis=1, keepdims=True)
    corr = np.nan_to_num(np.corrcoef(values.T), nan=0.0)
    np.fill_diagonal(corr, 0.0)
    corr = np.maximum(corr, 0.0)
    adjacency = np.zeros_like(corr)
    neighbors = []
    for row in range(len(cities)):
        candidates = np.arange(len(cities))
        candidates = candidates[candidates != row]
        nearest = candidates[np.argsort(corr[row, candidates])[-TOP_K:][::-1]]
        weight = corr[row, nearest]
        if weight.sum() > 0:
            adjacency[row, nearest] = weight / weight.sum()
        for rank, other in enumerate(nearest, start=1):
            neighbors.append({
                "city": cities[row], "neighbor": cities[other], "rank": rank,
                "training_correlation": round(float(corr[row, other]), 4),
                "graph_weight": round(float(adjacency[row, other]), 4),
            })
    return adjacency.astype(np.float32), pd.DataFrame(neighbors)


class GraphTemporalNet(nn.Module):
    def __init__(self, city_count: int, adjacency: np.ndarray):
        super().__init__()
        self.register_buffer("adjacency", torch.from_numpy(adjacency))
        self.temporal = nn.Sequential(
            nn.Conv1d(2, 12, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size=3, padding=1), nn.ReLU(),
        )
        self.city_embedding = nn.Embedding(city_count, 4)
        self.head = nn.Sequential(nn.Linear(12 + 12 + 4 + 2 + 2, 24), nn.ReLU(),
                                  nn.Dropout(0.15), nn.Linear(24, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, standardized: torch.Tensor, last_new: torch.Tensor,
                context: torch.Tensor, calendar: torch.Tensor) -> torch.Tensor:
        batch, cities, window, channels = standardized.shape
        z = standardized.permute(0, 1, 3, 2).reshape(batch * cities, channels, window)
        hidden = self.temporal(z).mean(dim=-1).reshape(batch, cities, 12)
        neighbors = torch.einsum("ij,bjh->bih", self.adjacency, hidden)
        city_id = self.city_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        common = torch.cat([context, calendar], dim=1).unsqueeze(1).expand(-1, cities, -1)
        delta = self.head(torch.cat([hidden, neighbors, city_id, common], dim=-1)).squeeze(-1)
        return last_new + delta


def make_tensors(examples: dict, fit_indices: np.ndarray) -> dict:
    mean = examples["x"][fit_indices].mean(axis=(0, 1, 2))
    std = examples["x"][fit_indices].std(axis=(0, 1, 2)).clip(min=0.05)
    return {
        "x": torch.from_numpy((examples["x"] - mean) / std),
        "last": torch.from_numpy(examples["x"][:, :, -1, 0]),
        "context": torch.from_numpy(examples["context"]),
        "calendar": torch.from_numpy(examples["calendar"]),
        "y": torch.from_numpy(examples["y"]),
        "mean": mean, "std": std,
    }


def predict(model: GraphTemporalNet, tensors: dict, indices: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        result = model(tensors["x"][indices], tensors["last"][indices],
                       tensors["context"][indices], tensors["calendar"][indices])
    return result.numpy()


def train_one(seed: int, adjacency: np.ndarray, tensors: dict, fit_indices: np.ndarray,
              validation_indices: np.ndarray | None = None, fixed_epochs: int | None = None) -> tuple[GraphTemporalNet, int, float | None]:
    torch.manual_seed(seed)
    model = GraphTemporalNet(tensors["x"].shape[1], adjacency)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.006, weight_decay=0.02)
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    epochs = fixed_epochs if fixed_epochs is not None else MAX_EPOCHS
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        output = model(tensors["x"][fit_indices], tensors["last"][fit_indices],
                       tensors["context"][fit_indices], tensors["calendar"][fit_indices])
        loss = nn.functional.smooth_l1_loss(output, tensors["y"][fit_indices], beta=0.25)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if validation_indices is not None:
            validation_pred = predict(model, tensors, validation_indices)
            validation_mae = float(np.mean(np.abs(validation_pred - tensors["y"][validation_indices].numpy())))
            if validation_mae < best_loss - 0.0001:
                best_loss, best_epoch, stale = validation_mae, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if epoch >= 30 and stale >= PATIENCE:
                break
    if validation_indices is not None:
        if best_state is None:
            raise RuntimeError("No validation checkpoint")
        model.load_state_dict(best_state)
        return model, best_epoch, best_loss
    return model, epochs, None


def fit_ensemble(examples: dict, panel: np.ndarray, periods: list[str], cities: list[str],
                 fit_idx: np.ndarray, target_idx: np.ndarray, graph_mode: str,
                 graph_known_through: str, epochs: list[int] | None = None,
                 validation_idx: np.ndarray | None = None) -> tuple[np.ndarray, list[int], list[float | None], pd.DataFrame]:
    if graph_known_through > examples["targets"][int(fit_idx[-1])]:
        raise ValueError("The graph uses information after the training cutoff")
    adjacency, neighbors = correlation_graph(
        panel, periods, graph_known_through, cities, residualize=graph_mode == "residual_graph"
    )
    if graph_mode == "without_graph":
        adjacency = np.zeros_like(adjacency)
    elif graph_mode not in ("raw_graph", "residual_graph"):
        raise ValueError(graph_mode)
    tensors = make_tensors(examples, fit_idx)
    forecasts, chosen_epochs, validation_losses = [], [], []
    for number, seed in enumerate(SEEDS):
        model, used_epoch, validation_loss = train_one(
            seed, adjacency, tensors, fit_idx, validation_idx,
            None if epochs is None else epochs[number],
        )
        forecasts.append(predict(model, tensors, target_idx))
        chosen_epochs.append(used_epoch)
        validation_losses.append(validation_loss)
    return np.mean(forecasts, axis=0), chosen_epochs, validation_losses, neighbors


def mae(actual: np.ndarray, forecast: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    panel, periods, cities = load_panel()
    source_links = pd.read_csv(SOURCE).set_index(["period", "city"])["source_url"].to_dict()
    examples = make_examples(panel, periods)
    targets = examples["targets"][:-1]
    train = np.array([i for i, month in enumerate(targets) if month <= "2023-12"])
    validation = np.array([i for i, month in enumerate(targets) if "2024-01" <= month <= "2024-12"])
    test = np.array([i for i, month in enumerate(targets) if "2025-01" <= month <= "2026-01"])
    if [len(train), len(validation), len(test)] != [25, 12, 13]:
        raise ValueError("Unexpected time split")
    validation_results = {}
    for name in ("without_graph", "raw_graph", "residual_graph"):
        pred, epochs, seed_maes, _ = fit_ensemble(
            examples, panel, periods, cities, train, validation, name, "2023-12",
            validation_idx=validation,
        )
        validation_results[name] = {
            "mae_pp": mae(examples["y"][validation], pred),
            "best_epochs_by_seed": epochs,
            "individual_seed_mae_pp": [round(float(x), 4) for x in seed_maes],
        }
        print(f"Validation {name}: {validation_results[name]}", flush=True)
    selected = min(validation_results, key=lambda name: validation_results[name]["mae_pp"])

    early_graph, early_neighbors = correlation_graph(panel, periods, "2023-12", cities, residualize=True)
    later_graph, _ = correlation_graph(panel, periods, "2024-12", cities, residualize=True)
    early_neighbors.to_csv(RESULTS / "residual_graph_neighbors_train_through_2023.csv", index=False, encoding="utf-8-sig")
    overlaps = []
    for city_index in range(len(cities)):
        before = set(np.flatnonzero(early_graph[city_index]))
        after = set(np.flatnonzero(later_graph[city_index]))
        overlaps.append(len(before & after) / len(before | after) if before | after else 1.0)

    train_val = np.concatenate([train, validation])
    exploratory_test = {}
    predictions = {}
    for name in ("without_graph", "raw_graph", "residual_graph"):
        epochs = validation_results[name]["best_epochs_by_seed"]
        pred, _, _, neighbors = fit_ensemble(
            examples, panel, periods, cities, train_val, test, name, "2024-12", epochs=epochs,
        )
        predictions[name] = pred
        exploratory_test[name] = {"mae_pp": mae(examples["y"][test], pred)}
        if name != "without_graph":
            neighbors.to_csv(RESULTS / f"{name}_neighbors_train_through_2024.csv", index=False, encoding="utf-8-sig")
        print(f"Exploratory historical test {name}: {exploratory_test[name]}", flush=True)

    records = []
    for row, month_index in enumerate(test):
        target_month = targets[month_index]
        input_month = periods[periods.index(target_month) - 1]
        for city_index, city in enumerate(cities):
            records.append({
                "input_month": input_month, "target_month": target_month, "city": city,
                "actual_change_pct": float(examples["y"][month_index, city_index]),
                "pred_without_graph": float(predictions["without_graph"][row, city_index]),
                "pred_raw_graph": float(predictions["raw_graph"][row, city_index]),
                "pred_residual_graph": float(predictions["residual_graph"][row, city_index]),
                "input_source_url": source_links[(input_month, city)],
                "target_source_url": source_links[(target_month, city)],
            })
    pd.DataFrame(records).to_csv(RESULTS / "advanced_backtest_exploratory.csv", index=False, encoding="utf-8-sig")

    all_known = np.concatenate([train, validation, test])
    forecast, _, _, _ = fit_ensemble(
        examples, panel, periods, cities, all_known, np.array([len(all_known)]),
        selected, "2026-01", epochs=validation_results[selected]["best_epochs_by_seed"],
    )
    pd.DataFrame({
        "city": cities, "input_month": "2026-01", "forecast_month": "2026-02",
        "predicted_change_pct": forecast[0],
        "input_source_url": [source_links[("2026-01", city)] for city in cities],
    }).to_csv(
        RESULTS / "advanced_forecast_2026_02.csv", index=False, encoding="utf-8-sig"
    )
    report = {
        "model": "compact graph-temporal convolution with city embeddings and residual prediction",
        "paper_relation": "Architecture inspired by AGCRN and MTGNN; it is not an official reproduction.",
        "window_months": WINDOW, "graph_top_k": TOP_K, "seeds": SEEDS,
        "training_loss": "Huber with beta=0.25", "selection_metric": "2024 validation MAE",
        "residual_graph_neighbor_jaccard_2023_to_2024": round(float(np.mean(overlaps)), 4),
        "validation": validation_results, "selected_on_validation": selected,
        "exploratory_historical_test": exploratory_test,
        "methodological_note": "The 2025-Jan2026 period was inspected during earlier model work, so new-model scores are exploratory, not a fresh blind test.",
    }
    (RESULTS / "advanced_model_results.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
