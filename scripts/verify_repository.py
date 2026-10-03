"""Read-only checks for the published experiment, documentation and checkpoints."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "research"
RESULTS = RESEARCH / "results/research"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(actual, expected, message, tolerance=1e-9):
    require(np.isclose(actual, expected, atol=tolerance, rtol=0),
            f"{message}: computed={actual}, recorded={expected}")


def verify_metrics(frame, recorded):
    actual = frame.actual_change_pct.to_numpy()
    for case, record in recorded.items():
        prediction = frame["pred_" + case].to_numpy()
        error = prediction - actual
        for key, value in (
            ("mae_pp", np.abs(error).mean()),
            ("bias_pp", error.mean()),
            ("rising_mae_pp", np.abs(error[actual > 0]).mean()),
            ("falling_mae_pp", np.abs(error[actual < 0]).mean()),
        ):
            close(value, record[key], f"{case} {key}")
        if "monthly_mae_pp" in record:
            for month, group in frame.groupby("target_month"):
                value = np.abs(group["pred_" + case] - group.actual_change_pct).mean()
                close(value, record["monthly_mae_pp"][month], f"{case} {month} MAE")
    return len(recorded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-model", action="store_true", help="Skip saved-weight inference.")
    args = parser.parse_args()
    summary = {}
    manifest = read(RESULTS / "completion_manifest.json")
    data_manifest = read(RESEARCH / "data_manifest.json")
    registry = read(RESULTS / "locked_experiment_registry.json")
    panel_path = RESEARCH / "data/official_nbs_70city.csv"
    raw = pd.read_csv(panel_path)
    months = list(pd.period_range("2021-05", "2026-01", freq="M").astype(str))
    require(list(sorted(raw.period.unique())) == months, "Raw month coverage")
    require(len(raw) == 3990 and raw.city.nunique() == 70, "Raw panel dimensions")
    require(not raw.duplicated(["period", "city"]).any(), "Duplicate city-months")
    require(not raw.isna().any().any(), "Missing input fields")
    require(np.isfinite(raw[["new_mom_pct", "second_mom_pct"]]).all().all(), "Nonfinite inputs")
    require((raw.groupby("period").size() == 70).all(), "Incomplete monthly panel")
    require((raw.groupby("city").size() == 57).all(), "Incomplete city series")
    require(digest(panel_path).upper() == manifest["data_sha256"] ==
            registry["data_sha256"] == data_manifest["cleaned_data_sha256"], "Data hash")
    sources = pd.read_csv(RESEARCH / "source_index.csv")
    require(digest(RESEARCH / "source_index.csv").upper() == data_manifest["source_index_sha256"],
            "Source-index hash")
    require(len(sources) == 57 and not sources.period.duplicated().any(), "Source coverage")
    links = sources.set_index("period").source_url.to_dict()
    require(raw.source_url.eq(raw.period.map(links)).all(), "Per-row source URL")
    require(sources.source_url.map(lambda u: urlparse(u).hostname == "www.stats.gov.cn").all(),
            "Official URL host")
    calendar = pd.read_csv(RESULTS / "release_calendar.csv")
    require(list(calendar.period) == months, "Release calendar coverage")
    require(calendar.source_url.eq(calendar.period.map(links)).all(), "Calendar source")
    url_dates = calendar.source_url.str.extract(r"/t(\d{8})_", expand=False)
    require(pd.to_datetime(url_dates, format="%Y%m%d").eq(
            pd.to_datetime(calendar.official_release_date)).all(), "Official URL date stamp")
    require((calendar.official_release_date <= "2026-02-28").all(), "Post-cutoff release")
    next_month = calendar.period.map(lambda m: str(pd.Period(m, freq="M") + 1))
    require(calendar.prediction_target_month.eq(next_month).all(), "Earliest target month")
    require(calendar.earliest_origin_date.eq(calendar.official_release_date).all(), "Origin date")
    summary["panel"] = {"rows": len(raw), "cities": 70, "months": 57, "cutoff": "2026-02-28"}

    for path, expected in manifest["sha256_by_artifact"].items():
        require(digest(RESEARCH / path) == expected, f"Snapshot hash: {path}")
    h = hashlib.sha256()
    for path in ("research_models.py", "run_research_experiments.py", "data/official_nbs_70city.csv"):
        h.update((RESEARCH / path).read_bytes())
    require(h.hexdigest() == registry["code_and_data_fingerprint"], "Code/data fingerprint")
    require(len(registry["configurations"]) == 20 and registry["ensemble_configurations"] == 2,
            "Configuration count")
    summary["snapshot_hashes"] = len(manifest["sha256_by_artifact"]) + 2

    frames = {}
    for filename, count, first, last in (
        ("validation_predictions_2024.csv", 840, "2024-01", "2024-12"),
        ("retrospective_predictions.csv", 910, "2025-01", "2026-01"),
        ("forecast_2026_02.csv", 70, "2026-02", "2026-02"),
        ("validation_screening_three_seeds.csv", 840, "2024-01", "2024-12"),
    ):
        frame = pd.read_csv(RESULTS / filename)
        frames[filename] = frame
        require(len(frame) == count and not frame.duplicated(["target_month", "city"]).any(),
                f"Prediction dimensions: {filename}")
        require(sorted(frame.target_month.unique()) ==
                list(pd.period_range(first, last, freq="M").astype(str)), f"Coverage: {filename}")
        require((frame.groupby("target_month").size() == 70).all(), f"City coverage: {filename}")
        require(np.isfinite(frame.select_dtypes(include=np.number)).all().all(),
                f"Finite predictions: {filename}")
        expected_input = frame.target_month.map(lambda m: str(pd.Period(m, freq="M") - 1))
        require(frame.input_month.eq(expected_input).all(), f"Input cutoff: {filename}")
        require(frame.input_source_url.eq(frame.input_month.map(links)).all(), f"Input URL: {filename}")
        if "actual_change_pct" in frame:
            labels = raw.set_index(["period", "city"]).new_mom_pct
            original = np.array([labels.loc[(m, c)] for m, c in zip(frame.target_month, frame.city)])
            np.testing.assert_allclose(frame.actual_change_pct, original, atol=1e-12, rtol=0)
        else:
            require(filename == "forecast_2026_02.csv", "Unexpected unlabeled table")
    val, test, screen, ev = (read(RESULTS / f) for f in (
        "validation_results.json", "retrospective_results.json",
        "validation_screening_results.json", "ensemble_validation.json"))
    require(set(val) == set(screen) == set(registry["configurations"]), "Validation cases")
    require(len(test) == 11, "Eight representatives, two ensembles, and last-month baseline")
    verify_metrics(frames["validation_predictions_2024.csv"], val)
    verify_metrics(frames["validation_screening_three_seeds.csv"], screen)
    verify_metrics(frames["retrospective_predictions.csv"], test)
    # The validation ensembles are not duplicated as columns in the validation CSV.
    require(np.isfinite([ev["online"]["mae_pp"], ev["uniform"]["mae_pp"]]).all(),
            "Ensemble validation summary")
    comparison = pd.read_csv(ROOT / "docs/tables/model_results.csv").set_index("case")
    for case, record in val.items():
        close(comparison.loc[case, "final_validation_mae_pp"], record["mae_pp"], "Published table")
    city_table = pd.read_csv(ROOT / "docs/tables/all_city_mae.csv")
    require(len(city_table) == 770 and not city_table.duplicated(["city", "case"]).any(),
            "All-city score table")
    retrospective = frames["retrospective_predictions.csv"]
    grouped = retrospective.groupby("city")
    for row in city_table.itertuples():
        group = grouped.get_group(row.city)
        close(np.abs(group["pred_" + row.case] - group.actual_change_pct).mean(), row.mae_pp,
              f"City score {row.city} {row.case}")
    selection = read(RESULTS / "selection_before_retrospective.json")
    require(selection["primary"] == "ridge_25", "Primary-model selection")
    require(not any(v["passes_predeclared_price_gate"] for v in selection["price_gates"].values()),
            "Replacement-gate statement")
    require(not any(v["passes_direction_gate"] for v in selection["direction_gates"].values()),
            "Direction-gate statement")
    summary["metrics"] = {"screened_cases": 20, "final_validation_cases": 20,
                          "retrospective_methods": 11, "city_scores": len(city_table)}

    audit = pd.read_csv(RESULTS / "training_cutoff_audit.csv")
    require(len(audit) == 315, "Training-cutoff record count")
    require(audit.first_train_target.eq("2021-12").all(), "Initial supervised month")
    require(audit.last_train_target.eq(
            audit.forecast_month.map(lambda m: str(pd.Period(m, freq="M") - 1))).all(),
            "Training cutoff equals previous statistical month")
    require(not audit.train_future_target_used.any(), "Future label used")
    expected_months = audit.last_train_target.map(
        lambda m: pd.Period(m, freq="M").ordinal - pd.Period("2021-12", freq="M").ordinal + 1)
    require(audit.train_months.eq(expected_months).all(), "Training month count")
    require(audit.train_city_rows.eq(expected_months * 70).all(), "Training row count")
    original_checks = read(RESULTS / "protocol_verification.json")
    require(all(v for v in original_checks.values() if isinstance(v, bool)), "Original protocol checks")
    require(sum(k.endswith("_future_invariance") for k in original_checks) == 8,
            "Recorded future-perturbation checks")
    require(read(RESULTS / "quarterly_causal_ensemble_check.json")["uses_original_three_seed_screening_only"],
            "Quarterly selection excludes confirmation seeds")
    summary["training_cutoffs"] = len(audit)

    modern_docs = [ROOT / "README.md", ROOT / "README.en.md", *ROOT.glob("docs/**/*.md")]
    local_link_count = 0
    for doc in modern_docs:
        text = doc.read_text(encoding="utf-8")
        for target in re.findall(r"!?\[[^\]]*\]\(([^)\n]+)\)", text):
            target = target.strip("<>").split("#", 1)[0]
            if not target or urlparse(target).scheme:
                continue
            require((doc.parent / unquote(target)).exists(), f"Broken link in {doc.name}: {target}")
            local_link_count += 1
    figure_count = 0
    for lang in ("zh", "en"):
        directory = ROOT / "docs/figures" / lang
        require(len(list(directory.glob("*.png"))) == 8, f"{lang} PNG count")
        require(len(list(directory.glob("*.svg"))) == 8, f"{lang} SVG count")
        for png in directory.glob("*.png"):
            with Image.open(png) as img:
                require(img.width >= 1400 and img.height >= 700, f"Figure size {png.name}")
                img.verify()
            svg = png.with_suffix(".svg")
            require(ET.parse(svg).getroot().tag.endswith("svg"), f"SVG format {svg.name}")
            figure_count += 2
    figure_manifest = ROOT / "docs/figure_manifest.json"
    if figure_manifest.exists():
        for path, expected in read(figure_manifest)["sha256_by_file"].items():
            require(digest(ROOT / path) == expected, f"Figure provenance hash {path}")
    summary["documentation"] = {"local_links": local_link_count, "figure_files": figure_count}

    metadata = read(RESULTS / "model_metadata.json")
    require(set(metadata) == {"ridge_25", "fair_tcn"}, "Exported model set")
    require(metadata["fair_tcn"]["seeds"] == [13, 31, 47, 61, 79], "Final neural seeds")
    for case, meta in metadata.items():
        require(meta["checkpoint_data_sha256"] == manifest["data_sha256"], f"{case} data hash")
        require(meta["fit_through"] == "2026-01" and meta["forecast_month"] == "2026-02",
                f"{case} final origin")
    if args.skip_model:
        summary["checkpoint_inference"] = "skipped by explicit option"
    else:
        sys.path.insert(0, str(RESEARCH))
        from predict_saved_models import predict_saved
        from research_models import ContextNet, load_research_data
        data = load_research_data()
        require(data["seq"].shape == (51, 70, 7, 2), "Sequence dimensions")
        require(data["features"].shape == (51, 70, 19), "Context dimensions")
        require(sum(p.numel() for p in ContextNet("fair").parameters()) == 1697, "CNN parameter count")
        reference = frames["forecast_2026_02.csv"].set_index("city").loc[data["cities"]]
        errors = {}
        for case, meta in metadata.items():
            prediction = predict_saved(data, case, meta)
            error = float(np.max(np.abs(prediction - reference["pred_" + case])))
            require(error < 2e-7, f"Checkpoint inference tolerance: {case} {error}")
            errors[case] = error
        summary["checkpoint_max_difference_pp"] = errors
    summary["status"] = "passed"
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
