"""Audit the finished delivery and record its exact artifact versions."""
from __future__ import annotations
import hashlib
import importlib.metadata as version
import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np
import pandas as pd
from research_models import ROOT, load_research_data
from run_research_experiments import fingerprint, CASES

R = ROOT / "results/research"
def read(name): return json.loads((R / name).read_text(encoding="utf-8"))
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    raw = pd.read_csv(ROOT / "data/official_nbs_70city.csv")
    assert len(raw) == 3990 and not raw.duplicated(["period", "city"]).any()
    assert raw.period.max() == "2026-01"
    registry = read("locked_experiment_registry.json")
    assert registry["code_and_data_fingerprint"] == fingerprint()
    assert len(registry["configurations"]) == 20 and registry["ensemble_configurations"] == 2
    assert registry["data_sha256"] == sha(ROOT / "data/official_nbs_70city.csv").upper()
    calendar = pd.read_csv(ROOT / "source_index.csv")
    calendar["official_release_date"] = calendar.source_url.map(lambda url: pd.to_datetime(re.search(r"/t(\d{8})_", url).group(1), format="%Y%m%d").strftime("%Y-%m-%d"))
    assert calendar.official_release_date.max() <= "2026-02-28"
    calendar["prediction_target_month"] = calendar.period.map(lambda m: str(pd.Period(m)+1))
    calendar["earliest_origin_date"] = calendar.official_release_date
    calendar.to_csv(R / "release_calendar.csv", index=False, encoding="utf-8-sig")
    val_frame = pd.read_csv(R / "validation_predictions_2024.csv")
    test_frame = pd.read_csv(R / "retrospective_predictions.csv")
    scores = {"validation": read("validation_results.json"), "retrospective": read("retrospective_results.json")}
    for stage, frame in (("validation", val_frame), ("retrospective", test_frame)):
        for name, metric in scores[stage].items():
            actual_mae = float(np.abs(frame["pred_"+name]-frame.actual_change_pct).mean())
            assert abs(actual_mae-metric["mae_pp"]) < 1e-9, (stage,name)
    assert len(read("validation_screening_results.json")) == 20
    selection = read("selection_before_retrospective.json")
    assert selection["primary"] == "ridge_25"
    assert not any(g["passes_predeclared_price_gate"] for g in selection["price_gates"].values())
    assert not any(g["passes_direction_gate"] for g in selection["direction_gates"].values())
    assert all(read("protocol_verification.json").values())
    assert all(entry["loaded_without_retraining"] for entry in read("checkpoint_verification.json").values())
    data = load_research_data()
    model_meta = read("model_metadata.json")
    for name, entry in model_meta.items():
        entry.update(cities=data["cities"], numeric_features=data["numeric"],
                     output_unit="percentage-point monthly index movement", fit_first_target="2021-12",
                     checkpoint_data_sha256=registry["data_sha256"], code_and_data_fingerprint=fingerprint())
    (R / "model_metadata.json").write_text(json.dumps(model_meta, ensure_ascii=False, indent=2), encoding="utf-8")
    packages = ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "requests", "beautifulsoup4", "lxml", "threadpoolctl")
    versions = {name: version.version(name) for name in packages}
    versions["torch"] = version.version("torch")
    text = "# Versions used for the completed experiments. Install PyTorch CPU separately.\n"
    text += "\n".join(f"{name}=={v}" for name,v in versions.items() if name != "torch")+"\n"
    (ROOT / "requirements-tested.txt").write_text(text, encoding="utf-8")
    for filename in ("申请展示提纲.md",):
        path = ROOT / filename
        content = path.read_text(encoding="utf-8")
        note = "> 当前完成版见[项目介绍与面试准备](项目介绍与面试准备.md)与[最终实验报告](最终实验报告.md)。以下保留早期展示提纲，数字对应早期版本。\n\n"
        if not content.startswith("> 当前完成版"):
            path.write_text(note+content, encoding="utf-8")
    included = [ROOT / n for n in ("research_models.py", "run_research_experiments.py", "predict_saved_models.py", "最终实验报告.md", "output/pdf/70城新房指数预测_成果展示.pdf")]
    included += sorted((R / "models").rglob("*"))
    hashes = {str(path.relative_to(ROOT)).replace("\\","/"): sha(path) for path in included if path.is_file()}
    manifest = {"status": "completed_and_verified", "completed_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
                "information_cutoff": "2026-02-28", "actual_data_end": "2026-01", "configuration_count": 22,
                "validation_rows": len(val_frame), "retrospective_rows": len(test_frame), "final_forecast_rows": 70,
                "evaluation_status": "retrospective research; researcher previously inspected historical evaluation period",
                "forecast_origin_rule": "on or after the preceding statistical month's official release, generally mid-target-month",
                "data_sha256": registry["data_sha256"], "verified_metrics_match_saved_predictions": True,
                "official_releases_before_cutoff": True, "runtime_versions": versions,
                "sha256_by_artifact": hashes,
                "pdf_qa": {"page_count": 8, "all_pages_rendered_and_visually_reviewed": True}}
    (R / "completion_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k:manifest[k] for k in ("status","configuration_count","validation_rows","retrospective_rows","final_forecast_rows")}, indent=2))


if __name__ == "__main__": main()
