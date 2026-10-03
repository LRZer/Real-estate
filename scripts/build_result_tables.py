"""Generate bilingual result documents and portable tables from saved outputs."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "research/results/research"
DOCS = ROOT / "docs"


def load(filename):
    return json.loads((RESULTS / filename).read_text(encoding="utf-8"))


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *["| " + " | ".join(map(str, row)) + " |" for row in rows]])


def num(value):
    return "—" if value is None else f"{value:.4f}"


def main():
    registry = load("locked_experiment_registry.json")
    screen, final, test = (load(f) for f in ("validation_screening_results.json",
                                           "validation_results.json", "retrospective_results.json"))
    ev = load("ensemble_validation.json")
    selection = load("selection_before_retrospective.json")
    segments = load("segment_diagnostics.json")
    ci = load("gain_vs_strong_baseline.json")
    quarterly = load("quarterly_causal_ensemble_check.json")["quarters"]
    frame = pd.read_csv(RESULTS / "retrospective_predictions.csv")
    valframe = pd.read_csv(RESULTS / "validation_predictions_2024.csv")
    raw = pd.read_csv(ROOT / "research/data/official_nbs_70city.csv").set_index(["period", "city"])
    last_val = np.array([raw.loc[(m, c), "new_mom_pct"] for m, c in
                         zip(valframe.input_month, valframe.city)])
    naive_val = float(np.abs(last_val - valframe.actual_change_pct).mean())
    core = ["last_month", "ridge_25", "fair_tcn", "decomp_ridge", "online_ensemble"]
    DOCS.joinpath("tables").mkdir(exist_ok=True)
    rows = []
    for case, config in registry["configurations"].items():
        rows.append({"case": case, "group": config["group"],
                     "screening_mae_pp": screen[case]["mae_pp"],
                     "final_validation_mae_pp": final[case]["mae_pp"],
                     "final_seed_count": len(final[case]["seed_mae_pp"]),
                     "validation_individual_seed_sd_pp": final[case]["seed_mae_std_pp"],
                     "retrospective_mae_pp": test.get(case, {}).get("mae_pp"),
                     "price_gate_passed": selection["price_gates"][case]["passes_predeclared_price_gate"]})
    with (DOCS / "tables/model_results.csv").open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Every city, not just the four Shandong examples.
    city_rows = []
    for city, group in frame.groupby("city"):
        for case in test:
            city_rows.append({"city": city, "case": case, "months": len(group),
                              "mae_pp": float(np.abs(group["pred_" + case] - group.actual_change_pct).mean())})
    pd.DataFrame(city_rows).to_csv(DOCS / "tables/all_city_mae.csv", index=False, encoding="utf-8")
    for lang in ("zh", "en"):
        zh = lang == "zh"
        def t(a, b): return a if zh else b
        parts = [
            t("# 完整实验结果", "# Complete experiment results"),
            f"[中文](results.zh.md) · [English](results.en.md) · [{t('方法','Methods')}](methods.{lang}.md)",
            t("本页由保存的 JSON / CSV 自动生成。MAE、RMSE、偏差及增益均以百分点（pp）计；MAE越低越好。2025-01至2026-01属于已观察过历史期间的回顾性评估，不是新的盲测。",
              "This page is generated from saved JSON/CSV outputs. MAE, RMSE, bias and gains use percentage points (pp); lower MAE is better. Jan 2025–Jan 2026 is a previously inspected historical period, so evaluation is retrospective, not a new blind test."),
            t("## 20个单模型配置", "## All 20 single-model configurations"),
            t("筛选列使用原始3种子13/31/47；确认列仅对七个月网络和加权方向模型增至5种子，其他配置保持原预算。种子标准差是各独立模型MAE的离散程度，不是种子平均预测的置信区间。— 表示该配置未进入后续代表模型评估。",
              "Screening uses the original seeds 13/31/47. Confirmation adds two seeds only for the contextual CNN and weighted direction candidate; other budgets remain unchanged. Seed SD describes individual-model MAE dispersion, not a confidence interval for averaged predictions. — means the configuration was not selected for representative evaluation."),
            table(t(["配置ID","组别","3种子筛选MAE","最终验证MAE","种子数*","种子MAE标准差","回顾性MAE"],
                    ["Case ID","Group","3-seed screen MAE","Final val MAE","Seeds*","Seed MAE SD","Retrospective MAE"]),
                  [[r["case"],r["group"],num(r["screening_mae_pp"]),num(r["final_validation_mae_pp"]),
                    r["final_seed_count"],num(r["validation_individual_seed_sd_pp"]),num(r["retrospective_mae_pp"])] for r in rows]),
            t("*确定性Ridge/树模型的计数1表示一个结果，未因元数据中的种子列表重复拟合。",
              "*A count of 1 for deterministic Ridge/tree models denotes one result; metadata seed lists do not imply repeated deterministic fits."),
            t("[下载数值表](tables/model_results.csv)。逐种子成绩、逐月MAE和完整分类指标保留在[最终验证JSON](../research/results/research/validation_results.json)与[回顾性JSON](../research/results/research/retrospective_results.json)。",
              "[Download the numeric table](tables/model_results.csv). Individual-seed scores, monthly MAE and classification metrics remain in [validation JSON](../research/results/research/validation_results.json) and [retrospective JSON](../research/results/research/retrospective_results.json)."),
            f"![{t('20配置验证比较','20-configuration validation comparison')}](figures/{lang}/04_validation.png)",
            t("## 无学习基线与组合", "## No-learning control and ensembles"),
            table(t(["配置","验证MAE","回顾性MAE"],["Case","Validation MAE","Retrospective MAE"]),
                  [["last_month",num(naive_val),num(test["last_month"]["mae_pp"])],
                   ["uniform_ensemble",num(ev["uniform"]["mae_pp"]),num(test["uniform_ensemble"]["mae_pp"])],
                   ["online_ensemble",num(ev["online"]["mae_pp"]),num(test["online_ensemble"]["mae_pp"])]]),
            t("组合专家为Ridge、七个月网络和沿用上月；前两者由完整2024验证期选出。因此上表组合验证成绩包含专家选择不确定性。在线权重只用已经实现的过去误差更新；更严格的季度选择结果见下。",
              "Experts are Ridge, the contextual CNN and last month; the first two were chosen using all of 2024. Ensemble validation scores therefore include expert-selection uncertainty. Online weights update only from realized past errors; the stricter quarterly selection check appears below."),
            t("## 选择结论与配对不确定性", "## Selection and paired uncertainty"),
            t("全部20个配置均未通过替换门槛：验证MAE至少改善3%、至少8个月改善、上下半年均不恶化超过5%。主模型保留ridge_25。四个方向配置也均未通过方向替换门槛。双分支Ridge的后续成绩最低，但不以此倒推选择主模型。",
              "None of the 20 configurations passes the replacement gate: at least 3% validation MAE reduction, at least eight improved months, and no half-year deterioration over 5%. ridge_25 remains primary. None of the four direction cases passes its gate. Decomposed Ridge has the lowest later-period MAE, but is not promoted using that outcome."),
            table(t(["相对Ridge的配置","平均月度MAE增益","95%区间下界","上界","改善月份/13"],
                    ["Compared with Ridge","Mean monthly MAE gain","95% lower","Upper","Improved months / 13"]),
                  [[c,num(v["mean_monthly_mae_gain_pp"]),num(v["circular_3month_block_95pct_interval_pp"][0]),
                    num(v["circular_3month_block_95pct_interval_pp"][1]),v["months_beating_baseline"]] for c,v in ci.items()]),
            t("增益=Ridge误差−候选误差；正数代表候选更好。按月份成块的3个月循环自助法，20,000次重采样。只有13个时间点且存在模型选择，这些是描述性区间，不构成独立盲测的显著性证明。",
              "Gain = Ridge MAE minus candidate MAE; positive favors the candidate. Intervals use 20,000 circular three-month block resamples. With only 13 temporal points and model selection, these are descriptive intervals, not evidence from an independent blind test."),
            f"![{t('误差与区间','Errors and intervals')}](figures/{lang}/05_performance.png)",
            t("## 分段表现", "## Period breakdown"),
            table([t("期间","Period"),*core],
                  [[s,*[num(v[c]["mae_pp"]) for c in core]] for s,v in segments.items()]),
            t("2025年6—8月，七个月网络MAE比Ridge低14.6%；2026年1月网络未优于Ridge。2025全年与2026年1月单独报告，以观察基期/分类权重调整月份的敏感性。",
              "In June–August 2025, CNN MAE is 14.6% below Ridge. In January 2026 the CNN does not outperform Ridge. The 2025-only and January 2026 rows expose sensitivity to the base/category-weight adjustment month."),
            f"![{t('季度和基期敏感性','Quarterly and rebasing sensitivity')}](figures/{lang}/08_robustness.png)",
            t("## 方向与反转", "## Direction and reversals"),
            t("回顾性实际标签：下跌705、上涨160、持平45。预测回归值的三分类采用±0.05 pp持平区间；二分类上涨为>0，平衡准确率排除实际为0的记录。辅助分类头的argmax结果单独报告。",
              "Retrospective actual labels: 705 falling, 160 rising and 45 flat. Three-way regression signs use a ±0.05 pp flat band; binary rise uses >0, with balanced accuracy excluding actual zeros. Auxiliary-head argmax predictions are reported separately."),
            table(t(["配置","上涨MAE","下跌MAE","回归值二类平衡准确率"],["Case","Rising MAE","Falling MAE","Regression binary balanced accuracy"]),
                  [[c,num(test[c]["rising_mae_pp"]),num(test[c]["falling_mae_pp"]),
                    f'{test[c]["balanced_binary_rise_accuracy_excluding_actual_zero"]*100:.2f}%']
                   for c in ("last_month","ridge_25","fair_tcn","decomp_tcn","multitask_0.05_weighted")]),
            t("加权方向分类头回顾性上涨召回率56.25%、上涨精确率49.18%；真实下跌/持平/上涨三行的预测数量分别为[628,0,77]、[29,0,16]、[70,0,90]。该头未输出持平类别；其验证平衡准确率未通过方向门槛，不能作为主模型替代。",
              "The weighted direction head has retrospective rise recall 56.25% and rise precision 49.18%. Counts for actual fall/flat/rise are [628,0,77], [29,0,16] and [70,0,90] across predicted fall/flat/rise. It never predicts flat and does not pass its validation direction gate, so it does not replace the primary model."),
            f"![{t('方向诊断','Direction diagnostics')}](figures/{lang}/06_direction.png)",
            t("## 全部城市与山东案例", "## All cities and Shandong examples"),
            t("[70城×11方法的城市MAE表](tables/all_city_mae.csv)含770行。山东四城成绩如下，每城13个月；济宁的学习模型均未优于沿用上月。",
              "The [70-city × 11-method MAE table](tables/all_city_mae.csv) contains 770 rows. The four Shandong examples below each have 13 months; learned models underperform last month in Jining."),
        ]
        sd = pd.read_csv(RESULTS / "shandong_metrics.csv")
        parts.append(table([t("城市","City"),"last_month","ridge_25","fair_tcn","online_ensemble"],
                           [[city,*[num(group.set_index("case").loc[c,"mae_pp"]) for c in
                                    ("last_month","ridge_25","fair_tcn","online_ensemble")]]
                            for city,group in sd.groupby("city")]))
        parts += [
            f"![{t('山东案例','Shandong cases')}](figures/{lang}/07_shandong.png)",
            t("## 只使用过去季度选择的检查", "## Selection using past quarters only"),
            table(t(["选择信息截至","评估季度起点","基线专家","神经专家","在线MAE","均匀MAE","Ridge25 MAE"],
                    ["Choose through","Evaluate from","Baseline expert","Neural expert","Online MAE","Uniform MAE","Ridge25 MAE"]),
                  [[q["choose_through_month"],q["evaluate_quarter_start"],q["baseline_expert"],q["nonlinear_expert"],
                    num(q["online_mae_pp"]),num(q["uniform_mae_pp"]),num(q["ridge25_mae_pp"])] for q in quarterly]),
            t("上述选择仅用原始3种子筛选预测。5种子全年确认不会参与更早季度的选择。在线组合在第二季度未改善，在第三、四季度小幅改善，不能把完整验证期的专家选择成绩当作纯时间外检验。",
              "These choices use original three-seed screening predictions only. Five-seed full-year confirmation never enters earlier-quarter selection. Online averaging does not improve Q2 and gives small gains in Q3/Q4; full-validation expert selection is not a pure out-of-time check."),
            t("## 输出与审计索引", "## Outputs and audits"),
            t("- [原始3种子筛选](../research/results/research/validation_screening_results.json)\n- [2024全部预测](../research/results/research/validation_predictions_2024.csv)\n- [回顾性全部预测](../research/results/research/retrospective_predictions.csv)\n- [2026-02预测，无实际标签](../research/results/research/forecast_2026_02.csv)\n- [315条训练截止审计](../research/results/research/training_cutoff_audit.csv)\n- [误差分解与72次由非上涨转上涨诊断](../research/results/research/error_decomposition.json)\n- [选择记录](../research/results/research/selection_before_retrospective.json)\n- [主模型和神经模型最终权重](../research/results/research/models/)",
              "- [Original three-seed screening](../research/results/research/validation_screening_results.json)\n- [All 2024 predictions](../research/results/research/validation_predictions_2024.csv)\n- [All retrospective predictions](../research/results/research/retrospective_predictions.csv)\n- [February 2026 output without actual labels](../research/results/research/forecast_2026_02.csv)\n- [315 training-cutoff audit records](../research/results/research/training_cutoff_audit.csv)\n- [Error decomposition and 72 non-rise-to-rise cases](../research/results/research/error_decomposition.json)\n- [Selection record](../research/results/research/selection_before_retrospective.json)\n- [Final primary/neural checkpoints](../research/results/research/models/)"),
        ]
        (DOCS / f"results.{lang}.md").write_text("\n\n".join(parts) + "\n", encoding="utf-8")
    print("Generated two result documents, 20-configuration table and 770 city/method scores.")


if __name__ == "__main__":
    main()
