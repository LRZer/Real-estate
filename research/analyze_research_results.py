"""Diagnostics and research figures from the locked experiment outputs."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from research_models import ROOT
from run_research_experiments import CASES, NONLINEAR_CANDIDATES, online_mix
from train_adaptive_backtest import metrics, block_interval

OUT = ROOT / "results/research"
LABELS = {"last_month": "沿用上月", "ridge_25": "逐月 Ridge", "ridge_5": "Ridge α=5",
          "ridge_100": "Ridge α=100", "hgb_15": "梯度提升树 15叶", "hgb_7": "梯度提升树 7叶",
          "nlinear": "NLinear 形式", "legacy_tcn": "六个月原型网络", "fair_tcn": "七个月+特征网络",
          "revin_tcn": "窗口归一化网络", "decomp_ridge": "双分支 Ridge", "decomp_tcn": "双分支网络",
          "decomp_revin": "归一化双分支", "online_ensemble": "在线组合", "uniform_ensemble": "均匀组合"}


def label(name):
    if name.startswith("bias_"):
        return "偏差校正 " + name[5:]
    if name.startswith("multitask_"):
        return "方向任务 " + name[10:].replace("weighted", "加权").replace("plain", "不加权")
    return LABELS.get(name, name)


def read_json(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write_json(name, payload):
    (OUT / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def frame_arrays(frame, column):
    return frame.pivot(index="target_month", columns="city", values=column).sort_index().sort_index(axis=1).to_numpy()


def main():
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "font.size": 10, "figure.facecolor": "white"})
    selection = read_json("selection_before_retrospective.json")
    val = read_json("validation_results.json")
    test = read_json("retrospective_results.json")
    primary, neural = selection["primary"], selection["best_neural_screened"]
    table = []
    for name, s in val.items():
        table.append({"case": name, "display_name": label(name), "group": CASES[name]["group"],
                      "validation_mae_pp": s["mae_pp"], "retrospective_mae_pp": test.get(name, {}).get("mae_pp"),
                      "price_gate_passed": selection["price_gates"][name]["passes_predeclared_price_gate"],
                      "seed_mae_std_pp": s["seed_mae_std_pp"]})
    pd.DataFrame(table).to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    frame = pd.read_csv(OUT / "retrospective_predictions.csv")
    actual = frame_arrays(frame, "actual_change_pct")
    preds = {n: frame_arrays(frame, "pred_"+n) for n in test}
    month_list = sorted(frame.target_month.unique())
    ranges = {"2025_Q1": ("2025-01", "2025-03"), "2025_Q2": ("2025-04", "2025-06"),
              "2025_Q3": ("2025-07", "2025-09"), "2025_Q4": ("2025-10", "2025-12"),
              "2025_Jun_Aug": ("2025-06", "2025-08"), "2025_only": ("2025-01", "2025-12"),
              "2026_Jan_only": ("2026-01", "2026-01")}
    segments = {}
    for segment, (lo, hi) in ranges.items():
        keep = [i for i, m in enumerate(month_list) if lo <= m <= hi]
        segments[segment] = {n: metrics(actual[keep], p[keep]) for n, p in preds.items()}
    write_json("segment_diagnostics.json", segments)
    baseline = preds[selection["best_simple_baseline"]]
    write_json("gain_vs_strong_baseline.json", {n: block_interval(actual, baseline, p) for n, p in preds.items() if n != primary})
    detailed = {}
    previous = frame_arrays(frame, "pred_last_month")
    turning = (previous <= 0) & (actual > 0)
    for name, p in preds.items():
        market_error = p.mean(axis=1)-actual.mean(axis=1)
        residual_error = (p-p.mean(axis=1, keepdims=True))-(actual-actual.mean(axis=1, keepdims=True))
        residual_month_mae = np.abs(residual_error).mean(axis=1)
        detailed[name] = {"rmse_pp": float(np.sqrt(np.mean((p-actual)**2))),
                         "market_mae_pp": float(np.abs(market_error).mean()),
                         "residual_mae_pp": float(np.abs(residual_error).mean()),
                         "market_abs_error_vs_residual_mae_monthly_corr": float(np.corrcoef(np.abs(market_error), residual_month_mae)[0, 1]),
                         "turn_to_rise_count": int(turning.sum()), "turn_to_rise_mae_pp": float(np.abs(p-actual)[turning].mean()),
                         "turn_to_rise_recall": float((p[turning]>0).mean())}
    write_json("error_decomposition.json", detailed)
    sd = frame[frame.city.isin(("济南", "青岛", "烟台", "济宁"))].copy()
    sd.to_csv(OUT / "shandong_retrospective.csv", index=False, encoding="utf-8-sig")
    city_scores = []
    for city, city_rows in sd.groupby("city"):
        for name in ("last_month", primary, neural, "online_ensemble"):
            city_scores.append({"city": city, "case": name, "mae_pp": float(np.abs(city_rows["pred_"+name]-city_rows.actual_change_pct).mean())})
    pd.DataFrame(city_scores).to_csv(OUT / "shandong_metrics.csv", index=False, encoding="utf-8-sig")
    # The following quarter analysis uses only original three-seed predictions.
    screening = pd.read_csv(OUT / "validation_screening_three_seeds.csv")
    yv = frame_arrays(screening, "actual_change_pct")
    pv = {n: frame_arrays(screening, "pred_"+n) for n in CASES}
    latest = pd.read_csv(ROOT / "data/official_nbs_70city.csv").pivot(index="period", columns="city", values="new_mom_pct").sort_index(axis=1)
    lastv = latest.loc[pd.period_range("2023-12", "2024-11", freq="M").astype(str)].to_numpy()
    quarters = []
    for start in (3, 6, 9):
        base = min([n for n, c in CASES.items() if c["group"] == "baseline"], key=lambda n: np.abs(pv[n][:start]-yv[:start]).mean())
        expert = min(NONLINEAR_CANDIDATES, key=lambda n: np.abs(pv[n][:start]-yv[:start]).mean())
        experts = np.stack([pv[base], pv[expert], lastv], axis=-1)
        _, risk, _ = online_mix(experts[:start], yv[:start])
        online, _, _ = online_mix(experts[start:start+3], yv[start:start+3], risk)
        quarters.append({"choose_through_month": f"2024-{start:02d}", "evaluate_quarter_start": f"2024-{start+1:02d}",
                         "baseline_expert": base, "nonlinear_expert": expert,
                         "online_mae_pp": metrics(yv[start:start+3], online)["mae_pp"],
                         "uniform_mae_pp": metrics(yv[start:start+3], experts[start:start+3].mean(axis=-1))["mae_pp"],
                         "ridge25_mae_pp": metrics(yv[start:start+3], pv["ridge_25"][start:start+3])["mae_pp"]})
    write_json("quarterly_causal_ensemble_check.json", {"quarters": quarters, "uses_original_three_seed_screening_only": True})

    # Main validation ablation: the full grid remains available in the CSV.
    names = ["ridge_25", "nlinear", "hgb_7", "legacy_tcn", "fair_tcn", "revin_tcn", "bias_0.5_0.5", "decomp_ridge", "decomp_tcn", selection["direction_representative"]]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    screening_scores = read_json("validation_screening_results.json")
    vals = [screening_scores[n]["mae_pp"] for n in names]
    positions = np.arange(len(names))
    ax.scatter(vals[::-1], positions, s=65, color=["#007F82" if n == neural else "#233B53" if n == primary else "#879CA8" for n in names][::-1])
    ax.set_yticks(positions, [label(n) for n in names][::-1])
    ax.axvline(val[primary]["mae_pp"], color="#D07745", linestyle="--", label="主模型验证误差")
    for position, number in zip(positions, vals[::-1]): ax.text(number+0.0015, position, f"{number:.4f}", va="center", fontsize=9)
    ax.set_xlim(0.25, max(vals)+0.025); ax.set_xlabel("2024 滚动验证 MAE（百分点，越低越好）")
    ax.set_title("2024 输入与结构对照：所有神经配置均使用三个种子")
    ax.grid(axis="x", alpha=0.15)
    ax.spines[["top", "right"]].set_visible(False); fig.tight_layout()
    fig.savefig(OUT / "validation_ablation.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [2, 1]}, sharex=True)
    for n, color in (("last_month", "#91A1AB"), (primary, "#233B53"), (neural, "#007F82"), ("online_ensemble", "#D07745")):
        axes[0].plot(month_list, np.abs(preds[n]-actual).mean(axis=1), marker="o", label=label(n), color=color, linewidth=1.6, markersize=4)
    axes[0].set_ylabel("逐月 MAE（百分点）"); axes[0].legend(ncol=2, loc="upper left", fontsize=9)
    weights = read_json("ensemble_weights.json")
    w = np.array([weights["months"][m]["weights_before_forecast"] for m in month_list])
    for j, color in enumerate(("#233B53", "#007F82", "#91A1AB")):
        axes[1].plot(month_list, w[:, j], color=color, label=label(weights["experts"][j]))
    axes[1].set_ylabel("预测前组合权重"); axes[1].set_ylim(0, 1); axes[1].tick_params(axis="x", rotation=35)
    for ax in axes: ax.grid(alpha=0.2); ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("逐月更新及专家互补：回顾性评估（2025-01 至 2026-01）")
    fig.tight_layout(); fig.savefig(OUT / "monthly_performance.png", dpi=190); plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True, sharey=True)
    for ax, city in zip(axes.flat, ("济南", "青岛", "烟台", "济宁")):
        s = sd[sd.city.eq(city)].sort_values("target_month")
        ax.plot(s.target_month, s.actual_change_pct, label="实际", color="#D07745", linewidth=2)
        ax.plot(s.target_month, s["pred_"+primary], label="主模型", color="#233B53")
        ax.plot(s.target_month, s["pred_"+neural], label="神经网络", color="#007F82", linestyle="--")
        ax.set_title(city); ax.axhline(0, color="#ADB5BD", linewidth=0.7); ax.grid(alpha=0.2)
        ax.set_xticks(s.target_month.iloc[::3]); ax.tick_params(axis="x", rotation=30)
    axes[0, 0].legend(ncol=3, fontsize=8); fig.suptitle("山东四城：下一月新房指数环比预测（%）")
    fig.tight_layout(); fig.savefig(OUT / "shandong_cases.png", dpi=190); plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6)); ax.set_xlim(0, 11); ax.set_ylim(0, 6); ax.axis("off")
    def box(x, y, width, height, title, subtitle, color="#E9F1F4"):
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.06", facecolor=color, edgecolor="#BBCAD1"))
        ax.text(x+width/2, y+height*0.69, title, ha="center", va="center", weight="bold", fontsize=11, color="#233B53")
        ax.text(x+width/2, y+height*0.28, subtitle, ha="center", va="center", fontsize=9, color="#405665")
    def arrow(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, linewidth=1.3, color="#657D89"))
    box(.2, 3.6, 2.5, 1.5, "已公布的七个月窗口", "新房 + 二手房\n70城 × 7月 × 2通道")
    box(3.3, 3.6, 2.4, 1.5, "时间卷积编码", "2层 Conv1d + ReLU\n12维城市时序表示")
    box(3.3, 1.2, 2.4, 1.5, "相同上下文", "19项数值特征\n+ 4维城市嵌入")
    box(6.3, 2.3, 2, 1.5, "残差预测头", "24维隐藏层\n预测下一期修正量")
    box(8.9, 2.3, 1.8, 1.5, "输出", "上一期变化\n+ 预测修正量", "#E4F3ED")
    arrow((2.7, 4.35), (3.3, 4.35)); arrow((5.7, 4.35), (6.3, 3.2)); arrow((5.7, 1.95), (6.3, 2.8)); arrow((8.3, 3.05), (8.9, 3.05))
    ax.text(.2, .4, "独立消融：窗口可逆归一化 / 全国均值与城市残差双分支 / 三分类辅助任务", fontsize=10, color="#233B53")
    ax.text(.2, 5.65, "可展示的神经模型：输入公平的短序列预测网络（1,697 参数）", fontsize=15, weight="bold", color="#233B53")
    fig.tight_layout(); fig.savefig(OUT / "model_architecture.png", dpi=190); plt.close(fig)
    print(json.dumps({"primary": primary, "neural": neural, "diagnostics": "completed"}, ensure_ascii=False))


if __name__ == "__main__": main()
