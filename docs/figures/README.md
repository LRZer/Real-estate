# 图表索引 / Figure index

[中文首页](../../README.md) · [English README](../../README.en.md) · [完整结果 / Results](../results.en.md)

8类图表，每类均提供中文、英文、PNG及SVG。PNG用于GitHub展示；SVG适合放大、报告和幻灯片。所有曲线、误差与区间使用提交的数据和实验输出，架构图对应已实现的代码。

Eight figure types are available in Chinese/English and PNG/SVG. PNG renders in GitHub; SVG supports scaling for reports and slides. Curves, scores and intervals use committed experimental outputs; architecture diagrams describe the implementation.

| 图表 / Figure | 内容 / Contents | 中文 | English |
| --- | --- | --- | --- |
| 01 数据趋势 / Market | 57个月新房与二手房均值、涨跌城市占比 / Means and direction shares | [PNG](zh/01_market.png) · [SVG](zh/01_market.svg) | [PNG](en/01_market.png) · [SVG](en/01_market.svg) |
| 02 时间协议 / Protocol | 初始训练、验证、回顾评估、发布后更新顺序 / Origins and causal update order | [PNG](zh/02_protocol.png) · [SVG](zh/02_protocol.svg) | [PNG](en/02_protocol.png) · [SVG](en/02_protocol.svg) |
| 03 模型结构 / Architecture | 输入维度、卷积与预测头、独立对照模块 / Tensor shapes and separate ablations | [PNG](zh/03_architecture.png) · [SVG](zh/03_architecture.svg) | [PNG](en/03_architecture.png) · [SVG](en/03_architecture.svg) |
| 04 配置验证 / Validation | 20配置、单种子离散程度、逐月增益热图 / All cases, seed scores and monthly gains | [PNG](zh/04_validation.png) · [SVG](zh/04_validation.svg) | [PNG](en/04_validation.png) · [SVG](en/04_validation.svg) |
| 05 误差与区间 / Performance | 逐月MAE、配对区间、误差散点、在线权重 / Monthly errors, intervals and weights | [PNG](zh/05_performance.png) · [SVG](zh/05_performance.svg) | [PNG](en/05_performance.png) · [SVG](en/05_performance.svg) |
| 06 方向诊断 / Direction | 方向分组误差、混淆矩阵、平衡准确率 / Sign errors, confusion and balanced scores | [PNG](zh/06_direction.png) · [SVG](zh/06_direction.svg) | [PNG](en/06_direction.png) · [SVG](en/06_direction.svg) |
| 07 山东案例 / Shandong | 济南、青岛、烟台、济宁的实际/预测曲线 / Four city cases | [PNG](zh/07_shandong.png) · [SVG](zh/07_shandong.svg) | [PNG](en/07_shandong.png) · [SVG](en/07_shandong.svg) |
| 08 分段敏感性 / Sensitivity | 2025各季度、2026基期调整、过去季度选择 / Quarters, rebasing and causal selection | [PNG](zh/08_robustness.png) · [SVG](zh/08_robustness.svg) | [PNG](en/08_robustness.png) · [SVG](en/08_robustness.svg) |

## 可追溯性 / Traceability

- [生成程序 / Builder](../../scripts/build_figures.py)
- [图表与输入哈希 / Figure and input hashes](../figure_manifest.json)
- [原始面板 / Input panel](../../research/data/official_nbs_70city.csv)
- [验证结果 / Validation outputs](../../research/results/research/validation_screening_results.json)
- [回顾预测 / Retrospective predictions](../../research/results/research/retrospective_predictions.csv)
- [季度检查 / Quarterly check](../../research/results/research/quarterly_causal_ensemble_check.json)

`04_validation` 使用原始3种子筛选，最终5种子确认成绩单独注明。`05_performance` 的区间是描述性自助法区间。`06_direction` 的分类头混淆矩阵与回归值涨跌判断不是同一指标。

`04_validation` uses original three-seed screening and separately notes five-seed confirmation. Intervals in `05_performance` are descriptive bootstrap intervals. The classifier confusion matrix in `06_direction` is distinct from regression-sign evaluation.
