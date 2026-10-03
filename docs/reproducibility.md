# 复现指南 / Reproduction runbook

[中文首页](../README.md) · [English README](../README.en.md)

## 1. 环境 / Environment

实测环境为 Python 3.11.7、CPU PyTorch 2.5.1，完整依赖版本见[锁定依赖](../research/requirements-tested.txt)。建议使用 Python 3.11 的独立虚拟环境。所有下列命令从仓库根目录运行。

The recorded environment uses Python 3.11.7 and CPU PyTorch 2.5.1. [Tested dependency pins](../research/requirements-tested.txt) are included. Use an isolated Python 3.11 environment and run every command below from the repository root.

```bash
git clone https://github.com/LRZer/Real-estate.git
cd Real-estate
python -m venv .venv
```

Windows PowerShell 激活 / activate:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux / macOS 激活 / activate:

```bash
source .venv/bin/activate
```

安装 / install:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

## 2. 直接核验保存结果 / Verify recorded results

```bash
python scripts/verify_repository.py
```

无需网络数据下载或模型训练。核验器只读取文件：验证面板/日期/来源、原始代码与权重哈希、全部模型MAE与预测一致性、315条训练截止记录、文档链接与图表，并使用保存的Ridge及5个神经网络权重重现2026-02预测。

No download or training is required. The read-only verifier checks the panel/dates/sources, original code/checkpoint hashes, MAE against predictions, 315 cutoff records, documentation links and figures. It then reproduces February 2026 outputs from saved Ridge and five neural checkpoints.

```bash
# 元数据、指标、文件检查；跳过权重推理 / Skip checkpoint inference
python scripts/verify_repository.py --skip-model
# 输出重新推理的预测CSV / Write reproduced forecast CSV
python research/predict_saved_models.py
```

第二条推理命令写入 `research/results/research/reproduced_forecast_2026_02.csv` 和权重核验JSON。它仅支持仓库内保存权重的 `ridge_25` 与 `fair_tcn`；其他代表模型保留完整预测与指标，但没有导出最终权重。

The inference command writes `reproduced_forecast_2026_02.csv` and checkpoint-check JSON inside the research result directory. Only `ridge_25` and `fair_tcn` have exported checkpoints; other representative methods have complete saved predictions/metrics.

## 3. 完整重跑 / Full experimental rerun

```bash
python research/run_research_experiments.py
python research/analyze_research_results.py
python research/verify_research_protocol.py
python research/predict_saved_models.py
python scripts/build_result_tables.py
python scripts/build_figures.py --language both
```

此流程会重新训练、改写实验输出并创建未纳入Git的 `research/results/research/cache/`。模型使用固定种子、逐月扩展训练、每次60个epoch；树和Ridge使用确定性配置。八种模型另执行未来数据扰动检查。CPU耗时受机器影响，明显长于保存权重推理。

This retrains models, replaces experimental outputs and creates an ignored fit cache. Seeds are fixed; training expands monthly with 60 epochs per neural fit. Tree/Ridge configurations are deterministic. Eight model types additionally receive future-data perturbation checks. Runtime depends on hardware and is substantially longer than checkpoint inference.

**保留原始仓库副本再重跑。** 原始完成清单、代码/数据指纹及权重哈希对应已提交的完成版本；跨平台重训可能有浮点差异，重跑之后应保存新的实验版本和校验清单。核验器不会悄悄更新旧哈希以接受变更。

**Keep an untouched copy before retraining.** Completion manifests, fingerprints and checkpoint hashes identify the committed version. Cross-platform retraining may introduce numerical differences. Record a new experiment version and manifest for changed artifacts; the verifier never silently replaces historical hashes.

2026-02的实际值始终排除。历史评估期此前已被研究者观察，不应在重跑后宣称为新的独立盲测。

February 2026 actuals remain excluded. Rerunning a previously inspected historical period does not make it a new independent blind test.

## 4. 数据获取 / Reacquire official data

清洗数据已经纳入仓库，复现不依赖官方站点可用性。如需重新解析原始表格：

Cleaned data are included; reproduction does not depend on live NBS availability. To parse the linked official pages again:

```bash
python research/download_official_data.py
```

脚本读取 `research/source_index.csv`，下载HTML到被忽略的 `research/data/official_pages/`，并重写数据文件。页面、解析库或换行格式变化可能使哈希改变；原始版本的出处与获取日期以数据清单为准。不要追加截止日期后的实际数据来重跑当前研究。

The script reads `research/source_index.csv`, caches ignored HTML under `research/data/official_pages/` and rewrites the cleaned CSV. Page, parser or line-ending changes may alter hashes; the manifest identifies the recorded retrieval. Do not append post-cutoff actuals to this experiment.

## 5. 图表与报告 / Figures and report

```bash
python scripts/build_result_tables.py
python scripts/build_figures.py --language en
python scripts/build_figures.py --language both
```

共8类图，每类有中英文及PNG/SVG版本。图表使用保存的数据和预测，未平滑或修改模型结果。SVG将字体转为路径，下载后不依赖本机字体；重新生成中文需 Microsoft YaHei、SimHei 或 Noto Sans CJK SC/Noto Sans SC。没有中文字体时可单独生成英文版。GitHub核验读取已有图表，不要求CI安装中文字体。

Eight figure types each have Chinese/English PNG and SVG outputs. Figures use recorded data/predictions without smoothing or altering model results. SVG text is converted to paths. Regenerating Chinese requires Microsoft YaHei, SimHei or Noto Sans CJK SC/Noto Sans SC; English can be regenerated separately. CI verifies committed figures and does not require CJK fonts.

此前完成的[8页中文成果PDF](../research/output/pdf/70城新房指数预测_成果展示.pdf)和[详细中文报告](../research/最终实验报告.md)直接纳入仓库。PDF源脚本为 `research/build_showcase_pdf.py`，附加依赖见 `requirements-docs.txt`；字体路径与Windows环境相关，已有PDF无需重新构建。

The completed [eight-page Chinese PDF](../research/output/pdf/70城新房指数预测_成果展示.pdf) and [Chinese report](../research/最终实验报告.md) are included. Optional PDF rebuilding uses `research/build_showcase_pdf.py` and `requirements-docs.txt`; its font paths depend on the original Windows environment.

## 6. 目录与版本边界 / Directory and version boundaries

| 路径 / Path | 内容 / Contents |
| --- | --- |
| `research/data/`、`source_index.csv` | 原始清洗面板与来源 / Recorded clean panel and sources |
| `research/results/research/` | 当前20配置、2组合、逐月预测、选择与审计 / Current study outputs |
| `research/results/research/models/` | Ridge系数、神经标准化参数、5个权重 / Saved final checkpoints |
| `research/results/adaptive/`、其他旧结果 / other legacy results | 前序探索，不能与当前协议混合比较 / Earlier experiments with different protocols |
| `docs/`、`scripts/` | 发布文档、图表、结果表和仓库核验 / Publication documents and verification |

`research/` 保留完成时的源文件字节，Git属性关闭该目录的文本换行转换，以保护原实验指纹。没有上传虚拟环境、缓存、密钥或下载HTML。仓库没有新增开源许可证，`CITATION.cff` 提供引用元数据。

The research snapshot keeps its original bytes; Git attributes disable line-ending conversion within it to protect experiment fingerprints. Virtual environments, caches, keys and downloaded HTML are excluded. No new open-source license has been added; `CITATION.cff` supplies citation metadata.
