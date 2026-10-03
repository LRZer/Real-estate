# Adaptive Forecasting of New-Home Price Movements Across 70 Chinese Cities

[中文](README.md) · [English](README.en.md) · [Methods](docs/methods.en.md) · [All results](docs/results.en.md) · [Dataset](docs/data.en.md) · [Figures](docs/figures/README.md)

Predict the next statistical month's new-home price-index movement in each city using official National Bureau of Statistics of China (NBS) releases. The project addresses regional market analysis for a property developer through data reconstruction, feature engineering, monthly model updates, neural design and temporal evaluation.

The input panel contains **57 months, 70 cities and 3,990 records, from May 2021 through January 2026**. The information cutoff is **28 February 2026**. The target is a city-level index movement: a reported index of 99.7 with previous month=100 means -0.3%. It is not an individual property valuation or a price per square metre.

## Results

MAE is the mean absolute difference between predicted and observed monthly movements, in **percentage points (pp)**; lower is better. The final neural candidate averages five fixed seeds.

| Method | 2024 rolling-validation MAE | Jan 2025-Jan 2026 retrospective MAE | Role |
| --- | ---: | ---: | --- |
| Last-month baseline | 0.3377 | 0.2487 | No-learning control |
| Monthly Ridge | 0.2781 | 0.2135 | Selected primary model |
| Seven-month contextual CNN | 0.2803 | 0.2067 | Five-seed neural candidate |
| Bias-corrected Ridge | 0.2760 | 0.2104 | Past-error calibration |
| Decomposed Ridge | 0.2913 | 0.2032 | Decomposed linear control |
| Online ensemble | 0.2769 | 0.2057 | Ensemble of selected experts |

The CNN's retrospective MAE is **16.9%** below last month and **3.2%** below monthly Ridge. Its five-seed validation MAE is 0.2803 versus Ridge's 0.2781. It does not pass the replacement gate, so Ridge remains the primary model. The CNN's paired MAE gain over Ridge has a descriptive 95% three-month block interval of **[-0.0049, 0.0206] pp**, crossing zero.

![Monthly errors, paired gain intervals and weights before prediction](docs/figures/en/05_performance.png)

The researcher had already inspected this historical evaluation period before proposing the new methods; these comparisons are retrospective. Decomposed Ridge has a lower later-period score but is not retrospectively promoted. Values are traceable to the [machine-readable outputs](research/results/research/).

## Data and forecast timing

The panel is reconstructed from [NBS monthly housing-price releases](https://www.stats.gov.cn/sj/zxfb/202602/t20260213_1962617.html). Every city-month retains its official URL. There are no missing values or duplicate city-months.

| Field | Definition |
| --- | --- |
| `period`, `city` | Statistical month and city |
| `new_mom_pct` | New-home index MoM movement: reported index minus 100 |
| `second_mom_pct` | Second-hand index MoM movement: reported index minus 100 |
| `source_url` | Official release page for that statistical month |

![Market means and city direction shares over 57 months](docs/figures/en/01_market.png)

Forecasts are issued after the previous statistical month's release, generally after the target month has started. January 2026 indices were released on **13 February 2026**; February's full-month outcome is forecast after that date. January data are not assumed available on 1 February. February 2026 actuals are excluded from fitting, selection and scoring.

## Temporal protocol

| Stage | Target months | Initial training-label cutoff | Origins / city rows |
| --- | --- | --- | ---: |
| Rolling validation and selection | Jan-Dec 2024 | Dec 2023; 25 months / 1,750 rows | 12 / 840 |
| Retrospective rolling evaluation | Jan 2025-Jan 2026 | Dec 2024; 37 months / 2,590 rows | 13 / 910 |
| As-of output | Feb 2026 | Jan 2026; 50 months / 3,500 rows | 1 / 70 |

At each origin, models, scalers, class weights and error states use released past information only. All cities advance together. Realized labels may affect the next origin after release. Additional quarterly checks select next-quarter configurations and experts using earlier quarters only.

![Temporal intervals and causal update order](docs/figures/en/02_protocol.png)

## Models and implemented work

The contextual temporal CNN reads a seven-month new/second-hand sequence, the same 19 numeric features as Ridge, and a four-dimensional city embedding. Two convolution layers produce a 12-dimensional representation. A 35→24→1 head predicts a correction added to the last known new-home movement. There are **1,697 trainable parameters**.

![Neural architecture and separate ablation modules](docs/figures/en/03_architecture.png)

| Experiment group | Implementation | Single-model configurations |
| --- | --- | ---: |
| Baselines and input controls | Three Ridge penalties, two tree sizes, NLinear-style model, six-month prototype and seven-month CNN | 8 |
| Reversible window normalization | Known-window statistics; restore the target-channel scale | 1 |
| Past-error calibration | EWMA of Ridge out-of-sample signed errors; two decay factors × two correction strengths | 4 |
| Market/city decomposition | Equal-weight 70-city mean plus centered city residual; linear and neural controls | 3 |
| Auxiliary direction learning | Rise/flat/fall cross-entropy; two loss weights × class weighting on/off | 4 |

Two ensembles use uniform or past-error online weights. All 20 single models receive 2024 validation. Eight locked representatives, both ensembles and the no-learning baseline receive later-period evaluation. Training uses 60 epochs per origin, AdamW learning rate 0.006, weight decay 0.02 and Huber beta 0.25. Screening seeds are 13/31/47; confirmation adds 61/79.

![All 20 validation configurations and monthly gains](docs/figures/en/04_validation.png)

The primary-model gate requires at least 3% lower 2024 MAE, improvement in at least eight months, and no half-year deterioration above 5%. No additional single model passes. The best bias correction lowers validation MAE by 0.75%. Normalization, decomposition and multitask outcomes, including non-improvements and individual seed scores, are retained.

## Shandong cases and direction diagnostics

![Observed and predicted movements in Jinan, Qingdao, Yantai and Jining](docs/figures/en/07_shandong.png)

The retrospective panel contains 705 falling, 160 rising and 45 flat records. Rising-case MAE is 0.2888 for last month, 0.3491 for Ridge and 0.3247 for the CNN. The CNN improves on Ridge but remains behind last month on rising cases. Learned models also underperform last month in Jining.

[Direction confusion and period comparisons](docs/figures/en/06_direction.png) · [Quarterly and rebased-month sensitivity](docs/figures/en/08_robustness.png) · [All city and segment results](docs/results.en.md)

## Reproduction

Use Python 3.11 in an isolated environment; see the [runbook](docs/reproducibility.md). Cleaned data and final checkpoints are included. Saved-model inference requires neither redownloading data nor retraining.

```bash
git clone https://github.com/LRZer/Real-estate.git
cd Real-estate
python -m pip install -r requirements.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python research/predict_saved_models.py
python scripts/verify_repository.py
```

```bash
# Full experiments; February 2026 actuals remain excluded
python research/run_research_experiments.py
python research/analyze_research_results.py
python research/verify_research_protocol.py
# Rebuild repository figures from recorded outputs
python scripts/build_figures.py --language both
```

Checks cover dataset/checkpoint SHA256, reported MAE against saved predictions, 315 training-cutoff records and checkpoint inference. Eight model types passed future-data perturbation checks. Saved-model reproduction differs by less than 2e-7 pp. GitHub Actions runs data, metric, documentation-link and checkpoint checks.

## Repository map

| Path | Contents |
| --- | --- |
| [research/data/official_nbs_70city.csv](research/data/official_nbs_70city.csv) | 3,990 cleaned records |
| [research/source_index.csv](research/source_index.csv) | 57 original releases; [release calendar](research/results/research/release_calendar.csv) |
| [research/research_models.py](research/research_models.py) | Neural, Ridge, tree and decomposition implementations |
| [research/run_research_experiments.py](research/run_research_experiments.py) | Locked configurations, rolling fits, selection and ensembles |
| [research/results/research/](research/results/research/) | Configurations, predictions, metrics, seeds and audits; `models/` contains final checkpoints |
| [docs/](docs/) | Bilingual methods, dataset cards, results and PNG/SVG figures |
| [Complete Chinese report](research/最终实验报告.md) | [Eight-page Chinese PDF](research/output/pdf/70城新房指数预测_成果展示.pdf) |

`research/` preserves the completed snapshot and earlier frozen, graph-temporal and adaptive studies. Current conclusions are stated here and in `results/research/`. Environments, downloaded HTML pages and fit caches are excluded.

## References and scope

References include [RevIN, ICLR 2022](https://github.com/ts-kim/RevIN), [NLinear, AAAI 2023](https://github.com/cure-lab/LTSF-Linear) and [OneNet, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html). These are independent adaptations to this panel, not full replications of the papers or their benchmark scores.

The completed version was developed in September-October 2026, reconstructing the information set of 28 February 2026. Official pages were retrieved later; the absence of revisions cannot be established. January 2026 base/category-weight changes are reported separately. No company transaction, marketing or client data were used, so property-level valuation, sales uplift and deployment effects were not evaluated.
