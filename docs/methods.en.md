# Models and temporal evaluation

[中文](methods.zh.md) · [English](methods.en.md) · [Complete results](results.en.md)

## Target and information set

For city c and target month m, the input month is t=m-1. The target is $y_{c,m}=I^{new}_{c,m}-100$, the next statistical month's new-home index movement. Inputs end at released month t. Forecasting starts after that month's publication, generally partway through m.

The first complete input window is May-November 2021, with target December 2021. There are 50 known target months. February 2026 has predictions but no actual label in the project.

## Features and information controls

| Numeric inputs | Count | Definition |
| --- | ---: | --- |
| Current values | 2 | City new/second-hand movements in t |
| Panel context | 3 | Equal-weight 70-city means of both series; new-home rising share |
| Calendar | 2 | Sine/cosine of the input month |
| Lags | 8 | Both channels at lags 1, 2, 3 and 6 |
| Rolling means | 4 | Both channels over 3 and 6 months, including t |

Ridge receives these 19 features plus a 70-dimensional city one-hot code. Neural models receive the same numeric features, a four-dimensional city embedding, and the full seven-month two-channel sequence t-6…t. The network therefore covers Ridge's lag-6 and explicit context, but representations differ and the CNN additionally reads every raw value in the window.

The six-month prototype only covers t-5…t. The new version changes both the window and context head. Its three-seed validation improvement from 0.2951 to 0.2791 pp is a configuration comparison, not a causal attribution to one additional month.

### Feature columns and sample alignment

Columns follow the order in [the feature builder](../research/train_forecast.py) and are recorded in [checkpoint metadata](../research/results/research/model_metadata.json):

| Group | Actual column names, in group order |
| --- | --- |
| Current values | `new_mom_pct`, `second_mom_pct` |
| Panel context | `national_new_mean`, `national_second_mean`, `share_cities_rising` |
| Calendar | `calendar_sin`, `calendar_cos` |
| New-home lags | `new_lag1`, `new_lag2`, `new_lag3`, `new_lag6` |
| Second-hand lags | `second_lag1`, `second_lag2`, `second_lag3`, `second_lag6` |
| Means | `new_mean3`, `new_mean6`, `second_mean3`, `second_mean6` |

Rolling means include t. Calendar codes are sin(2πq/12) and cos(2πq/12), where q is the input month number. The label and its source URL shift to the next month within each city's series.

| Sample role | Known input window | Label / forecast month |
| --- | --- | --- |
| First supervised sample | May–Nov 2021 | Dec 2021 |
| Last training sample for Jan 2025 | May–Nov 2024 | Dec 2024, released |
| Jan 2025 retrospective forecast | Jun–Dec 2024 | Jan 2025, unread at prediction |
| Final as-of forecast | Jul 2025–Jan 2026 | Feb 2026, no actual label |

Each origin uses all supervised samples with target months earlier than the forecast month. From December 2021 onward, this gives 25 training months for January 2024, 37 for January 2025, 49 for January 2026 and 50 for February 2026; each month has 70 rows. Scalers and models are refitted at every origin. Neural models are reinitialized rather than inheriting previous-origin weights.

### Primary model and non-neural controls

Ridge applies `StandardScaler` to the 19 numeric columns using training rows, then appends the 70-dimensional city one-hot code. With 89 columns and an intercept, it minimizes:

$$
\min_{w,b}\sum_{(c,j)\in \mathcal T_m}
(y_{c,j}-b-z_{c,j}^{\mathsf T}w)^2+\alpha\lVert w\rVert_2^2,
\qquad \mathcal T_m=\{(c,j):c=1,\ldots,70,\ j\in\mathcal M_m\}.
$$

Here $\mathcal M_m$ contains known label months from December 2021 onward, earlier than m. Penalties 5, 25 and 100 are screened; the primary uses 25. City one-hot coefficients are also regularized. Last month outputs $\hat y_{c,m}=y_{c,m-1}$ with no fitted parameters.

The tree control uses 19 unscaled numeric columns plus city one-hot codes: `HistGradientBoostingRegressor`, 120 iterations, learning rate 0.05, maximum leaves 7 or 15, minimum leaf samples 30, L2=0.5, early stopping disabled and random_state=42.

## Contextual temporal CNN

![Architecture](figures/en/03_architecture.png)

| Component | Operation | Shape or count |
| --- | --- | --- |
| Sequence scaling | Channel statistics from past training windows only | B×70×7×2 |
| First convolution | Conv1d 2→12, kernel 3, padding 1, ReLU | 12×7 per city |
| Second convolution | Conv1d 12→12, kernel 3, padding 1, ReLU | 12×7 per city |
| Temporal pooling | Mean across time | 12 features per city |
| Context | Concatenate 19 scaled features and 4-d city embedding | 35 features |
| Head | Linear 35→24, ReLU, Dropout 0.15, Linear 24→1 | Correction δ |
| Output | $\hat y_{c,m}=y_{c,t}+\delta_{c,m}$ | One movement per city |

Parameters: embedding 280, convolutions 84+444, head 864+25, totaling 1,697. The output layer is zero-initialized. This network does not use dilated convolutions or a Transformer.

B counts target months. Inputs are reshaped to (B×70)×2×7 for shared per-city convolutions, with symmetric padding within the known window. Pooling restores B×70×12. Known panel means/rising shares provide cross-city context; the current convolutional architecture has no city-graph message passing.

Channel means and standard deviations are computed across training months × cities × sequence positions, with a standard deviation floor of 0.05. Overlapping windows contribute according to their actual occurrences. The 19 numeric columns separately use `StandardScaler` on training rows. Forecast windows only apply those fitted statistics. Inference loads `preprocessing.npz` together with city/feature ordering metadata.

Neural fits use AdamW learning rate 0.006, weight decay 0.02, gradient clipping 1, and 60 fixed epochs per origin. Each epoch makes one full-batch update using all past training months × 70 cities. Target-month outcomes do not control early stopping. Screening uses seeds 13/31/47; final CNN and direction representatives add 61/79. Inference disables Dropout and averages predictions. Seed MAE dispersion describes repeated fits, not a temporal sampling interval.

Core regression uses PyTorch `smooth_l1_loss` with β=0.25. For error e=prediction−actual, the loss and reduction are:

$$
\ell_\beta(e)=
\begin{cases}
e^2/(2\beta), & |e|<\beta,\\
|e|-\beta/2, & |e|\ge\beta,
\end{cases}
\qquad
L_{\mathrm{city}}=\frac1{70|\mathcal M_m|}
\sum_{j\in\mathcal M_m}\sum_{c=1}^{70}\ell_\beta(\hat y_{c,j}-y_{c,j}).
$$

Errors, β and corrections use the monthly movement's percentage-point scale. Zero output initialization starts the correction at zero; the loss is computed on the final prediction.

## Separate ablations

### Reversible window normalization

Tests whether differences in city-window levels and fluctuation scales affect fitting.

Following the idea in [RevIN's author implementation](https://github.com/ts-kim/RevIN), compute each city's channel means μ and standard deviations σ from its known seven-month window, with affine parameters disabled and a standard deviation floor of 0.05. Encode $(x-\mu)/\sigma$; scale the predicted correction by the new-home σ before adding the original-scale last value. There are two input channels but one target channel. The same 19 context features remain available.

### Past-error calibration

Use monthly Ridge alpha 25. Forecast month m with the existing past state:

$$
\hat y^{corr}_{c,m}=\hat y^{\mathrm{Ridge}}_{c,m}-\lambda b_{m-1}.
$$

After month m's actual release, compute $e_m=\frac1{70}\sum_c(\hat y^{\mathrm{Ridge}}_{c,m}-y_{c,m})$ from the **uncorrected Ridge forecast**, then update $b_m=\rho b_{m-1}+(1-\rho)e_m$ for month m+1.

rho is 0.5/0.8; lambda is 0.5/1. The state starts at zero in January 2024, updates after predicting, and carries through 2025 and the final forecast. It uses past rolling out-of-sample errors, not training residuals.

### Market/city decomposition

Tests separate learning of common market movements and city deviations.

For training targets, define $g_m=\frac1{70}\sum_c y_{c,m}$ and $r_{c,m}=y_{c,m}-g_m$. Forecast $\hat y_{c,m}=\hat g_m+\hat r_{c,m}$ and center predicted city residuals to zero. At inference, use predicted g, never the target month's observed mean.

The market branch receives 17 inputs: 14 values from seven months of both panel means, one rising share and two calendar codes. Its neural head is 17→12→1. Loss is final-city Huber plus 0.25×market-mean Huber. Both decomposed Ridge branches fix alpha 25; the market branch has one target per month rather than 70 copies.

The market mean is a descriptive panel statistic, not an NBS national index.

### Auxiliary direction learning

Tests how an auxiliary classification constraint and class weighting affect rise detection and magnitude error.

Add a three-class fall/flat/rise head to the unnormalized two-branch CNN's shared 24-dimensional representation. Training labels use actual movements below, equal to or above zero. Add eta×cross-entropy, with eta 0.01/0.05. Class weighting is disabled or uses inverse square-root training frequencies, first capped at 3 and then normalized to mean 1 over training labels.

The classifier does not overwrite regression. Three-way regression signs use a fixed ±0.05 flat band. Binary rise is >0; balanced binary scores exclude actual flat cases. Classifier and regression directions are scored separately, including their disagreement rate.

### Linear control and ensembles

The [NLinear-style control](https://github.com/cure-lab/LTSF-Linear) subtracts each standardized channel's last value, flattens 14 sequence values, concatenates 19 features and the 4-d embedding, and uses one linear correction head. Add the correction to the original-scale last new-home value. There are 318 parameters; this is a contextual short-sequence adaptation.

The online ensemble is an independent EWMA+softmax simplification inspired by [OneNet](https://proceedings.neurips.cc/paper_files/paper/2023/hash/dd6a47bc0aad6f34aa5e77706d90cdc4-Abstract-Conference.html), without its reinforcement-learning mechanism. Experts are the best validation simple baseline, the nonlinear CNN selected on three seeds, and last month. Risk updates as $R_m=0.5R_{m-1}+0.5MAE_m$; weights before prediction are $w_m=softmax(-R_{m-1}/0.02)$. A uniform control is also evaluated.

## Selection and evaluation

![Temporal protocol](figures/en/02_protocol.png)

The [locked registry](../research/results/research/locked_experiment_registry.json) contains 20 single-model configurations and 2 ensembles. No configurations are added based on later-period outcomes. All single models receive 2024 rolling validation; [representative selection](../research/results/research/selection_before_retrospective.json) is saved before later scoring.

Primary replacement requires ≥3% lower 2024 MAE, improvement in at least 8 months and ≤5% deterioration in either half-year. The direction gate allows ≤2% overall MAE deterioration versus the unnormalized two-branch control, requires ≥0.05 higher balanced binary accuracy and ≥10% lower rising MAE. These are engineering gates, not definitions of statistical significance.

Five-seed confirmation does not enter earlier quarterly selection. Original three-seed records select the next quarter from preceding 3/6/9 months. Full-year ensemble validation uses experts selected from the full year and has additional selection uncertainty.

Paired intervals preserve all 70 cities within each month and use 20,000 circular three-month block resamples. They do not correct researcher exposure to the later period, configuration selection or possible official revisions. NBS base/category-weight changes in January 2026 motivate separate 2025-only and January 2026 results.

### Metric definitions

For forecast-month set $\mathcal E$, MAE is $\frac1{70|\mathcal E|}\sum_{m\in\mathcal E}\sum_c|\hat y_{c,m}-y_{c,m}|$. Since every month has 70 cities, this equals the equal-weight mean of monthly MAE. RMSE is the square root of mean squared error; signed bias is mean prediction−actual. Rising/falling MAE uses observations with actual >0/<0 respectively.

Binary balanced accuracy averages rising recall and the fraction of falling observations predicted as non-rising; actual zeros are excluded. Magnitude, classifier-head and regression-sign results are scored separately. Aggregate MAE, group errors, monthly stability and paired intervals are reported together; primary selection uses 2024 records only.

## Implementation

| File | Contents |
| --- | --- |
| [Models](../research/research_models.py) | Past ranges, scaling, convolutions, normalization, branches and losses |
| [Experiment runner](../research/run_research_experiments.py) | Configurations, monthly fits, seeds, selection, ensembles and final output |
| [Protocol verifier](../research/verify_research_protocol.py) | Future perturbations, causal states and output integrity |
| [Saved-model inference](../research/predict_saved_models.py) | Reproduce February 2026 without retraining |

The original Chinese report and earlier studies remain in `research/`. These methods are independent adaptations, not full paper replications.
