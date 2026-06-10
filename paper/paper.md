# Stability-Aware Hybrid AI Framework for Supply Chain Demand Forecasting

## Abstract

Demand forecasts in supply chains are acted on repeatedly: they shape replenishment, inventory positioning, production schedules, labor plans, and transportation capacity. In such settings, a forecast can be accurate on average and still be operationally disruptive if it oscillates from day to day. We study a stability-aware hybrid forecasting framework that combines TimeMixer-style temporal embeddings, structured calendar and price features, and an XGBoost forecasting layer. The training objective augments forecast loss with a penalty on consecutive within-series prediction changes. On selected M5 demand series at 1000, 3000, and 4000-series scales, the final hybrid stability-aware model improves Forecast Stability Score over XGBoost by 6.91%, 6.66%, and 7.68%, respectively, while RMSE changes remain within 0.72% across three random seeds. Post-hoc smoothing reduces raw forecast movement more aggressively, but with a larger RMSE cost; the proposed objective occupies a more accuracy-preserving region and performs better under normalized stability.

## 1. Introduction

Demand forecasts are no longer passive reports in many supply chains. They feed replenishment systems, inventory policies, production schedules, labor rosters, and transportation plans. A model that changes its forecast sharply from one planning cycle to the next can therefore create operational churn even when its aggregate error is acceptable. For intermittent retail demand, where zeros, local spikes, promotions, and price changes are common, this distinction between statistical accuracy and operational stability becomes difficult to ignore.

Recent deep forecasting models have improved long-horizon sequence modeling through Transformer, MLP, and multiscale temporal architectures [1]-[5]. Retail demand forecasting, however, is not only a sequence modeling problem. Calendar effects, sell prices, events, product hierarchy, and store identity often carry information that is awkward to represent with temporal history alone. The M5 benchmark makes this tension explicit by combining item-store demand histories with calendar and price tables [7]. Tree-based learners such as XGBoost remain strong in this setting because they handle heterogeneous tabular covariates well [8], but they rely heavily on engineered lag and rolling features.

This paper examines a hybrid alternative. A lightweight TimeMixer-style encoder summarizes recent demand history, structured operational features provide calendar and price context, and an XGBoost layer produces point forecasts. We then add a stability-aware training term that penalizes excessive consecutive forecast movement within each item-store series. The purpose is not to maximize smoothness at all costs. Rather, we ask whether a model can reduce forecast volatility while staying close to the accuracy profile of a strong structured baseline.

## 2. Related Work

### 2.1 Deep Time Series Forecasting

Recent deep forecasting research has developed specialized architectures for long-horizon temporal representation learning. TimeMixer models temporal variation through decomposable multiscale mixing, allowing information from different temporal resolutions to interact within a unified forecasting architecture [1]. PatchTST introduced a patch-based Transformer formulation in which subseries patches serve as temporal tokens, improving long-context modeling while reducing sequence length [2]. Informer addressed the computational burden of long-sequence forecasting through ProbSparse attention and a generative decoder [3], while Autoformer incorporated decomposition and auto-correlation mechanisms for long-term forecasting [4]. TSMixer showed that all-MLP time and feature mixing can be competitive with attention-based models [5].

These architectures have advanced the accuracy frontier for temporal forecasting, but their standard evaluation protocols are still dominated by point-error metrics. In retail and supply-chain settings, this can leave a gap between benchmark performance and operational usefulness. A forecast that is accurate on average but unstable across consecutive periods may still create planning churn. Moreover, temporal-only models may underuse structured operational signals such as price, calendar events, product hierarchy, and store identity.

### 2.2 Supply Chain and Retail Demand Forecasting

Retail demand forecasting is a high-dimensional operational problem involving sparse SKU-store series, heterogeneous seasonality, pricing variation, promotional events, and hierarchy. Forecasting competitions have shaped empirical forecasting practice for decades [6]; the M5 Accuracy Competition extended that tradition to large-scale retail demand using Walmart sales, calendar, and sell-price data [7]. Retail forecasting practice requires methods that incorporate product, store, calendar, price, promotion, and replenishment-relevant signals at scale [10]. Work at the forecasting--operations interface also stresses that forecasting models should be evaluated in the decision contexts where they are deployed [9], [11].

Tree-based machine learning remains competitive in such settings because structured demand drivers are often nonlinear and heterogeneous. XGBoost provides scalable gradient-boosted trees with efficient handling of sparse features and nonlinear interactions [8]. However, structured learners typically depend on manually specified lag and rolling-window features. This motivates hybrid models that retain the tabular strength of boosted trees while adding learned representations of recent demand history.

### 2.3 Hybrid Forecasting Frameworks

Hybrid forecasting methods combine temporal representation learning with structured covariate modeling. Temporal encoders such as TimeMixer can learn multiscale demand patterns [1], while structured models such as XGBoost exploit price, calendar, event, and categorical covariates [8]. Evidence from TSMixer and M5 further suggests that auxiliary and cross-variate information is important in realistic retail benchmarks [5], [7].

The present framework follows this hybrid direction but emphasizes operational stability in addition to accuracy. Rather than treating temporal encoders and structured learners as competing alternatives, it uses temporal embeddings as learned demand-history summaries and combines them with structured operational features before tree-based prediction.

### 2.4 Forecast Stability and Robust Forecasting

Supply chain planning is sensitive not only to forecast error but also to forecast instability. The operational cost of forecast volatility is related to the bullwhip effect, where distorted or amplified demand information propagates upstream and increases order variability [15]. Forecasting and lead times contribute directly to bullwhip behavior [16], and forecasting method choice can affect demand amplification under replenishment policies [17]. Intermittent demand has long been treated as an inventory-relevant forecasting problem, from Croston-style stock-control methods [12] to later work on intermittent-demand accuracy and stock-control performance [13], [14]. Recent work formalizes forecast stability and the accuracy-stability trade-off [20]. Robust forecasting emphasizes reliability under noise, outliers, and structural change [18], while multi-step evaluation work shows that aggregate error metrics can hide forecast update behavior [21].

To our knowledge, limited prior forecasting work has jointly optimized operational forecast stability and hybrid temporal-structured forecasting for large-scale retail demand. We therefore position this work as a cautious empirical step toward stability-aware hybrid demand forecasting, rather than as a claim that stability regularization is universally superior across all forecasting settings.

## 3. Methodology

### 3.1 Problem Formulation

Let $y_{i,t}$ denote observed demand for series $i$ at time $t$, where each series corresponds to an item-store demand stream. Given a historical window $\mathbf{y}_{i,t-L:t-1}$ and structured covariates $\mathbf{x}_{i,t}$, the forecasting task is to estimate demand $\hat{y}_{i,t}$ over a validation horizon.

### 3.2 Stability-Aware Objective

```math
\mathcal{L}_{\mathrm{forecast}} = \frac{1}{N}\sum_{i,t}(y_{i,t} - \hat{y}_{i,t})^2
```

```math
\mathcal{L}_{\mathrm{stability}} = \frac{1}{N_s}\sum_i\sum_{t>1} |\hat{y}_{i,t} - \hat{y}_{i,t-1}|
```

```math
\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{forecast}} + \lambda \mathcal{L}_{\mathrm{stability}}.
```

### 3.3 Implementation of the Stability-Aware XGBoost Objective

Predictions are grouped by item-store series, and adjacent forecast differences are computed only within the same series. The stability loss is computed only on training-window predictions; validation and test targets are never used. The goal is controlled within-series regularization rather than global smoothing across unrelated products or stores. The leakage audit confirms chronological train-validation separation and same-series training-only stability pairs.

### 3.4 Normalized Stability

```math
\mathrm{nFSS}_i = \frac{\frac{1}{T_i-1}\sum_{t>1} |\hat{y}_{i,t}-\hat{y}_{i,t-1}|}{\frac{1}{T_i}\sum_t |y_{i,t}| + \epsilon}, \quad \epsilon=10^{-6}
```

```math
\mathrm{nFSS} = \frac{1}{M}\sum_{i=1}^M \mathrm{nFSS}_i.
```

## 4. Experiments

### 4.1 Dataset and Setup

We use selected subsets of 1000, 3000, and 4000 M5 demand series. Each scale is run with full selected training rows, 300 XGBoost trees, identical feature engineering, the same TimeMixer-style encoder settings, and $\lambda=0.05$. The split is chronological, and repeated experiments use seeds 42, 123, and 2024.

Raw MAPE is not used because M5 contains many zero and near-zero demand observations. We report WAPE instead. Because many M5 series are intermittent and low-volume, nFSS can take large absolute values when the average realized demand in the denominator is small. We therefore interpret nFSS as a relative robustness metric rather than as an absolute operational threshold.

### 4.2 Main Results

| Series | Method | Runs | MAE | RMSE | WAPE (%) | FSS | Volatility |
|---|---|---:|---|---|---|---|---|
| 1000 | XGBoost | 3 | 0.926 ± 0.034 | 1.774 ± 0.185 | 70.02 ± 4.82 | 0.220 ± 0.028 | 0.313 ± 0.043 |
| 1000 | TimeMixer | 3 | 0.970 ± 0.044 | 1.930 ± 0.316 | 73.28 ± 4.30 | 0.347 ± 0.076 | 0.391 ± 0.073 |
| 1000 | Hybrid | 3 | 0.926 ± 0.034 | 1.781 ± 0.198 | 70.02 ± 4.76 | 0.222 ± 0.030 | 0.315 ± 0.044 |
| 1000 | Hybrid + Stable | 3 | 0.927 ± 0.035 | 1.787 ± 0.208 | 70.08 ± 4.71 | 0.205 ± 0.030 | 0.302 ± 0.042 |
| 3000 | XGBoost | 3 | 0.953 ± 0.006 | 1.852 ± 0.081 | 69.81 ± 1.47 | 0.228 ± 0.007 | 0.326 ± 0.012 |
| 3000 | TimeMixer | 3 | 0.993 ± 0.012 | 2.002 ± 0.123 | 72.79 ± 1.37 | 0.401 ± 0.040 | 0.439 ± 0.027 |
| 3000 | Hybrid | 3 | 0.953 ± 0.007 | 1.850 ± 0.078 | 69.84 ± 1.43 | 0.229 ± 0.008 | 0.328 ± 0.013 |
| 3000 | Hybrid + Stable | 3 | 0.954 ± 0.007 | 1.854 ± 0.082 | 69.91 ± 1.45 | 0.213 ± 0.009 | 0.315 ± 0.012 |
| 4000 | XGBoost | 3 | 0.955 ± 0.013 | 1.876 ± 0.012 | 69.53 ± 1.00 | 0.232 ± 0.007 | 0.331 ± 0.006 |
| 4000 | TimeMixer | 3 | 0.998 ± 0.020 | 2.014 ± 0.037 | 72.65 ± 1.34 | 0.350 ± 0.027 | 0.399 ± 0.031 |
| 4000 | Hybrid | 3 | 0.956 ± 0.012 | 1.882 ± 0.012 | 69.56 ± 0.94 | 0.231 ± 0.005 | 0.330 ± 0.005 |
| 4000 | Hybrid + Stable | 3 | 0.956 ± 0.012 | 1.881 ± 0.014 | 69.58 ± 0.96 | 0.214 ± 0.006 | 0.318 ± 0.007 |

![Main results](updated_scalability_figures/main_results_rmse_fss.png)

### 4.3 Stability-Accuracy Trade-off

The stability-aware hybrid model improves FSS over XGBoost by 6.91%, 6.66%, and 7.68% at 1000, 3000, and 4000 series. Across these scales, the largest absolute RMSE change is 0.72%.

| Series | FSS Improvement (%) | RMSE Change (%) | Volatility Improvement (%) | Seeds Improved | Runs |
|---:|---:|---:|---:|---:|---:|
| 1000 | 6.91 | 0.72 | 3.77 | 3 | 3 |
| 3000 | 6.66 | 0.10 | 3.18 | 3 | 3 |
| 4000 | 7.68 | 0.24 | 3.85 | 3 | 3 |

![Accuracy-stability tradeoff](updated_scalability_figures/accuracy_stability_tradeoff.png)

### 4.4 Smoothing Baselines and Normalized Stability

Post-hoc smoothing with alpha=0.5 obtains the lowest raw FSS, but it pays a larger RMSE cost. The stability-aware objective does not minimize raw FSS; it occupies a more accuracy-preserving region and performs better under nFSS.

| Series | Method | alpha | RMSE | WAPE (%) | FSS | nFSS | RMSE Δ (%) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | Hybrid + Stable | -- | 1.787 ± 0.208 | 70.08 ± 4.71 | 0.205 ± 0.030 | 388.7 ± 113.8 | 0.72 |
| 1000 | XGBoost + Smooth | 0.5 | 1.812 ± 0.217 | 70.54 ± 4.56 | 0.116 ± 0.015 | 449.0 ± 132.9 | 2.15 |
| 1000 | Hybrid + Smooth | 0.5 | 1.815 ± 0.223 | 70.53 ± 4.55 | 0.116 ± 0.015 | 423.4 ± 94.7 | 2.33 |
| 3000 | Hybrid + Stable | -- | 1.854 ± 0.082 | 69.91 ± 1.46 | 0.213 ± 0.008 | 373.5 ± 56.1 | 0.10 |
| 3000 | XGBoost + Smooth | 0.5 | 1.887 ± 0.090 | 70.41 ± 1.43 | 0.118 ± 0.004 | 452.1 ± 90.3 | 1.87 |
| 3000 | Hybrid + Smooth | 0.5 | 1.883 ± 0.087 | 70.41 ± 1.39 | 0.119 ± 0.004 | 437.3 ± 50.0 | 1.66 |
| 4000 | Hybrid + Stable | -- | 1.881 ± 0.014 | 69.58 ± 0.96 | 0.214 ± 0.006 | 377.5 ± 73.7 | 0.24 |
| 4000 | XGBoost + Smooth | 0.5 | 1.911 ± 0.016 | 70.12 ± 0.92 | 0.121 ± 0.004 | 416.9 ± 74.6 | 1.88 |
| 4000 | Hybrid + Smooth | 0.5 | 1.917 ± 0.021 | 70.15 ± 0.88 | 0.120 ± 0.003 | 423.3 ± 90.1 | 2.16 |

![Smoothing tradeoff](figures/smoothing_accuracy_stability_tradeoff.png)

![Normalized FSS robustness](figures/normalized_fss_comparison.png)

### 4.5 Runtime and Robustness

Relative to XGBoost, the final stability-aware hybrid method incurs runtime multipliers of 4.39x, 7.71x, and 9.72x at 1000, 3000, and 4000 series. The cost is material, but it is reported under consistent full training configurations.

![Runtime scalability](updated_scalability_figures/runtime_scalability.png)

![Forecast visualization](updated_scalability_figures/hybrid_forecast_visualization.png)

## 5. Discussion

These results do not imply that objective-based regularization dominates smoothing in every setting. Smoothing is more aggressive in reducing raw forecast movement, and it remains a practical baseline when raw smoothness is the primary target. The proposed objective gives a different operating point: it preserves more of the accuracy-oriented forecast structure and performs better under scale-normalized stability.

Intermittent retail demand contains zeros, sparse spikes, and short-lived local shocks. Penalizing within-series prediction changes may reduce reaction to some local noise while still allowing the model to use calendar, price, and temporal context. The runtime cost is real, so deployment should weigh stability gains against computational budget and planning cost.

## 6. Limitations

The study is limited to selected M5 retail demand series and does not evaluate downstream inventory outcomes. The largest full-configuration main experiment uses 4000 series; the capped 5000-series diagnostic is appendix-only. nFSS is useful for relative comparison but can be large for low-volume intermittent series. A stronger operational evaluation would connect forecast stability to service level, stockouts, holding cost, and schedule changes.

## 7. Conclusion

The final evaluation supports a cautious conclusion: stability-aware hybrid forecasting can improve forecast stability while preserving comparable predictive accuracy. Post-hoc smoothing is stronger for raw FSS, but objective-based regularization offers a more balanced and scale-robust operating point.

## Appendix

### Full Smoothing Alpha Ablation

| Series | Base | alpha | RMSE | WAPE (%) | FSS | nFSS | Volatility |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1000 | Hybrid | 0.5 | 1.815 ± 0.223 | 70.53 ± 4.55 | 0.116 ± 0.015 | 423.4 ± 94.7 | 0.234 ± 0.033 |
| 1000 | Hybrid | 0.7 | 1.784 ± 0.199 | 70.08 ± 4.71 | 0.158 ± 0.021 | 520.3 ± 116.5 | 0.270 ± 0.038 |
| 1000 | Hybrid | 0.9 | 1.777 ± 0.194 | 69.97 ± 4.77 | 0.199 ± 0.027 | 599.5 ± 137.1 | 0.301 ± 0.042 |
| 1000 | XGBoost | 0.5 | 1.812 ± 0.217 | 70.54 ± 4.56 | 0.116 ± 0.015 | 449.0 ± 132.9 | 0.233 ± 0.033 |
| 1000 | XGBoost | 0.7 | 1.780 ± 0.190 | 70.08 ± 4.75 | 0.157 ± 0.020 | 545.9 ± 159.8 | 0.268 ± 0.038 |
| 1000 | XGBoost | 0.9 | 1.771 ± 0.182 | 69.97 ± 4.82 | 0.198 ± 0.025 | 615.3 ± 178.5 | 0.299 ± 0.041 |
| 3000 | Hybrid | 0.5 | 1.883 ± 0.087 | 70.41 ± 1.39 | 0.119 ± 0.004 | 437.3 ± 50.0 | 0.246 ± 0.010 |
| 3000 | Hybrid | 0.7 | 1.858 ± 0.079 | 69.97 ± 1.43 | 0.162 ± 0.006 | 539.9 ± 62.4 | 0.282 ± 0.011 |
| 3000 | Hybrid | 0.9 | 1.849 ± 0.077 | 69.82 ± 1.44 | 0.206 ± 0.007 | 624.0 ± 74.4 | 0.313 ± 0.012 |
| 3000 | XGBoost | 0.5 | 1.887 ± 0.090 | 70.41 ± 1.43 | 0.118 ± 0.004 | 452.1 ± 90.3 | 0.244 ± 0.010 |
| 3000 | XGBoost | 0.7 | 1.862 ± 0.083 | 69.96 ± 1.46 | 0.161 ± 0.005 | 561.5 ± 109.0 | 0.280 ± 0.011 |
| 3000 | XGBoost | 0.9 | 1.852 ± 0.081 | 69.80 ± 1.47 | 0.204 ± 0.006 | 644.4 ± 114.9 | 0.311 ± 0.012 |
| 4000 | Hybrid | 0.5 | 1.917 ± 0.021 | 70.15 ± 0.88 | 0.120 ± 0.003 | 423.3 ± 90.1 | 0.248 ± 0.002 |
| 4000 | Hybrid | 0.7 | 1.891 ± 0.015 | 69.72 ± 0.93 | 0.164 ± 0.004 | 529.1 ± 115.5 | 0.284 ± 0.003 |
| 4000 | Hybrid | 0.9 | 1.881 ± 0.013 | 69.56 ± 0.96 | 0.207 ± 0.005 | 616.3 ± 130.4 | 0.315 ± 0.005 |
| 4000 | XGBoost | 0.5 | 1.911 ± 0.016 | 70.12 ± 0.92 | 0.121 ± 0.004 | 416.9 ± 74.6 | 0.248 ± 0.003 |
| 4000 | XGBoost | 0.7 | 1.885 ± 0.013 | 69.68 ± 0.98 | 0.164 ± 0.005 | 520.6 ± 96.5 | 0.284 ± 0.004 |
| 4000 | XGBoost | 0.9 | 1.875 ± 0.013 | 69.52 ± 1.00 | 0.208 ± 0.006 | 604.8 ± 112.2 | 0.316 ± 0.006 |

![Raw smoothing diagnostic](figures/smoothing_vs_stability_objective.png)

### Normalized FSS Table

| Series | Method | RMSE | FSS | nFSS | Volatility |
| --- | --- | --- | --- | --- | --- |
| 1000 | XGBoost | 1.774 ± 0.185 | 0.220 ± 0.028 | 650.2 ± 184.9 | 0.313 ± 0.043 |
| 1000 | TimeMixer | 1.930 ± 0.316 | 0.347 ± 0.076 | 1016.7 ± 331.5 | 0.391 ± 0.073 |
| 1000 | Hybrid | 1.781 ± 0.198 | 0.222 ± 0.030 | 645.8 ± 149.6 | 0.315 ± 0.044 |
| 1000 | Hybrid + Stable | 1.787 ± 0.208 | 0.205 ± 0.030 | 388.7 ± 113.8 | 0.302 ± 0.042 |
| 1000 | XGBoost + Smoothing | 1.812 ± 0.217 | 0.116 ± 0.015 | 449.0 ± 132.9 | 0.233 ± 0.033 |
| 1000 | Hybrid + Smoothing | 1.815 ± 0.223 | 0.116 ± 0.015 | 423.4 ± 94.7 | 0.234 ± 0.033 |
| 3000 | XGBoost | 1.852 ± 0.081 | 0.228 ± 0.007 | 687.4 ± 120.8 | 0.326 ± 0.012 |
| 3000 | TimeMixer | 2.002 ± 0.123 | 0.401 ± 0.040 | 931.9 ± 252.5 | 0.439 ± 0.027 |
| 3000 | Hybrid | 1.850 ± 0.078 | 0.229 ± 0.008 | 669.6 ± 86.7 | 0.328 ± 0.013 |
| 3000 | Hybrid + Stable | 1.854 ± 0.082 | 0.213 ± 0.008 | 373.5 ± 56.1 | 0.316 ± 0.012 |
| 3000 | XGBoost + Smoothing | 1.887 ± 0.090 | 0.118 ± 0.004 | 452.1 ± 90.3 | 0.244 ± 0.010 |
| 3000 | Hybrid + Smoothing | 1.883 ± 0.087 | 0.119 ± 0.004 | 437.3 ± 50.0 | 0.246 ± 0.010 |
| 4000 | XGBoost | 1.876 ± 0.012 | 0.232 ± 0.007 | 650.1 ± 121.5 | 0.331 ± 0.006 |
| 4000 | TimeMixer | 2.013 ± 0.039 | 0.350 ± 0.028 | 868.7 ± 194.8 | 0.399 ± 0.032 |
| 4000 | Hybrid | 1.881 ± 0.012 | 0.231 ± 0.005 | 664.6 ± 136.2 | 0.330 ± 0.005 |
| 4000 | Hybrid + Stable | 1.880 ± 0.015 | 0.214 ± 0.006 | 377.5 ± 73.7 | 0.318 ± 0.007 |
| 4000 | XGBoost + Smoothing | 1.911 ± 0.016 | 0.121 ± 0.004 | 416.9 ± 74.6 | 0.248 ± 0.003 |
| 4000 | Hybrid + Smoothing | 1.917 ± 0.021 | 0.120 ± 0.003 | 423.3 ± 90.1 | 0.248 ± 0.002 |

### Leakage Audit

| Check | Status | Evidence |
| --- | --- | --- |
| Time split | PASS | train max 2016-03-27; validation min 2016-03-28 |
| Validation overlap | PASS | no validation rows or validation dates in training |
| Lag features | PASS | lag features use grouped historical demand shifts |
| Rolling features | PASS | rolling means computed after shift(1) |
| Encoding/scaling | PASS | categorical encoders and TimeMixer scaler fit on training data only |
| Stability objective | PASS | same-series consecutive training predictions only |

### Capped 5000-Series Diagnostic

The capped 5000-series result is retained only as supplementary constrained-environment evidence and is not used for main scalability claims.

## References

[1] S. Wang, H. Wu, X. Shi, T. Hu, H. Luo, L. Ma, J. Zhang, and J. Zhou, "TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting," ICLR, 2024.
[2] Y. Nie, N. H. Nguyen, P. Sinthong, and J. Kalagnanam, "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers," ICLR, 2023.
[3] H. Zhou et al., "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting," AAAI, 2021.
[4] H. Wu, J. Xu, J. Wang, and M. Long, "Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting," NeurIPS, 2021.
[5] S.-A. Chen, C.-L. Li, N. Yoder, S. O. Arik, and T. Pfister, "TSMixer: An All-MLP Architecture for Time Series Forecasting," TMLR, 2023.
[6] S. Makridakis and M. Hibon, "The M3-Competition: Results, Conclusions and Implications," International Journal of Forecasting, 2000.
[7] S. Makridakis, E. Spiliotis, and V. Assimakopoulos, "M5 Accuracy Competition: Results, Findings, and Conclusions," International Journal of Forecasting, 2022.
[8] T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD, 2016.
[9] R. Fildes, K. Nikolopoulos, S. F. Crone, and A. A. Syntetos, "Forecasting and Operational Research: A Review," Journal of the Operational Research Society, 2008.
[10] R. D. Fildes, S. Ma, and S. Kolassa, "Retail Forecasting: Research and Practice," International Journal of Forecasting, 2022.
[11] F. Petropoulos et al., "Forecasting: Theory and Practice," International Journal of Forecasting, 2022.
[12] J. D. Croston, "Forecasting and Stock Control for Intermittent Demands," Operational Research Quarterly, 1972.
[13] A. A. Syntetos and J. E. Boylan, "The Accuracy of Intermittent Demand Estimates," International Journal of Forecasting, 2005.
[14] A. A. Syntetos and J. E. Boylan, "On the Stock Control Performance of Intermittent Demand Estimators," International Journal of Production Economics, 2006.
[15] H. L. Lee, V. Padmanabhan, and S. Whang, "Information Distortion in a Supply Chain: The Bullwhip Effect," Management Science, 1997.
[16] F. Chen, Z. Drezner, J. K. Ryan, and D. Simchi-Levi, "Quantifying the Bullwhip Effect in a Simple Supply Chain," Management Science, 2000.
[17] X. Zhang, "The Impact of Forecasting Methods on the Bullwhip Effect," International Journal of Production Economics, 2004.
[18] D. Barrow, N. Kourentzes, R. Sandberg, and J. Niklewski, "Automatic Robust Estimation for Exponential Smoothing," Expert Systems with Applications, 2020.
[19] R. J. Hyndman and A. B. Koehler, "Another Look at Measures of Forecast Accuracy," International Journal of Forecasting, 2006.
[20] R. Godahewa et al., "On Forecast Stability," International Journal of Forecasting, 2025.
[21] E. Strom and O. E. Gundersen, "Performance Metrics for Multi-Step Forecasting Measuring Win-Loss, Seasonal Variance and Forecast Stability," Applied Intelligence, 2024.
