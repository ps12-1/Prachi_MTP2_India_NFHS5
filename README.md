# India NFHS-5 — ECD Proxy Analysis (4-Domain Pipeline)
## Modelling Early Childhood Development using the UNICEF ECDI2030 Framework

**Thesis**: MTP2 — "Modelling Early Childhood Development using the UNICEF ECDI2030 Framework"  
**Institute**: Indian Institute of Technology Kharagpur  
**Data**: India DHS-7 / NFHS-5 (2019–21), Kids Recode (`IAKR7EFL.DTA`)  
**Sample**: 133,750 alive children aged 24–59 months across 36 states and Union Territories

---

## Framework: 4-Domain Proxy Composite

NFHS-5 does not contain individual-level measurements for ECDI2030 Literacy-Numeracy or
Socio-emotional domains (EC6–EC8, EC13–EC15). This pipeline constructs a 4-domain composite
by merging external survey data into each NFHS-5 child record on `v024` (state of residence),
following the standard small-area-estimation approach (Pfeffermann 2013).

| ECDI2030 Domain | Proxy | Source |
|---|---|---|
| Physical | Not stunted (HAZ > −2 SD) | NFHS-5 child (`hw70`) |
| Learning | Regular anganwadi / ECCE attendance | NFHS-5 child (`s562`) |
| Literacy-Numeracy | State ASER letter+number recognition ≥ 50% OR mother has secondary+ education | ASER 2019 + NFHS-5 |
| Socio-emotional | State women's empowerment composite ≥ median AND household not in lowest wealth quintile | NFHS-5 Factsheet + KR |

**Composite rule (UNICEF ECDI2030)**: child is "on track" if ≥ 3 of 4 domains pass.

**External survey merges (real data, not hardcoded)**:
- ASER 2019 Early Years — parsed from the public PDF (`external_data/parse_aser_2019.py`)
- NFHS-5 State Factsheets (Phase I + II) — via [jvargh7/nfhs5_factsheets](https://github.com/jvargh7/nfhs5_factsheets)

---

## Results Summary

### Domain On-Track Rates

| Domain | On-Track Rate | n |
|---|---|---|
| Physical (not stunted) | 62.2% | 126,250 |
| Learning (regular anganwadi) | 40.7% | 91,719 |
| Literacy-Numeracy (ASER 2019 + mother edu) | 63.6% | 86,999 |
| Socio-emotional (factsheet + wealth) | 41.2% | 86,999 |
| **4-domain composite (≥ 3 / 4)** | **44.6%** unweighted · **44.4%** survey-weighted | 86,999 |

---

### Supervised ML — ECD Proxy Composite Prediction

Models: Logistic Regression (baseline) and HistGradientBoostingClassifier, evaluated with
5-fold stratified cross-validation. SHAP values computed via `TreeExplainer` on the full
dataset (n = 86,999).

| Model | ROC-AUC (5-fold CV) | F1 |
|---|---|---|
| Logistic Regression | 0.762 ± 0.004 | — |
| HistGradientBoosting | 0.790 ± 0.003 | — |

**Top 5 SHAP predictors** (mean |SHAP value|):

| Rank | Feature | Mean |SHAP| |
|---|---|---|
| 1 | Child age (months) | 0.140 |
| 2 | State: Nagaland | 0.127 |
| 3 | State: Uttar Pradesh | 0.114 |
| 4 | Mother's education (ordinal) | 0.085 |
| 5 | Wealth quintile (1–5) | 0.076 |

---

### Unsupervised ML — Developmental Profile Clustering

K-Means on HAZ, WAZ, and anganwadi attendance (standardised), silhouette criterion on the
full dataset (n = 86,834). Optimal k = 2.

| Cluster | Label | n | Physical on-track | Learning on-track |
|---|---|---|---|---|
| 0 | No ECCE attendance | 51,470 | ~67% | ~0% |
| 1 | Regular ECCE attendance | 35,364 | ~55% | ~100% |

Physical stunting prevalence is statistically indistinguishable across clusters; dual
deprivation (stunted + no ECCE) is the highest-risk profile within Cluster 0.

---

### Geospatial Analysis

- **Moran's I = 0.239** (p = 0.026) — positive spatial autocorrelation across 31
  contiguous states (5 island states/UTs excluded), 999 permutations, row-standardised
  queen contiguity
- **High-high cluster**: Southern belt (Tamil Nadu, Kerala, Andhra Pradesh)
- **Low-low cluster**: North-Central belt (UP, Bihar, Rajasthan, MP)

---

### Econometric Analysis — 13 Specifications (A–M)

#### Core specifications

| Spec | Method | Wealth | Mother's edu | Urban |
|---|---|---|---|---|
| A | Survey-weighted descriptive | — | — | 44.4% on-track (weighted) |
| B | PSU-clustered logit (state FEs) | +1.030*** | +0.879*** | +0.469*** |
| C | Probit AME | +7.9 pp*** | +6.7 pp*** | +3.7 pp*** |
| D | Multilevel LPM (ICC = 0.263) | +0.090*** | +0.066*** | +0.049*** |
| E | Heckman 2-step (IMR = −1.301***) | positive*** | positive*** | — |
| F | Oaxaca-Blinder (gap = 30.9 pp) | — | — | endowments explain 80.4% |

#### Robustness suite

| Spec | Method | Wealth | Mother's edu | Urban |
|---|---|---|---|---|
| G | Two-way clustering (PSU + state) | +1.030*** | +0.879*** | +0.469** |
| H | Threshold sensitivity sweep | 0.543–1.406*** | stable | stable |
| I | IV / 2SLS (LOO, F_KP = 11,943) | — | — | β_2SLS = +0.790*** |
| J | Quantile regression (τ = 0.50) | +0.211*** | +0.231*** | +0.215*** |
| K | Drop 12 imputed states | +1.041*** | +0.878*** | +0.505*** |
| L | Wild-cluster bootstrap | — | — | p_WCB = 0.504 (ns) |
| M | Survey-weighted logit | +1.126*** | +0.919*** | +0.559*** |

\*** p < 0.001, ** p < 0.01, ns = not significant. Spec M uses DHS analytic weights; all others unweighted.

#### VIF diagnostics (Spec B regressors)

All individual-level regressors have VIF between 1.00 and 1.50 — no multicollinearity concern.

#### IV / 2SLS detail (Spec I)

- **Instrument**: Leave-one-out (LOO) state mean of anganwadi attendance — removes own-observation mechanical inflation
- **Kleibergen-Paap F = 11,943** (state-clustered SEs) — strong instrument; reflects genuine cross-state ICDS penetration heterogeneity
- **β_2SLS = +0.790** (SE = 0.143, p < 0.001) vs OLS β = +0.112 — ~7× upward revision consistent with disadvantaged households selecting into programme
- **Durbin-Wu-Hausman t = −5.228** (p < 0.001) — formally rejects exogeneity of attendance
- *Caveat*: the exclusion restriction (state ICDS supply affects ECD only through attendance) is untestable and constitutes the primary limitation of this specification

#### Key findings

- **Household wealth** and **maternal education** are the most stable and economically meaningful predictors across all 13 specifications
- **Urban coefficient** is not robust under state-level inference (wild-cluster bootstrap p = 0.504)
- Survey-weighted estimates (Spec M) confirm unweighted results are not materially misleading; weighted coefficients are marginally larger, consistent with over-representation of disadvantaged children in the raw sample

---

## Pipeline Scripts

| Script | Description | Outputs |
|---|---|---|
| `01_data_loading.py` | Load KR file, filter 24–59 months, recode variables | `india_merged.parquet` |
| `02_ecd_scoring.py` | Construct 2-domain proxy composite (physical + learning) | `india_scored.parquet`, `figures/01–03` |
| `04_supervised_ml.py` | LR + HGB prediction; SHAP via TreeExplainer on full dataset | `figures/08–10`, `shap_importance.csv`, `X_imputed.parquet` |
| `05_clustering.py` | K-Means clustering (silhouette on full dataset); PCA visualisation | `figures/11–13`, `india_clustered.parquet` |
| `06_geospatial.py` | State-level rates, Moran's I (31 contiguous states), choropleth | `figures/06–07`, `state_ecd_rates.csv` |
| `07_proxy_integration.py` | Merge ASER 2019 + NFHS-5 factsheet state proxies | `india_enriched.parquet`, `state_proxy_table.csv` |
| `08_expanded_composite.py` | 4-domain proxy + UNICEF ≥3/4 composite rule | `india_4dom.parquet`, `figures/14–15` |
| `09_econometrics.py` | 13 econometric specs (A–M): logit, probit AME, multilevel LPM, Heckman, Oaxaca, IV/2SLS with LOO instrument, quantile regression, wild-cluster bootstrap, survey-weighted logit; VIF diagnostics and DWH endogeneity test | `econ_*.csv`, `figures/16–20` |
| `build_report.py` | Compile DOCX methodology report | `India_NFHS5_ECD_Methodology_Report.docx` |

**Run order**: `01 → 02 → 04 → 05 → 06 → 07 → 08 → 09 → build_report.py`

---

## Output Files

| File | Contents |
|---|---|
| `econ_logit_coefs.csv` | Spec B logit coefficients and clustered SEs |
| `econ_probit_AME.csv` | Spec C probit average marginal effects |
| `econ_decomposition.csv` | Spec F Oaxaca-Blinder decomposition |
| `econ_robustness_sweep.csv` | Spec H threshold sensitivity results |
| `econ_iv_results.csv` | Spec I 2SLS vs OLS comparison, LOO instrument, F_KP |
| `econ_quantile_regression.csv` | Spec J quantile regression coefficients |
| `econ_drop_imputed.csv` | Spec K sample restriction results |
| `econ_wcb_bootstrap.csv` | Spec L wild-cluster bootstrap p-values |
| `econ_weighted_logit.csv` | Spec M survey-weighted logit vs unweighted comparison |
| `econ_vif.csv` | VIF values for Spec B regressors |
| `econ_dwh_test.csv` | Durbin-Wu-Hausman test results (LOO instrument, state-clustered) |
| `shap_importance.csv` | Mean absolute SHAP values for all features |
| `state_proxy_table.csv` | State-level proxy domain rates |
| `state_ecd_rates.csv` | State-level composite on-track rates |

---

## Data Variables Used

### NFHS-5 KR file (child level)

| Variable | Description |
|---|---|
| `hw1` | Child age in months |
| `b4` | Child sex |
| `hw70` | Height-for-age z-score (DHS units ×0.01 SD) |
| `hw71` | Weight-for-age z-score (DHS units ×0.01 SD) |
| `s558` | Received any anganwadi/ICDS benefit in last 12 months |
| `s562` | Frequency of anganwadi attendance |
| `v024` | State (36 states and UTs) — merge key for external surveys |
| `v025` | Urban/rural residence |
| `v190` | Wealth index quintile (combined) |
| `v149` | Mother's educational attainment |
| `v012` | Mother's age |
| `v005` | Women's sample weight (used in Spec A and Spec M) |
| `v001` | PSU cluster ID (used for cluster-robust SEs) |

### External data merged on `v024` (state)

| Source | Variable | Description |
|---|---|---|
| ASER 2019 Early Years | `aser_letters_4to5` | State % aged 4–5 recognising letters |
| ASER 2019 Early Years | `aser_numbers_4to5` | State % aged 4–5 recognising 1-digit numbers |
| NFHS-5 Factsheet | `women_decisions_state` | State % married women in 3+ household decisions |
| NFHS-5 Factsheet | `spousal_violence_state` | State % women reporting 12-month spousal violence |
| NFHS-5 Factsheet | `women_work_state` | State % married women employed in last 12 months |

---

## Limitations

1. **State-level imputation for two domains** — every child in a state inherits the same Literacy-Numeracy and Socio-emotional proxy values; within-state variation in these domains is zero by construction (ecological-fallacy risk, partially mitigated by individual-level corroborators in the AND-rule).
2. **ASER 2019 coverage** — 12 states/UTs not sampled by ASER 2019 receive their region's ASER mean as an imputed value; ASER draws from a single rural district per state, so urban areas are not represented.
3. **Regressor-in-outcome overlap** — `wealth_index` and `mother_edu` enter both the Literacy-Numeracy / Socio-emotional domain constructions and the regression RHS. Coefficients on these variables in Specifications B–L represent upper bounds on their true partial effects.
4. **Heckman correction scope** — Specification E addresses selection into regular attendance conditional on ICDS enrolment. The ~33.5% of children with no enrolment record are excluded from the analytic sample entirely; Heckman cannot recover this larger selection stage.
5. **IV exclusion restriction** — the state ICDS penetration instrument cannot be tested for exclusion; Specification I should be interpreted as evidence of endogeneity direction and magnitude, not a clean causal estimate.
6. **Cross-sectional design** — all econometric specifications estimate associations, not causal effects (except the IV LATE under its maintained exclusion restriction).

---

## Requirements

```
pandas numpy matplotlib seaborn scikit-learn shap pyarrow
statsmodels linearmodels python-docx scipy libpysal esda
```

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap pyarrow statsmodels linearmodels python-docx scipy libpysal esda
```

---

*MTP2 Thesis — Indian Institute of Technology Kharagpur*

