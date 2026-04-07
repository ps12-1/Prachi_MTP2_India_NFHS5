# India NFHS-5 — ECD Proxy Analysis
## Modelling Early Childhood Development using UNICEF ECDI2030 Framework

**Data Source**: India DHS-7 / NFHS-5 (2019–21), Kids Recode (IAKR7EFL.DTA)  
**Sample**: 133,750 alive children aged 24–59 months  
**Country**: India (36 states and Union Territories)

---

## Methodological Note: Proxy Indicators

NFHS-5 does not administer direct ECDI2030 cognitive or behavioural assessments
(equivalent to Nepal MICS6 items EC6–EC15). This analysis constructs the best
available proxy for ECDI2030 domains using NFHS-5 variables:

| ECDI2030 Domain | NFHS-5 Proxy | Variable | Coverage |
|---|---|---|---|
| Physical | Not stunted (HAZ > −2 SD) | hw70 | 94% |
| Learning/ECCE | Regular anganwadi attendance | s562 | 69% |
| Literacy-Numeracy | *No valid proxy* | — | — |
| Socio-emotional | *No valid proxy* | — | — |

**Proxy Composite**: Child is "on track" if **both** physical (not stunted)
AND learning (regular anganwadi attendance) proxy domains are satisfied.
This is a conservative two-domain composite; it covers a subset of the
full ECDI2030 framework.

---

## Results Summary

### Proxy Domain On-Track Rates
| Domain | Rate | n |
|---|---|---|
| Physical (not stunted) | **62.2%** | 126,250 |
| Nutritional (not underweight) | **67.2%** | 127,318 |
| Learning (regular anganwadi) | **40.7%** | 91,719 |
| **Proxy composite (both)** | **25.0%** | 86,999 |

### Supervised ML Performance (5-fold CV)
| Model | ROC-AUC | F1 |
|---|---|---|
| Logistic Regression | **0.656 ± 0.003** | 0.438 ± 0.003 |
| HistGradientBoosting | **0.656 ± 0.004** | 0.434 ± 0.004 |

Note: HAZ/WAZ excluded from feature set — HAZ > −2 is a direct component of the outcome (physical proxy domain), so inclusion would create circular prediction. The AUC of ~0.656 reflects honest predictive power of socioeconomic context variables alone.

### Top 5 SHAP Predictors (HistGradientBoosting)
1. **Child age (months)** — older children more likely to be on track
2. **State: Meghalaya** — very low ECCE attendance
3. **State: Nagaland** — near-zero anganwadi attendance
4. **Mother's education (ordinal)** — strong protective factor
5. **State: Uttar Pradesh** — large state with low rates

### Geospatial Analysis (State-Level)
- **Moran's I = 0.421**, p = 0.001 (999 permutations) — significant spatial clustering
- **Highest on-track states**: Andaman & Nicobar Islands (47%), West Bengal (43%), Odisha (42%), Gujarat (40%)
- **Lowest on-track states**: Nagaland (0.3%), Manipur (4.9%), Meghalaya (8.5%)
  - *Note: Very low values in Northeast states reflect near-zero anganwadi attendance in this sample,
    possibly due to alternative childcare systems (crèches, church-based centres) or survey coverage*

### Clustering (k = 2, silhouette-optimal)
K-Means on [HAZ, WAZ, anganwadi attendance] identifies 2 clusters:
- **Cluster 0** (n=51,470): No regular anganwadi attendance; similar nutritional status
- **Cluster 1** (n=35,364): Regular anganwadi attendance; similar nutritional status

The binary learning proxy dominates clustering — anthropometric differences between
clusters are small (mean HAZ: −1.49 vs −1.53), reflecting that ECCE access and
nutritional status are largely independent in India.

---

## Pipeline Scripts

| Script | Description | Output |
|---|---|---|
| `01_data_loading.py` | Load KR file, filter 24–59 months, recode variables | `india_merged.parquet` |
| `02_ecd_scoring.py` | Construct proxy domains and composite, visualisations | `india_scored.parquet`, `figures/01–03` |
| `04_supervised_ml.py` | LR + HGB prediction, SHAP interpretability | `figures/08–10`, `shap_importance.csv` |
| `05_clustering.py` | K-Means clustering, PCA, profile characterisation | `figures/11–13`, `india_clustered.parquet` |
| `06_geospatial.py` | State-level rates, Moran's I, scatter plots | `figures/06–07`, `state_ecd_rates.csv` |

---

## Data Variables Used

| Variable | Description |
|---|---|
| `hw1` | Child age in months |
| `b4` | Child sex |
| `hw70` | Height-for-age z-score (DHS units: ×0.01 SD) |
| `hw71` | Weight-for-age z-score (DHS units: ×0.01 SD) |
| `hw72` | Weight-for-height z-score |
| `s558` | Received any anganwadi/ICDS benefit in last 12 months |
| `s562` | Frequency of anganwadi attendance (regularly/occasionally/not at all) |
| `v024` | State (36 states and UTs) |
| `v025` | Urban/rural residence |
| `v190` | Wealth index quintile (combined) |
| `v149` | Mother's educational attainment |
| `v012` | Mother's age |
| `v005` | Women's sample weight |

---

## Comparison: Nepal vs India ECD Analysis

| Metric | Nepal MICS6 | India NFHS-5 |
|---|---|---|
| Framework | ECDI2030 (direct items) | ECDI2030 proxy (2 domains) |
| Sample size | 2,799 (complete ECDI) | 86,999 (proxy composite) |
| On-track rate | 12.6% | 25.0% |
| HGB AUC | 0.663 | 0.656 |
| Top predictor | Child age (SHAP) | Child age / state (SHAP) |
| Spatial clustering | Moran's I=0.098, p=0.166 | Moran's I=0.421, p=0.001 |
| Administrative unit | Province (7) | State (36) |

---

## Limitations

1. **No direct cognitive/socio-emotional assessment** — proxy domains cover physical
   health and ECCE access only; not comparable to direct ECDI2030 measurement.
2. **Composite overlap**: HAZ is both a predictor feature (in ML model) and a component
   of the outcome — SHAP values for HAZ reflect this circularity.
3. **Anganwadi coverage gap**: s562 is only recorded for children whose mothers report
   receiving anganwadi benefits (s558='yes'); children not enrolled are coded as NaN,
   creating selection bias in the learning proxy.
4. **Northeast states anomaly**: Several Northeast states show near-zero anganwadi
   attendance in NFHS-5, possibly due to alternative ECD delivery systems not captured
   by the anganwadi-specific questions.
5. **Survey design**: Complex multi-stage sampling weights (v005) not applied in this
   analysis — point estimates should be treated as unweighted descriptives.

---

## Requirements

```
pandas, numpy, matplotlib, seaborn, scikit-learn, shap, pyarrow
```

Install: `pip install pandas numpy matplotlib seaborn scikit-learn shap pyarrow`

---

*Part of MTP2 thesis: "Modelling Early Childhood Development using the UNICEF ECDI2030 Framework"*
