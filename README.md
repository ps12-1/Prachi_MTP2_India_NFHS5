# India NFHS-5 — ECD Proxy Analysis (Expanded 4-Domain Pipeline)
## Modelling Early Childhood Development using UNICEF ECDI2030 Framework

**Data Source**: India DHS-7 / NFHS-5 (2019–21), Kids Recode (IAKR7EFL.DTA)
**External survey merges (real data, not hardcoded)**:
- ASER 2019 Early Years — parsed directly from the public PDF (`external_data/parse_aser_2019.py`)
- NFHS-5 published State Factsheets (Phase I + II) — via [jvargh7/nfhs5_factsheets](https://github.com/jvargh7/nfhs5_factsheets) parsed CSV
**Sample**: 133,750 alive children aged 24–59 months
**Country**: India only — 36 states and Union Territories

---

## Methodology: Expanded 4-Domain Proxy Composite

NFHS-5 contains no individual-level measurements for ECDI2030 Literacy-Numeracy or
Socio-emotional items (EC6–EC8, EC13–EC15). To build a 4-domain composite covering
all of UNICEF's ECDI2030 framework, this pipeline merges external survey data into
each NFHS-5 child record using v024 (state of residence) as the demographic
merge key — the standard small-area-estimation approach (Pfeffermann 2013).

| ECDI2030 Domain | Proxy | Source |
|---|---|---|
| Physical | Not stunted (HAZ > −2 SD) | NFHS-5 child (hw70) |
| Learning | Regular anganwadi attendance | NFHS-5 child (s562) |
| Literacy-Numeracy | State-level ASER letter+number recognition ≥ 50%, OR mother has complete-secondary+ education | ASER 2019 + NFHS-5 |
| Socio-emotional | State women's empowerment composite ≥ median, AND household not in lowest wealth quintile | NFHS-5 factsheet + KR |

**Composite rule (UNICEF ECDI2030 standard)**: Child is "on track" if ≥ 3 of 4 domains pass.

---

## Results Summary

### Proxy Domain On-Track Rates (4-domain framework, real merged data)
| Domain | Rate | n |
|---|---|---|
| Physical (not stunted) | **62.2%** | 126,250 |
| Learning (regular anganwadi) | **40.7%** | 91,719 |
| Literacy-Numeracy (ASER 2019 parsed + mother edu) | **63.6%** | 86,999 |
| Socio-emotional (NFHS-5 factsheet + wealth) | **41.2%** | 86,999 |
| **4-domain composite (≥ 3 / 4)** | **44.6%** (unweighted), **44.4%** (survey-weighted) | 86,999 |

### Econometric Analysis (Section 10 of Report — real-data run)
| Specification | Key result |
|---|---|
| Survey-weighted descriptive | 44.4% on track (weighted) vs 44.6% unweighted |
| Cluster-robust logit (PSU clustered at v001, n=25,706 PSUs) | Wealth β=+0.496, mother_edu β=+0.124, urban β=−0.189 (urban paradox), all p<0.001 |
| Probit AME (Δ probability) | wealth +7.9 pp / quintile, mother_edu +2.0 pp / level, urban −3.0 pp |
| Multilevel LPM (state random intercept) | **ICC = 0.124** — 12% of residual variance is between states (real proxies absorb most of the cross-state heterogeneity) |
| Heckman two-step (anganwadi selection) | IMR coefficient = **−0.6441 (p<0.001)** — confirms ICDS targets disadvantaged children |
| Oaxaca-Blinder (urban-rural gap of 15.5 pp) | **117%** of gap explained by endowments (chiefly wealth +12 pp, plus the two state-level proxies +2.6 pp each); structural urban premium ≈ 0 |

### Geospatial Analysis (State-Level, 4-domain real-data composite)
- **Moran's I = 0.421**, p = 0.001 — significant positive spatial clustering
- **Top states**: Tamil Nadu (83.9%), Andhra Pradesh (83.5%), Kerala (82.8%), Andaman & Nicobar (82.2%), Puducherry (82.1%)
- **Bottom states**: Rajasthan (2.4%), UP (16.6%), Ladakh (18.3%), J&K (18.8%), MP (21.4%), Bihar (22.6%)

---

## Pipeline Scripts

| Script | Description | Output |
|---|---|---|
| `01_data_loading.py` | Load KR file, filter 24–59 months, recode variables | `india_merged.parquet` |
| `02_ecd_scoring.py` | Construct 2-domain proxy composite | `india_scored.parquet`, `figures/01–03` |
| `04_supervised_ml.py` | LR + HGB prediction, SHAP interpretability | `figures/08–10`, `shap_importance.csv` |
| `05_clustering.py` | K-Means clustering, PCA, profile characterisation | `figures/11–13`, `india_clustered.parquet` |
| `06_geospatial.py` | State-level rates, Moran's I, scatter plots | `figures/06–07`, `state_ecd_rates.csv` |
| `07_proxy_integration.py` | **NEW** — merge ASER 2019 + NFHS-5 factsheet state-level proxies | `india_enriched.parquet`, `state_proxy_table.csv` |
| `08_expanded_composite.py` | **NEW** — 4-domain proxy + UNICEF ≥3/4 composite rule | `india_4dom.parquet`, `figures/14–15` |
| `09_econometrics.py` | **NEW** — survey-weighted, cluster-robust logit, probit AME, multilevel, Heckman, Oaxaca | `econ_*.csv`, `figures/16–17` |
| `build_report.py` | Compile DOCX methodology report (Sections 1–13) | `India_NFHS5_ECD_Methodology_Report.docx` |

Run order: `01 → 02 → 04 → 05 → 06 → 07 → 08 → 09 → build_report.py`

---

## Data Variables Used

### NFHS-5 KR file (child level)
| Variable | Description |
|---|---|
| `hw1` | Child age in months |
| `b4` | Child sex |
| `hw70` | Height-for-age z-score (DHS units: ×0.01 SD) |
| `hw71` | Weight-for-age z-score (DHS units: ×0.01 SD) |
| `s558` | Received any anganwadi/ICDS benefit in last 12 months |
| `s562` | Frequency of anganwadi attendance |
| `v024` | State (36 states and UTs) — merge key for external surveys |
| `v025` | Urban/rural residence |
| `v190` | Wealth index quintile (combined) |
| `v149` | Mother's educational attainment |
| `v012` | Mother's age |
| `v005` | Women's sample weight (applied in Section 10.1) |
| `v001` | PSU cluster ID (used for cluster-robust SEs) |

### External — merged on v024 (state)
| Source | Variable | Description |
|---|---|---|
| ASER 2019 Early Years | `aser_letters_4to5` | State % aged 4–5 recognising letters |
| ASER 2019 Early Years | `aser_numbers_4to5` | State % aged 4–5 recognising 1-digit numbers |
| NFHS-5 Factsheet | `women_decisions_state` | State % married women in 3+ household decisions |
| NFHS-5 Factsheet | `spousal_violence_state` | State % women reporting 12-mo spousal violence |
| NFHS-5 Factsheet | `women_work_state` | State % married women employed in last 12 months |

---

## Limitations

1. **State-level imputation for two domains** — every child in a state inherits the same Literacy-Numeracy
   and Socio-emotional proxy values; within-state variation in these domains is by construction zero
   (ecological-fallacy risk; partially mitigated by AND-rule with individual-level corroborators).
2. **ASER 2019 covers 24 of 36 states** — 12 states/UTs not sampled by ASER 2019 receive their region's
   ASER mean as an imputed value (north / south / east / west / central / north-east). Within-sampled
   states, ASER drew from a single rural district per state, so urban areas are not represented.
3. **Anganwadi selection bias** is quantified via Heckman two-step (IMR = −0.64, p<0.001) but cannot
   be eliminated; children entirely outside ICDS are excluded from the analytic sample.
4. **HAZ as Physical proxy** measures nutritional history, not fine motor skills (EC9) or energy (EC10).
5. **Cross-sectional design** — Section 10's econometric models estimate associations, not causal effects.

---

## Requirements

```
pandas, numpy, matplotlib, seaborn, scikit-learn, shap, pyarrow,
statsmodels, linearmodels, python-docx, scipy
```

Install: `pip install pandas numpy matplotlib seaborn scikit-learn shap pyarrow statsmodels linearmodels python-docx scipy`

---

*Part of MTP2 thesis: "Modelling Early Childhood Development using the UNICEF ECDI2030 Framework"*
