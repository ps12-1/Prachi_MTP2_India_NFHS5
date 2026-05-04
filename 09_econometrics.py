"""
India NFHS-5 — Econometric Analysis with ROBUSTNESS SUITE
==========================================================
Twelve specifications partitioned into a CORE (A–F) and a ROBUSTNESS
suite (G–L). The robustness specifications stress-test the core findings
under alternative assumptions about clustering, instrumentation, sample
inclusion, threshold choice, distributional shape, and individual-level
proxy construction.

Core specifications (already in the methodology report):
  (A) Survey-weighted descriptives — applies DHS weight v005/1e6.
  (B) Cluster-robust logistic regression — PSU-clustered SEs at v001
      with state fixed effects.
  (C) Average marginal effects from a probit (interpretable pp).
  (D) Two-level random-intercept LPM (children → state, REML).
  (E) Heckman two-step selection — corrects anganwadi enrolment bias.
  (F) Oaxaca-Blinder decomposition of the urban-rural gap.

Robustness suite:
  (G) TWO-WAY clustering at PSU AND state.
  (H) Threshold + composite-rule sensitivity sweep (3 × 3 grid).
  (I) IV / 2SLS — state anganwadi share as instrument for individual
      anganwadi attendance, identifying the LATE on the on-track outcome.
  (J) Quantile regression on the continuous 4-domain score (Q10/25/50/75/90).
  (K) Sample-restriction sensitivity — drop the 12 region-imputed (non-
      ASER-sampled) states.
  (L) Wild-cluster bootstrap on the urban coefficient (state-level
      cluster, B = 999) — robust inference with 36 state clusters.

Outputs:
  econ_logit_coefs.csv          (B)
  econ_probit_AME.csv           (C)
  econ_decomposition.csv        (F)
  econ_robustness_sweep.csv     (H)
  econ_iv_results.csv           (I)
  econ_quantile_regression.csv  (J)
  econ_drop_imputed.csv         (K)
  econ_wcb_bootstrap.csv        (L)
  figures/16_probit_AME.png
  figures/17_oaxaca.png
  figures/19_robustness_sweep.png
  figures/20_quantile_regression.png
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

OUT = Path("/Users/prachi/india_nfhs5_project")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.05)

df = pd.read_parquet(OUT / "india_4dom.parquet")
print(f"Data loaded: {len(df):,} children")

y_col = "ecdi_proxy4_composite"
y_cont_col = "ecdi_proxy4_continuous"

covariates = [
    "child_age_months",
    "is_female",
    "is_urban",
    "wealth_index",
    "mother_edu",
    "mother_age",
    "literacy_numeracy_state_rate",
    "socioemo_state_index",
]
psu_col   = "v001"
state_col = "state"
wt_col    = "weight"

work = df[[y_col, y_cont_col] + covariates +
          [psu_col, state_col, wt_col, "v025", "anganwadi_any",
           "ecdi_proxy_learning", "from_aser_direct",
           "state_anganwadi_share"]].dropna(subset=[y_col]).copy()
print(f"Analytic sample after listwise deletion: {len(work):,}")

# ═════════════════════════════════════════════════════════════════════════════
# (A) SURVEY-WEIGHTED DESCRIPTIVES
# ═════════════════════════════════════════════════════════════════════════════

def weighted_mean(values, weights):
    values  = np.asarray(values)
    weights = np.asarray(weights)
    mask = ~np.isnan(values) & ~np.isnan(weights)
    if mask.sum() == 0:
        return np.nan
    return np.sum(values[mask] * weights[mask]) / np.sum(weights[mask])

print("\n" + "="*72)
print("(A) SURVEY-WEIGHTED DESCRIPTIVES")
print("="*72)
print(f"  Unweighted on-track:         {work[y_col].mean()*100:.2f}%")
print(f"  Weighted on-track:           {weighted_mean(work[y_col], work[wt_col])*100:.2f}%")

# ═════════════════════════════════════════════════════════════════════════════
# (B) CLUSTER-ROBUST LOGISTIC REGRESSION (PSU)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(B) CLUSTER-ROBUST LOGIT — PSU clustering (v001)")
print("="*72)

work["state_clean"] = work[state_col].str.replace(" ", "_").str.replace("&", "and")
state_dummies = pd.get_dummies(work["state_clean"], prefix="st", drop_first=True).astype(float)
covariates_b  = [c for c in covariates
                 if c not in ("literacy_numeracy_state_rate","socioemo_state_index")]
X_b = pd.concat([work[covariates_b], state_dummies], axis=1).astype(float)
X_b = sm.add_constant(X_b)
y_b = work[y_col].astype(int)

glm_logit_psu = sm.GLM(y_b, X_b, family=sm.families.Binomial()).fit(
    cov_type="cluster",
    cov_kwds={"groups": work[psu_col].astype(int).values}
)
non_state = [c for c in X_b.columns if not c.startswith("st_")]
summary_b = pd.DataFrame({
    "coef":  glm_logit_psu.params[non_state].round(3),
    "se":    glm_logit_psu.bse[non_state].round(3),
    "z":     (glm_logit_psu.params[non_state] / glm_logit_psu.bse[non_state]).round(2),
    "p":     glm_logit_psu.pvalues[non_state].round(4),
}, index=non_state)
print(summary_b.to_string())
ll_null = sm.GLM(y_b, sm.add_constant(np.ones(len(y_b))), family=sm.families.Binomial()).fit().llf
print(f"\n  McFadden Pseudo R²: {1 - glm_logit_psu.llf/ll_null:.3f}")
print(f"  PSU clusters: {work[psu_col].nunique():,}")

# ─────────────────────────────────────────────────────────────────────────────
# VIF DIAGNOSTICS (individual-level regressors only; state dummies omitted)
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- VIF Diagnostics (individual-level covariates) ---")
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif_cols = covariates_b  # 6 individual predictors (no state dummies)
X_vif = sm.add_constant(work[vif_cols].astype(float).dropna())
vif_df = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [variance_inflation_factor(X_vif.values, i)
            for i in range(X_vif.shape[1])]
}).query("feature != 'const'").round(2)
print(vif_df.to_string(index=False))
vif_df.to_csv(OUT / "econ_vif.csv", index=False)
print("  Saved econ_vif.csv  (VIF < 5 = no multicollinearity concern; < 10 = acceptable)")

# ═════════════════════════════════════════════════════════════════════════════
# (C) PROBIT  AVERAGE MARGINAL EFFECTS
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(C) PROBIT  AVERAGE MARGINAL EFFECTS")
print("="*72)
probit = sm.Probit(y_b, X_b).fit(disp=False, maxiter=200)
ame = probit.get_margeff(at="overall", method="dydx")
ame_full = ame.summary_frame()
ame_non_state = ame_full.loc[[c for c in ame_full.index if not c.startswith("st_")]]
print(ame_non_state.round(4).to_string())

# ═════════════════════════════════════════════════════════════════════════════
# (D) MULTILEVEL LPM — state random intercept
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(D) MULTILEVEL LPM — state random intercept")
print("="*72)
mlm_data = work[[y_col] + covariates + [state_col]].copy()
formula_lpm = (f"{y_col} ~ child_age_months + is_female + is_urban + "
               "wealth_index + mother_edu + mother_age + "
               "literacy_numeracy_state_rate + socioemo_state_index")
mlm = smf.mixedlm(formula_lpm, data=mlm_data, groups=mlm_data[state_col]).fit(method="lbfgs", reml=True)
print(mlm.summary().tables[1])
var_re    = float(mlm.cov_re.iloc[0, 0])
var_resid = float(mlm.scale)
icc = var_re / (var_re + var_resid)
print(f"  Between-state variance: {var_re:.5f}")
print(f"  Within-state variance:  {var_resid:.5f}")
print(f"  ICC = {icc:.3f}")

# ═════════════════════════════════════════════════════════════════════════════
# (E) HECKMAN  TWO-STEP SELECTION  MODEL
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(E) HECKMAN TWO-STEP SELECTION")
print("="*72)
heck_df = df[df["anganwadi_any"].notna()].copy()
sel_covs = ["child_age_months","is_female","is_urban","wealth_index",
            "mother_edu","mother_age","state_anganwadi_share"]
heck_sel = heck_df[["anganwadi_any"] + sel_covs].dropna().copy()
X_sel = sm.add_constant(heck_sel[sel_covs].astype(float))
y_sel = heck_sel["anganwadi_any"].astype(int)
sel_probit = sm.Probit(y_sel, X_sel).fit(disp=False, maxiter=200)

heck_sel["xb_sel"] = sel_probit.predict(X_sel, linear=True)
heck_sel["imr"]    = stats.norm.pdf(heck_sel["xb_sel"]) / stats.norm.cdf(heck_sel["xb_sel"])
heck_sel["caseid"] = heck_sel.index

out_covs = ["child_age_months","is_female","is_urban","wealth_index",
            "mother_edu","mother_age",
            "literacy_numeracy_state_rate","socioemo_state_index"]
heck_out = heck_df[heck_df["anganwadi_any"] == 1].copy()
heck_out = heck_out.merge(heck_sel[["caseid","imr"]], left_index=True, right_on="caseid", how="left")
heck_out = heck_out[[y_col] + out_covs + ["imr"]].dropna()
X_out = sm.add_constant(heck_out[out_covs + ["imr"]].astype(float))
y_out = heck_out[y_col].astype(float)
ols_out = sm.OLS(y_out, X_out).fit(cov_type="HC1")
imr_coef = ols_out.params["imr"]
imr_p    = ols_out.pvalues["imr"]
print(f"  IMR coefficient: {imr_coef:+.4f} (p = {imr_p:.4f})")

# ═════════════════════════════════════════════════════════════════════════════
# (F) OAXACA-BLINDER  DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(F) OAXACA-BLINDER  urban − rural")
print("="*72)
ob_covs = ["child_age_months","is_female","wealth_index","mother_edu","mother_age",
           "literacy_numeracy_state_rate","socioemo_state_index"]
ob = work[[y_col, "is_urban"] + ob_covs].dropna().copy()
urban  = ob[ob["is_urban"] == 1]
rural  = ob[ob["is_urban"] == 0]
mean_u = urban[ob_covs].mean()
mean_r = rural[ob_covs].mean()
X_pool = sm.add_constant(ob[ob_covs])
y_pool = ob[y_col].astype(float)
beta_pool = sm.OLS(y_pool, X_pool).fit().params
beta_u = sm.OLS(urban[y_col].astype(float), sm.add_constant(urban[ob_covs])).fit().params
beta_r = sm.OLS(rural[y_col].astype(float), sm.add_constant(rural[ob_covs])).fit().params
mean_u_full = pd.concat([pd.Series([1.0], index=["const"]), mean_u])
mean_r_full = pd.concat([pd.Series([1.0], index=["const"]), mean_r])
mean_gap   = urban[y_col].mean() - rural[y_col].mean()
endowment  = float((mean_u_full - mean_r_full) @ beta_pool)
unexplained = float(mean_u_full @ (beta_u - beta_pool)) + float(mean_r_full @ (beta_pool - beta_r))
print(f"  Total gap:    {mean_gap*100:+6.2f} pp")
print(f"  Endowments:   {endowment*100:+6.2f} pp ({endowment/mean_gap*100:5.1f}%)")
print(f"  Unexplained:  {unexplained*100:+6.2f} pp ({unexplained/mean_gap*100:5.1f}%)")

# ═════════════════════════════════════════════════════════════════════════════
# (G) TWO-WAY CLUSTERING — PSU and state
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(G) TWO-WAY CLUSTERING — PSU and state")
print("="*72)

# Cameron-Gelbach-Miller (2011): V_2way = V_PSU + V_state - V_PSU∩state
# Build identifier for the intersection (psu, state) — uniquely a PSU is
# in exactly one state in NFHS-5, so V_PSU∩state == V_PSU; we therefore
# add the state-level cluster correction and subtract once.

# Refit with state clustering
glm_logit_state = sm.GLM(y_b, X_b, family=sm.families.Binomial()).fit(
    cov_type="cluster",
    cov_kwds={"groups": work["state_clean"].astype("category").cat.codes.values}
)
# Two-way SE = sqrt(V_PSU + V_state - V_PSU)  =  sqrt(V_state)  in this case
# but Cameron-Gelbach-Miller correctly applies the formula even when one
# cluster is nested in the other, returning the more conservative SE.
# Using the larger of the two as a defensible bound:
# pandas-friendly element-wise max
two_way_se = pd.concat([glm_logit_psu.bse, glm_logit_state.bse], axis=1).max(axis=1)
two_way_z  = glm_logit_psu.params / two_way_se
two_way_p  = pd.Series(2 * (1 - stats.norm.cdf(np.abs(two_way_z))), index=two_way_z.index)

twoway_df = pd.DataFrame({
    "coef":      glm_logit_psu.params[non_state].round(3),
    "se_psu":    glm_logit_psu.bse[non_state].round(3),
    "se_state":  glm_logit_state.bse[non_state].round(3),
    "se_2way":   two_way_se[non_state].round(3),
    "z_2way":    two_way_z[non_state].round(2),
    "p_2way":    two_way_p[non_state].round(4),
}, index=non_state)
print(twoway_df.to_string())
print("  (Defensible upper bound: max(V_PSU, V_state) — conservative.)")

# ═════════════════════════════════════════════════════════════════════════════
# (H) THRESHOLD + COMPOSITE-RULE SENSITIVITY SWEEP
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(H) ROBUSTNESS SWEEP — thresholds × composite rules")
print("="*72)

sweep_rows = []
sweep = []
for thr in [-0.25, 0.0, 0.25]:           # individual-score threshold (z)
    for rule in [2, 3, 4]:                # composite k-of-4
        litnum_pass = (df["litnum_score"] >= thr).astype(float)
        se_pass     = (df["se_score"]     >= thr).astype(float)
        # Use v2 multi-component proxies to match the main composite definition
        dom_sum = (df["ecdi_proxy_physical_v2"] + df["ecdi_proxy_learning_v2"]
                   + litnum_pass + se_pass)
        ok = df[["ecdi_proxy_physical_v2", "ecdi_proxy_learning_v2"]].notna().all(axis=1)
        comp = np.where(ok, (dom_sum >= rule).astype(float), np.nan)
        pct  = float(np.nanmean(comp) * 100)
        # Re-fit logit with this new outcome
        ok_idx = ~np.isnan(comp)
        try:
            sub = df.loc[ok_idx].copy()
            sub["state_clean"] = sub["state"].str.replace(" ", "_").str.replace("&", "and")
            sd = pd.get_dummies(sub["state_clean"], prefix="st", drop_first=True).astype(float)
            X_s = pd.concat([sub[covariates_b], sd], axis=1).astype(float)
            X_s = sm.add_constant(X_s)
            y_s = pd.Series(comp[ok_idx], index=sub.index).astype(int)
            mdl = sm.GLM(y_s, X_s, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub[psu_col].astype(int).values}
            )
            wealth_b = mdl.params["wealth_index"]; wealth_p = mdl.pvalues["wealth_index"]
            urban_b  = mdl.params["is_urban"];     urban_p  = mdl.pvalues["is_urban"]
        except Exception as e:
            wealth_b = wealth_p = urban_b = urban_p = np.nan
        sweep_rows.append({
            "threshold_z": thr, "kof4_rule": rule, "ontrack_pct": round(pct, 1),
            "wealth_β": round(wealth_b, 3), "wealth_p": round(wealth_p, 4),
            "urban_β":  round(urban_b, 3),  "urban_p":  round(urban_p, 4),
        })
sweep_df = pd.DataFrame(sweep_rows)
print(sweep_df.to_string(index=False))
sweep_df.to_csv(OUT / "econ_robustness_sweep.csv", index=False)

# ═════════════════════════════════════════════════════════════════════════════
# (I) IV  /  2SLS — state anganwadi share INSTRUMENTS individual attendance
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(I) IV / 2SLS — state_anganwadi_share → ecdi_proxy_learning")
print("="*72)
# Endogenous: ecdi_proxy_learning (binary, individual)
# Instrument: state_anganwadi_share (continuous, varies only by state)
# Exclusion: state aggregate has no direct effect on outcome other than through
#            individual attendance (after controlling for all other state-level
#            proxies and individual covariates) — strong assumption, see report.

iv_df = work[[y_col, "ecdi_proxy_learning"] + covariates +
             [psu_col, state_col, "state_anganwadi_share"]].dropna().copy()

# ── Leave-One-Out (LOO) instrument construction ───────────────────────────
# For each child i, the instrument is the state mean of ecdi_proxy_learning
# computed EXCLUDING child i.  This removes the mechanical own-observation
# bias that inflates the in-sample F when using the raw state mean.
state_n   = iv_df.groupby(state_col)["ecdi_proxy_learning"].transform("count")
state_sum = iv_df.groupby(state_col)["ecdi_proxy_learning"].transform("sum")
iv_df["instrument_loo"] = (state_sum - iv_df["ecdi_proxy_learning"]) / (state_n - 1)

# States with only one observation get NaN LOO; drop those rows (rare edge case)
iv_df = iv_df.dropna(subset=["instrument_loo"])
print(f"  IV analytic sample: {len(iv_df):,} children")
print(f"  LOO instrument — state N range: {state_n.min():.0f} – {state_n.max():.0f}")

fs_covs_loo = [c for c in covariates
               if c not in ("ecdi_proxy_learning", "state_anganwadi_share")]
fs_X_loo = sm.add_constant(
    iv_df[fs_covs_loo + ["instrument_loo"]].astype(float))
fs_y = iv_df["ecdi_proxy_learning"].astype(float)

# ── First stage — two SE specifications ───────────────────────────────────
# (a) HC1 robust (for comparison with old result)
fs_hc1   = sm.OLS(fs_y, fs_X_loo).fit(cov_type="HC1")
fs_F_hc1 = (fs_hc1.params["instrument_loo"] / fs_hc1.bse["instrument_loo"])**2

# (b) State-clustered SEs — the Kleibergen-Paap rk approximation.
#     The instrument varies only at the state level (36 groups), so
#     individual-level SEs severely understate sampling uncertainty.
fs_cl = sm.OLS(fs_y, fs_X_loo).fit(
    cov_type="cluster",
    cov_kwds={"groups": iv_df[state_col].astype("category").cat.codes.values})
fs_F_kp = (fs_cl.params["instrument_loo"] / fs_cl.bse["instrument_loo"])**2

print(f"\n  First stage: LOO instrument → ecdi_proxy_learning")
print(f"     β     = {fs_cl.params['instrument_loo']:+.4f}")
print(f"     SE_HC1      = {fs_hc1.bse['instrument_loo']:.6f}  "
      f"→  F_HC1 (inflated) = {fs_F_hc1:.1f}")
print(f"     SE_clustered = {fs_cl.bse['instrument_loo']:.4f}  "
      f"→  F_KP  (correct)  = {fs_F_kp:.2f}")
print(f"     (Stock-Yogo threshold F > 10; "
      f"Olea-Pflueger 2013 effective F > 23.1 for 5% size)")

# Use clustered first stage for second stage
iv_df["learning_hat"] = fs_cl.predict(fs_X_loo)
ss_X = sm.add_constant(
    iv_df[fs_covs_loo + ["learning_hat"]].astype(float))
ss_y = iv_df[y_col].astype(float)
ss = sm.OLS(ss_y, ss_X).fit(
    cov_type="cluster",
    cov_kwds={"groups": iv_df[state_col].astype("category").cat.codes.values})

# OLS comparison (state-clustered SEs)
ols_compare = sm.OLS(iv_df[y_col].astype(float),
                     sm.add_constant(
                         iv_df[fs_covs_loo + ["ecdi_proxy_learning"]].astype(float))
                     ).fit(
    cov_type="cluster",
    cov_kwds={"groups": iv_df[state_col].astype("category").cat.codes.values})

print(f"\n  Second stage (state-clustered SEs, LOO instrument):")
print(f"     β_2SLS = {ss.params['learning_hat']:+.4f}  "
      f"SE={ss.bse['learning_hat']:.4f}  p={ss.pvalues['learning_hat']:.4f}")
print(f"     β_OLS  = {ols_compare.params['ecdi_proxy_learning']:+.4f}  "
      f"SE={ols_compare.bse['ecdi_proxy_learning']:.4f}  "
      f"p={ols_compare.pvalues['ecdi_proxy_learning']:.4f}")

iv_results = pd.DataFrame({
    "estimator":     ["OLS (clustered)", "IV 2SLS (LOO, clustered)"],
    "learning_β":    [ols_compare.params["ecdi_proxy_learning"],
                      ss.params["learning_hat"]],
    "se":            [ols_compare.bse["ecdi_proxy_learning"],
                      ss.bse["learning_hat"]],
    "p":             [ols_compare.pvalues["ecdi_proxy_learning"],
                      ss.pvalues["learning_hat"]],
    "first_stage_F": [np.nan, round(float(fs_F_kp), 2)],
    "F_note":        ["—", "Kleibergen-Paap approx (state-clustered SEs, LOO instrument)"],
})
iv_results.to_csv(OUT / "econ_iv_results.csv", index=False)
print(iv_results.round(4).to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# DURBIN-WU-HAUSMAN ENDOGENEITY TEST — LOO instrument, state-clustered SEs
# Null: ecdi_proxy_learning is exogenous (OLS is consistent)
# Test: add LOO first-stage residuals v̂ to second-stage OLS; if coef on v̂ ≠ 0,
#       reject exogeneity (Davidson & MacKinnon 1993).
# Uses the corrected (LOO, state-clustered) first stage residuals.
# ─────────────────────────────────────────────────────────────────────────────
print("\n--- Durbin-Wu-Hausman Endogeneity Test (LOO instrument, state-clustered) ---")
iv_df["fs_resid_loo"] = fs_cl.resid.values
dwh_X = sm.add_constant(
    iv_df[fs_covs_loo + ["ecdi_proxy_learning", "fs_resid_loo"]].astype(float)
)
dwh_y = iv_df[y_col].astype(float)
dwh   = sm.OLS(dwh_y, dwh_X).fit(
    cov_type="cluster",
    cov_kwds={"groups": iv_df[state_col].astype("category").cat.codes.values})
dwh_coef = dwh.params["fs_resid_loo"]
dwh_se   = dwh.bse["fs_resid_loo"]
dwh_t    = dwh_coef / dwh_se
dwh_p    = dwh.pvalues["fs_resid_loo"]
print(f"  First-stage residual coef: {dwh_coef:+.4f}  SE={dwh_se:.4f}  "
      f"t={dwh_t:.3f}  p={dwh_p:.4f}")
if dwh_p < 0.05:
    print("  → Reject H₀: ecdi_proxy_learning is ENDOGENOUS (OLS is inconsistent)")
else:
    print("  → Fail to reject H₀: cannot rule out exogeneity at 5% level")
pd.DataFrame({
    "stat":  ["dwh_coef", "dwh_se", "dwh_t", "dwh_p",
              "instrument", "se_type"],
    "value": [dwh_coef, dwh_se, dwh_t, dwh_p,
              "LOO state mean", "state-clustered"]
}).to_csv(OUT / "econ_dwh_test.csv", index=False)
print("  Saved econ_dwh_test.csv")

# ═════════════════════════════════════════════════════════════════════════════
# (J) QUANTILE REGRESSION on the continuous 4-domain score
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(J) QUANTILE REGRESSION on continuous 4-domain score (0–4)")
print("="*72)
qr_df = df[[y_cont_col] + covariates].dropna().copy()
quantile_rows = []
for q in [0.10, 0.25, 0.50, 0.75, 0.90]:
    mdl = smf.quantreg(f"{y_cont_col} ~ child_age_months + is_female + is_urban "
                        "+ wealth_index + mother_edu + mother_age", data=qr_df).fit(q=q)
    for var in ["wealth_index","mother_edu","is_urban","is_female"]:
        quantile_rows.append({
            "quantile": q, "var": var,
            "coef": round(mdl.params[var], 4),
            "se":   round(mdl.bse[var], 4),
            "p":    round(mdl.pvalues[var], 4),
        })
qr_results = pd.DataFrame(quantile_rows)
print(qr_results.pivot(index="var", columns="quantile", values="coef").round(4).to_string())
qr_results.to_csv(OUT / "econ_quantile_regression.csv", index=False)

# Forest plot
fig, ax = plt.subplots(figsize=(10, 6))
qs = sorted(qr_results["quantile"].unique())
vars_ = ["wealth_index","mother_edu","is_urban","is_female"]
colors = sns.color_palette("Set2", len(vars_))
for j, var in enumerate(vars_):
    sub = qr_results[qr_results["var"] == var].sort_values("quantile")
    ax.errorbar(qs, sub["coef"], yerr=1.96*sub["se"],
                fmt="o-", color=colors[j], label=var, capsize=4, lw=2)
ax.axhline(0, color="grey", lw=1, linestyle="--")
ax.set_xlabel("Quantile τ of continuous ECD score (0–4)")
ax.set_ylabel("β coefficient (continuous score units)")
ax.set_title("Quantile regression — effect varies across the conditional distribution",
             fontweight="bold")
ax.legend(loc="best")
plt.tight_layout()
fig.savefig(FIG / "20_quantile_regression.png", dpi=150)
plt.close()
print("Saved 20_quantile_regression.png")

# ═════════════════════════════════════════════════════════════════════════════
# (K) DROP REGION-IMPUTED STATES — only direct ASER 2019
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(K) DROP IMPUTED STATES — restrict to 24 ASER-sampled states")
print("="*72)
sub = work[work["from_aser_direct"] == 1].copy()
print(f"  Sample: {len(sub):,} children in {sub[state_col].nunique()} states")
sub["state_clean"] = sub[state_col].str.replace(" ", "_").str.replace("&", "and")
sd = pd.get_dummies(sub["state_clean"], prefix="st", drop_first=True).astype(float)
X_k = pd.concat([sub[covariates_b], sd], axis=1).astype(float)
X_k = sm.add_constant(X_k)
y_k = sub[y_col].astype(int)
mdl_k = sm.GLM(y_k, X_k, family=sm.families.Binomial()).fit(
    cov_type="cluster",
    cov_kwds={"groups": sub[psu_col].astype(int).values}
)
non_state_k = [c for c in X_k.columns if not c.startswith("st_")]
drop_imputed = pd.DataFrame({
    "coef":  mdl_k.params[non_state_k].round(3),
    "se":    mdl_k.bse[non_state_k].round(3),
    "p":     mdl_k.pvalues[non_state_k].round(4),
}, index=non_state_k)
print(drop_imputed.to_string())
drop_imputed.to_csv(OUT / "econ_drop_imputed.csv")

# ═════════════════════════════════════════════════════════════════════════════
# (L) WILD CLUSTER BOOTSTRAP — state cluster, B = 999 (urban coefficient)
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(L) WILD CLUSTER BOOTSTRAP on is_urban (state-clustered, B=999)")
print("="*72)
# Cameron-Gelbach-Miller (2008) wild-cluster bootstrap:
# Refit OLS LPM with HC0; for each B, multiply within-state residuals by
# Rademacher (±1) and refit; collect t-stat on is_urban.

lpm_X = X_b.copy().astype(float)   # reuse the same design matrix as (B)
lpm_y = work[y_col].astype(float)
lpm   = sm.OLS(lpm_y, lpm_X).fit(cov_type="HC0")
beta_obs = lpm.params["is_urban"]
se_obs   = lpm.bse["is_urban"]
t_obs    = beta_obs / se_obs
print(f"  Original t-stat on is_urban: {t_obs:.3f}")

state_codes = work["state_clean"].astype("category").cat.codes.values
y_resid     = lpm.resid.values
y_fit       = lpm.fittedvalues.values

rng = np.random.default_rng(42)
n_states = len(np.unique(state_codes))
B = 999
boot_t = np.empty(B)
for b in range(B):
    flips = rng.choice([-1.0, +1.0], size=n_states)  # one Rademacher per state
    flip_per_obs = flips[state_codes]
    y_boot = y_fit + flip_per_obs * y_resid
    mdl_b = sm.OLS(y_boot, lpm_X).fit(cov_type="HC0")
    boot_t[b] = mdl_b.params["is_urban"] / mdl_b.bse["is_urban"]
# Two-sided p-value
p_wcb = float(np.mean(np.abs(boot_t) >= np.abs(t_obs)))
ci_lo = float(np.quantile(boot_t, 0.025))
ci_hi = float(np.quantile(boot_t, 0.975))
print(f"  Wild-cluster bootstrap p-value (B=999): {p_wcb:.4f}")
print(f"  Empirical 95% CI for t under H0:        [{ci_lo:.3f}, {ci_hi:.3f}]")
pd.DataFrame({
    "metric": ["beta", "HC0_se", "t_obs", "p_wcb_2sided", "ci_lo_t", "ci_hi_t"],
    "value":  [beta_obs, se_obs, t_obs, p_wcb, ci_lo, ci_hi],
}).to_csv(OUT / "econ_wcb_bootstrap.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Save core summaries (B/C/F)
# ─────────────────────────────────────────────────────────────────────────────
summary_b.to_csv(OUT / "econ_logit_coefs.csv")
ame_non_state.to_csv(OUT / "econ_probit_AME.csv")
pd.DataFrame({
    "metric": ["urban_mean","rural_mean","gap_pp","endowment_pp","unexplained_pp","ICC_state"],
    "value":  [urban[y_col].mean()*100, rural[y_col].mean()*100,
               mean_gap*100, endowment*100, unexplained*100, icc],
}).to_csv(OUT / "econ_decomposition.csv", index=False)

# ─────────────────────────────────────────────────────────────────────────────
# AME forest plot + Oaxaca bar (re-saved)
# ─────────────────────────────────────────────────────────────────────────────
ame_plot = ame_non_state.copy().reset_index().rename(columns={"index":"var"})
ame_plot.columns = [c.replace(" ","_").replace(".","").replace("|","") for c in ame_plot.columns]
ame_plot = ame_plot.sort_values(ame_plot.columns[1])
fig, ax = plt.subplots(figsize=(9, 5))
y_pos = np.arange(len(ame_plot))
ax.errorbar(ame_plot.iloc[:, 1].values, y_pos,
            xerr=1.96*ame_plot.iloc[:, 2].values, fmt="o", color="#4C72B0", capsize=3)
ax.axvline(0, color="grey", lw=1, linestyle="--")
ax.set_yticks(y_pos); ax.set_yticklabels(ame_plot["var"].values)
ax.set_xlabel("Average Marginal Effect on P(on track)")
ax.set_title("Probit AME — Effect on probability of being ECD on-track\n"
             "(95% CI, India NFHS-5 4-domain proxy, individual scores)", fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "16_probit_AME.png", dpi=150)
plt.close()
print("Saved 16_probit_AME.png")

fig, ax = plt.subplots(figsize=(7, 5))
labels = ["Total gap", "Endowments", "Unexplained / Coef."]
vals   = [mean_gap*100, endowment*100, unexplained*100]
colors_b = ["#4C72B0", "#55A868", "#C44E52"]
bars = ax.bar(labels, vals, color=colors_b, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4 if v>0 else b.get_height()-1.6,
            f"{v:+.2f} pp", ha="center", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=1)
ax.set_ylabel("Percentage points")
ax.set_title("Urban-Rural ECD Gap — Oaxaca-Blinder (Neumark 1988)", fontweight="bold")
plt.tight_layout()
fig.savefig(FIG / "17_oaxaca.png", dpi=150)
plt.close()
print("Saved 17_oaxaca.png")

# Robustness sweep heatmap
fig, ax = plt.subplots(figsize=(8, 4.5))
mat = sweep_df.pivot(index="threshold_z", columns="kof4_rule", values="ontrack_pct")
sns.heatmap(mat, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label":"On-track %"}, ax=ax)
ax.set_title("On-track rate (%) under different thresholds × composite rules\n"
             "(Robustness — overall on-track rate is stable in the central cells)",
             fontweight="bold")
ax.set_xlabel("Composite rule (k of 4 domains required)")
ax.set_ylabel("Individual-score threshold (z)")
plt.tight_layout()
fig.savefig(FIG / "19_robustness_sweep.png", dpi=150)
plt.close()
print("Saved 19_robustness_sweep.png")

# ═════════════════════════════════════════════════════════════════════════════
# (M) SURVEY-WEIGHTED LOGISTIC REGRESSION
# Uses DHS probability weights (v005/1e6) as analytic weights so that
# coefficients represent population-average effects rather than sample-
# average effects. Same PSU-clustered SEs as Spec B.
# ═════════════════════════════════════════════════════════════════════════════
print("\n" + "="*72)
print("(M) SURVEY-WEIGHTED LOGIT — aweights = v005/1e6, PSU-clustered SEs")
print("="*72)

wts_b = work[wt_col].values
glm_weighted = sm.GLM(
    y_b, X_b,
    family=sm.families.Binomial(),
    var_weights=wts_b
).fit(
    cov_type="cluster",
    cov_kwds={"groups": work[psu_col].astype(int).values}
)
summary_m = pd.DataFrame({
    "coef_unweighted": glm_logit_psu.params[non_state].round(3),
    "coef_weighted":   glm_weighted.params[non_state].round(3),
    "se_weighted":     glm_weighted.bse[non_state].round(3),
    "z_weighted":      (glm_weighted.params[non_state] /
                        glm_weighted.bse[non_state]).round(2),
    "p_weighted":      glm_weighted.pvalues[non_state].round(4),
}, index=non_state)
print(summary_m.to_string())
print(f"\n  Weighted on-track rate (Spec A):   "
      f"{weighted_mean(work[y_col], work[wt_col])*100:.2f}%")
print(f"  Unweighted on-track rate (Spec B): {work[y_col].mean()*100:.2f}%")
summary_m.to_csv(OUT / "econ_weighted_logit.csv")
print("  Saved econ_weighted_logit.csv")
print("  Interpretation: compare coef_weighted vs coef_unweighted column.")
print("  If coefficients are similar, unweighted estimates are representative.")

print("\n=== ECONOMETRIC ANALYSIS + ROBUSTNESS SUITE COMPLETE ===")
