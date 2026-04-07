"""
India NFHS-5 — Supervised ML: Predicting ECD Proxy Composite
=============================================================
Outcome: ecdi_proxy_composite (1 = on track in BOTH physical and learning
         proxy domains; 0 = not on track in one or both)

Models:
  1. Logistic Regression (baseline, interpretable)
  2. HistGradientBoostingClassifier (sklearn gradient boosting)

Evaluation:
  - 5-fold stratified cross-validation (ROC-AUC, F1)
  - Cross-validated ROC curve (honest out-of-fold predictions)
  - Confusion matrix on OOF predictions

Interpretability:
  - SHAP values via PermutationExplainer (subsample=500)
  - Global importance bar chart
  - Beeswarm plot

Note on class imbalance:
  class_weight="balanced" used in both models.

Note on proxy outcome:
  The composite outcome covers only 2 of 4 ECDI2030 domains. Findings
  should be interpreted as predictors of physical non-stunting AND regular
  ECCE attendance, not full ECDI2030 on-track status.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay, confusion_matrix)
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
import shap

OUT = Path("/Users/prachi/india_nfhs5_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

# ── Load data ─────────────────────────────────────────────────────────────────

df = pd.read_parquet("/Users/prachi/india_nfhs5_project/india_scored.parquet")
df = df[df["ecdi_proxy_composite"].notna()].copy()
print(f"Analysis sample: {len(df):,} children with complete proxy composite")

# ── 1. Feature engineering ────────────────────────────────────────────────────

# State one-hot (Kerala is reference / dropped — generally best-performing state)
# First rename state col to ensure no spaces
df["state_clean"] = df["state"].str.replace(" ", "_").str.replace("&","and")
df = pd.get_dummies(df, columns=["state_clean"], prefix="st", drop_first=True, dummy_na=False)

# ── 2. Feature matrix ─────────────────────────────────────────────────────────

base_features = [
    "child_age_months",    # age in months
    "is_female",           # sex (female=1)
    "is_urban",            # urban area
    "wealth_index",        # 1–5 wealth quintile
    # HAZ and WAZ deliberately excluded: HAZ > -2 is a direct component of the
    # outcome (ecdi_proxy_physical), so including it would create a circular
    # predictor that partially determines the label it is trying to predict.
    # WAZ is also excluded for the same reason (highly correlated with HAZ and
    # partially overlapping with the physical domain definition).
    "mother_edu",          # mother's education (0–5 ordinal)
    "mother_higher_ed",    # mother: complete secondary or higher
    "mother_age",          # mother's age in years
    # anganwadi_any excluded: it equals 1 for all children in the analysis
    # sample (s562 is non-null only when s558='yes'), so it is a constant
    # feature with zero predictive variance.
]

state_dummies = [c for c in df.columns if c.startswith("st_")]
feature_cols = base_features + state_dummies
feature_cols = [c for c in feature_cols if c in df.columns]

print(f"Features: {len(feature_cols)}")

X = df[feature_cols].copy().astype(float)
y = df["ecdi_proxy_composite"].astype(int)
print(f"On-track rate: {y.mean()*100:.1f}%  |  "
      f"Class 0: {(y==0).sum():,}  Class 1: {(y==1).sum():,}")

# Imputed matrix (for SHAP)
imputer = SimpleImputer(strategy="median")
X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

# ── 3. Cross-validation setup ─────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ── 4. Logistic Regression baseline ──────────────────────────────────────────

lr_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale",  StandardScaler()),
    ("model",  LogisticRegression(max_iter=1000, random_state=42,
                                  C=1.0, class_weight="balanced")),
])

lr_auc = cross_val_score(lr_pipe, X, y, cv=cv, scoring="roc_auc")
lr_f1  = cross_val_score(lr_pipe, X, y, cv=cv, scoring="f1")
lr_oof = cross_val_predict(lr_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"\nLogistic Regression (5-fold CV)")
print(f"  AUC: {lr_auc.mean():.3f} ± {lr_auc.std():.3f}")
print(f"  F1:  {lr_f1.mean():.3f} ± {lr_f1.std():.3f}")

# ── 5. HistGradientBoosting ───────────────────────────────────────────────────

hgb_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("model",  HistGradientBoostingClassifier(
        max_iter=300, max_depth=5, learning_rate=0.05,
        random_state=42, class_weight="balanced",
    )),
])

hgb_auc = cross_val_score(hgb_pipe, X, y, cv=cv, scoring="roc_auc")
hgb_f1  = cross_val_score(hgb_pipe, X, y, cv=cv, scoring="f1")
hgb_oof = cross_val_predict(hgb_pipe, X, y, cv=cv, method="predict_proba")[:, 1]

print(f"\nHistGradientBoosting (5-fold CV)")
print(f"  AUC: {hgb_auc.mean():.3f} ± {hgb_auc.std():.3f}")
print(f"  F1:  {hgb_f1.mean():.3f} ± {hgb_f1.std():.3f}")

# ── 6. Cross-validated ROC curve ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(7, 6))
for oof, label, color in [
    (lr_oof,  f"Logistic Regression (AUC={roc_auc_score(y, lr_oof):.3f})",  "#C44E52"),
    (hgb_oof, f"HistGradientBoosting (AUC={roc_auc_score(y, hgb_oof):.3f})", "#4C72B0"),
]:
    fpr, tpr, _ = roc_curve(y, oof)
    ax.plot(fpr, tpr, lw=2, label=label, color=color)
ax.plot([0,1],[0,1], "k--", lw=1, label="Random classifier")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("Cross-validated ROC — ECD Proxy Composite Prediction\n"
             "India NFHS-5  (out-of-fold, 5-fold CV)", fontweight="bold")
ax.legend(loc="lower right")
plt.tight_layout()
fig.savefig(OUT / "08_roc_curve.png", dpi=150)
plt.close()
print("\nSaved 08_roc_curve.png")

# ── 7. Confusion matrix (OOF, HGB) ───────────────────────────────────────────

hgb_oof_pred = (hgb_oof >= 0.5).astype(int)
cm = confusion_matrix(y, hgb_oof_pred)
fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(cm, display_labels=["Not on track","On track"])
disp.plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Confusion Matrix — HistGradientBoosting (OOF)\nIndia NFHS-5", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "08b_confusion_matrix.png", dpi=150)
plt.close()
print("Saved 08b_confusion_matrix.png")

print("\nClassification Report (OOF predictions — HistGradientBoosting):")
print(classification_report(y, hgb_oof_pred,
                             target_names=["Not on track","On track"]))

# ── 8. Fit final model on full data (for SHAP) ────────────────────────────────

hgb_final = HistGradientBoostingClassifier(
    max_iter=300, max_depth=5, learning_rate=0.05,
    random_state=42, class_weight="balanced",
)
hgb_final.fit(X_imp, y)

# ── 9. SHAP — subsample 500 rows for speed ────────────────────────────────────

print("\nComputing SHAP values (n=500 subsample) …")
rng = np.random.default_rng(42)
idx = rng.choice(len(X_imp), size=min(500, len(X_imp)), replace=False)
X_shap = X_imp.iloc[idx].reset_index(drop=True)

masker   = shap.maskers.Independent(X_shap, max_samples=50)
explainer = shap.PermutationExplainer(hgb_final.predict_proba, masker)
sv_raw   = explainer(X_shap)
sv       = sv_raw[..., 1]   # class-1 (on-track) SHAP values

# Human-readable feature names
feature_labels = {
    "child_age_months":  "Child age (months)",
    "is_female":         "Sex (female=1)",
    "is_urban":          "Urban area",
    "wealth_index":      "Wealth quintile (1–5)",
    "HAZ":               "Height-for-age z-score",
    "WAZ":               "Weight-for-age z-score",
    "mother_edu":        "Mother's education (ordinal)",
    "mother_higher_ed":  "Mother: secondary+ education",
    "mother_age":        "Mother's age (years)",
    "anganwadi_any":     "Any anganwadi benefit received",
}
# State dummies: shorten
def shorten_state(c):
    return "State: " + c.replace("st_","").replace("_"," ").title()

display_names = [feature_labels.get(c, shorten_state(c)) for c in feature_cols]
sv.feature_names = display_names

# SHAP bar chart (non-state features only for clarity)
mean_shap = pd.Series(
    np.abs(sv.values).mean(axis=0),
    index=display_names,
).sort_values(ascending=True)

# Show only non-state features + top 5 state effects
non_state = mean_shap[~mean_shap.index.str.startswith("State: ")]
state_only = mean_shap[mean_shap.index.str.startswith("State: ")].tail(5)
plot_shap = pd.concat([non_state, state_only]).sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 7))
plot_shap.plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_xlabel("Mean |SHAP value| (impact on on-track probability)")
ax.set_title("Feature Importance (SHAP) — ECD Proxy Composite\n"
             "India NFHS-5, HistGradientBoosting, n=500 subsample", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "09_shap_bar.png", dpi=150)
plt.close()
print("Saved 09_shap_bar.png")

# SHAP beeswarm (top 15 features)
# Use full sv but limit to display
fig = plt.figure(figsize=(11, 8))
shap.plots.beeswarm(sv, max_display=15, show=False)
plt.title("SHAP Beeswarm — Direction & Magnitude of Predictors\n"
          "India NFHS-5 ECD Proxy Composite (red=high value, blue=low)",
          fontweight="bold", pad=15)
plt.tight_layout()
fig.savefig(OUT / "10_shap_beeswarm.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 10_shap_beeswarm.png")

# ── 10. Save outputs ──────────────────────────────────────────────────────────

importance_df = (mean_shap
                 .sort_values(ascending=False)
                 .reset_index()
                 .rename(columns={"index": "feature", 0: "mean_abs_shap"}))
importance_df.columns = ["feature", "mean_abs_shap"]
importance_df.to_csv("/Users/prachi/india_nfhs5_project/shap_importance.csv", index=False)
X_imp.to_parquet("/Users/prachi/india_nfhs5_project/X_imputed.parquet", index=False)

# ── 11. Summary table ─────────────────────────────────────────────────────────

print("\n" + "="*60)
print("MODEL COMPARISON — 5-FOLD CROSS-VALIDATED PERFORMANCE")
print("="*60)
print(f"{'Model':<30} {'ROC-AUC':>9} {'F1':>8}")
print("-"*60)
print(f"{'Logistic Regression':<30} "
      f"{lr_auc.mean():>7.3f}±{lr_auc.std():.3f}  "
      f"{lr_f1.mean():>6.3f}±{lr_f1.std():.3f}")
print(f"{'HistGradientBoosting':<30} "
      f"{hgb_auc.mean():>7.3f}±{hgb_auc.std():.3f}  "
      f"{hgb_f1.mean():>6.3f}±{hgb_f1.std():.3f}")
print("\nTop 5 SHAP predictors:")
print(importance_df.head(5).to_string(index=False))
print("\nSaved: shap_importance.csv, X_imputed.parquet")
