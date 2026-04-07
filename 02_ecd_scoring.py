"""
India NFHS-5 — ECD Proxy Scoring
==================================
NFHS-5 does not contain direct ECDI2030 cognitive/behavioural assessment items
(EC6–EC15). This script constructs the best available proxy for each ECDI2030
domain using NFHS-5 indicators, following the approach in the literature on
ECD measurement using DHS-type surveys.

Proxy Domain Mapping
--------------------
ECDI2030 Domain        | NFHS-5 Proxy                         | Coverage
-----------------------|--------------------------------------|----------
Literacy-Numeracy      | (no valid proxy available)           | —
Physical               | Not stunted: HAZ > −2 SD             | 94%
Learning               | Regular anganwadi/ICDS attendance    | 69%
Socio-emotional        | (no valid proxy available)           | —

Two-domain proxy composite (on track if BOTH domains are on track):
  ecdi_proxy_physical  = 1 if HAZ > −2 SD
  ecdi_proxy_learning  = 1 if regular anganwadi attendance (s562='regularly')
  ecdi_proxy_composite = 1 if BOTH physical AND learning proxies = 1
                        (restricted to children with data on both domains)

Additional nutritional indicator (reported separately, not in composite):
  not_underweight      = 1 if WAZ > −2 SD

Limitations (documented):
  - Composite covers only 2 of 4 ECDI2030 domains; not directly comparable
    to Nepal composite
  - HAZ reflects past growth faltering; not a direct motor-skill assessment
  - Anganwadi attendance ≠ learning outcomes; reflects programme access
  - Data collected from different sub-samples (HAZ from measurement roster;
    anganwadi from youngest-child questionnaire) → composite sample smaller

Reference: Rajaratnam et al. (2018); Headey & Hoddinott (2015);
           WHO Multicentre Growth Reference Study Group (2006)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("/Users/prachi/india_nfhs5_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_parquet("/Users/prachi/india_nfhs5_project/india_merged.parquet")
print(f"Loaded: {len(df):,} children aged 24–59 months")

# ── 1. Physical proxy: not stunted ───────────────────────────────────────────

df["ecdi_proxy_physical"] = np.where(
    df["HAZ"].isnull(), np.nan,
    (df["HAZ"] > -2.0).astype(float)
)

# ── 2. Nutritional proxy: not underweight (reported separately) ───────────────

df["ecdi_proxy_nutritional"] = np.where(
    df["WAZ"].isnull(), np.nan,
    (df["WAZ"] > -2.0).astype(float)
)

# ── 3. Learning proxy: regular anganwadi attendance ──────────────────────────

# s562: regularly / occasionally / not at all / NaN (not enrolled)
# "regularly" = on track for structured early learning
df["ecdi_proxy_learning"] = np.where(
    df["s562"].isnull(), np.nan,
    (df["s562"] == "regularly").astype(float)
)

# ── 4. Two-domain proxy composite ────────────────────────────────────────────

# Only children with BOTH HAZ and anganwadi data
has_both = df["ecdi_proxy_physical"].notna() & df["ecdi_proxy_learning"].notna()
df["ecdi_proxy_composite"] = np.where(
    ~has_both, np.nan,
    ((df["ecdi_proxy_physical"] == 1) & (df["ecdi_proxy_learning"] == 1)).astype(float)
)

# ── 5. Report ─────────────────────────────────────────────────────────────────

print("\n" + "="*60)
print("NFHS-5 ECD PROXY SCORES — India")
print("="*60)

n_phys  = df["ecdi_proxy_physical"].notna().sum()
n_learn = df["ecdi_proxy_learning"].notna().sum()
n_comp  = df["ecdi_proxy_composite"].notna().sum()

print(f"\nProxy domain on-track rates:")
print(f"  Physical (not stunted, HAZ>-2):    "
      f"{df['ecdi_proxy_physical'].mean()*100:.1f}%  (n={n_phys:,})")
print(f"  Nutritional (not underweight):     "
      f"{df['ecdi_proxy_nutritional'].mean()*100:.1f}%  "
      f"(n={df['ecdi_proxy_nutritional'].notna().sum():,})")
print(f"  Learning (regular anganwadi):      "
      f"{df['ecdi_proxy_learning'].mean()*100:.1f}%  (n={n_learn:,})")
print(f"\nProxy composite (both physical+learning on track):")
print(f"  On track: {df['ecdi_proxy_composite'].mean()*100:.1f}%  (n={n_comp:,})")

print("\nBy sex:")
print(df.groupby("b4")["ecdi_proxy_composite"].agg(["mean","count"]).round(3))

print("\nBy area (urban/rural):")
print(df.groupby("v025")["ecdi_proxy_composite"].agg(["mean","count"]).round(3))

print("\nBy wealth quintile:")
print(df.groupby("v190")["ecdi_proxy_composite"].agg(["mean","count"]).round(3))

print("\nBy mother's education:")
print(df.groupby("v149")["ecdi_proxy_composite"].agg(["mean","count"]).round(3))

# Top 10 states by on-track rate
print("\nTop 10 states — composite on-track rate:")
state_rates = (df.groupby("state")["ecdi_proxy_composite"]
               .agg(["mean","count"])
               .query("count >= 100")
               .sort_values("mean", ascending=False))
print(state_rates.head(10).round(3))

print("\nBottom 10 states — composite on-track rate:")
print(state_rates.tail(10).round(3))

# ── 6. Visualisations ─────────────────────────────────────────────────────────

# Domain on-track rates bar chart
domains_data = {
    "Physical\n(not stunted)": df["ecdi_proxy_physical"].mean() * 100,
    "Nutritional\n(not underweight)": df["ecdi_proxy_nutritional"].mean() * 100,
    "Learning\n(regular anganwadi)": df["ecdi_proxy_learning"].mean() * 100,
    "Proxy composite\n(physical+learning)": df["ecdi_proxy_composite"].mean() * 100,
}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(domains_data.keys(), domains_data.values(),
              color=["#4C72B0","#55A868","#C44E52","#8172B2"], alpha=0.85)
for bar, val in zip(bars, domains_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{val:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("% On Track")
ax.set_ylim(0, 110)
ax.set_title("NFHS-5 ECD Proxy On-Track Rates — India (24–59 months)\n"
             "(ECDI2030 proxies; see methodology note)", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "01_proxy_domain_rates.png", dpi=150)
plt.close()
print("\nSaved 01_proxy_domain_rates.png")

# HAZ distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, col, label, cutoff in [
    (axes[0], "HAZ", "Height-for-Age Z-score (HAZ)", -2),
    (axes[1], "WAZ", "Weight-for-Age Z-score (WAZ)", -2),
]:
    data = df[col].dropna()
    axes_ref = ax
    axes_ref.hist(data, bins=80, color="#4C72B0", alpha=0.7, edgecolor="none")
    axes_ref.axvline(cutoff, color="red", lw=2, linestyle="--",
                     label=f"Cut-off ({cutoff} SD)")
    axes_ref.set_xlabel(label)
    axes_ref.set_ylabel("Count")
    stunted_pct = (data < cutoff).mean() * 100
    axes_ref.set_title(f"{label}\nMean={data.mean():.2f} | "
                       f"Below {cutoff}SD: {stunted_pct:.1f}%", fontweight="bold")
    axes_ref.legend()

plt.suptitle("Anthropometric Distributions — NFHS-5 India (24–59 months)", y=1.02, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "02_anthropometric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_anthropometric_distributions.png")

# Composite rate by wealth quintile
fig, ax = plt.subplots(figsize=(8, 5))
wq_data = (df.groupby("v190")["ecdi_proxy_composite"].mean() * 100).reindex(
    ["poorest","poorer","middle","richer","richest"]
)
wq_data.plot(kind="bar", ax=ax, color="#4C72B0", alpha=0.85)
ax.set_xlabel("Wealth Quintile")
ax.set_ylabel("% On Track (proxy composite)")
ax.set_title("ECD Proxy Composite by Wealth Quintile — NFHS-5 India", fontweight="bold")
ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
for i, v in enumerate(wq_data):
    if not np.isnan(v):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "03_composite_by_wealth.png", dpi=150)
plt.close()
print("Saved 03_composite_by_wealth.png")

# ── 7. Save ───────────────────────────────────────────────────────────────────

df.to_parquet("/Users/prachi/india_nfhs5_project/india_scored.parquet", index=False)
print(f"\nSaved india_scored.parquet  ({len(df):,} rows)")
