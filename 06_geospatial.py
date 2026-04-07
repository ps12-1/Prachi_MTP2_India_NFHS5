"""
India NFHS-5 — Geospatial Analysis
=====================================
State-level ECD proxy composite rates for India.

Steps:
  1. Aggregate proxy composite by state (36 states/UTs)
  2. Bar chart of state-level on-track rates (sorted)
  3. Moran's I spatial autocorrelation using hand-coded adjacency
     (no shapefile required — state-level adjacency matrix)

Note: Geographic boundary shapefiles are not available in this pipeline
(the DHS Geographic Data file could not be downloaded). State-level analysis
uses the v024 state variable from the KR file.

Moran's I: adjacency is defined by land-border sharing between Indian states.
           Only includes major states (excluding small UTs that share no land
           borders for practical purposes).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

OUT = Path("/Users/prachi/india_nfhs5_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_parquet("/Users/prachi/india_nfhs5_project/india_scored.parquet")

# ── 1. State-level aggregation ───────────────────────────────────────────────

state_df = (df.groupby("state")
            .agg(
                n_total   = ("ecdi_proxy_composite", "count"),
                on_track  = ("ecdi_proxy_composite", "mean"),
                n_physical = ("ecdi_proxy_physical", "count"),
                phys_rate  = ("ecdi_proxy_physical", "mean"),
                n_learning = ("ecdi_proxy_learning", "count"),
                learn_rate = ("ecdi_proxy_learning", "mean"),
                mean_HAZ   = ("HAZ", "mean"),
                mean_WAZ   = ("WAZ", "mean"),
            )
            .reset_index()
)
state_df["on_track_pct"] = state_df["on_track"] * 100
state_df["phys_pct"]     = state_df["phys_rate"] * 100
state_df["learn_pct"]    = state_df["learn_rate"] * 100

# Filter to states with ≥ 50 children in composite sample
state_df = state_df[state_df["n_total"] >= 50].copy()
print(f"States with ≥50 observations: {len(state_df)}")
print(state_df[["state","n_total","on_track_pct","phys_pct","learn_pct"]]
      .sort_values("on_track_pct", ascending=False).to_string(index=False))

# ── 2. Bar chart — state-level composite rates ────────────────────────────────

state_sorted = state_df.sort_values("on_track_pct", ascending=True)

fig, ax = plt.subplots(figsize=(10, 12))
colors_bar = ["#4C72B0" if v >= state_df["on_track_pct"].median() else "#C44E52"
              for v in state_sorted["on_track_pct"]]
ax.barh(state_sorted["state"], state_sorted["on_track_pct"],
        color=colors_bar, alpha=0.85)
ax.axvline(state_df["on_track_pct"].median(), color="black",
           linestyle="--", lw=1.5, label=f"Median ({state_df['on_track_pct'].median():.1f}%)")
ax.set_xlabel("% On Track (proxy composite)")
ax.set_title("ECD Proxy Composite by State — India NFHS-5\n"
             "(Physical + Learning proxy domains)", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "06_state_rates.png", dpi=150)
plt.close()
print("\nSaved 06_state_rates.png")

# Physical domain bar chart
state_phys = state_df.sort_values("phys_pct", ascending=True)
fig, ax = plt.subplots(figsize=(10, 12))
ax.barh(state_phys["state"], state_phys["phys_pct"],
        color="#55A868", alpha=0.85)
ax.axvline(state_df["phys_pct"].median(), color="black", linestyle="--", lw=1.5,
           label=f"Median ({state_df['phys_pct'].median():.1f}%)")
ax.set_xlabel("% Not Stunted (HAZ > -2 SD)")
ax.set_title("Physical Proxy (Non-Stunting) by State — India NFHS-5", fontweight="bold")
ax.legend()
plt.tight_layout()
fig.savefig(OUT / "06b_state_physical.png", dpi=150)
plt.close()
print("Saved 06b_state_physical.png")

# ── 3. Scatter: physical vs learning proxy by state ──────────────────────────

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(state_df["phys_pct"], state_df["learn_pct"],
           s=state_df["n_total"] / 20, alpha=0.7, color="#4C72B0")
for _, row in state_df.iterrows():
    ax.annotate(row["state"].title()[:15],
                (row["phys_pct"], row["learn_pct"]),
                fontsize=7, alpha=0.75,
                xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("Physical Proxy — % Not Stunted")
ax.set_ylabel("Learning Proxy — % Regular Anganwadi Attendance")
ax.set_title("Physical vs Learning ECD Proxy by State — India NFHS-5\n"
             "(bubble size = sample size)", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "07_state_scatter.png", dpi=150)
plt.close()
print("Saved 07_state_scatter.png")

# ── 4. Moran's I (hand-coded state adjacency) ─────────────────────────────────

# Land-border adjacency for major Indian states
# Source: geographical adjacency; UTs/island territories omitted
adjacency = {
    "jammu & kashmir":   ["himachal pradesh","ladakh","punjab"],
    "ladakh":            ["himachal pradesh","jammu & kashmir"],
    "himachal pradesh":  ["jammu & kashmir","ladakh","punjab","haryana","uttarakhand","uttar pradesh"],
    "punjab":            ["jammu & kashmir","himachal pradesh","haryana","rajasthan"],
    "haryana":           ["punjab","himachal pradesh","uttarakhand","uttar pradesh","rajasthan","nct of delhi"],
    "nct of delhi":      ["haryana","uttar pradesh","rajasthan"],
    "rajasthan":         ["punjab","haryana","nct of delhi","uttar pradesh","madhya pradesh","gujarat"],
    "uttarakhand":       ["himachal pradesh","haryana","uttar pradesh"],
    "uttar pradesh":     ["uttarakhand","haryana","nct of delhi","rajasthan","madhya pradesh",
                          "chhattisgarh","jharkhand","bihar"],
    "bihar":             ["uttar pradesh","jharkhand","west bengal"],
    "jharkhand":         ["uttar pradesh","bihar","west bengal","odisha","chhattisgarh"],
    "west bengal":       ["bihar","jharkhand","odisha","sikkim","assam","meghalaya"],
    "sikkim":            ["west bengal"],
    "assam":             ["west bengal","meghalaya","arunachal pradesh","nagaland","manipur","mizoram","tripura"],
    "meghalaya":         ["west bengal","assam"],
    "arunachal pradesh": ["assam","nagaland"],
    "nagaland":          ["assam","arunachal pradesh","manipur"],
    "manipur":           ["assam","nagaland","mizoram"],
    "mizoram":           ["assam","manipur","tripura"],
    "tripura":           ["assam","mizoram"],
    "odisha":            ["west bengal","jharkhand","chhattisgarh","andhra pradesh"],
    "chhattisgarh":      ["uttar pradesh","jharkhand","odisha","andhra pradesh","telangana","madhya pradesh"],
    "madhya pradesh":    ["rajasthan","uttar pradesh","chhattisgarh","maharashtra","gujarat"],
    "gujarat":           ["rajasthan","madhya pradesh","maharashtra"],
    "maharashtra":       ["gujarat","madhya pradesh","chhattisgarh","telangana","andhra pradesh","karnataka","goa"],
    "goa":               ["maharashtra","karnataka"],
    "karnataka":         ["goa","maharashtra","andhra pradesh","telangana","kerala","tamil nadu"],
    "telangana":         ["maharashtra","chhattisgarh","andhra pradesh","karnataka"],
    "andhra pradesh":    ["odisha","chhattisgarh","telangana","karnataka","tamil nadu"],
    "tamil nadu":        ["andhra pradesh","karnataka","kerala"],
    "kerala":            ["karnataka","tamil nadu"],
}

# Build weight matrix
state_rate_dict = state_df.set_index("state")["on_track_pct"].to_dict()
states_in_adj = [s for s in adjacency.keys() if s in state_rate_dict]

n = len(states_in_adj)
W = np.zeros((n, n))
for i, s1 in enumerate(states_in_adj):
    for j, s2 in enumerate(states_in_adj):
        if s2 in adjacency.get(s1, []):
            W[i, j] = 1

# Row-standardise
row_sums = W.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # avoid division by zero
W_std = W / row_sums

y_vec = np.array([state_rate_dict[s] for s in states_in_adj])
y_mean = y_vec.mean()
y_dev  = y_vec - y_mean

# Moran's I
numerator   = (y_dev @ W_std @ y_dev)
denominator = (y_dev @ y_dev)
I = (n / W.sum()) * (numerator / denominator)

# Permutation test
np.random.seed(42)
n_perm = 999
I_perm = []
for _ in range(n_perm):
    y_perm = np.random.permutation(y_vec)
    y_perm_dev = y_perm - y_perm.mean()
    I_perm.append((n / W.sum()) *
                  (y_perm_dev @ W_std @ y_perm_dev) / (y_perm_dev @ y_perm_dev))
p_value = (np.sum(np.array(I_perm) >= I) + 1) / (n_perm + 1)

print(f"\nMoran's I Spatial Autocorrelation")
print(f"  States in adjacency matrix: {n}")
print(f"  Moran's I = {I:.4f}")
print(f"  Permutation p-value = {p_value:.3f} (999 permutations)")
if p_value < 0.05:
    print("  → Significant positive spatial clustering of ECD proxy rates")
else:
    print("  → No significant spatial autocorrelation detected")

# Moran scatter plot
fig, ax = plt.subplots(figsize=(7, 6))
Wy = W_std @ y_vec
ax.scatter(y_vec, Wy, alpha=0.7, color="#4C72B0")
for i, s in enumerate(states_in_adj):
    ax.annotate(s.title()[:12], (y_vec[i], Wy[i]), fontsize=7, alpha=0.7,
                xytext=(3, 3), textcoords="offset points")
# Regression line
m, b = np.polyfit(y_vec, Wy, 1)
xr = np.linspace(y_vec.min(), y_vec.max(), 100)
ax.plot(xr, m*xr + b, "r-", lw=2)
ax.axvline(y_mean, color="gray", lw=1, linestyle="--")
ax.axhline(np.mean(Wy), color="gray", lw=1, linestyle="--")
ax.set_xlabel("ECD Proxy On-Track Rate (%)")
ax.set_ylabel("Spatially Lagged Rate (row-standardised W)")
ax.set_title(f"Moran Scatter Plot — State ECD Proxy Rates\n"
             f"Moran's I = {I:.4f}, p = {p_value:.3f}", fontweight="bold")
plt.tight_layout()
fig.savefig(OUT / "07b_moran_scatter.png", dpi=150)
plt.close()
print("Saved 07b_moran_scatter.png")

# ── 5. Save state summary ─────────────────────────────────────────────────────

state_df.to_csv("/Users/prachi/india_nfhs5_project/state_ecd_rates.csv", index=False)
print("Saved state_ecd_rates.csv")
