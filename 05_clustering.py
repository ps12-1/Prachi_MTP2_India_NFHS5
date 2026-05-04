"""
India NFHS-5 — Unsupervised ML: Clustering Developmental Profiles
==================================================================
Clusters children by their proxy ECD domain scores to identify
distinct developmental profiles.

Domains used for clustering:
  ecdi_proxy_physical    : not stunted (HAZ > -2 SD)     [0/1]
  ecdi_proxy_nutritional : not underweight (WAZ > -2 SD) [0/1]
  ecdi_proxy_learning    : regular anganwadi attendance   [0/1]

Since these are binary, we also include the continuous z-scores
(HAZ, WAZ) to give richer distance information.

Methods:
  1. K-Means on continuous features (HAZ, WAZ + binary proxies)
  2. Silhouette score to choose k
  3. PCA visualisation
  4. Profile characterisation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer

OUT = Path("/Users/prachi/india_nfhs5_project/figures")
OUT.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", font_scale=1.1)

df = pd.read_parquet("/Users/prachi/india_nfhs5_project/india_scored.parquet")

# Use children with all three proxy domain values available
domains = ["ecdi_proxy_physical", "ecdi_proxy_nutritional", "ecdi_proxy_learning"]
domain_labels = ["Physical\n(not stunted)", "Nutritional\n(not underweight)",
                  "Learning\n(regular ECCE)"]

# Also include continuous z-scores for richer clustering
cluster_features = ["HAZ", "WAZ", "ecdi_proxy_learning"]
df_c = df[df[cluster_features].notna().all(axis=1)].copy()
print(f"Complete clustering data: {len(df_c):,} children")

X_raw = df_c[cluster_features].values

# Standardise (z-scores are already on comparable scale; binary learning proxy needs scaling)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

# Full dataset for silhouette — sklearn uses chunked pairwise distances so
# memory stays bounded; runtime is O(n²) but feasible for ~86k rows.
print(f"Running silhouette on full dataset ({len(X):,} rows) …")
X_sample = X

# ── 1. Elbow method + Silhouette scores ───────────────────────────────────────

inertias, sil_scores = [], []
K_range = range(2, 9)

for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sample)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_sample, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(list(K_range), inertias, "o-", color="#4C72B0", linewidth=2)
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia")
ax1.set_title("Elbow Method", fontweight="bold")

ax2.plot(list(K_range), sil_scores, "o-", color="#55A868", linewidth=2)
best_k = list(K_range)[np.argmax(sil_scores)]
ax2.axvline(best_k, color="red", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
ax2.set_xlabel("Number of Clusters (k)")
ax2.set_ylabel("Silhouette Score")
ax2.set_title("Silhouette Score", fontweight="bold")
ax2.legend()

plt.suptitle(f"Optimal Number of Clusters — NFHS-5 ECD Proxy Domains (n={len(X):,})", fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT / "11_cluster_selection.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved 11_cluster_selection.png  (best k={best_k})")

# ── 2. Fit K-Means with optimal k ─────────────────────────────────────────────

K = best_k
km_final = KMeans(n_clusters=K, random_state=42, n_init=20)
df_c["cluster"] = km_final.fit_predict(X)

print(f"\nCluster sizes:")
print(df_c["cluster"].value_counts().sort_index())

# ── 3. Cluster profiles ───────────────────────────────────────────────────────

# Compute mean of binary proxy domains per cluster (in %)
cluster_means_bin = df_c.groupby("cluster")[domains].mean() * 100
print("\nCluster mean proxy on-track rates (%):")
print(cluster_means_bin.round(1))

# Compute mean z-scores per cluster
cluster_means_cont = df_c.groupby("cluster")[["HAZ","WAZ"]].mean()
print("\nCluster mean z-scores:")
print(cluster_means_cont.round(2))

def name_cluster(row):
    """Assign descriptive label based on proxy domain rates."""
    phys  = row["ecdi_proxy_physical"]
    nutr  = row["ecdi_proxy_nutritional"]
    learn = row["ecdi_proxy_learning"]
    if phys >= 70 and learn >= 60:
        return "High overall"
    elif phys < 40 and learn < 30:
        return "Low overall"
    elif phys >= 70 and learn < 30:
        return "Well-nourished, low ECCE"
    elif phys < 40 and learn >= 60:
        return "Stunted, attending ECCE"
    elif learn >= 60:
        return "High ECCE attendance"
    elif phys < 40:
        return "High stunting"
    else:
        return f"Profile {int(row.name) + 1}"

cluster_means_bin["label"] = cluster_means_bin.apply(name_cluster, axis=1)
label_map = cluster_means_bin["label"].to_dict()
df_c["cluster_label"] = df_c["cluster"].map(label_map)

# Bar chart: proxy domain rates per cluster
fig, ax = plt.subplots(figsize=(11, 6))
x = np.arange(len(domain_labels))
width = 0.8 / K
colors = sns.color_palette("Set2", K)

for i, (cidx, row) in enumerate(cluster_means_bin.iterrows()):
    vals = row[domains].values
    n = (df_c["cluster"] == cidx).sum()
    label_name = row["label"]
    ax.bar(x + i*width, vals, width,
           label=f"{label_name} (n={n:,})", color=colors[i], alpha=0.85)

ax.set_xticks(x + width*(K-1)/2)
ax.set_xticklabels(domain_labels)
ax.set_ylabel("% On Track")
ax.set_title(f"ECD Proxy Domain Profiles by Cluster (k={K}) — India NFHS-5", fontweight="bold")
ax.legend(loc="lower right", fontsize=8)
ax.set_ylim(0, 115)
plt.tight_layout()
fig.savefig(OUT / "12_cluster_profiles.png", dpi=150)
plt.close()
print("Saved 12_cluster_profiles.png")

# ── 4. PCA visualisation ─────────────────────────────────────────────────────

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)
df_c["pca1"] = X_pca[:, 0]
df_c["pca2"] = X_pca[:, 1]

fig, ax = plt.subplots(figsize=(9, 7))
for i, (cidx, row) in enumerate(cluster_means_bin.iterrows()):
    mask = df_c["cluster"] == cidx
    # Subsample for plotting speed (large dataset)
    sub = df_c[mask].sample(min(2000, mask.sum()), random_state=42)
    ax.scatter(sub["pca1"], sub["pca2"],
               c=[colors[i]], alpha=0.3, s=10, label=row["label"])

centroids_pca = pca.transform(km_final.cluster_centers_)
ax.scatter(centroids_pca[:,0], centroids_pca[:,1], c="black",
           marker="X", s=200, zorder=5, label="Centroids")

ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
ax.set_title("PCA — NFHS-5 ECD Proxy Developmental Profiles", fontweight="bold")
ax.legend(markerscale=3, fontsize=8)
plt.tight_layout()
fig.savefig(OUT / "13_cluster_pca.png", dpi=150)
plt.close()
print("Saved 13_cluster_pca.png")

# ── 5. Cluster characteristics ────────────────────────────────────────────────

char_cols = ["wealth_index","is_urban","child_age_months","HAZ","WAZ","mother_edu"]
char_cols = [c for c in char_cols if c in df_c.columns]

print("\nCluster characteristics (means):")
print(df_c.groupby("cluster_label")[char_cols].mean().round(2))

print("\nCluster × Urban/Rural:")
print(pd.crosstab(df_c["cluster_label"], df_c["v025"],
                  normalize="index").round(2))

print("\nCluster × Wealth quintile:")
print(pd.crosstab(df_c["cluster_label"], df_c["v190"],
                  normalize="index").round(2))

# ── 6. Save ───────────────────────────────────────────────────────────────────

df_c.to_parquet("/Users/prachi/india_nfhs5_project/india_clustered.parquet", index=False)
print(f"\nSaved india_clustered.parquet  ({len(df_c):,} rows)")
