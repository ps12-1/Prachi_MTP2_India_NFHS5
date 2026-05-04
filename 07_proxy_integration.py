"""
India NFHS-5 — State-Level Demographic Proxy Integration  (REAL DATA)
======================================================================
This pipeline merges PARSED external survey data into the NFHS-5 child
record on demographic key v024 (state). Replaces the earlier hardcoded
version. Two real external sources:

1. ASER 2019 'Early Years' — extracted from the public PDF report by
   parse_aser_2019.py (pdfplumber). Provides 24 states (one rural
   district per state, except 2 each in MP and UP). Indicators:
     preschool_age5         — % age-5 children in any pre-school (Anganwadi
                              + Govt pre-primary + Pvt LKG/UKG)
     age5_picture_desc      — % age-5 children completing the picture-
                              description early-language task
     age5_counting          — % age-5 children completing the counting-
                              objects early-numeracy task
     age5_listening         — % age-5 children completing listening
                              comprehension
     age5_relative_comp     — % age-5 children completing relative-
                              comparison numeracy
     age5_cognitive_mean    — average of 5 cognitive task pass rates
     learning_composite_age5_pct — mean of preschool, picture_desc, counting
   Source PDF: archive.org / asercentre.org
   Parser:    external_data/parse_aser_2019.py
   Output:    external_data/aser2019_state.csv

2. NFHS-5 published State Compendium of Factsheets — sourced from the
   community repo jvargh7/nfhs5_factsheets which extracted the 131
   indicators from the Phase-I and Phase-II compendium PDFs (IIPS 2021).
   Three indicators used:
     #119  Currently married women who participate in 3+ household decisions
     #120  Women who worked in last 12 months and were paid in cash
     #125  Ever-married women age 18-49 who experienced spousal violence
   Source CSV: external_data/nfhs5_states.csv (downloaded from
   https://github.com/jvargh7/nfhs5_factsheets)
   Coverage: all 36 states/UTs.

The NFHS-5 factsheet covers ALL 36 states. ASER 2019 covers only 24 of
36. For the 12 missing states/UTs we use a regional-mean imputation
(documented in the report) rather than fabricating values.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("/Users/prachi/india_nfhs5_project")
EXT = OUT / "external_data"

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load existing NFHS-5 scored data
# ─────────────────────────────────────────────────────────────────────────────

df = pd.read_parquet(OUT / "india_scored.parquet")
print(f"NFHS-5 children loaded: {len(df):,}")
df["state_key"] = df["state"].str.lower().str.strip()

# ─────────────────────────────────────────────────────────────────────────────
# 2. ASER 2019 — REAL parsed values from external_data/aser2019_state.csv
# ─────────────────────────────────────────────────────────────────────────────

aser_raw = pd.read_csv(EXT / "aser2019_state.csv")
aser_raw["state_key"] = aser_raw["state"].str.lower().str.strip()

# Convert ASER % rates to 0-1 scale for consistency with old format
aser_raw["aser_preschool_4to5"] = (aser_raw["preschool_age4"]
                                   + aser_raw["preschool_age5"]) / 200.0
aser_raw["aser_letters_4to5"]   = (aser_raw["age4_picture_desc"]
                                   + aser_raw["age5_picture_desc"]) / 200.0
aser_raw["aser_numbers_4to5"]   = (aser_raw["age4_counting"]
                                   + aser_raw["age5_counting"]) / 200.0
# Composite literacy-numeracy proxy = mean of those 3 dimensions, on 0-1 scale
aser_raw["literacy_numeracy_state_rate"] = aser_raw[
    ["aser_preschool_4to5", "aser_letters_4to5", "aser_numbers_4to5"]
].mean(axis=1)
# Direct child socio-emotional assessment: % correctly identifying all 4 emotions
# Averaged across ages 4 and 5, converted to 0-1 scale.
# Source: ASER 2019 Table 4 (Happy/Sad/Angry/Afraid recognition task).
aser_raw["aser_emotions_4to5"] = aser_raw["aser_emotions_4to5_pct"] / 100.0

aser = aser_raw[["state_key", "aser_preschool_4to5",
                 "aser_letters_4to5", "aser_numbers_4to5",
                 "literacy_numeracy_state_rate",
                 "aser_emotions_4to5"]].copy()
print(f"ASER 2019 states (parsed from PDF): {len(aser)}")

# Regional imputation for ASER-missing states/UTs.
# Map each NFHS-5 state to its census region; missing states get the
# mean of ASER values within their region.
region_map = {
    # NORTH
    "jammu & kashmir": "north", "ladakh": "north", "himachal pradesh": "north",
    "punjab": "north", "haryana": "north", "uttarakhand": "north",
    "rajasthan": "north", "uttar pradesh": "north", "nct of delhi": "north",
    "chandigarh": "north",
    # CENTRAL
    "madhya pradesh": "central", "chhattisgarh": "central",
    # EAST
    "bihar": "east", "jharkhand": "east", "odisha": "east", "west bengal": "east",
    "sikkim": "east",
    # NORTH-EAST
    "assam": "ne", "arunachal pradesh": "ne", "manipur": "ne",
    "meghalaya": "ne", "mizoram": "ne", "nagaland": "ne", "tripura": "ne",
    # WEST
    "gujarat": "west", "maharashtra": "west", "goa": "west",
    "dadra & nagar haveli and daman & diu": "west",
    # SOUTH
    "kerala": "south", "tamil nadu": "south", "karnataka": "south",
    "andhra pradesh": "south", "telangana": "south", "puducherry": "south",
    "lakshadweep": "south", "andaman & nicobar islands": "south",
}

aser["region"] = aser["state_key"].map(region_map)

# All-state list from the NFHS-5 KR data
nfhs_states = pd.DataFrame({"state_key": sorted(df["state_key"].unique())})
nfhs_states["region"] = nfhs_states["state_key"].map(region_map)

# Region means from ASER
region_means = (aser.groupby("region")
                    [["aser_preschool_4to5","aser_letters_4to5",
                      "aser_numbers_4to5","literacy_numeracy_state_rate",
                      "aser_emotions_4to5"]]
                    .mean().reset_index())
print("\nRegion means (ASER):")
print(region_means.round(3).to_string(index=False))

# Merge ASER values; impute missing with region mean
aser_full = nfhs_states.merge(aser.drop(columns=["region"]),
                              on="state_key", how="left")
aser_full["from_aser_direct"] = aser_full["literacy_numeracy_state_rate"].notna().astype(int)
aser_full = aser_full.merge(
    region_means.rename(columns={
        "aser_preschool_4to5":"_imp_preschool",
        "aser_letters_4to5":"_imp_letters",
        "aser_numbers_4to5":"_imp_numbers",
        "literacy_numeracy_state_rate":"_imp_litnum",
        "aser_emotions_4to5":"_imp_emotions",
    }), on="region", how="left")

for raw, imp in [("aser_preschool_4to5","_imp_preschool"),
                 ("aser_letters_4to5","_imp_letters"),
                 ("aser_numbers_4to5","_imp_numbers"),
                 ("literacy_numeracy_state_rate","_imp_litnum"),
                 ("aser_emotions_4to5","_imp_emotions")]:
    aser_full[raw] = aser_full[raw].fillna(aser_full[imp])
aser_full = aser_full.drop(columns=[c for c in aser_full.columns if c.startswith("_imp_")])
aser_full = aser_full.drop(columns=["region"])

print(f"\nASER coverage after regional imputation:")
print(f"  Direct ASER: {(aser_full['from_aser_direct']==1).sum()} states")
print(f"  Regional imputation: {(aser_full['from_aser_direct']==0).sum()} states")
print(f"  Total: {len(aser_full)} states")

# ─────────────────────────────────────────────────────────────────────────────
# 3. NFHS-5 Published State Factsheets — REAL parsed CSV
#    (community repo jvargh7/nfhs5_factsheets)
# ─────────────────────────────────────────────────────────────────────────────

fs_raw = pd.read_csv(EXT / "nfhs5_states.csv")
fs_raw["state_key"] = fs_raw["state"].str.lower().str.strip()

# Indicator number lookup
def pull_indicator(num, new_name):
    rows = fs_raw[fs_raw["Indicator"].str.match(rf"^\s*{num}\.\s")]
    out = rows[["state_key", "Total"]].rename(columns={"Total": new_name})
    out[new_name] = pd.to_numeric(out[new_name], errors="coerce") / 100.0
    return out

dec  = pull_indicator(119, "women_decisions_state")
work = pull_indicator(120, "women_work_state")
viol = pull_indicator(125, "spousal_violence_state")

factsheet = (dec
             .merge(work, on="state_key", how="outer")
             .merge(viol, on="state_key", how="outer"))
print(f"\nNFHS-5 factsheet states (real parsed CSV): {len(factsheet)}")

# Composite socio-emotional household environment index — z-scored mean
def zscore_rescale(s):
    z = (s - s.mean()) / s.std()
    return (z - z.min()) / (z.max() - z.min())
factsheet["no_violence"] = 1.0 - factsheet["spousal_violence_state"]
for c in ["women_decisions_state","no_violence","women_work_state"]:
    factsheet[c + "_z"] = zscore_rescale(factsheet[c])
factsheet["socioemo_state_index"] = factsheet[
    [f"{c}_z" for c in ["women_decisions_state","no_violence","women_work_state"]]
].mean(axis=1)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Merge state-level proxies into the NFHS-5 child record by v024 (state)
# ─────────────────────────────────────────────────────────────────────────────

merge_cols_aser = ["state_key", "aser_preschool_4to5", "aser_letters_4to5",
                   "aser_numbers_4to5", "literacy_numeracy_state_rate",
                   "aser_emotions_4to5", "from_aser_direct"]
merge_cols_fs   = ["state_key", "women_decisions_state", "spousal_violence_state",
                   "women_work_state", "socioemo_state_index"]

df = df.merge(aser_full[merge_cols_aser], on="state_key", how="left")
df = df.merge(factsheet[merge_cols_fs],   on="state_key", how="left")

missing_aser = df["literacy_numeracy_state_rate"].isna().sum()
missing_fs   = df["socioemo_state_index"].isna().sum()
print(f"\nMerge diagnostics:")
print(f"  Children missing ASER value:       {missing_aser:,} ({missing_aser/len(df)*100:.1f}%)")
print(f"  Children missing factsheet value:  {missing_fs:,} ({missing_fs/len(df)*100:.1f}%)")
if missing_aser > 0:
    unmatched = df.loc[df["literacy_numeracy_state_rate"].isna(), "state_key"].unique()
    print(f"  Unmatched states (ASER): {list(unmatched)}")
if missing_fs > 0:
    unmatched_fs = df.loc[df["socioemo_state_index"].isna(), "state_key"].unique()
    print(f"  Unmatched states (factsheet): {list(unmatched_fs)}")

# ─────────────────────────────────────────────────────────────────────────────
# 5. Within-NFHS-5 derived state aggregates (control covariates)
# ─────────────────────────────────────────────────────────────────────────────

state_agg = df.groupby("state_key").agg(
    state_mother_edu_mean = ("mother_edu",         "mean"),
    state_urban_share     = ("is_urban",           "mean"),
    state_stunting_rate   = ("ecdi_proxy_physical","mean"),
    state_anganwadi_share = ("ecdi_proxy_learning","mean"),
).reset_index()
state_agg["state_stunting_rate"] = 1.0 - state_agg["state_stunting_rate"]
df = df.merge(state_agg, on="state_key", how="left")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Diagnostics
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "="*72)
print("STATE-LEVEL DEMOGRAPHIC PROXIES — distribution after real-data merge")
print("="*72)
diag_cols = ["literacy_numeracy_state_rate", "aser_emotions_4to5",
             "women_decisions_state", "spousal_violence_state", "socioemo_state_index",
             "state_mother_edu_mean", "state_urban_share", "state_stunting_rate"]
print(df[diag_cols].describe().round(3).to_string())

state_view = (df.groupby("state")
              .agg(litnum=("literacy_numeracy_state_rate","first"),
                   socioemo=("socioemo_state_index","first"),
                   from_aser=("from_aser_direct","first"),
                   n=("ecdi_proxy_composite","size"))
              .sort_values("litnum", ascending=False))
print("\nTop 5 states (literacy-numeracy proxy):")
print(state_view.head(5).round(3).to_string())
print("\nBottom 5 states (literacy-numeracy proxy):")
print(state_view.tail(5).round(3).to_string())

# ─────────────────────────────────────────────────────────────────────────────
# 7. Save enriched dataset and the public state-level proxy table
# ─────────────────────────────────────────────────────────────────────────────

df.to_parquet(OUT / "india_enriched.parquet", index=False)
print(f"\nSaved india_enriched.parquet ({len(df):,} rows × {df.shape[1]} cols)")

proxy_table = aser_full.merge(
    factsheet[["state_key","women_decisions_state","spousal_violence_state",
               "women_work_state","socioemo_state_index"]],
    on="state_key", how="outer"
).merge(state_agg, on="state_key", how="outer")
proxy_table.to_csv(OUT / "state_proxy_table.csv", index=False)
print(f"Saved state_proxy_table.csv ({len(proxy_table)} states)")
