"""
India NFHS-5 — Data Loading
============================
Source: DHS-7 / NFHS-5 (2019-21) Kids Recode (KR) file
File  : IAKR7EFL.DTA (Stata format, 232,920 birth records)

Filter: alive children aged 24–59 months (ECDI2030 target age range)
        b5 == 'yes' AND hw1 in [24, 59]

Key variable groups:
  hw1          : child age in months
  b4           : sex of child (male/female)
  hw70, hw71   : HAZ, WAZ (DHS units: integer × 0.01 SD, e.g. -200 = -2.0 SD)
  hw72         : WHZ (same units)
  s558, s562   : anganwadi/ICDS benefits received; attendance frequency
  v024         : state (36 states/UTs)
  v025         : type of place of residence (urban/rural)
  v190         : wealth index quintile (combined)
  v149         : educational attainment (mother)
  v005         : women's sample weight (6 decimals → divide by 1,000,000 for weight)
  v001, v002   : cluster and household IDs (for linking to HR file)
  h2, h7, h9   : BCG, DPT3, Measles1 vaccination status

Note: NFHS-5 KR file already contains all mother-level variables (v*) for
each birth record — no separate merge with IR file is necessary for our
core analysis. We merge the HR file only for additional household variables.
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("/Users/prachi/india_nfhs5_project")
OUT.mkdir(exist_ok=True)

KR_PATH = "/Users/prachi/Downloads/DHS Data/IAKR7EDT/IAKR7EFL.DTA"
HR_PATH = "/Users/prachi/Downloads/DHS Data/IAHR7EDT/IAHR7EFL.DTA"

# ── 1. Load KR file ───────────────────────────────────────────────────────────

print("Loading NFHS-5 KR file …")
kr_cols = [
    "v001", "v002", "b16",          # IDs for HR merge
    "v005",                          # sample weight
    "hw1",                           # age in months
    "b4",                            # sex
    "b5",                            # alive?
    "hw70", "hw71", "hw72",          # HAZ, WAZ, WHZ
    "s558", "s562",                  # anganwadi benefits; attendance
    "v024",                          # state
    "v025",                          # urban/rural
    "v190",                          # wealth index
    "v149",                          # mother's education
    "h2", "h7", "h8", "h9",         # BCG, DPT3, Polio3, Measles1
    "v012",                          # mother's age
    "v101",                          # region/state (backup)
]
# Some columns may be absent in certain file versions — load safely
df_kr = pd.read_stata(KR_PATH, columns=kr_cols)
print(f"  KR loaded: {df_kr.shape[0]:,} rows × {df_kr.shape[1]} cols")

# ── 2. Filter to alive children 24–59 months ─────────────────────────────────

df = df_kr[(df_kr["b5"] == "yes") &
           (df_kr["hw1"] >= 24) &
           (df_kr["hw1"] <= 59)].copy()
print(f"  After filter (alive, 24–59 months): {len(df):,} children")

# ── 3. Recode / standardise variables ────────────────────────────────────────

# Age (keep as hw1, numeric)
df["child_age_months"] = df["hw1"].astype(float)

# Sex
df["is_female"] = (df["b4"] == "female").astype(float)

# Urban/rural
df["is_urban"] = (df["v025"] == "urban").astype(float)

# Wealth: ordinal 1–5
wealth_map = {"poorest": 1, "poorer": 2, "middle": 3, "richer": 4, "richest": 5}
df["wealth_index"] = df["v190"].map(wealth_map).astype(float)

# Mother's education: ordinal
edu_map = {
    "no education": 0,
    "incomplete primary": 1,
    "complete primary": 2,
    "incomplete secondary": 3,
    "complete secondary": 4,
    "higher": 5,
}
df["mother_edu"] = df["v149"].map(edu_map).astype(float)
df["mother_higher_ed"] = (df["mother_edu"] >= 4).astype(float)  # complete secondary+

# Sample weight (DHS convention: divide by 1,000,000)
df["weight"] = df["v005"].astype(float) / 1_000_000

# Anthropometrics: convert DHS integer units to actual z-scores (÷ 100)
for raw_col, new_col in [("hw70", "HAZ"), ("hw71", "WAZ"), ("hw72", "WHZ")]:
    numeric = pd.to_numeric(df[raw_col], errors="coerce")
    # Flag extreme / implausible values (DHS flags ≥ 9996 or categories)
    df[new_col] = numeric.where((numeric >= -600) & (numeric <= 600), np.nan) / 100.0

# Anganwadi/ICDS attendance
# s558: received any benefits (yes/no)
df["anganwadi_any"] = (df["s558"] == "yes").astype(float)
# s562: attendance frequency (NaN if s558='no')
df["anganwadi_regular"] = (df["s562"] == "regularly").astype(float)
df["anganwadi_attending"] = df["s562"].isin(["regularly", "occasionally"]).astype(float)

# State label
df["state"] = df["v024"].astype(str).str.strip()

# Mother's age
df["mother_age"] = df["v012"].astype(float)

# ── 4. Vaccination: BCG + DPT3 + Measles1 ────────────────────────────────────
# Only ~33% of children have vaccination data (DHS collects for youngest eligible child)
# "vaccination date on card" or "vaccination marked on card" or "reported by mother"
# all count as 'received'

def received(col):
    """Returns 1 if vaccination recorded (any positive category), 0 if 'no', NaN if missing."""
    got = col.isin(["vaccination date on card",
                    "vaccination marked on card",
                    "vaccination identified",
                    "reported by mother"])
    no_vax = col == "no"
    return np.where(got, 1, np.where(no_vax, 0, np.nan))

for vax_col, new_col in [("h2", "bcg"), ("h7", "dpt3"), ("h9", "measles1")]:
    df[new_col] = received(df[vax_col])

# Fully vaccinated: BCG + DPT3 + Measles1 all received
df["fully_vaccinated"] = np.where(
    df[["bcg","dpt3","measles1"]].isnull().any(axis=1), np.nan,
    (df[["bcg","dpt3","measles1"]].sum(axis=1) == 3).astype(float)
)

# ── 5. Summary ────────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("NFHS-5 SAMPLE SUMMARY")
print("="*55)
print(f"Children 24–59 months: {len(df):,}")
print(f"  Sex: Female = {df['is_female'].mean()*100:.1f}%")
print(f"  Urban = {df['is_urban'].mean()*100:.1f}%")
print(f"  With HAZ:         {df['HAZ'].notna().sum():,} ({df['HAZ'].notna().mean()*100:.1f}%)")
print(f"  With WAZ:         {df['WAZ'].notna().sum():,} ({df['WAZ'].notna().mean()*100:.1f}%)")
print(f"  With anganwadi:   {df['s562'].notna().sum():,} ({df['s562'].notna().mean()*100:.1f}%)")
print(f"  With vaccination: {df['fully_vaccinated'].notna().sum():,} ({df['fully_vaccinated'].notna().mean()*100:.1f}%)")

print(f"\nMean HAZ: {df['HAZ'].mean():.2f} SD")
print(f"Mean WAZ: {df['WAZ'].mean():.2f} SD")
print(f"Not stunted (HAZ>-2): {(df['HAZ']>-2).mean()*100:.1f}%")
print(f"Not underweight (WAZ>-2): {(df['WAZ']>-2).mean()*100:.1f}%")
print(f"Anganwadi regular: {(df['s562']=='regularly').sum():,} / {df['s562'].notna().sum():,} "
      f"= {(df['s562']=='regularly').sum()/df['s562'].notna().sum()*100:.1f}% among enrolled")

print("\nBy state (top 5 by count):")
print(df["state"].value_counts().head(5))

print("\nWealth distribution:")
print(df["v190"].value_counts().sort_index())

# ── 6. Save ───────────────────────────────────────────────────────────────────

# Convert any remaining Stata categorical columns to strings before saving
for col in df.columns:
    if hasattr(df[col], "cat"):
        df[col] = df[col].astype(str).replace("nan", np.nan)

df.to_parquet("/Users/prachi/india_nfhs5_project/india_merged.parquet", index=False)
print(f"\nSaved india_merged.parquet  ({len(df):,} rows × {df.shape[1]} cols)")
