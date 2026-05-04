"""
Parse ASER 2019 'Early Years' PDF — extract state-level early-learning indicators
==================================================================================
ASER 2019 sampled 26 districts across 24 states (one rural district per state,
two each in MP and UP). This script:
  1. Locates each district page via its "STATE: District" header.
  2. Detects tables by content rather than fixed index, so it is robust to
     layout variation across district pages.
  3. Extracts from each data page:
       Table 1 — pre-school enrollment (Anganwadi + Govt pre-primary + Pvt LKG)
                 for ages 4 and 5.
       Table 3 — cognitive (5 tasks), early-language (picture-description,
                 listening), early-numeracy (counting, relative-comparison)
                 for ages 4 and 5.
       Table 4 — socio-emotional recognition (% correctly identifying all 4
                 emotions: happy, sad, angry, afraid) for ages 4 and 5.
  4. Aggregates within state (averaging the two MP and UP districts).
  5. Writes a CSV with all indicators plus:
       learning_composite_age5_pct — mean(preschool_age5, age5_picture_desc,
                                     age5_counting); operational lit-num proxy.
       aser_emotions_4to5_pct      — mean(age4_all4emotions, age5_all4emotions);
                                     direct child socio-emotional assessment.

NOTE: ASER 2019 'Early Years' sampled rural districts only; no data exist in
this PDF for the 12 states/UTs not represented (Andaman & Nicobar, Arunachal
Pradesh, Chandigarh, Dadra & Nagar Haveli & Daman & Diu, Goa, Jammu &
Kashmir, Ladakh, Lakshadweep, Mizoram, NCT of Delhi, Puducherry, Sikkim).
Those states receive regional-mean imputation in 07_proxy_integration.py.

Output: /Users/prachi/india_nfhs5_project/external_data/aser2019_state.csv
"""

import re
import pdfplumber
import pandas as pd
import numpy as np
from pathlib import Path

PDF = Path("/Users/prachi/india_nfhs5_project/external_data/aser2019.pdf")
OUT = Path("/Users/prachi/india_nfhs5_project/external_data/aser2019_state.csv")


def find_district_pages(pdf):
    """Return dict: state -> list of (pdf_page_idx, district_name).
    Skips multi-state section-cover pages (which contain >1 ':' in first line).
    Pages 60–134 (0-indexed) cover all 24 sampled states; story/illustration
    pages that share the "State: District" header are filtered later by content.
    """
    out = {}
    for i in range(60, 135):
        txt = pdf.pages[i].extract_text() or ""
        first_line = (txt.split("\n", 1)[0] if txt else "").strip()
        if first_line.count(":") > 1:
            continue
        m = re.match(r"([A-Za-z][A-Za-z &]+):\s*([A-Za-z][A-Za-z &()]+)$", first_line)
        if m:
            state = m.group(1).strip()
            district = m.group(2).strip()
            if len(state.split()) <= 3 and "ASER" not in state and len(state) > 3:
                out.setdefault(state, []).append((i, district))
    return out


def fnum(s):
    """Float-cast a cell; return NaN on bad input."""
    if s is None:
        return np.nan
    s = str(s).strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _header_text(table, n_rows=2):
    """Flatten the first n_rows of a table into a single lower-case string."""
    return " ".join(
        str(cell or "")
        for row in table[:n_rows]
        for cell in (row or [])
    ).lower()


def _is_enrollment_table(t):
    h = _header_text(t)
    return ("pre-school" in h or "angan" in h) and (
        any((r[0] or "").strip() in ("Age 4", "Age 5") for r in t[1:] if r)
    )


def _is_cognitive_table(t):
    h = _header_text(t)
    return "cognitive" in h and "early language" in h and (
        any((r[0] or "").strip() in ("Age 4", "Age 5") for r in t[1:] if r)
    )


def _is_emotions_table(t):
    h = _header_text(t)
    return "happy" in h and "sad" in h and (
        any((r[0] or "").strip() in ("Age 4", "Age 5") for r in t[1:] if r)
    )


def extract_district(pdf, page_idx):
    """Return dict of metrics from one district's early-years data page."""
    page = pdf.pages[page_idx]
    tables = page.extract_tables() or []
    out = {"page": page_idx + 1}

    for t in tables:
        if not t or len(t) < 3:
            continue

        # ── Enrollment table ────────────────────────────────────────────────
        if _is_enrollment_table(t):
            for row in t:
                if not row:
                    continue
                r0 = (row[0] or "").strip()
                if r0 in ("Age 4", "Age 5"):
                    # Columns: Anganwadi | Govt pre-primary | Pvt LKG/UKG | ...
                    anganwadi = fnum(row[1])
                    govt_pp   = fnum(row[2])
                    pvt_lkg   = fnum(row[3])
                    val = (anganwadi or 0) + (govt_pp or 0) + (pvt_lkg or 0)
                    if r0 == "Age 4":
                        out["preschool_age4"] = val
                    else:
                        out["preschool_age5"] = val

        # ── Cognitive / language / numeracy table ───────────────────────────
        elif _is_cognitive_table(t):
            for row in t:
                if not row:
                    continue
                r0 = (row[0] or "").strip()
                if r0 in ("Age 4", "Age 5"):
                    vals = [fnum(c) for c in row[1:]]
                    # 5 cognitive | picture_desc | listening | counting | relative_comp
                    cog_mean = float(np.nanmean(vals[:5])) if len(vals) >= 5 else np.nan
                    pic  = vals[5] if len(vals) > 5 else np.nan
                    lst  = vals[6] if len(vals) > 6 else np.nan
                    cnt  = vals[7] if len(vals) > 7 else np.nan
                    rel  = vals[8] if len(vals) > 8 else np.nan
                    if r0 == "Age 4":
                        out.update({
                            "age4_cognitive_mean": cog_mean,
                            "age4_picture_desc":   pic,
                            "age4_listening":      lst,
                            "age4_counting":       cnt,
                            "age4_relative_comp":  rel,
                        })
                    else:
                        out.update({
                            "age5_cognitive_mean": cog_mean,
                            "age5_picture_desc":   pic,
                            "age5_listening":      lst,
                            "age5_counting":       cnt,
                            "age5_relative_comp":  rel,
                        })

        # ── Socio-emotional recognition table (emotions) ────────────────────
        elif _is_emotions_table(t):
            for row in t:
                if not row:
                    continue
                r0 = (row[0] or "").strip()
                if r0 in ("Age 4", "Age 5"):
                    # Columns: Happy | Sad | Angry | Afraid | All 4 emotions
                    all4 = fnum(row[-1]) if row else np.nan
                    if r0 == "Age 4":
                        out["age4_all4emotions"] = all4
                    else:
                        out["age5_all4emotions"] = all4

    return out


def main():
    pdf = pdfplumber.open(PDF)
    district_pages = find_district_pages(pdf)
    print(f"States with district pages found: {len(district_pages)}")

    rows = []
    for state, pages in district_pages.items():
        chosen = None
        for pdf_idx, district in pages:
            m = extract_district(pdf, pdf_idx)
            # Accept this page if it has cognitive/language data
            if not pd.isna(m.get("age5_picture_desc", np.nan)):
                m["state"]    = state
                m["district"] = district
                chosen = m
                break
        if chosen is None:
            # Fall back to the first page for the state
            pdf_idx, district = pages[0]
            m = extract_district(pdf, pdf_idx)
            m["state"]    = state
            m["district"] = district
            chosen = m
        rows.append(chosen)
    pdf.close()

    df = pd.DataFrame(rows)

    # MP and UP have 2 districts each — collapse to state mean
    df_state = (
        df.groupby("state")
          .agg({c: "mean" for c in df.columns if c not in ("state", "district", "page")})
          .reset_index()
    )
    df_state["n_districts"] = df.groupby("state").size().values

    # Composite literacy-numeracy proxy (% at age 5 on the three core tasks)
    cols_comp = ["preschool_age5", "age5_picture_desc", "age5_counting"]
    df_state["learning_composite_age5_pct"] = df_state[cols_comp].mean(axis=1)

    # Composite socio-emotional proxy: mean of age-4 and age-5 "all 4 emotions" rate
    df_state["aser_emotions_4to5_pct"] = (
        df_state[["age4_all4emotions", "age5_all4emotions"]].mean(axis=1)
    )

    col_order = [
        "state", "n_districts",
        "preschool_age4", "preschool_age5",
        "age4_cognitive_mean", "age4_picture_desc", "age4_listening",
        "age4_counting", "age4_relative_comp",
        "age5_cognitive_mean", "age5_picture_desc", "age5_listening",
        "age5_counting", "age5_relative_comp",
        "age4_all4emotions", "age5_all4emotions",
        "learning_composite_age5_pct", "aser_emotions_4to5_pct",
    ]
    df_state = df_state.reindex(columns=col_order)
    df_state.to_csv(OUT, index=False)

    print(f"\nSaved {OUT}  ({len(df_state)} states)")
    print(
        df_state[[
            "state", "n_districts",
            "preschool_age5", "age5_picture_desc",
            "age5_counting", "learning_composite_age5_pct",
            "age4_all4emotions", "age5_all4emotions", "aser_emotions_4to5_pct",
        ]]
        .sort_values("learning_composite_age5_pct", ascending=False)
        .round(1)
        .to_string(index=False)
    )

    # Spot-check missing values
    missing = df_state[df_state["age5_picture_desc"].isna()]["state"].tolist()
    if missing:
        print(f"\nWARNING: No cognitive data extracted for: {missing}")
    missing_se = df_state[df_state["aser_emotions_4to5_pct"].isna()]["state"].tolist()
    if missing_se:
        print(f"WARNING: No emotions data extracted for: {missing_se}")


if __name__ == "__main__":
    main()
