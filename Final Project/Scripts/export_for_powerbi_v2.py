"""
=============================================================================
EXPORT FOR POWER BI — EXPANDED
=============================================================================
    python export_for_powerbi_v2.py

Replaces the earlier 14-column export. That one was cut too hard: it left
the dashboard able to display results but not to compute anything, which
made DAX ornamental rather than load-bearing.

WHAT CHANGED AND WHY

The old rule was "the dashboard shows the model's results rather than
recomputing the model". That is right about the MODEL and wrong about
everything else. Without maternal age, parity and comorbidity the dashboard
cannot:

  - reproduce the standardisation, which is the project's central method
  - let anyone slice the Robson table by anything clinical
  - support a what-if parameter on the group 5 share
  - build the VBAC page

So the export now carries every column the dashboard can genuinely use,
raw AND banded. The raw value lets DAX compute; the band gives clean slicer
values that match the bands used in the Python analysis exactly, so a
figure produced in Power BI and a figure produced in the pipeline are
comparable rather than merely similar.

WHAT IS STILL LEFT OUT

Model outputs — predicted probability, SHAP values, performance metrics.
Those come from summary tables, because recomputing a fitted model in DAX
is neither possible nor desirable, and the model card restricts how its
output may be used.

BANDS ARE COMPUTED HERE, NOT IN DAX

Deliberately. If the age bands were defined again in DAX they could drift
from the ones used in stage 3 and stage 4, and two versions of "under 20"
in one project is the kind of quiet inconsistency nobody notices until an
examiner does.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

OUT = Path("powerbi"); OUT.mkdir(exist_ok=True)

SOURCES = [
    ("natality_2023_extended.parquet", 2023),
    ("natality_2024_classified.parquet", 2024),
]

# raw columns to carry through, grouped by what they enable in the dashboard
# Every column here has a page that uses it. Ten were dropped after review
# because they had no page: height, pre-pregnancy weight, weight gain,
# education, marital status, antenatal timing, smoking, fertility treatment,
# ART, previous preterm birth, interpregnancy interval and number of previous
# caesareans. Most were adjustment variables in the model, not dashboard
# content, and a field list carrying a third of its columns unused reads as
# undisciplined.
WANT = [
    # Robson classification and data quality           -> pages 2, 8
    "Robson_group", "Robson_group_sort_order", "Classification_status",
    "Missing_variable",
    # outcome                                          -> everywhere
    "is_cesarean", "RDMETH_REC", "ME_TRIAL", "RF_CESAR",
    # the six Robson criteria, so a viewer can see how the group was reached
    "ME_PRES", "DPLURAL", "OEGest_Comb", "LD_INDL", "PRIORLIVE", "PRIORDEAD",
    # maternal characteristics for WITHIN-group analysis -> page 6
    "MAGER", "BMI",
    # comorbidity, kept separate on purpose             -> page 6
    "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE",
    # practice setting                                  -> page 6
    "payer", "attendant", "BFACIL", "weekend", "time_of_day",
    # outcomes                                          -> page 5
    "APGAR5", "AB_NICU", "near_miss_flag",
]

SENTINELS = {
    "MAGER": [], "BMI": [99.9], "PRIORLIVE": [99], "PRIORDEAD": [99],
    "OEGest_Comb": [99], "DPLURAL": [9], "APGAR5": [99],
}
YN = ["RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE",
      "RF_CESAR", "ME_TRIAL", "LD_INDL"]

frames = []
for path, year in SOURCES:
    if not Path(path).exists():
        print(f"[skip] {path} not found")
        continue
    print(f"Reading {path} ...")
    d = pd.read_parquet(path)
    have = [c for c in WANT if c in d.columns]
    missing = [c for c in WANT if c not in d.columns]
    if missing:
        print(f"  absent from this file: {missing}")
    s = d[have].copy()
    s["year"] = year
    frames.append(s)
    print(f"  {len(s):,} rows, {len(have)} columns kept\n")

if not frames:
    raise SystemExit("No source files found.")

df = pd.concat(frames, ignore_index=True)
print(f"Combined: {len(df):,} rows\n")

# ---------------------------------------------------------------------------
print("Cleaning sentinel codes ...")
for col, codes in SENTINELS.items():
    if col not in df.columns:
        continue
    x = pd.to_numeric(df[col], errors="coerce")
    if codes:
        n = int(x.isin(codes).sum())
        x = x.where(~x.isin(codes))
        if n:
            print(f"  {col:<14} {n:>10,} unknown codes removed")
    df[col] = x

for col in YN:
    if col not in df.columns:
        continue
    s = df[col].astype(str).str.strip().str.upper()
    v = pd.Series(np.nan, index=df.index)
    v[s == "Y"] = 1.0
    v[s == "N"] = 0.0
    df[col] = v

# ---------------------------------------------------------------------------
print("\nBuilding bands (same cut points as the Python analysis) ...")

df["age_band"] = pd.cut(df["MAGER"], [0, 19, 24, 29, 34, 99],
                        labels=["<20", "20-24", "25-29", "30-34", "35+"])
df["age_band_sort"] = df["age_band"].cat.codes

df["bmi_band"] = pd.cut(df["BMI"], [0, 18.5, 25, 30, 35, 99],
                        labels=["Underweight", "Normal", "Overweight",
                                "Obese I", "Obese II+"])
df["bmi_band_sort"] = df["bmi_band"].cat.codes

df["parity_band"] = pd.cut(df["PRIORLIVE"], [-1, 0, 1, 3, 99],
                           labels=["First birth", "Second", "3rd-4th", "5th+"])
df["parity_band_sort"] = df["parity_band"].cat.codes

df["gestation_band"] = pd.cut(df["OEGest_Comb"], [0, 33, 36, 38, 40, 50],
                              labels=["<34", "34-36", "37-38", "39-40", "41+"])
df["gestation_band_sort"] = df["gestation_band"].cat.codes

df["any_diabetes"] = ((df.get("RF_PDIAB", 0) == 1) |
                      (df.get("RF_GDIAB", 0) == 1)).astype("int8")
df["any_hypertension"] = ((df.get("RF_PHYPE", 0) == 1) |
                          (df.get("RF_GHYPE", 0) == 1) |
                          (df.get("RF_EHYPE", 0) == 1)).astype("int8")
df["any_comorbidity"] = ((df["any_diabetes"] == 1) |
                         (df["any_hypertension"] == 1)).astype("int8")

# readable labels for slicers
df["presentation"] = df["ME_PRES"].astype(str).str.strip().map(
    {"1": "Cephalic", "2": "Breech", "3": "Other"}).fillna("Unknown")

bf = df["BFACIL"].astype(str).str.strip() if "BFACIL" in df.columns else pd.Series("", index=df.index)
df["setting"] = np.select(
    [bf == "1", bf == "2", bf.isin(["3", "4", "5"]), bf.isin(["6", "7"])],
    ["Hospital", "Birth centre", "Home", "Clinic or other"], default="Unknown")

rd = df["RDMETH_REC"].astype(str).str.strip() if "RDMETH_REC" in df.columns else pd.Series("", index=df.index)
df["delivery_detail"] = np.select(
    [rd == "1", rd == "2", rd == "3", rd == "4", rd.isin(["5", "6"])],
    ["Vaginal", "VBAC", "Primary caesarean", "Repeat caesarean", "Vaginal after caesarean, other"],
    default="Unknown")

# TOLAC: achieved VBAC, or a repeat caesarean after a recorded trial of labour
df["tolac_attempted"] = np.where(
    (df.get("RF_CESAR", 0) == 1) & rd.isin(["2", "4"]),
    ((rd == "2") | ((rd == "4") & (df.get("ME_TRIAL", 0) == 1))).astype(float),
    np.nan)
df["vbac_success"] = np.where(
    (df.get("RF_CESAR", 0) == 1) & rd.isin(["2", "4"]),
    (rd == "2").astype(float), np.nan)

df["apgar5_below7"] = (df["APGAR5"] < 7).astype("float")
df["nicu_admission"] = (df["AB_NICU"].astype(str).str.strip() == "Y").astype("int8")
df["is_classified"] = (df["Classification_status"] == "Classified").astype("int8")
df["robson_group_10"] = df["Robson_group"].replace(
    {"2a": "2", "2b": "2", "4a": "4", "4b": "4", "5.1": "5", "5.2": "5"})

df = df.drop(columns=[c for c in ["ME_PRES", "BFACIL", "RDMETH_REC", "APGAR5",
                                  "AB_NICU"] if c in df.columns])

for c in df.select_dtypes("category").columns:
    df[c] = df[c].astype(str).replace("nan", "Unknown")

out = OUT / "births.parquet"
df.to_parquet(out, index=False)
mb = out.stat().st_size / 1e6

print(f"\nWritten {out}")
print(f"  {len(df):,} rows, {len(df.columns)} columns, {mb:,.1f} MB\n")

print("=" * 66)
print("REPRODUCE THESE IN POWER BI")
print("=" * 66)
for year in sorted(df["year"].unique()):
    y = df[df.year == year]
    c = y[y.is_classified == 1]
    n = c[c["Robson_group"].isin(["1", "2a", "2b"])]
    print(f"\n  {year}   classified {len(c):>10,}   "
          f"CS {100*c.is_cesarean.mean():.2f}%   NTSV {100*n.is_cesarean.mean():.2f}%")

print("\n" + "=" * 66)
print("WHAT THE NEW COLUMNS UNLOCK")
print("=" * 66)
print("""
  age_band, bmi_band, parity_band, gestation_band
      slice the Robson table by anything clinical, and drive the
      standardisation measure

  any_diabetes, any_hypertension, any_comorbidity
      a page on what the recorded indications actually explain

  delivery_detail, tolac_attempted, vbac_success
      the VBAC page: how often a trial of labour is offered, and how
      often it succeeds

  setting, payer, attendant
      practice patterns, with the caveat that attendant is unreliable for
      caesareans by NCHS's own documentation

  presentation, DPLURAL, OEGest_Comb
      the Robson criteria themselves, so a viewer can see how the
      classification was reached rather than taking it on trust
""")
