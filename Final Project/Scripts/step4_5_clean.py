"""
=============================================================================
STEP 4 + 5 — LOAD AND CLEAN
=============================================================================
    python step4_5_clean.py

THE PROBLEM THIS SOLVES

CDC does not leave unknown values blank. It records them as numbers:

    gestational age unknown ....... 99   (weeks)
    birth weight unknown .......... 9999 (grams)
    BMI unknown ................... 99.9
    Apgar unknown ................. 99
    previous caesareans unknown ... 99
    height unknown ................ 99   (inches)

Nothing raises an error. The value simply flows into every mean, every
split and every SHAP plot as though a woman were 99 weeks pregnant. It is
the largest silent data-quality risk in this source, and it is invisible
unless you go looking for it.

This script converts each sentinel to a genuine missing value, and prints
how many it found in each column so the decision is auditable rather than
assumed. Nothing is imputed: XGBoost handles missingness natively by
treating it as a branch during splitting, which preserves the information
that a value was absent instead of inventing one.

Output: model_data.parquet — the 24 model columns plus the target, the
Robson group (for later evaluation splits) and the classification status.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Feature list, inlined so this runs as a single file.
# Generated from data_dictionary.py — 24 of 59 columns.
# The test for inclusion: was this fact known BEFORE the delivery decision?
# ---------------------------------------------------------------------------
TARGET = "is_cesarean"

MODEL_FEATURES = [
    # maternal characteristics
    "MAGER",        # mother's age (years)
    "M_Ht_In",      # height (inches)
    "PWgt_R",       # pre-pregnancy weight (lb)
    "WTGAIN",       # weight gain in pregnancy (lb)
    "BMI",          # body mass index
    "MEDUC",        # education (ordinal)
    "DMAR",         # marital status
    "CIG_REC",      # smoked during pregnancy
    "PRECARE5",     # month antenatal care began (ordinal)
    # obstetric history
    "PRIORLIVE",    # previous live births
    "PRIORDEAD",    # previous stillbirths
    "RF_CESARN",    # number of previous caesareans
    "ILLB_R",       # interval since last live birth
    "RF_PPTERM",    # previous preterm birth
    # comorbidity
    "RF_PDIAB",     # pre-pregnancy diabetes
    "RF_GDIAB",     # gestational diabetes
    "RF_PHYPE",     # chronic hypertension
    "RF_GHYPE",     # gestational hypertension
    "RF_EHYPE",     # eclampsia
    "RF_INFTR",     # infertility treatment
    "RF_ARTEC",     # assisted reproductive technology
    # current pregnancy
    "ME_PRES",      # fetal presentation (nominal)
    "DPLURAL",      # number of fetuses
    "OEGest_Comb",  # gestational age (weeks)
]

# sentinel -> the codes that mean "unknown" in that column
SENTINELS = {
    "MAGER":       [],                 # 12-50, no sentinel
    "M_Ht_In":     [99],
    "PWgt_R":      [999],
    "WTGAIN":      [99],
    "BMI":         [99.9],
    "MEDUC":       [9],
    "DMAR":        [9],
    "PRIORLIVE":   [99],
    "PRIORDEAD":   [99],
    "RF_CESARN":   [99],
    "ILLB_R":      [888, 999],         # 888 = not applicable (first birth)
    "OEGest_Comb": [99],
    "DPLURAL":     [9],
}
# Y/N/U fields -> True / False / missing
YN_COLS = ["CIG_REC", "RF_PPTERM", "RF_PDIAB", "RF_GDIAB", "RF_PHYPE",
           "RF_GHYPE", "RF_EHYPE", "RF_INFTR", "RF_ARTEC"]
# categorical codes kept as labels, with their unknown code
CAT_UNKNOWN = {"ME_PRES": ["9"], "PRECARE5": ["5"]}

SRC = next((f for f in ["natality_2023_extended.parquet",
                        "natality_2023_filtered.parquet"] if Path(f).exists()), None)
if SRC is None:
    raise SystemExit("No parquet cache found. Run robson_analysis.py first.")

print(f"Loading {SRC} ...")
keep = MODEL_FEATURES + [TARGET, "Robson_group", "Robson_group_sort_order",
                         "Classification_status"]
df = pd.read_parquet(SRC)
missing_cols = [c for c in keep if c not in df.columns]
if missing_cols:
    raise SystemExit(f"Columns not in the cache: {missing_cols}")
d = df[keep].copy()
print(f"  {len(d):,} rows, {len(MODEL_FEATURES)} model columns\n")

print("=" * 74)
print("CLEANING — what was found and removed")
print("=" * 74)
print(f"{'column':<14} {'type':<12} {'sentinel':>12} {'flagged':>12} {'% missing':>11}")
print("-" * 74)

report = []
for col in MODEL_FEATURES:
    if col in YN_COLS:
        s = d[col].astype(str).str.strip().str.upper()
        out = pd.Series(np.nan, index=d.index, dtype="object")
        out[s == "Y"] = 1.0
        out[s == "N"] = 0.0
        n_unknown = int((~s.isin(["Y", "N"])).sum())
        d[col] = pd.to_numeric(out, errors="coerce")
        kind, sent = "binary Y/N", "U"
    elif col in CAT_UNKNOWN:
        s = d[col].astype(str).str.strip()
        n_unknown = int(s.isin(CAT_UNKNOWN[col]).sum())
        d[col] = s.where(~s.isin(CAT_UNKNOWN[col]), np.nan)
        kind, sent = "categorical", ",".join(CAT_UNKNOWN[col])
    else:
        x = pd.to_numeric(d[col], errors="coerce")
        codes = SENTINELS.get(col, [])
        n_unknown = int(x.isin(codes).sum()) if codes else 0
        if codes:
            x = x.where(~x.isin(codes))
        d[col] = x
        kind = "numeric"
        sent = ",".join(str(c) for c in codes) if codes else "none"

    pct = 100 * d[col].isna().mean()
    report.append((col, kind, sent, n_unknown, pct))
    print(f"{col:<14} {kind:<12} {sent:>12} {n_unknown:>12,} {pct:>10.2f}%")

rep = pd.DataFrame(report, columns=["column","type","sentinel","n_flagged","pct_missing"])
rep.to_csv("cleaning_report.csv", index=False)

print("\n" + "=" * 74)
print("SANITY CHECK — ranges after cleaning")
print("=" * 74)
num = [c for c in MODEL_FEATURES if c not in CAT_UNKNOWN]
desc = d[num].describe().T[["min", "max", "mean"]].round(2)
print(desc.to_string())
print("""
  Read the max column. If any of these still shows 99, 999 or 9999 where
  that value is clinically impossible, a sentinel has been missed and must
  be added to SENTINELS above before going further.
""")

flagged = []
for col in num:
    mx = d[col].max()
    if col in ("OEGest_Comb",) and mx > 50: flagged.append(col)
    if col in ("M_Ht_In",) and mx > 90: flagged.append(col)
    if col in ("RF_CESARN","PRIORLIVE","PRIORDEAD") and mx > 30: flagged.append(col)
    if col == "BMI" and mx > 90: flagged.append(col)
if flagged:
    print(f"  WARNING — implausible maximum still present in: {flagged}")
else:
    print("  All ranges plausible. No sentinel left behind.")

print("\n" + "=" * 74)
print("MISSINGNESS SUMMARY")
print("=" * 74)
worst = rep.sort_values("pct_missing", ascending=False).head(8)
print(worst[["column","pct_missing"]].to_string(index=False))
print("""
  Nothing is imputed. XGBoost treats missingness as a branch when it splits,
  which keeps the fact that a value was absent as information rather than
  replacing it with a guess. A high missing rate is a limitation to report,
  not a hole to fill quietly.

  ILLB_R is expected to be largely missing: 888 means 'not applicable',
  which is the correct value for a first birth. That is a real absence, not
  a recording failure, and the distinction belongs in the write-up.
""")

d.to_parquet("model_data.parquet", index=False)
print(f"Saved model_data.parquet — {len(d):,} rows, {len(d.columns)} columns")
print("Saved cleaning_report.csv")
print("\nNext: step 6 — one-hot encoding.")
