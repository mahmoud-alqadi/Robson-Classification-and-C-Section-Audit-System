"""
=============================================================================
STEP 6 — VERIFY THE CONDITIONAL FIELDS, THEN ENCODE
=============================================================================
    python step6_encode.py

TWO THINGS BEFORE ENCODING

1. RF_ARTEC came out 97.61% missing with 77% "yes" among the remainder.
   Assisted reproduction is about 2% of US births, not 77%, so one of two
   things is true: the field position is wrong, or the field is CONDITIONAL
   — asked only of women who had infertility treatment. If it is
   conditional, "missing" means "not applicable", and a woman who had no
   infertility treatment did not have ART. Recording that as unknown throws
   away a fact we hold.

   This script cross-tabulates RF_ARTEC against RF_INFTR and decides from
   the data rather than from assumption. If the pattern confirms
   conditionality, it recodes; if it does not, it stops and says so.

2. ILLB_R mixes two different absences. 888 means "not applicable" — the
   correct value for a first birth — while 999 means genuinely unknown.
   Step 5 lumped them. They are separated here, because "she has no
   previous birth" and "we failed to record the interval" are different
   facts, and a model that cannot tell them apart is being lied to.

THEN ENCODING

Nominal columns become one-hot: a separate 0/1 column per category. Without
this, a numeric code tells the model that breech (2) is larger than
cephalic (1), which is meaningless and lets the tree split on an ordering
that does not exist.

Ordinal columns stay numeric. MEDUC and PRECARE5 have a real order — more
education is more, later antenatal booking is later — so the ordering is
information, not an artefact.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

SRC = "model_data.parquet"
if not Path(SRC).exists():
    raise SystemExit(f"{SRC} not found. Run step4_5_clean.py first.")

d = pd.read_parquet(SRC)
print(f"Loaded {len(d):,} rows, {len(d.columns)} columns\n")

# =============================================================================
print("=" * 74)
print("1. IS RF_ARTEC CONDITIONAL ON RF_INFTR?")
print("=" * 74)

inf = d["RF_INFTR"]
art = d["RF_ARTEC"]
ct = pd.crosstab(inf.fillna(-1).map({1.0: "infertility treatment: YES",
                                     0.0: "infertility treatment: NO",
                                     -1: "infertility treatment: unknown"}),
                 art.isna().map({True: "ARTEC missing", False: "ARTEC recorded"}))
print(ct.to_string())

no_treat = d[inf == 0.0]
pct_missing_when_no = 100 * no_treat["RF_ARTEC"].isna().mean() if len(no_treat) else np.nan
yes_treat = d[inf == 1.0]
pct_missing_when_yes = 100 * yes_treat["RF_ARTEC"].isna().mean() if len(yes_treat) else np.nan

print(f"\n  ARTEC missing where there was NO infertility treatment : {pct_missing_when_no:.2f}%")
print(f"  ARTEC missing where there WAS infertility treatment    : {pct_missing_when_yes:.2f}%")

CONDITIONAL = pct_missing_when_no > 90 and pct_missing_when_yes < 25
if CONDITIONAL:
    print("""
  CONFIRMED conditional. The field is asked only of women who had
  infertility treatment, so "missing" means "not applicable".

  Recoding: no infertility treatment -> ART = 0. This is a factual recode,
  not imputation: a woman who had no fertility treatment did not conceive
  by assisted reproduction.
""")
    d.loc[inf == 0.0, "RF_ARTEC"] = 0.0
    print(f"  RF_ARTEC missing after recode: {100*d['RF_ARTEC'].isna().mean():.2f}%")
    print(f"  RF_ARTEC prevalence now      : {100*d['RF_ARTEC'].mean():.2f}%   "
          f"(published US ART share of births is roughly 2%)")
else:
    print("""
  NOT the expected conditional pattern. Do not recode. The field position
  or the code values need re-checking against the User Guide before
  RF_ARTEC is used at all. It is dropped from the feature set for now.
""")
    d = d.drop(columns=["RF_ARTEC"])

# =============================================================================
print("\n" + "=" * 74)
print("2. SEPARATING THE TWO KINDS OF ABSENCE IN ILLB_R")
print("=" * 74)
print("""
  Step 5 turned both 888 (not applicable — no previous birth) and 999
  (unknown) into the same missing value. PRIORLIVE already tells us which
  women have no previous birth, so a companion flag restores the
  distinction without re-reading the raw file.
""")
d["ILLB_first_birth"] = (pd.to_numeric(d["PRIORLIVE"], errors="coerce") == 0).astype(float)
print(f"  First births (no previous live birth): {100*d['ILLB_first_birth'].mean():.2f}%")
print(f"  ILLB_R missing                       : {100*d['ILLB_R'].isna().mean():.2f}%")
print("""
  The two should be close. Where ILLB_R is missing but this is NOT a first
  birth, the interval is genuinely unrecorded — that residual is the real
  missingness, and it is what belongs in the limitations.
""")
resid = ((d["ILLB_R"].isna()) & (d["ILLB_first_birth"] == 0)).mean()
print(f"  Genuinely unrecorded interval: {100*resid:.2f}% of all births")

# =============================================================================
print("\n" + "=" * 74)
print("3. ONE-HOT ENCODING")
print("=" * 74)

NOMINAL = ["ME_PRES"]          # cephalic / breech / other — no ordering
ORDINAL = ["MEDUC", "PRECARE5", "DPLURAL"]

print("  Nominal (one-hot):", NOMINAL)
print("  Ordinal (left numeric):", ORDINAL)
print("""
  ME_PRES is the only truly nominal clinical field in the set. Encoding it
  numerically would let a tree split on "presentation <= 1.5", which asserts
  an ordering between cephalic, breech and other that does not exist.
""")

for col in NOMINAL:
    s = d[col].astype(str).str.strip()
    labels = {"1": "cephalic", "2": "breech", "3": "other"}
    for code, name in labels.items():
        d[f"{col}_{name}"] = (s == code).astype(float)
        # keep genuine unknowns as missing rather than as three zeros
        d.loc[d[col].isna(), f"{col}_{name}"] = np.nan
    print(f"  {col} -> " + ", ".join(f"{col}_{n}" for n in labels.values()))
    d = d.drop(columns=[col])

# =============================================================================
FEATURES = [c for c in d.columns
            if c not in ["is_cesarean", "Robson_group", "Robson_group_sort_order",
                         "Classification_status"]]

print("\n" + "=" * 74)
print("FINAL FEATURE SET")
print("=" * 74)
print(f"  {len(FEATURES)} features\n")
for i, c in enumerate(FEATURES, 1):
    miss = 100 * d[c].isna().mean()
    print(f"  {i:>2}. {c:<22} {miss:>6.2f}% missing")

d.to_parquet("model_ready.parquet", index=False)
pd.Series(FEATURES).to_csv("feature_list.csv", index=False, header=["feature"])
print(f"\nSaved model_ready.parquet — {len(d):,} rows")
print("Saved feature_list.csv")
print("\nNext: step 7 — restrict the training population.")
