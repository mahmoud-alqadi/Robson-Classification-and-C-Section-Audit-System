"""
=============================================================================
EGYPT — THE SAME TEST, RUN ON EGYPTIAN DATA  (v2)
=============================================================================
    python egypt_standardisation.py

TWO FIXES OVER THE FIRST VERSION

1. Reads EGIR61FL.DTA directly instead of the intermediate CSV, which had
   duplicated column names.

2. Maternal age is now computed EXACTLY at the date of each birth, from the
   century-month codes: (child's birth CMC - mother's birth CMC) / 12.
   The first version fell back to age at interview, which is wrong here and
   not harmlessly wrong: a woman interviewed in 2014 about a birth in 2010
   is recorded four years too old, and since age drives caesarean risk that
   error would have leaked straight into the standardisation it is meant to
   remove.

WHAT THIS TESTS
On US data the crude caesarean gap between payers inside Robson group 1 did
not survive adjustment — private 14.25% and self-pay 8.30% became 13.26%
and 13.56%. EDHS 2014 shows a crude private-versus-public gap of 21.5
points. Egyptian private-sector mothers are wealthier, more urban and lower
parity, all of which raise caesarean risk on their own, so the crude figure
means nothing yet.

direct_standardise is IMPORTED from stage3_confounding.py, not rewritten.
The two countries are analysed by the same function, not merely by similar
methods.

Home births are excluded: 13.2% of Egyptian births with a 0.00% caesarean
rate by definition, since a home birth that becomes a caesarean is recorded
as a facility birth. This mirrors the hospital-only restriction used on the
US data.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
try:
    from stage3_confounding import direct_standardise
    print("direct_standardise imported from stage3_confounding.py")
    print("  — the identical function applied to the CDC data.\n")
except Exception as e:
    raise SystemExit(f"Could not import stage3_confounding.py: {e}")

DTA = next((p for p in Path(".").glob("*.DTA") if "IR61" in p.name.upper()), None)
if DTA is None:
    raise SystemExit("EGIR61FL.DTA not found in this folder.")

WOMAN = ["caseid", "v005", "v011", "v024", "v025", "v106", "v190"]
BIRTH = ["m17", "m15", "bord", "b0", "b3"]
want = (WOMAN + [f"{x}_{i}" for x in BIRTH for i in range(1, 7)]
              + [f"{x}_{i:02d}" for x in BIRTH for i in range(1, 7)])

print(f"Loading {DTA.name} ...")
head = pd.read_stata(DTA, convert_categoricals=False, chunksize=1)
available = set(next(head).columns)
use = [c for c in want if c in available]
raw = pd.read_stata(DTA, columns=use, convert_categoricals=False)
print(f"  {len(raw):,} women, {len(use)} variables kept\n")

# ---- reshape to one row per birth ----------------------------------------
frames = []
for i in range(1, 7):
    ren = {}
    for x in BIRTH:
        for cand in (f"{x}_{i}", f"{x}_{i:02d}"):
            if cand in raw.columns:
                ren[cand] = x
                break
    if "m17" not in ren.values():
        continue
    keep = [c for c in WOMAN if c in raw.columns]
    sub = raw[keep + list(ren)].rename(columns=ren).copy()
    sub["birth_index"] = i
    frames.append(sub)

b = pd.concat(frames, ignore_index=True)
b = b.loc[:, ~b.columns.duplicated()]
b = b[b["m17"].notna()].copy()

b["weight"] = pd.to_numeric(b["v005"], errors="coerce") / 1_000_000
b["cs"] = (pd.to_numeric(b["m17"], errors="coerce") == 1).astype(float)
print(f"Births with a recorded mode of delivery: {len(b):,}")
print(f"Weighted caesarean rate (all births)   : "
      f"{100*(b['cs']*b['weight']).sum()/b['weight'].sum():.2f}%   (published 51.8%)\n")

# ---- sector ---------------------------------------------------------------
m15 = pd.to_numeric(b["m15"], errors="coerce")
b["sector"] = np.select(
    [m15.between(11, 12), m15.between(21, 26), m15.between(31, 36)],
    ["Home", "Public", "Private"], default="Other")

# ---- EXACT maternal age at the birth --------------------------------------
b3   = pd.to_numeric(b["b3"],   errors="coerce")
v011 = pd.to_numeric(b["v011"], errors="coerce")
b["age_at_birth"] = (b3 - v011) / 12.0
ok = b["age_at_birth"].between(10, 55)
print(f"Maternal age at birth computed for {100*ok.mean():.1f}% of births "
      f"(median {b.loc[ok,'age_at_birth'].median():.1f} years)\n")
b.loc[~ok, "age_at_birth"] = np.nan
b["age_band"] = pd.cut(b["age_at_birth"], [0, 19, 24, 29, 34, 99],
                       labels=["<20", "20-24", "25-29", "30-34", "35+"])

# ---- parity, wealth, residence -------------------------------------------
b["parity_band"] = pd.cut(pd.to_numeric(b["bord"], errors="coerce"),
                          [0, 1, 3, 99], labels=["1st", "2nd-3rd", "4th+"])
b["wealth"] = pd.to_numeric(b["v190"], errors="coerce")
b["urban"]  = pd.to_numeric(b["v025"], errors="coerce")

b["stratum"] = (b["wealth"].astype(str) + "|" + b["age_band"].astype(str)
                + "|" + b["parity_band"].astype(str))

f = b[b["sector"].isin(["Public", "Private"])].copy()
print(f"Facility births analysed: {len(f):,} "
      f"({100*len(f)/len(b):.1f}% of births; home births excluded)\n")

# =============================================================================
print("=" * 74)
print("1. CAESAREAN RATE BY SECTOR — CRUDE vs STANDARDISED")
print("   Strata: wealth quintile x maternal age at birth x parity")
print("=" * 74)
r = direct_standardise(f, "sector", "cs", min_cell=25)
print(r.round(2).to_string())

if {"Private", "Public"} <= set(r.index):
    crude = r.loc["Private","crude_%"] - r.loc["Public","crude_%"]
    std   = r.loc["Private","standardised_%"] - r.loc["Public","standardised_%"]
    print(f"\n  Private minus public, crude        : {crude:+.2f} pp")
    print(f"  Private minus public, standardised : {std:+.2f} pp")
    if crude:
        print(f"  Explained by case mix              : {crude-std:+.2f} pp "
              f"({100*(crude-std)/crude:.0f}% of the crude gap)")

print("\n" + "-" * 74)
print("Sensitivity — urban/rural added to the strata")
print("-" * 74)
f2 = f.copy(); f2["stratum"] = f2["stratum"] + "|" + f2["urban"].astype(str)
print(direct_standardise(f2, "sector", "cs", min_cell=20).round(2).to_string())

print("\n" + "-" * 74)
print("Sensitivity — first births only (the cleanest comparison available)")
print("-" * 74)
f3 = f[f["parity_band"] == "1st"].copy()
f3["stratum"] = f3["wealth"].astype(str) + "|" + f3["age_band"].astype(str)
print(f"  n = {len(f3):,}")
print(direct_standardise(f3, "sector", "cs", min_cell=20).round(2).to_string())
print("""
  First births carry no previous caesarean, so this removes the group 5
  effect entirely and isolates how each sector treats a woman with no
  uterine scar. It is the closest Egyptian equivalent to Robson group 1
  that DHS can support.
""")

# =============================================================================
print("=" * 74)
print("2. AUDIT — what differs between the sectors")
print("=" * 74)
for v, name in [("wealth","wealth quintile"), ("age_band","maternal age at birth"),
                ("parity_band","parity"), ("urban","residence (1 urban, 2 rural)")]:
    print(f"\n  {name} (% within sector):")
    print((100*pd.crosstab(f["sector"], f[v], normalize="index")).round(1).to_string())

print("\n" + "=" * 74)
print("3. THE COMPARISON THE PROJECT RESTS ON")
print("=" * 74)
print("""
  United States, Robson group 1, by payer
    crude          private 14.25%   Medicaid 11.97%   self-pay  8.30%
    standardised   private 13.26%   Medicaid 13.90%   self-pay 13.56%
    -> the gap disappears

  Egypt, facility births, by sector — see section 1 above.
""")

f.to_csv("egypt_sector_analysis.csv", index=False)
print("Saved egypt_sector_analysis.csv")
