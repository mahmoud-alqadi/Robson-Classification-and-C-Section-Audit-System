"""
=============================================================================
EGYPT HEALTH ISSUES SURVEY 2015 — what it actually contains
=============================================================================
    python ehis_2015.py

TWO JOBS

1. Test why 43.8% and 51.8% disagree.
   EDHS 2014 measures caesareans among births in the five years before the
   survey. EHIS 2015 has no birth history at all — it asks total children
   ever born (a609) and how many of those were caesareans (a625). That is a
   LIFETIME measure. Because the caesarean rate has risen steeply, a
   lifetime measure carries older, lower-rate births and is dragged down.
   If sum(a625)/sum(a609) lands near 43.8%, the discrepancy is a difference
   of definition, not a contradiction — and that is worth stating, because
   the two figures are quoted side by side as though comparable.

2. Extract the national attitude and decision-timing questions.
   These are the demand side, on a nationally representative sample:
   whether a woman believes vaginal birth is possible after a caesarean,
   when the decision was taken, and the stated reasons.
=============================================================================
"""

import pandas as pd
import numpy as np

FILE = "EGIQ73FL.DTA"
print(f"Loading {FILE} ...")
raw = pd.read_stata(FILE, convert_categoricals=False)
lab = pd.read_stata(FILE, convert_categoricals=False, iterator=True).variable_labels()
print(f"  {len(raw):,} women, {len(raw.columns):,} variables\n")

# ---- weight ---------------------------------------------------------------
wcol = next((c for c in ["v005","awfactt","a005","weight"] if c in raw.columns), None)
if wcol:
    raw["w"] = pd.to_numeric(raw[wcol], errors="coerce") / 1_000_000
    if raw["w"].mean() < 0.001 or raw["w"].mean() > 100:
        raw["w"] = pd.to_numeric(raw[wcol], errors="coerce")
    print(f"Weight variable: {wcol} (mean {raw['w'].mean():.3f})\n")
else:
    raw["w"] = 1.0
    print("WARNING: no weight variable found — results are unweighted\n")

num = lambda c: pd.to_numeric(raw[c], errors="coerce") if c in raw.columns else None

# ---- 1. the lifetime caesarean measure ------------------------------------
births = num("a609")      # total children ever born
cs     = num("a625")      # of those, how many by caesarean
print("=" * 68)
print("1. LIFETIME CAESAREAN MEASURE")
print("=" * 68)
if births is None or cs is None:
    print("  a609 or a625 missing — check the variable names")
else:
    # women who have given birth; treat missing caesarean count as zero only
    # where the woman was asked and answered nothing meaningful
    d = raw.assign(births=births, cs=cs)
    d = d[d["births"].notna() & (d["births"] > 0) & (d["births"] < 30)]
    d["cs"] = d["cs"].where(d["cs"].between(0, 30), np.nan)
    known = d[d["cs"].notna()]

    tot_b = (known["births"] * known["w"]).sum()
    tot_c = (known["cs"]     * known["w"]).sum()
    print(f"  Women who have given birth, with a usable caesarean count: {len(known):,}")
    print(f"  Weighted total births      : {tot_b:,.0f}")
    print(f"  Weighted total caesareans  : {tot_c:,.0f}")
    print(f"  LIFETIME caesarean rate    : {100*tot_c/tot_b:.2f}%")
    print(f"  Unweighted equivalent      : {100*known['cs'].sum()/known['births'].sum():.2f}%")
    print(f"\n  Published EHIS 2015 figure : 43.8%")
    print(f"  Published EDHS 2014 figure : 51.8%  (births in the last five years)")
    print("""
  If the lifetime figure lands near 43.8%, the two national numbers are
  measuring different things and should never be plotted on one trend line
  without saying so.
""")
    # proportion of mothers who have ever had a caesarean
    ever = (known["cs"] > 0)
    print(f"  Mothers who have EVER had a caesarean: "
          f"{100*(ever*known['w']).sum()/known['w'].sum():.2f}%")

# ---- 2. attitudes and decision timing -------------------------------------
print("\n" + "=" * 68)
print("2. NATIONAL ATTITUDES AND DECISION TIMING")
print("=" * 68)

def show(var):
    if var not in raw.columns:
        print(f"  [{var}] not present"); return
    s = pd.to_numeric(raw[var], errors="coerce")
    d = raw[s.notna()]
    if not len(d):
        print(f"  [{var}] no valid responses"); return
    t = (d.groupby(s[s.notna()]).apply(lambda x: 100*x["w"].sum()/d["w"].sum())
           .round(2).rename("pct").to_frame())
    t["n"] = d.groupby(s[s.notna()]).size()
    print(f"\n  {var} — {lab.get(var,'')}")
    print(t.to_string())

for v in ["a623", "a626", "a627", "a630", "a628", "a631",
          "a622a", "a622b", "a622c", "a622d", "a622e", "a621"]:
    show(v)

print("""

  Codes are numeric. To read them, open the survey's recode map, or
  re-read the file with convert_categoricals=True for these columns.

  a623 is the one to look at first. Benha's authors named a
  "once a caesarean always a caesarean" culture as a reason for a 100%
  caesarean rate in Robson group 5. This variable measures that belief
  on a national sample — the belief and the practice, from two
  independent Egyptian sources.

  a626 gives decision timing nationally, which is the counterpart to the
  Qena finding that 77% of births were pre-labour caesareans.
""")

# ---- 3. save --------------------------------------------------------------
cols = [c for c in ["caseid","w","a602","a609","a625","a621","a623","a626",
                    "a627","a628","a630","a631","a622a","a622b","a622c",
                    "a622d","a622e","a110","v025","v190","v024"]
        if c in raw.columns]
raw[cols].to_csv("ehis_2015_extract.csv", index=False)
print(f"Saved ehis_2015_extract.csv ({len(cols)} columns)")
