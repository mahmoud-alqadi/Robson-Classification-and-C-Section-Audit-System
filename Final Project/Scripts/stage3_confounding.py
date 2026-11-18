"""
=============================================================================
STAGE 3 — THE CLEAN TEST
Does a higher caesarean rate buy better outcomes?
=============================================================================

WHY THIS SCRIPT EXISTS

Stage 2 printed a line inviting the reader to compare safety outcomes ACROSS
Robson groups. That invitation was wrong and is withdrawn here.

Comparing group 2a against group 1 is confounded by indication. Labour is
induced FOR a reason — hypertension, diabetes, growth restriction,
post-dates. Worse neonatal outcomes in group 2a may reflect the indication
rather than the induction. The ARRIVE randomised trial found the opposite
for elective induction at 39 weeks, so observational data of this shape is
not evidence against induction. The same objection applies to comparing
group 1 against group 3: nulliparity is itself a risk factor.

Robson groups differ in exactly the characteristics that drive both
caesarean and outcome. Between-group causal claims are not available.

THE VALID COMPARISON is WITHIN one group, ACROSS practice settings. Robson
group 1 fixes the clinical population — nulliparous, term, singleton,
cephalic, spontaneous labour. Every woman arrived in labour on her own and
no caesarean in the group was scheduled. If one payer stratum operates at a
14.25% caesarean rate and another at 8.30%, and the babies do equally well,
the extra caesareans are not buying safety.

Residual confounding remains: payer correlates with maternal age, BMI and
comorbidity. So we do not compare crude rates. We standardise.

METHOD — DIRECT STANDARDISATION
Split group 1 into strata of maternal age x BMI x diabetes x hypertension.
Compute each payer's caesarean rate WITHIN each stratum, then reweight to
the risk profile of the whole group:

    standardised rate = SUM_over_strata ( w_s  x  rate_payer,s )
    where w_s = that stratum's share of all group 1 births

This answers: "if every payer served an identical mix of women, what would
each one's caesarean rate be?"

It is the same technique the project will use for Egypt in phase two, run
here on US data first as a rehearsal — and it is checkable, because unlike
the Egypt calculation we can see the true answer alongside it.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

CACHE = "natality_2023_filtered.parquet"


# =============================================================================
# RISK STRATA
# =============================================================================

def build_strata(d):
    """Maternal age x BMI x any diabetes x any hypertensive disorder."""
    age = pd.to_numeric(d["MAGER"], errors="coerce")
    d["age_band"] = pd.cut(age, [0, 19, 24, 29, 34, 39, 99],
                           labels=["<20", "20-24", "25-29", "30-34", "35-39", "40+"])

    bmi = pd.to_numeric(d["BMI"], errors="coerce")
    bmi = bmi.where(bmi < 99)                      # 99.9 = unknown
    d["bmi_band"] = pd.cut(bmi, [0, 18.5, 25, 30, 35, 99],
                           labels=["<18.5", "18.5-24", "25-29", "30-34", "35+"])

    yes = {"Y"}
    d["diabetes"] = (d["RF_PDIAB"].astype(str).str.strip().isin(yes) |
                     d["RF_GDIAB"].astype(str).str.strip().isin(yes))
    d["hypertension"] = (d["RF_PHYPE"].astype(str).str.strip().isin(yes) |
                         d["RF_GHYPE"].astype(str).str.strip().isin(yes) |
                         d["RF_EHYPE"].astype(str).str.strip().isin(yes))

    d["stratum"] = (d["age_band"].astype(str) + "|" + d["bmi_band"].astype(str)
                    + "|" + d["diabetes"].astype(str) + "|" + d["hypertension"].astype(str))
    return d


def direct_standardise(d, group_col, outcome_col, min_cell=30):
    """
    Reweight each group's stratum-specific rate to the population risk profile.
    Strata where a group has fewer than min_cell births are dropped for that
    group and its weights renormalised, so a thin cell cannot swing the result.
    """
    valid = d[~d["stratum"].str.contains("nan")]
    weights = valid["stratum"].value_counts(normalize=True)

    out = {}
    for g, sub in valid.groupby(group_col, observed=True):
        rates = sub.groupby("stratum", observed=True)[outcome_col].agg(["mean", "size"])
        rates = rates[rates["size"] >= min_cell]
        if rates.empty:
            continue
        w = weights.reindex(rates.index).fillna(0)
        w = w / w.sum()
        out[g] = {
            "crude_%": 100 * sub[outcome_col].mean(),
            "standardised_%": 100 * float((w * rates["mean"]).sum()),
            "n": len(sub),
            "strata_used": len(rates),
            "coverage_%": 100 * float(weights.reindex(rates.index).fillna(0).sum()),
        }
    return pd.DataFrame(out).T


# =============================================================================
if __name__ == "__main__":
    if not Path(CACHE).exists():
        raise SystemExit(f"{CACHE} not found — run robson_analysis.py first.")

    print(f"Loading {CACHE} ...")
    df = pd.read_parquet(CACHE)

    d = df[(df["Classification_status"] == "Classified") &
           (df["Robson_group"] == "1")].copy()
    print(f"Robson group 1: {len(d):,} births\n")

    d = build_strata(d)
    d["apgar5_lt7"] = pd.to_numeric(d["APGAR5"], errors="coerce").lt(7)
    d["nicu"] = d["AB_NICU"].astype(str).str.strip() == "Y"

    d = d[d["payer"].isin(["Private", "Medicaid", "Self-pay"])]

    # ---- 1. Is the caesarean difference real, or is it case mix? -----------
    print("=" * 76)
    print("1. CAESAREAN RATE BY PAYER, WITHIN ROBSON GROUP 1")
    print("   Standardised to the group 1 risk profile")
    print("   (maternal age x BMI x diabetes x hypertensive disorder)")
    print("=" * 76)
    cs = direct_standardise(d, "payer", "is_cesarean")
    print(cs.round(2).to_string())
    print("""
  Crude vs standardised is the whole point. If the gap between payers
  survives standardisation, it is not explained by the women — it is
  explained by the practice.
""")

    # ---- 2. Do the extra caesareans buy better outcomes? -------------------
    print("=" * 76)
    print("2. NEONATAL OUTCOMES BY PAYER, SAME STANDARDISATION")
    print("=" * 76)
    for outcome, label in [("apgar5_lt7", "5-minute Apgar below 7"),
                           ("nicu", "NICU admission"),
                           ("near_miss_flag", "Maternal near-miss")]:
        print(f"\n  {label}:")
        print(direct_standardise(d, "payer", outcome).round(3).to_string())

    print("""
  How to read this. If the payer stratum with the HIGHEST standardised
  caesarean rate does NOT show correspondingly better neonatal outcomes,
  the additional caesareans are not producing a safety benefit in a
  population where every woman entered labour spontaneously at term with
  a single cephalic fetus and no uterine scar.

  That is the strongest form this argument can take on observational
  data, and it is made entirely on US records — before Egypt is
  mentioned at all.
""")

    # ---- 3. Time of day, same treatment -----------------------------------
    print("=" * 76)
    print("3. CAESAREAN RATE BY TIME OF DAY, WITHIN GROUP 1 (standardised)")
    print("=" * 76)
    print(direct_standardise(d, "time_of_day", "is_cesarean").round(2).to_string())
    print("""
  A woman's physiology does not know the hour. Any surviving gradient is
  a property of the service, not of the labour.
""")

    # ---- 4. What the standardisation actually did -------------------------
    print("=" * 76)
    print("4. AUDIT — case mix by payer (why standardisation was needed)")
    print("=" * 76)
    for v in ["age_band", "bmi_band", "diabetes", "hypertension"]:
        t = pd.crosstab(d["payer"], d[v], normalize="index") * 100
        print(f"\n  {v} (% within payer):")
        print(t.round(1).to_string())

    print("\n" + "=" * 76)
    print("STAGE 3 COMPLETE.")
    print("=" * 76)
