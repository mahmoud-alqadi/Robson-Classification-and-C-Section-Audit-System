"""
=============================================================================
STAGE 4 — EXPANDED CONFOUNDING CONTROL
=============================================================================

WHAT STAGE 3 SHOWED, AND WHY WE ARE NOT DONE

Stage 3 standardised on maternal age, BMI, diabetes and hypertensive
disorder. The crude payer gradient (private 14.25% > Medicaid 11.97% >
self-pay 8.30%) did not survive: after standardisation Medicaid came out
HIGHEST (13.70% vs private 13.07%), because private insurance serves a
much older population and maternal age drives caesarean.

Two questions remain open, and both are answerable:

  1. Self-pay stood apart — lowest caesarean rate (9.04% standardised) but
     the HIGHEST maternal near-miss rate (0.630% vs 0.478% private). That
     pattern is what you would expect if a meaningful share of self-pay
     births are out-of-hospital. BFACIL (birth place) settles it.

  2. Age, BMI and comorbidity are not the only things that differ between
     payers. Smoking, timing of first prenatal visit and maternal education
     are all associated with both payer and caesarean risk. If the flat
     Apgar result survives adjustment for those too, it is a real finding.

NEW FIELDS ADDED (positions verified against UserGuide2023.pdf)

  BFACIL    pos 32       Birth place — 1 hospital, 2 birth centre,
                         3/4/5 home, 6 clinic, 7 other, 9 unknown
  CIG_REC   pos 269      Cigarette recode — Y/N/U
  PRECARE5  pos 227      Month prenatal care began, recoded —
                         1 = 1st trimester ... 4 = none, 5 = unknown
  MEDUC     pos 124      Mother's education, 1-8 (9 unknown)
  DMAR      pos 120      Marital status — 1 married, 2 unmarried
  RF_PPTERM pos 318      Previous preterm birth — Y/N/U

A DOCUMENTED LIMITATION, FROM NCHS ITSELF

The 2023 User Guide states that births attended by certified nurse
midwives are understated, because when more than one provider is present
the attendant is hard to identify — and that some hospitals require a
physician to be recorded as attendant even when no physician was
physically present at a midwife-attended birth.

This is the source agency confirming the artefact diagnosed in stage 2.
The attendant field is therefore reported descriptively only and is never
used as a practice-pattern comparator. Cite the User Guide for this, not
our own inference.

ALSO CONFIRMED BY THE USER GUIDE

NCHS defines its published low-risk caesarean rate as: singleton, term
(37+ weeks by obstetric estimate), cephalic caesareans to women having a
first birth, per 100 such births. That is exactly Robson groups 1 + 2a +
2b. Our 26.56% is therefore a direct comparison with a published official
statistic, not an approximation.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from stage3_confounding import direct_standardise

CACHE = "natality_2023_filtered.parquet"
CACHE_EXT = "natality_2023_extended.parquet"

NEW_COLSPECS = [
    (31, 32),     # BFACIL      pos 32      Birth Place
    (119, 120),   # DMAR        pos 120     Marital Status
    (123, 124),   # MEDUC       pos 124     Mother's Education
    (226, 227),   # PRECARE5    pos 227     Month Prenatal Care Began Recode
    (268, 269),   # CIG_REC     pos 269     Cigarette Recode
    (317, 318),   # RF_PPTERM   pos 318     Previous Preterm Birth
]
NEW_NAMES = ["BFACIL", "DMAR", "MEDUC", "PRECARE5", "CIG_REC", "RF_PPTERM"]


# =============================================================================
def extend_cache(raw_path):
    """
    Read ONLY the new columns from the raw file and attach them to the
    existing cache by row position.

    Row alignment is safe because both reads use the same file in the same
    order with no row filtering. The row-count assertion below is the guard:
    if it ever fails, do not proceed.
    """
    if Path(CACHE_EXT).exists():
        print(f"Loading {CACHE_EXT} ...")
        return pd.read_parquet(CACHE_EXT)

    if not Path(CACHE).exists():
        raise SystemExit(f"{CACHE} not found — run robson_analysis.py first.")

    print(f"Loading {CACHE} ...")
    base = pd.read_parquet(CACHE)
    print(f"  {len(base):,} rows")

    print(f"Reading {len(NEW_NAMES)} new columns from {raw_path} ...")
    frames = []
    for chunk in pd.read_fwf(raw_path, colspecs=NEW_COLSPECS, names=NEW_NAMES,
                             dtype=str, chunksize=500_000):
        frames.append(chunk)
    new = pd.concat(frames, ignore_index=True)
    print(f"  {len(new):,} rows")

    assert len(new) == len(base), (
        f"ROW COUNT MISMATCH: cache has {len(base):,}, new read has {len(new):,}. "
        "Do not proceed — the row alignment assumption is broken."
    )

    out = pd.concat([base.reset_index(drop=True), new.reset_index(drop=True)], axis=1)
    print(f"Saving {CACHE_EXT} ...")
    out.to_parquet(CACHE_EXT, index=False)
    print("Done.\n")
    return out


def build_strata_v2(d):
    """
    Wider adjustment set than stage 3, but deliberately coarser bands so the
    number of strata stays manageable and cell coverage stays high.
    A perfectly adjusted comparison with 3% coverage is worth nothing.
    """
    age = pd.to_numeric(d["MAGER"], errors="coerce")
    d["age_band"] = pd.cut(age, [0, 19, 24, 29, 34, 99],
                           labels=["<20", "20-24", "25-29", "30-34", "35+"])

    bmi = pd.to_numeric(d["BMI"], errors="coerce")
    bmi = bmi.where(bmi < 99)
    d["bmi_band"] = pd.cut(bmi, [0, 18.5, 25, 30, 99],
                           labels=["<18.5", "18.5-24", "25-29", "30+"])

    yes = {"Y"}
    diab = (d["RF_PDIAB"].astype(str).str.strip().isin(yes) |
            d["RF_GDIAB"].astype(str).str.strip().isin(yes))
    hyp = (d["RF_PHYPE"].astype(str).str.strip().isin(yes) |
           d["RF_GHYPE"].astype(str).str.strip().isin(yes) |
           d["RF_EHYPE"].astype(str).str.strip().isin(yes))
    d["comorbid"] = (diab | hyp)

    d["smoker"] = d["CIG_REC"].astype(str).str.strip() == "Y"

    pc = d["PRECARE5"].astype(str).str.strip()
    d["prenatal"] = np.select(
        [pc == "1", pc.isin(["2", "3"]), pc == "4"],
        ["1st trimester", "later", "none"],
        default="unknown")

    d["stratum"] = (d["age_band"].astype(str) + "|" + d["bmi_band"].astype(str)
                    + "|" + d["comorbid"].astype(str) + "|" + d["smoker"].astype(str)
                    + "|" + d["prenatal"])
    return d


# =============================================================================
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("path", nargs="?", default="Nat2023PublicUS.txt")
    args = p.parse_args()

    df = extend_cache(args.path)

    d = df[(df["Classification_status"] == "Classified") &
           (df["Robson_group"] == "1")].copy()
    d["apgar5_lt7"] = pd.to_numeric(d["APGAR5"], errors="coerce").lt(7)
    d["nicu"] = d["AB_NICU"].astype(str).str.strip() == "Y"

    bf = d["BFACIL"].astype(str).str.strip()
    d["setting"] = np.select(
        [bf == "1", bf == "2", bf.isin(["3", "4", "5"]), bf.isin(["6", "7"])],
        ["Hospital", "Birth centre", "Home", "Clinic/other"],
        default="Unknown")

    # ---- 1. Does birth place explain self-pay? ----------------------------
    print("=" * 78)
    print("1. WHERE DO THESE BIRTHS HAPPEN? (Robson group 1, by payer)")
    print("=" * 78)
    print(f"\nGroup 1 total: {len(d):,}\n")
    print("  Setting distribution overall:")
    print((100 * d["setting"].value_counts(normalize=True)).round(3).to_string())
    print("\n  Setting by payer (% within payer):")
    print((100 * pd.crosstab(d["payer"], d["setting"], normalize="index")).round(2).to_string())
    print("\n  Caesarean rate and near-miss by setting:")
    t = d.groupby("setting").agg(n=("is_cesarean", "size"),
                                 cs_rate=("is_cesarean", "mean"),
                                 near_miss=("near_miss_flag", "mean"),
                                 apgar5_lt7=("apgar5_lt7", "mean"))
    for c in ["cs_rate", "near_miss", "apgar5_lt7"]:
        t[c] = (100 * t[c]).round(3)
    print(t.to_string())
    print("""
  If self-pay is disproportionately out-of-hospital, its low caesarean
  rate and high near-miss rate are a property of the SETTING, not of the
  payment method — and the comparison must be re-run inside hospitals only.
""")

    # ---- 2. Hospital-only, wider adjustment -------------------------------
    h = d[d["setting"] == "Hospital"].copy()
    h = build_strata_v2(h)
    h = h[h["payer"].isin(["Private", "Medicaid", "Self-pay"])]

    print("=" * 78)
    print("2. HOSPITAL BIRTHS ONLY — WIDER ADJUSTMENT")
    print("   age x BMI x comorbidity x smoking x prenatal-care timing")
    print("=" * 78)
    print(f"\n  Hospital group 1 births: {len(h):,}\n")
    print("  Caesarean rate:")
    print(direct_standardise(h, "payer", "is_cesarean").round(2).to_string())

    for outcome, label in [("apgar5_lt7", "5-minute Apgar below 7"),
                           ("nicu", "NICU admission"),
                           ("near_miss_flag", "Maternal near-miss")]:
        print(f"\n  {label}:")
        print(direct_standardise(h, "payer", outcome).round(3).to_string())

    print("""
  THE TEST. Compare the caesarean row against the Apgar row.

  If the payer groups differ materially in standardised caesarean rate but
  not in standardised 5-minute Apgar below 7, then within hospitals, among
  nulliparous women at term with a single cephalic fetus who entered labour
  spontaneously and have no uterine scar, and holding age, BMI, comorbidity,
  smoking and prenatal care constant — the extra caesareans are not
  producing a measurable neonatal benefit.

  Caveat to keep: NICU admission is not a neutral outcome here. Caesarean
  delivery itself causes transient tachypnoea of the newborn and raises NICU
  admission. NICU is partly a CONSEQUENCE of the intervention, not an
  independent measure of whether it was needed. Apgar is the cleaner
  endpoint of the two.
""")

    # ---- 3. Time of day, hospital only ------------------------------------
    print("=" * 78)
    print("3. TIME OF DAY — HOSPITAL BIRTHS ONLY, WIDER ADJUSTMENT")
    print("=" * 78)
    print(direct_standardise(h, "time_of_day", "is_cesarean").round(2).to_string())
    print("""
  Clinical caveat, stated before anyone else raises it: spontaneous labour
  has a circadian rhythm, with onset and delivery clustering overnight. A
  lower caesarean rate in the small hours may be physiology rather than
  decision-making. The evening rise is the harder half to explain that way,
  but this analysis cannot separate fatigue, shift handover and staffing
  from chance. Report the gradient; do not name a cause.
""")

    # ---- 4. Audit -------------------------------------------------------
    print("=" * 78)
    print("4. AUDIT — what differs between payers, hospital births only")
    print("=" * 78)
    for v in ["age_band", "smoker", "prenatal", "comorbid"]:
        print(f"\n  {v} (% within payer):")
        print((100 * pd.crosstab(h["payer"], h[v], normalize="index")).round(1).to_string())

    print("\n" + "=" * 78)
    print("STAGE 4 COMPLETE.")
    print("=" * 78)
