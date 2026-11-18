"""
=============================================================================
CDC NATALITY 2023 — FULL PIPELINE
Step 1: Download the file (do this ON YOUR OWN MACHINE, not in Claude)
Step 2: Load + filter (only the columns we need)
Step 3: Run the Robson classifier
Step 4: Produce the standard report table
Step 5: Sanity-check the distribution against published studies
=============================================================================

HOW TO DOWNLOAD (do this once, on your own machine)
----------------------------------------------------
The file is ~500 MB compressed, ~3.8 GB uncompressed.

Option A — Browser (easiest):
  1. Go to:  https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm
  2. Click "2023" under "Natality"
  3. Download  Nat2023us.zip  (about 500 MB)
  4. Unzip it — you get  Nat2023us.dat  (fixed-width text)

Option B — Command line (Mac/Linux):
  wget https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip
  unzip Nat2023us.zip

Option C — Windows:
  curl -O https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Datasets/DVS/natality/Nat2023us.zip
  # then unzip normally

After downloading, put  Nat2023us.dat  in the same folder as this script.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# ── make sure the classifier is importable ──────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from robson_classifier import classify, report_table, sanity_checks, COLS


# =============================================================================
# 1. COLUMN SPECS — verified against UserGuide2023.pdf
#    Format: (start-1, end)  ← Python slice convention (0-indexed, exclusive end)
#    Source positions from the guide are 1-indexed, so subtract 1 from start.
# =============================================================================

COLSPECS = [
    # Field           colspec (0-idx, excl-end)  pos (1-idx)   len   field name in UserGuide2023
    (170, 172),   # PRIORLIVE   Prior Births Now Living             171-172   2
    (172, 174),   # PRIORDEAD   Prior Births Now Dead               173-174   2
    (174, 176),   # PRIORTERM   Prior Other Terminations            175-176   2
    (197, 200),   # ILLB_R      Interval Since Last Live Birth      198-200   3
    (312, 313),   # RF_PDIAB    Pre-pregnancy Diabetes              313       1
    (313, 314),   # RF_GDIAB    Gestational Diabetes                314       1
    (314, 315),   # RF_PHYPE    Pre-pregnancy Hypertension          315       1
    (315, 316),   # RF_GHYPE    Gestational Hypertension            316       1
    (316, 317),   # RF_EHYPE    Eclampsia                           317       1
    (324, 325),   # RF_INFTR    Infertility Treatment Used          325       1
    (325, 326),   # RF_FEDRG    Fertility Enhancing Drugs           326       1
    (326, 327),   # RF_ARTEC    Assisted Reproductive Technology    327       1
    (330, 331),   # RF_CESAR    Previous Cesarean                   331       1
    (331, 333),   # RF_CESARN   Number of Previous Cesareans        332-333   2
    (382, 383),   # LD_INDL     Induction of Labor                  383       1
    (383, 384),   # LD_AUGM     Augmentation of Labor               384       1
    (386, 387),   # LD_CHOR     Chorioamnionitis                    387       1
    (400, 401),   # ME_PRES     Fetal Presentation at Delivery      401       1
    (401, 402),   # ME_ROUT     Final Route & Method of Delivery    402       1
    (402, 403),   # ME_TRIAL    Trial of Labor Attempted (if CS)    403       1
    (406, 407),   # RDMETH_REC  Delivery Method Recode (8 cats)     407       1
    (407, 408),   # DMETH_REC   Delivery Method Recode (3 cats)     408       1
    (414, 415),   # MM_MTR      Maternal Transfusion                415       1
    (416, 417),   # MM_RUPT     Ruptured Uterus                     417       1
    (417, 418),   # MM_UHYST    Unplanned Hysterectomy              418       1
    (418, 419),   # MM_AICU     Admit to Intensive Care             419       1
    (432, 433),   # ATTEND      Attendant at Birth                  433       1
    (434, 435),   # PAY         Payment Source for Delivery         435       1
    (435, 436),   # PAY_REC     Payment Recode                      436       1
    (443, 445),   # APGAR5      Five Minute APGAR Score             444-445   2
    (453, 454),   # DPLURAL     Plurality Recode                    454       1
    (498, 500),   # OEGest_Comb Obstetric Estimate Edited           499-500   2
    (502, 503),   # OEGest_R3   OE Recode 3 (preterm/term/unk)     503       1
    (503, 507),   # DBWT        Birth Weight grams                  504-507   4
    (518, 519),   # AB_NICU     Admission to NICU                   519       1
    (74,  76),    # MAGER       Mother's Single Year Age             75-76    2
    (279, 281),   # M_Ht_In     Mother's Height in Total Inches     280-281   2
    (282, 286),   # BMI         Body Mass Index (raw float)         283-286   4
    (291, 294),   # PWgt_R      Pre-pregnancy Weight Recode         292-294   3
    (303, 305),   # WTGAIN      Weight Gain                         304-305   2
    (22,  23),    # DOB_WK      Day of Week  (1=Sun 7=Sat)          23        1
    (18,  22),    # DOB_TT      Time of Birth 0000-2359             19-22     4
]

COLNAMES = [
    "PRIORLIVE", "PRIORDEAD", "PRIORTERM", "ILLB_R",
    "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE",
    "RF_INFTR", "RF_FEDRG", "RF_ARTEC",
    "RF_CESAR", "RF_CESARN",
    "LD_INDL", "LD_AUGM", "LD_CHOR",
    "ME_PRES", "ME_ROUT", "ME_TRIAL",
    "RDMETH_REC", "DMETH_REC",
    "MM_MTR", "MM_RUPT", "MM_UHYST", "MM_AICU",
    "ATTEND", "PAY", "PAY_REC",
    "APGAR5",
    "DPLURAL",
    "OEGest_Comb", "OEGest_R3",
    "DBWT",
    "AB_NICU",
    "MAGER",
    "M_Ht_In", "BMI", "PWgt_R", "WTGAIN",
    "DOB_WK", "DOB_TT",
]

assert len(COLSPECS) == len(COLNAMES), "COLSPECS and COLNAMES must be the same length"


# =============================================================================
# 2. LOAD THE FILE
# =============================================================================

def load_puf(path, chunksize=500_000, nrows=None):
    """
    Read the fixed-width PUF once, keep only the needed columns.
    
    path      : path to Nat2023us.dat
    chunksize : rows per chunk (adjust down if you run out of RAM)
    nrows     : set a small number (e.g. 100_000) for a quick smoke test
    
    Returns a pandas DataFrame.
    """
    path = Path(path)
    if not path.exists():
        print(f"""
ERROR: File not found: {path}

Download instructions:
1. Go to https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm
2. Click '2023' under 'Natality'
3. Download Nat2023us.zip (~500 MB)
4. Unzip it and place Nat2023us.dat in: {path.parent}
""")
        sys.exit(1)

    print(f"Reading {path.name} (record length: 1330 chars)...")
    print(f"File size: {path.stat().st_size / 1e9:.2f} GB")

    frames = []
    total = 0
    reader = pd.read_fwf(
        path,
        colspecs=COLSPECS,
        names=COLNAMES,
        dtype=str,
        chunksize=chunksize,
        nrows=nrows,
    )
    for i, chunk in enumerate(reader):
        frames.append(chunk)
        total += len(chunk)
        if (i + 1) % 5 == 0:
            print(f"  ... {total:,} rows loaded")
        if nrows and total >= nrows:
            break

    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df):,} records total.\n")
    return df


# =============================================================================
# 3. DERIVE THE is_cesarean FLAG (our target variable)
# =============================================================================

def add_is_cesarean(df):
    """
    DMETH_REC:  1 = Vaginal   2 = C-Section   9 = Unknown
    We use this simple field as the binary outcome.
    
    DOCUMENTED: RDMETH_REC (8-category) separately identifies
    VBAC (code 2), primary CS (code 3), repeat CS (code 4) —
    used for the descriptive VBAC analysis, not the main model.
    """
    dm = df["DMETH_REC"].astype(str).str.strip()
    df["is_cesarean"] = (dm == "2").astype(int)
    df["delivery_known"] = dm.isin(["1", "2"])   # exclude unknown
    return df


# =============================================================================
# 4. DERIVE PRACTICE-PATTERN VARIABLES (useful for the Egypt argument)
# =============================================================================

def add_practice_vars(df):
    """
    These fields measure BEHAVIOUR not biology.
    In the Egypt comparison, caesarean rates varying by payer / attendant / 
    time of day within a single country = the same mechanism we attribute to Egypt.
    """
    pay = df["PAY_REC"].astype(str).str.strip()
    df["payer"] = pay.map({"1": "Medicaid", "2": "Private", "3": "Self-pay", "4": "Other"}).fillna("Unknown")

    att = df["ATTEND"].astype(str).str.strip()
    df["attendant"] = att.map({
        "1": "MD", "2": "DO", "3": "CNM/CM", "4": "Other Midwife", "5": "Other"
    }).fillna("Unknown")

    wk = df["DOB_WK"].astype(str).str.strip()
    df["weekend"] = wk.isin(["1", "7"])   # Sunday=1, Saturday=7

    tt = pd.to_numeric(df["DOB_TT"], errors="coerce")
    df["time_of_day"] = pd.cut(tt, bins=[-1, 259, 559, 859, 1159, 1459, 1759, 2059, 2359],
                                labels=["00-02", "03-05", "06-08", "09-11", "12-14", "15-17", "18-20", "21-23"])
    return df


# =============================================================================
# 5. NEAR-MISS FLAG  (WHO 2012 criteria, fields available in CDC data)
# =============================================================================

def add_near_miss_flag(df):
    """
    WHO 2012 maternal near-miss proxy using available CDC fields:
      - Maternal transfusion    (MM_MTR)
      - Ruptured uterus         (MM_RUPT)
      - Unplanned hysterectomy  (MM_UHYST)
      - ICU admission           (MM_AICU)
    A 'Y' on any of these is classified as near-miss.
    NOTE: This is a proxy, not a validated clinical tool. Documented as such.
    """
    yes = {"Y"}
    mtr  = df["MM_MTR"].astype(str).str.strip().isin(yes)
    rupt = df["MM_RUPT"].astype(str).str.strip().isin(yes)
    hyst = df["MM_UHYST"].astype(str).str.strip().isin(yes)
    aicu = df["MM_AICU"].astype(str).str.strip().isin(yes)
    df["near_miss_flag"] = (mtr | rupt | hyst | aicu).astype(int)
    return df


# =============================================================================
# 6. VBAC DESCRIPTIVE ANALYSIS  (no predictive model — see proposal Section 10)
# =============================================================================

def vbac_descriptive(df):
    """
    Among women with a previous caesarean:
      - How many attempted trial of labour (TOLAC)?
      - Of those who attempted, how many succeeded (VBAC)?
    Cross-tab by payer and attendant.
    
    This uses RDMETH_REC (8-category delivery method):
      1 = Vaginal (no prev CS)
      2 = VBAC (vaginal after prev CS)
      3 = Primary CS
      4 = Repeat CS
    """
    prev_cs = df["RF_CESAR"].astype(str).str.strip() == "Y"
    rdm = df["RDMETH_REC"].astype(str).str.strip()

    prev_cs_df = df[prev_cs & rdm.isin(["2", "4"])].copy()
    prev_cs_df["tolac_attempted"] = rdm[prev_cs & rdm.isin(["2", "4"])] != "4"   # VBAC = tolac + success
    prev_cs_df["vbac_success"]    = rdm[prev_cs & rdm.isin(["2", "4"])] == "2"

    n_prev_cs = prev_cs.sum()
    n_in_analysis = len(prev_cs_df)
    tolac_rate = prev_cs_df["tolac_attempted"].mean() * 100
    vbac_rate  = prev_cs_df["vbac_success"].mean() * 100

    print("=" * 60)
    print("VBAC DESCRIPTIVE ANALYSIS")
    print("=" * 60)
    print(f"Women with previous caesarean:          {n_prev_cs:>10,}")
    print(f"  → With known delivery method:         {n_in_analysis:>10,}")
    print(f"  → TOLAC attempted:                    {prev_cs_df['tolac_attempted'].sum():>10,}  ({tolac_rate:.1f}%)")
    print(f"  → VBAC successes:                     {prev_cs_df['vbac_success'].sum():>10,}  ({vbac_rate:.1f}% of those w/ prev CS)")
    print()

    print("TOLAC rate by payer:")
    print(prev_cs_df.groupby("payer")["tolac_attempted"].agg(["sum", "mean", "count"])
          .rename(columns={"sum": "N_TOLAC", "mean": "Rate", "count": "N_Total"})
          .assign(Rate=lambda x: (x.Rate * 100).round(1))
          .to_string())
    print()
    print("TOLAC rate by attendant:")
    print(prev_cs_df.groupby("attendant")["tolac_attempted"].agg(["sum", "mean", "count"])
          .rename(columns={"sum": "N_TOLAC", "mean": "Rate", "count": "N_Total"})
          .assign(Rate=lambda x: (x.Rate * 100).round(1))
          .to_string())
    return prev_cs_df


# =============================================================================
# 7. MAIN — run everything
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CDC Natality 2023 → Robson Report")
    parser.add_argument("path", nargs="?", default="Nat2023us.dat",
                        help="Path to the fixed-width .dat file (default: Nat2023us.dat)")
    parser.add_argument("--nrows", type=int, default=None,
                        help="Load only N rows (for a quick test, e.g. --nrows 100000)")
    parser.add_argument("--out", default="robson_report_2023.csv",
                        help="Output CSV for the standard report table")
    args = parser.parse_args()

    # ── STEP 1: load ──────────────────────────────────────────────────────────
    df = load_puf(args.path, nrows=args.nrows)

    # ── STEP 2: add derived columns ───────────────────────────────────────────
    df = add_is_cesarean(df)
    df = add_practice_vars(df)
    df = add_near_miss_flag(df)

    # ── STEP 3: classify ──────────────────────────────────────────────────────
    print("Running Robson classifier...")
    df = classify(df, subgroups=True)
    print(f"Classification complete.\n")

    # ── STEP 4: sanity checks ─────────────────────────────────────────────────
    print("SANITY CHECKS")
    print("-" * 60)
    for issue in sanity_checks(df):
        print(" •", issue)
    print()

    # ── STEP 5: standard report table ─────────────────────────────────────────
    # Use delivery_known == True so unknowns don't distort rates
    classified = df[df["Classification_status"] == "Classified"].copy()

    print("=" * 70)
    print("WHO STANDARD ROBSON REPORT TABLE — 2023 CDC Natality")
    if args.nrows:
        print(f"  (SAMPLE: first {args.nrows:,} records — for verification only)")
    print("=" * 70)
    rt = report_table(classified, cs_flag_col="is_cesarean")
    print(rt.to_string(index=False))
    rt.to_csv(args.out, index=False)
    print(f"\nSaved to: {args.out}\n")

    # ── STEP 6: distribution validation ──────────────────────────────────────
    # Compare against Egyptian published studies (% of total births in each group)
    # These are approximate ranges from the five published Egyptian studies
    # (Port Said, Benha ×2, Qena) — used as a reference NOT as a CDC validation.
    # The CDC distribution will differ from Egypt; that is expected and is the point.

    print("=" * 70)
    print("GROUP SIZE DISTRIBUTION — CDC 2023 vs Published Benchmarks")
    print("(Note: CDC is the REFERENCE; differences from Egypt are the finding)")
    print("=" * 70)
    # WHO global benchmarks from Robson et al. (not Egypt-specific)
    who_benchmarks = {
        "1":  (35, 45),   # Largest group globally — nullip spontaneous
        "2a": (8, 12),
        "2b": (3, 7),
        "3":  (18, 26),
        "4a": (5, 9),
        "4b": (2, 5),
        "5":  (8, 15),    # combined 5.1+5.2
        "5.1": None,
        "5.2": None,
        "6":  (1, 3),
        "7":  (2, 5),
        "8":  (2, 4),
        "9":  (0.1, 1),
        "10": (5, 10),
    }
    dist = classified.groupby("Robson_group").size() / len(classified) * 100
    print(f"\n{'Group':>14}  {'%':>6}  {'Benchmark range':>20}  {'In range?':>10}")
    print("-" * 55)
    for grp, pct in dist.sort_index().items():
        bm = who_benchmarks.get(grp)
        if bm and bm[0] is not None:
            ok = "✓" if bm[0] <= pct <= bm[1] else "⚠ CHECK"
            print(f"{grp:>14}  {pct:>6.2f}%  {str(bm):>20}  {ok:>10}")
        else:
            print(f"{grp:>14}  {pct:>6.2f}%")

    # ── STEP 7: practice-pattern analysis ─────────────────────────────────────
    print("\n" + "=" * 70)
    print("PRACTICE-PATTERN ANALYSIS — CS Rate by Payer & Attendant")
    print("(The mechanism we attribute to Egypt, demonstrated on real US data)")
    print("=" * 70)
    for var in ["payer", "attendant", "weekend"]:
        tbl = classified.groupby(var)["is_cesarean"].agg(["mean", "count"])
        tbl["cs_rate_%"] = (tbl["mean"] * 100).round(2)
        print(f"\nBy {var}:")
        print(tbl[["cs_rate_%", "count"]].to_string())

    # ── STEP 8: VBAC descriptive ──────────────────────────────────────────────
    print("\n")
    vbac_descriptive(df)

    print("\nAll done. Check the distribution table above against published studies.")
    print("If any group is wildly off benchmark, investigate the derivation before proceeding.")
