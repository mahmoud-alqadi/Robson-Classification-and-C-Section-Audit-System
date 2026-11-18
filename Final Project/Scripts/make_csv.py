"""
=============================================================================
MAKE CSV — converts the analysis dataset to CSV
=============================================================================
Run:   python make_csv.py

Reads natality_2023_extended.parquet (the 69 MB file you already have) and
writes two CSVs plus the report table.

WHY NOT THE RAW .txt FILE
The raw file is fixed-width: columns are defined by character POSITION, not
by commas. Renaming it to .csv changes nothing — every row still opens as a
single squashed column. It also has no Robson classification in it. The
parquet already holds the 48 extracted columns plus the Robson group for
every one of the 3.6 million births, which is the actual analysis dataset.

THE EXCEL TRAP
Excel stops at 1,048,576 rows. Our data is 3.6 million. Opening the full
CSV in Excel silently keeps the first million rows and discards the rest,
with no warning at all. That is why this script writes a separate sampled
file that fits.
=============================================================================
"""

import pandas as pd
from pathlib import Path

SRC = "natality_2023_extended.parquet"
if not Path(SRC).exists():
    SRC = "natality_2023_filtered.parquet"
if not Path(SRC).exists():
    raise SystemExit("No parquet file found. Run this in the Nat2023us folder.")

print(f"Loading {SRC} ...")
df = pd.read_parquet(SRC)
print(f"  {len(df):,} rows, {len(df.columns)} columns\n")

COLS = [
    "Robson_group", "Classification_status", "Missing_variable", "is_cesarean",
    "MAGER", "BMI", "M_Ht_In", "PWgt_R",
    "PRIORLIVE", "PRIORDEAD", "RF_CESAR", "RF_CESARN",
    "LD_INDL", "LD_AUGM", "ME_TRIAL", "ME_PRES", "ME_ROUT",
    "RDMETH_REC", "DMETH_REC", "DPLURAL", "OEGest_Comb", "DBWT",
    "APGAR5", "AB_NICU", "near_miss_flag",
    "RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE",
    "payer", "attendant", "weekend", "time_of_day",
]
COLS = [c for c in COLS if c in df.columns]
out = df

# ---- 1. Full dataset, written in chunks so memory stays flat -------------
print("Writing robson_full_dataset.csv (this is the big one) ...")
first = True
for i in range(0, len(out), 500_000):
    out.iloc[i:i+500_000].to_csv(
        "robson_full_dataset.csv", mode="w" if first else "a",
        header=first, index=False)
    first = False
    print(f"  {min(i+500_000, len(out)):,} / {len(out):,}")
mb = Path("robson_full_dataset.csv").stat().st_size / 1e6
print(f"  done — {mb:,.0f} MB\n")

# ---- 2. Excel-safe random sample ------------------------------------------
print("Writing robson_sample_200k.csv ...")
out.sample(n=min(200_000, len(out)), random_state=42) \
   .to_csv("robson_sample_200k.csv", index=False)
print(f"  done — {Path('robson_sample_200k.csv').stat().st_size/1e6:.0f} MB\n")

# ---- 3. The WHO report table ----------------------------------------------
print("Writing robson_report_table.csv ...")
d = df[df["Classification_status"] == "Classified"]
N, C = len(d), int(d["is_cesarean"].sum())
t = (d.groupby(["Robson_group_sort_order", "Robson_group"], observed=True)
       .agg(n_women=("is_cesarean", "size"), n_cesarean=("is_cesarean", "sum"))
       .reset_index().sort_values("Robson_group_sort_order")
       .drop(columns="Robson_group_sort_order"))
t["group_size_pct"]             = (100 * t.n_women / N).round(2)
t["group_cs_rate_pct"]          = (100 * t.n_cesarean / t.n_women).round(2)
t["absolute_contribution_pct"]  = (100 * t.n_cesarean / N).round(2)
t["relative_contribution_pct"]  = (100 * t.n_cesarean / C).round(2)
t.loc[len(t)] = ["TOTAL", N, C, 100.0, round(100*C/N, 2), round(100*C/N, 2), 100.0]
t.to_csv("robson_report_table.csv", index=False)
print("  done\n")

print("=" * 60)
print("FILES WRITTEN")
print("=" * 60)
print("  robson_report_table.csv    <- show this to your supervisor first")
print("  robson_sample_200k.csv     <- opens in Excel, safe")
print(f"  robson_full_dataset.csv    <- {mb:,.0f} MB, DO NOT open in Excel")
print("\nExcel truncates at 1,048,576 rows without warning.")
print("For the full file use Power BI, or read it in Python.")
