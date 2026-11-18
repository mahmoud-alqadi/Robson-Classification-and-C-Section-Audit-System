"""
=============================================================================
EXPORT TO CSV — run this on YOUR machine
=============================================================================

The nine CSVs supplied alongside this script contain every aggregate figure
the project has produced so far. They need no data and no setup.

This script produces the things that can only come from the 3.6 million
records sitting on your own disk. Run it in the same folder as
natality_2023_extended.parquet (created by stage4) or
natality_2023_filtered.parquet (created by robson_analysis.py).

    python export_csvs.py

Outputs go into a subfolder called csv_export/.

A note on the big one: 10_cdc_analysis_sample.csv is a 200,000-row random
sample, not the full file. Excel stops at 1,048,576 rows and would silently
truncate 3.6 million. If you want the whole thing for Power BI, uncomment
the Parquet line at the bottom — Power BI reads Parquet natively and it is
about twenty times smaller than the equivalent CSV.
"""

import pandas as pd
from pathlib import Path

OUT = Path("csv_export"); OUT.mkdir(exist_ok=True)

src = None
for candidate in ["natality_2023_extended.parquet", "natality_2023_filtered.parquet"]:
    if Path(candidate).exists():
        src = candidate; break
if src is None:
    raise SystemExit("No parquet cache found. Run robson_analysis.py first.")

print(f"Loading {src} ...")
df = pd.read_parquet(src)
print(f"{len(df):,} records\n")

d = df[df["Classification_status"] == "Classified"].copy()
d["apgar5_lt7"] = pd.to_numeric(d["APGAR5"], errors="coerce").lt(7)
d["nicu"] = d["AB_NICU"].astype(str).str.strip() == "Y"

# --- A. group x payer cross-tab -------------------------------------------
(d.groupby(["Robson_group_sort_order", "Robson_group", "payer"], observed=True)
   .agg(n=("is_cesarean", "size"), cs_rate_pct=("is_cesarean", "mean"))
   .reset_index().sort_values("Robson_group_sort_order")
   .drop(columns="Robson_group_sort_order")
   .assign(cs_rate_pct=lambda x: (100*x.cs_rate_pct).round(2))
   .to_csv(OUT/"A_group_by_payer.csv", index=False))

# --- B. group x time of day -----------------------------------------------
(d.groupby(["Robson_group_sort_order", "Robson_group", "time_of_day"], observed=True)
   .agg(n=("is_cesarean", "size"), cs_rate_pct=("is_cesarean", "mean"))
   .reset_index().sort_values("Robson_group_sort_order")
   .drop(columns="Robson_group_sort_order")
   .assign(cs_rate_pct=lambda x: (100*x.cs_rate_pct).round(2))
   .to_csv(OUT/"B_group_by_time_of_day.csv", index=False))

# --- C. maternal age band x group ------------------------------------------
age = pd.to_numeric(d["MAGER"], errors="coerce")
d["age_band"] = pd.cut(age, [0,19,24,29,34,39,99],
                       labels=["<20","20-24","25-29","30-34","35-39","40+"])
(d.groupby(["Robson_group_sort_order", "Robson_group", "age_band"], observed=True)
   .agg(n=("is_cesarean","size"), cs_rate_pct=("is_cesarean","mean"))
   .reset_index().sort_values("Robson_group_sort_order")
   .drop(columns="Robson_group_sort_order")
   .assign(cs_rate_pct=lambda x: (100*x.cs_rate_pct).round(2))
   .to_csv(OUT/"C_group_by_maternal_age.csv", index=False))

# --- D. data completeness --------------------------------------------------
mv = df["Missing_variable"].fillna("")
(mv[mv != ""].str.split(",").explode().value_counts()
   .rename_axis("missing_variable").reset_index(name="n_records")
   .assign(pct_of_all=lambda x: (100*x.n_records/len(df)).round(4))
   .to_csv(OUT/"D_missing_variable_counts.csv", index=False))

# --- E. VBAC / TOLAC -------------------------------------------------------
prev = df["RF_CESAR"].astype(str).str.strip() == "Y"
rdm  = df["RDMETH_REC"].astype(str).str.strip()
tri  = df["ME_TRIAL"].astype(str).str.strip()
v = df[prev & rdm.isin(["2","4"])].copy()
v["vbac"]  = rdm[prev & rdm.isin(["2","4"])] == "2"
v["tolac"] = v["vbac"] | ((rdm[prev & rdm.isin(["2","4"])]=="4") & (tri[prev & rdm.isin(["2","4"])]=="Y"))
(v.groupby("payer").agg(n=("tolac","size"), tolac_rate_pct=("tolac","mean"))
   .join(v[v.tolac].groupby("payer")["vbac"].mean().rename("vbac_success_pct"))
   .assign(tolac_rate_pct=lambda x:(100*x.tolac_rate_pct).round(2),
           vbac_success_pct=lambda x:(100*x.vbac_success_pct).round(2))
   .reset_index().to_csv(OUT/"E_tolac_vbac_by_payer.csv", index=False))

# --- F. analysis-ready sample ---------------------------------------------
keep = ["Robson_group","Classification_status","Missing_variable","is_cesarean",
        "MAGER","BMI","DPLURAL","ME_PRES","OEGest_Comb","DBWT","APGAR5",
        "RF_CESAR","RF_CESARN","LD_INDL","ME_TRIAL","ME_ROUT","RDMETH_REC",
        "RF_PDIAB","RF_GDIAB","RF_PHYPE","RF_GHYPE","RF_EHYPE",
        "payer","attendant","weekend","time_of_day","near_miss_flag","AB_NICU"]
keep = [c for c in keep if c in df.columns]
df[keep].sample(n=min(200_000, len(df)), random_state=42) \
        .to_csv(OUT/"F_cdc_sample_200k.csv", index=False)

# Full file for Power BI (Parquet — far smaller than CSV, and Power BI reads it):
# df[keep].to_parquet(OUT/"cdc_full_analysis_columns.parquet", index=False)

print("Written to csv_export/:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<34} {f.stat().st_size/1e6:>8.2f} MB")
