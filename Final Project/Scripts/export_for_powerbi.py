"""
=============================================================================
EXPORT FOR POWER BI
=============================================================================
    python export_for_powerbi.py

Writes powerbi/births.csv — one row per classified birth, both years, with
only the columns the dashboard needs.

WHY THE FULL FILE AND NOT A SAMPLE

Power BI handles 7 million rows without difficulty; the reason people
sample is usually Excel, not Power BI. Using the full file matters here for
a specific reason: when the dashboard recomputes the caesarean rate in DAX
and gets 32.35%, that is the same figure the Python pipeline produced,
independently, in a different tool. If it came out slightly different
because the dashboard was built on a sample, you would have two numbers for
the same thing in one project — which is worse than not using DAX at all.

So this is not just a data export. It is a second, independent check on the
whole pipeline, and it either agrees or it finds a bug.

WHY THESE COLUMNS AND NOT ALL 60

Everything here is either a Robson output, the target, or a practice
variable the dashboard displays. The clinical risk factors are left out:
they belong to the model, and the dashboard shows the model's results
rather than recomputing it.

TARGETS TO CHECK AFTER LOADING
    overall caesarean rate 2023 .... 32.35%
    NTSV (groups 1, 2a, 2b) 2023 ... 26.56%
    groups 2b and 4b ............... exactly 100%
    unclassifiable 2023 ............ 0.55%
=============================================================================
"""

import pandas as pd
from pathlib import Path

OUT = Path("powerbi"); OUT.mkdir(exist_ok=True)

SOURCES = [
    ("natality_2023_extended.parquet", 2023),
    ("natality_2024_classified.parquet", 2024),
]

KEEP = ["Robson_group", "Robson_group_sort_order", "Classification_status",
        "Missing_variable", "is_cesarean", "payer", "weekend", "time_of_day",
        "APGAR5", "AB_NICU", "near_miss_flag"]

frames = []
for path, year in SOURCES:
    if not Path(path).exists():
        print(f"[skip] {path} not found")
        continue
    print(f"Reading {path} ...")
    d = pd.read_parquet(path)
    cols = [c for c in KEEP if c in d.columns]
    missing = [c for c in KEEP if c not in d.columns]
    if missing:
        print(f"  note: not present in this file — {missing}")
    s = d[cols].copy()
    s["year"] = year

    # tidy the two outcome columns into something a visual can use directly
    if "APGAR5" in s.columns:
        a = pd.to_numeric(s["APGAR5"], errors="coerce")
        s["apgar5_below7"] = (a < 7).where(a.notna() & (a <= 10)).astype("float")
        s = s.drop(columns=["APGAR5"])
    if "AB_NICU" in s.columns:
        s["nicu_admission"] = (s["AB_NICU"].astype(str).str.strip() == "Y").astype("int8")
        s = s.drop(columns=["AB_NICU"])

    s["is_classified"] = (s["Classification_status"] == "Classified").astype("int8")
    frames.append(s)
    print(f"  {len(s):,} rows\n")

if not frames:
    raise SystemExit("No source files found. Run the pipeline first.")

df = pd.concat(frames, ignore_index=True)

# a compact grouping label for the pages that report ten groups rather than subgroups
df["robson_group_10"] = df["Robson_group"].replace(
    {"2a": "2", "2b": "2", "4a": "4", "4b": "4", "5.1": "5", "5.2": "5"})

out = OUT / "births.csv"
print(f"Writing {out} ...")
df.to_csv(out, index=False, encoding="utf-8-sig")
mb = out.stat().st_size / 1e6
print(f"  {len(df):,} rows, {len(df.columns)} columns, {mb:,.0f} MB\n")

# ---- the numbers the dashboard must reproduce -----------------------------
print("=" * 66)
print("CHECK THESE AFTER LOADING INTO POWER BI")
print("=" * 66)
for year in sorted(df["year"].unique()):
    y = df[df.year == year]
    c = y[y.is_classified == 1]
    ntsv = c[c["Robson_group"].isin(["1", "2a", "2b"])]
    print(f"\n  {year}")
    print(f"    births loaded        {len(y):>12,}")
    print(f"    classified           {len(c):>12,}")
    print(f"    unclassifiable       {100*(1-len(c)/len(y)):>11.2f}%")
    print(f"    caesarean rate       {100*c.is_cesarean.mean():>11.2f}%")
    print(f"    NTSV rate            {100*ntsv.is_cesarean.mean():>11.2f}%")
    for g in ("2b", "4b"):
        sub = c[c["Robson_group"] == g]
        if len(sub):
            print(f"    group {g:<3}            {100*sub.is_cesarean.mean():>11.2f}%")

print("""
If Power BI reproduces these to two decimal places, the dashboard and the
Python pipeline agree and either can be cited. If it does not, stop and find
out why before building anything on top — a mismatch here means one of them
is wrong, and it is worth knowing which.
""")
