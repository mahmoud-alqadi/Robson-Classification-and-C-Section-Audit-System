"""
=============================================================================
EGYPT GOVERNORATE DATA
=============================================================================
    python egypt_governorates.py

THE TRAP THIS SCRIPT EXISTS TO AVOID

Two different governorate datasets circulate together and are routinely
conflated:

  Egypt Family Health Survey 2021 (CAPMAS)
      precise per-governorate figures — Port Said 91.3%, Kafr El-Sheikh
      88.4%, Gharbeya 84.3%, Matrouh 39.3%

  Ministry of Health birth registration, 2025 (announced 11 Jan 2026)
      NO per-governorate figures at all. Only bands: three governorates
      "exceeded 90%", two were "below 60%", national rate about 80%.

The three governorates named in the 2025 statement are the same three that
topped the 2021 survey, so it is easy to attach the 2021 numbers to the 2025
announcement. That would be wrong, and in one case badly wrong: Matrouh was
39.3% in 2021 and is described as below 60% in 2025, which means it ROSE by
up to twenty points. Quoting 39.3% as a current figure would understate it.

This file keeps the two sources in separate tables, with a `source` column
on every row, so they cannot be silently merged in a dashboard.

WHAT IS AND IS NOT AVAILABLE

Per-governorate caesarean rates exist for 2021 only. For 2025 there are
bands. No governorate-level Robson distribution exists for any year, so the
standardisation cannot be run at governorate level — the governorate page is
descriptive, and should say so.
=============================================================================
"""

import pandas as pd

# =============================================================================
# 2021 — Egypt Family Health Survey (CAPMAS). Precise figures.
# =============================================================================
GOV_2021 = [
    ("Port Said",      91.3, "urban",       "highest recorded"),
    ("Kafr El-Sheikh", 88.4, "Lower Egypt", ""),
    ("Gharbeya",       84.3, "Lower Egypt", ""),
    ("Damietta",       82.5, "Lower Egypt", "100% attended by a health provider"),
    ("Matrouh",        39.3, "frontier",    "lowest recorded"),
]

REGION_2021 = [
    ("Lower Egypt governorates", 60.3, 78.5, "+18.2"),
    ("Urban governorates",       62.0, 75.0, "+13.0"),
    ("Upper Egypt governorates", 39.7, 66.4, "+26.7"),
]

# =============================================================================
# 2025 — Ministry of Health birth registration. Bands only.
# =============================================================================
GOV_2025 = [
    ("Kafr El-Sheikh", "above 90%", "named in the ministerial statement"),
    ("Port Said",      "above 90%", "named in the ministerial statement"),
    ("Gharbeya",       "above 90%", "named in the ministerial statement"),
    ("Matrouh",        "below 60%", "named in the ministerial statement"),
    ("South Sinai",    "below 60%", "named in the ministerial statement"),
]

NATIONAL_2025 = {
    "rate": 80.0,
    "coverage": "more than 5,000 health facilities",
    "system": "electronic birth and death registration",
    "announced": "11 January 2026",
    "private_share_of_caesareans": 93.0,
}

# =============================================================================
print("=" * 74)
print("2021 — Egypt Family Health Survey, per governorate")
print("=" * 74)
g21 = pd.DataFrame(GOV_2021, columns=["governorate", "cs_rate_pct", "region", "note"])
g21["source"] = "Egypt Family Health Survey 2021 (CAPMAS)"
print(g21[["governorate", "cs_rate_pct", "region", "note"]].to_string(index=False))
print(f"\n  spread across named governorates: "
      f"{g21.cs_rate_pct.max() - g21.cs_rate_pct.min():.1f} percentage points")

print("\n" + "=" * 74)
print("2021 — regional trend, against 2014")
print("=" * 74)
r21 = pd.DataFrame(REGION_2021, columns=["region", "rate_2014", "rate_2021", "change_pp"])
r21["source"] = "Egypt Family Health Survey 2021 vs EDHS 2014"
print(r21[["region", "rate_2014", "rate_2021", "change_pp"]].to_string(index=False))
print("""
  Upper Egypt rose fastest — 26.7 points in seven years, from the lowest
  regional base. The gap between regions is closing upward rather than the
  high regions coming down, which is the opposite of what a successful
  containment policy would produce.
""")

print("=" * 74)
print("2025 — Ministry of Health registration data")
print("=" * 74)
g25 = pd.DataFrame(GOV_2025, columns=["governorate", "band", "note"])
g25["source"] = "MoH electronic birth registration 2025 (announced Jan 2026)"
print(g25[["governorate", "band"]].to_string(index=False))
print(f"""
  National rate {NATIONAL_2025['rate']:.0f}%, from {NATIONAL_2025['coverage']}.
  {NATIONAL_2025['private_share_of_caesareans']:.0f}% of these caesareans were in the private sector.

  NOTE: the 2025 release gives BANDS, not figures. Any dashboard showing a
  precise number against a governorate for 2025 is showing a 2021 number
  with the wrong year attached.
""")

print("=" * 74)
print("WHERE THE TWO SOURCES CAN BE COMPARED")
print("=" * 74)
comp = []
band_map = {"above 90%": (90, 100), "below 60%": (0, 60)}
for gov, band, _ in GOV_2025:
    row = g21[g21.governorate == gov]
    v21 = float(row.cs_rate_pct.iloc[0]) if len(row) else None
    lo, hi = band_map[band]
    if v21 is None:
        direction = "no 2021 figure available"
    elif v21 < lo:
        direction = f"ROSE — was {v21:.1f}% in 2021, now {band}"
    elif v21 > hi:
        direction = f"fell — was {v21:.1f}% in 2021, now {band}"
    else:
        direction = f"consistent — {v21:.1f}% in 2021, {band} in 2025"
    comp.append({"governorate": gov, "rate_2021": v21, "band_2025": band,
                 "reading": direction})
    print(f"  {gov:<16} {direction}")
print("""
  Matrouh is the one to be careful with. At 39.3% in 2021 and below 60% in
  2025, the only safe statement is that it rose. Quoting 39.3% as current
  would understate it by up to twenty points.
""")

print("=" * 74)
print("WHAT CANNOT BE DONE AT GOVERNORATE LEVEL")
print("=" * 74)
print("""
  No Robson distribution exists for any Egyptian governorate. The
  standardisation that produces the expected-versus-observed gap needs group
  sizes, and those exist only for Benha and Qena — two hospitals, not
  governorates.

  So the governorate page is descriptive: it shows the spread and its
  trajectory, and it does NOT show an excess-caesarean estimate by
  governorate. Presenting one would imply a calculation that has not been
  done.

  What the spread does support is the argument itself. A range this wide
  inside one country, under one set of clinical guidelines and one training
  system, cannot be explained by differences in obstetric risk between
  governorates. It is variation in practice, which is what the project set
  out to measure.
""")

for df, name in [(g21, "egypt_governorates_2021.csv"),
                 (r21, "egypt_regions_2021.csv"),
                 (g25, "egypt_governorates_2025_bands.csv"),
                 (pd.DataFrame(comp), "egypt_governorates_comparison.csv")]:
    df.to_csv(name, index=False, encoding="utf-8-sig")
    print(f"Saved {name}")
