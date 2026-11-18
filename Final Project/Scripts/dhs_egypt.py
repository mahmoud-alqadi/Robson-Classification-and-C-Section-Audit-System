"""
=============================================================================
EGYPT DHS — first pass
=============================================================================
Put the two downloaded ZIPs (or their extracted .DTA files) in this folder
and run:

    pip install pandas pyreadstat
    python dhs_egypt.py

WHAT DHS CAN AND CANNOT GIVE US

Robson needs six criteria. DHS carries three of them:

    parity ................ yes, from birth order
    previous caesarean .... yes, but only within the recall window
    number of fetuses ..... yes, from the multiple-birth code
    fetal presentation .... NO
    onset of labour ....... NO
    gestational age ....... NO (only a crude size-at-birth report)

So DHS cannot produce a Robson table. What it CAN do is give a nationally
representative parity and previous-caesarean structure — which is exactly
what the two hospital studies cannot give, because they are tertiary
referral centres. The two sources are complements, not substitutes.

THE RECALL WINDOW MATTERS
Mode of delivery (m17) is asked only for births in the five years before
the survey. For a woman whose earlier births fall outside that window, we
cannot know whether they were caesareans — so "previous caesarean" is only
reliably derivable for women whose whole birth history sits inside it.
The script reports how large that usable subset is instead of hiding it.

WEIGHTS ARE NOT OPTIONAL
DHS uses a stratified cluster design. An unweighted percentage is not the
national figure. v005 is divided by 1,000,000 to give the weight.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import zipfile, io

pd.set_option("display.width", 200)


def find_dta(stem_hint):
    """Locate a .DTA on disk, or pull it straight out of its ZIP."""
    for f in Path(".").glob("*.DTA"):
        if stem_hint.lower() in f.name.lower():
            return f
    for f in Path(".").glob("*.dta"):
        if stem_hint.lower() in f.name.lower():
            return f
    for z in list(Path(".").glob("*.ZIP")) + list(Path(".").glob("*.zip")):
        if stem_hint.lower() not in z.name.lower():
            continue
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if name.upper().endswith(".DTA"):
                    print(f"  extracting {name} from {z.name}")
                    zf.extract(name, ".")
                    return Path(name)
    return None


def load(stem_hint, label):
    p = find_dta(stem_hint)
    if p is None:
        print(f"[skip] {label}: no .DTA or .ZIP matching '{stem_hint}' in this folder")
        return None
    print(f"Loading {label}: {p}")
    df = pd.read_stata(p, convert_categoricals=False)
    print(f"  {len(df):,} women, {len(df.columns):,} variables\n")
    return df


def births_long(df, label, max_births=6):
    """
    The Individual Recode is one row per WOMAN, with birth-history variables
    suffixed _1 .. _6 (most recent first). Reshape to one row per BIRTH.
    """
    keep_woman = [c for c in ["caseid","v005","v012","v024","v025","v106",
                              "v190","v201","v208","v212"] if c in df.columns]
    rows = []
    for i in range(1, max_births+1):
        cols = {}
        for base in ["m17","m15","bord","b0","b19","b5","m19","m18"]:
            c = f"{base}_{i}"
            if c in df.columns:
                cols[c] = base
        if "m17" not in cols.values():
            continue
        sub = df[keep_woman + list(cols)].rename(columns=cols).copy()
        sub["birth_index"] = i
        rows.append(sub)
    if not rows:
        print(f"[{label}] no m17_* variables found — check the file")
        return None
    b = pd.concat(rows, ignore_index=True)
    b = b[b["m17"].notna()]
    b["weight"] = b["v005"] / 1_000_000 if "v005" in b else 1.0
    b["cs"] = (pd.to_numeric(b["m17"], errors="coerce") == 1).astype(float)
    print(f"[{label}] {len(b):,} births with a recorded mode of delivery")
    return b


def wpct(d, col="cs"):
    w = d["weight"]
    return 100 * (d[col] * w).sum() / w.sum() if w.sum() else np.nan


# =============================================================================
ed14 = load("IR61", "EDHS 2014")
if ed14 is None:
    ed14 = load("EGIR", "EDHS 2014")
eh15 = load("IQ73", "EHIS 2015")
if eh15 is None:
    eh15 = load("EGIQ", "EHIS 2015")

results = {}
for df, label in [(ed14, "EDHS 2014"), (eh15, "EHIS 2015")]:
    if df is None:
        continue
    b = births_long(df, label)
    if b is None:
        continue
    results[label] = b

    print("=" * 70)
    print(f"{label}")
    print("=" * 70)
    print(f"  Caesarean rate, ALL births in the recall window")
    print(f"    weighted   : {wpct(b):.2f}%")
    print(f"    unweighted : {100*b['cs'].mean():.2f}%   <- not the national figure")

    last = b[b["birth_index"] == 1]
    print(f"  Caesarean rate, MOST RECENT birth only")
    print(f"    weighted   : {wpct(last):.2f}%   (n = {len(last):,})")
    print("""
    Published figures usually refer to one of these two definitions.
    If they differ here, that alone may explain part of the gap between
    the 51.8% and 43.8% national figures.
""")

    # --- parity -----------------------------------------------------------
    if "bord" in b.columns:
        bo = pd.to_numeric(b["bord"], errors="coerce")
        b["parity_group"] = np.where(bo == 1, "nulliparous (1st birth)",
                             np.where(bo.notna(), "multiparous", None))
        t = b.dropna(subset=["parity_group"]).groupby("parity_group").apply(
            lambda x: pd.Series({"n_births": len(x),
                                 "share_pct": round(100*x["weight"].sum()/b["weight"].sum(),2),
                                 "cs_rate_pct": round(wpct(x),2)}))
        print("  By parity at that birth (Robson's first criterion):")
        print(t.to_string(), "\n")

    # --- plurality --------------------------------------------------------
    if "b0" in b.columns:
        b["multiple"] = pd.to_numeric(b["b0"], errors="coerce").fillna(0) > 0
        print(f"  Multiple births: {100*b['multiple'].mean():.2f}% "
              f"| caesarean rate {wpct(b[b['multiple']]):.2f}% "
              f"vs singleton {wpct(b[~b['multiple']]):.2f}%\n")

    # --- previous caesarean, within the window ----------------------------
    b = b.sort_values(["caseid", "birth_index"], ascending=[True, False])
    b["prev_cs_in_window"] = (b.groupby("caseid")["cs"].cumsum() - b["cs"]) > 0
    if "v201" in b.columns and "bord" in b.columns:
        # usable only where every earlier birth is also observed
        b["history_complete"] = pd.to_numeric(b["bord"], errors="coerce") <= \
                                b.groupby("caseid")["birth_index"].transform("max")
        usable = b[b["history_complete"]]
        print(f"  Previous caesarean is derivable for {len(usable):,} of {len(b):,} births "
              f"({100*len(usable)/len(b):.1f}%)")
        if len(usable):
            print(f"    with a previous caesarean : {100*usable['prev_cs_in_window'].mean():.2f}% "
                  f"| caesarean rate {wpct(usable[usable.prev_cs_in_window]):.2f}%")
            print(f"    without                   : caesarean rate "
                  f"{wpct(usable[~usable.prev_cs_in_window]):.2f}%")
        print()

    # --- place of delivery, the public/private divide ---------------------
    if "m15" in b.columns:
        m15 = pd.to_numeric(b["m15"], errors="coerce")
        b["place"] = np.select(
            [m15.between(11,12), m15.between(21,26), m15.between(31,36)],
            ["home", "public facility", "private facility"], default="other/unknown")
        t = b.groupby("place").apply(lambda x: pd.Series({
            "share_pct": round(100*x["weight"].sum()/b["weight"].sum(),2),
            "cs_rate_pct": round(wpct(x),2)}))
        print("  By place of delivery:")
        print(t.to_string(), "\n")

    # --- residence and wealth --------------------------------------------
    for v, name in [("v025","urban/rural"), ("v190","wealth quintile"), ("v024","region")]:
        if v in b.columns:
            t = b.groupby(v).apply(lambda x: pd.Series({
                "share_pct": round(100*x["weight"].sum()/b["weight"].sum(),2),
                "cs_rate_pct": round(wpct(x),2)}))
            print(f"  By {name} (codes — see the DHS recode manual):")
            print(t.to_string(), "\n")

    b.to_csv(f"dhs_{label.replace(' ','_').lower()}_births.csv", index=False)
    print(f"  saved dhs_{label.replace(' ','_').lower()}_births.csv\n")

# --- the headline comparison ---------------------------------------------
if len(results) == 2:
    print("=" * 70)
    print("THE 2014 vs 2015 DISCREPANCY")
    print("=" * 70)
    for label, b in results.items():
        last = b[b["birth_index"] == 1]
        print(f"  {label:<12} all births {wpct(b):>6.2f}%   most recent only {wpct(last):>6.2f}%")
    print("""
  Published: EDHS 2014 = 51.8%, EHIS 2015 = 43.8% — eight points apart,
  one year apart. Candidate explanations to test from here:
    - a different definition (all births in the window vs most recent)
    - a different recall window (five years vs three, or vs two)
    - a different age range of women
    - EHIS being a smaller survey with a different sample design
  If the difference reproduces under one definition and vanishes under
  another, that is the answer — and nobody has published it.
""")
