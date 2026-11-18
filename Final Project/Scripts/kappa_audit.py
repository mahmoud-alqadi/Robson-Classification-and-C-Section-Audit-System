"""
=============================================================================
COHEN'S KAPPA AUDIT — manual verification of the Robson classifier
=============================================================================
    python kappa_audit.py            # build the blinded worksheet
    python kappa_audit.py --score    # score it after you have filled it in

WHY THIS IS WORTH DOING, AND WHAT IT IS NOT

The classifier is deterministic if/else code. A clinician working from the
same variables should reach the same group every time, so kappa = 1 is the
expected result, not a surprising one. A published audit of a comparable
rule-based classifier (IJRCOG 2024) reported exactly that.

The value therefore is not the coefficient. It is the disagreements. A
single mismatch means one of two things, and both are worth finding:

  - a derivation bug, most likely in onset of labour, which is the only
    Robson criterion not recorded directly in this data source; or
  - a genuine ambiguity in the WHO definitions as they meet this data,
    which belongs in the limitations section rather than being silently
    resolved by whichever reading the code happened to take.

BLINDING
The worksheet contains no algorithmic output. The answer key is written to
a separate file. Do not open it until the worksheet is complete, or the
audit is worthless.

SAMPLING
50 records in two parts, kept separate on purpose:
  - 40 drawn at random. Kappa is computed on these, because they are
    representative and the coefficient means something.
  - 10 drawn purposively from the rare and awkward groups (6, 7, 8, 9,
    unclassifiable). These are reported as an edge-case check and are
    EXCLUDED from the kappa, because hand-picking hard cases would
    deflate a coefficient that is supposed to describe routine work.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

WORKSHEET = "kappa_audit_worksheet.csv"
ANSWERKEY = "kappa_audit_answerkey.csv"
GUIDE     = "kappa_audit_guide.txt"

PRES = {"1": "Cephalic", "2": "Breech", "3": "Other", "9": "Unknown"}
ROUT = {"1": "Spontaneous vaginal", "2": "Forceps", "3": "Vacuum",
        "4": "Caesarean", "9": "Unknown"}
YN   = {"Y": "Yes", "N": "No", "U": "Unknown"}
PLUR = {"1": "Singleton", "2": "Twin", "3": "Triplet", "4": "Quadruplet",
        "5": "Quintuplet or higher"}


def decode(v, d):
    return d.get(str(v).strip(), f"[{v}]")


def build():
    src = next((f for f in ["natality_2023_extended.parquet",
                            "natality_2023_filtered.parquet"]
                if Path(f).exists()), None)
    if src is None:
        raise SystemExit("No parquet cache found. Run robson_analysis.py first.")

    print(f"Loading {src} ...")
    df = pd.read_parquet(src)
    rng = np.random.default_rng(2024)

    # 40 representative
    main = df.sample(n=40, random_state=2024)
    main = main.assign(part="random")

    # 10 edge cases, spread across the awkward groups
    edge_pool, want = [], ["6", "7", "8", "9", "Unclassifiable"]
    for g in want:
        sub = df[df["Robson_group"] == g]
        if len(sub):
            edge_pool.append(sub.sample(n=min(2, len(sub)), random_state=2024))
    edge = pd.concat(edge_pool).assign(part="edge") if edge_pool else main.iloc[:0]

    s = pd.concat([main, edge], ignore_index=True)
    s = s.sample(frac=1, random_state=7).reset_index(drop=True)   # shuffle
    s["audit_id"] = [f"A{i+1:03d}" for i in range(len(s))]

    ga = pd.to_numeric(s["OEGest_Comb"], errors="coerce")
    ga = ga.where(~ga.isin([99, 999]))

    def count(col):
        x = pd.to_numeric(s[col], errors="coerce")
        return x.where(~x.isin([99, 999])).apply(
            lambda v: "unknown" if pd.isna(v) else str(int(v)))

    ws = pd.DataFrame({
        "audit_id": s["audit_id"],
        "previous_live_births":  count("PRIORLIVE"),
        "previous_stillbirths":  count("PRIORDEAD"),
        "previous_caesarean":    s["RF_CESAR"].apply(lambda v: decode(v, YN)),
        "number_previous_caesareans": count("RF_CESARN"),
        "labour_induced":        s["LD_INDL"].apply(lambda v: decode(v, YN)),
        "trial_of_labour_attempted": s["ME_TRIAL"].apply(
            lambda v: decode(v, YN) + "  (recorded for caesareans only)"),
        "route_of_delivery":     s["ME_ROUT"].apply(lambda v: decode(v, ROUT)),
        "fetal_presentation":    s["ME_PRES"].apply(lambda v: decode(v, PRES)),
        "plurality":             s["DPLURAL"].apply(lambda v: decode(v, PLUR)),
        "gestational_age_weeks": ga.apply(
            lambda v: "unknown" if pd.isna(v) else str(int(v))),
        "YOUR_GROUP": "",
        "YOUR_NOTES": "",
    })
    ws.to_csv(WORKSHEET, index=False)

    s[["audit_id", "Robson_group", "Classification_status",
       "Missing_variable", "part"]].rename(
        columns={"Robson_group": "algorithm_group"}).to_csv(ANSWERKEY, index=False)

    Path(GUIDE).write_text(GUIDE_TEXT, encoding="utf-8")

    print(f"\nWritten:")
    print(f"  {WORKSHEET}   <- fill in the YOUR_GROUP column")
    print(f"  {GUIDE}       <- the WHO flowchart, in order")
    print(f"  {ANSWERKEY}   <- DO NOT OPEN until you are finished")
    print(f"\n{len(ws)} records: {(s['part']=='random').sum()} random, "
          f"{(s['part']=='edge').sum()} edge cases.")


def kappa(a, b):
    """Cohen's kappa for nominal categories, computed from first principles."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    m = np.zeros((len(cats), len(cats)))
    for x, y in zip(a, b):
        m[idx[x], idx[y]] += 1
    po = np.trace(m) / n
    pe = sum(m[i, :].sum() * m[:, i].sum() for i in range(len(cats))) / (n * n)
    k = (po - pe) / (1 - pe) if pe != 1 else 1.0
    return k, po, pe, pd.DataFrame(m, index=cats, columns=cats).astype(int)


def score():
    ws = pd.read_csv(WORKSHEET, dtype=str)
    key = pd.read_csv(ANSWERKEY, dtype=str)
    d = ws.merge(key, on="audit_id")

    d["YOUR_GROUP"] = d["YOUR_GROUP"].fillna("").str.strip()
    blank = (d["YOUR_GROUP"] == "").sum()
    if blank:
        print(f"WARNING: {blank} rows have no answer and are excluded.\n")
    d = d[d["YOUR_GROUP"] != ""].copy()
    if not len(d):
        raise SystemExit("No answers found. Fill in the YOUR_GROUP column first.")

    d["YOUR_GROUP"] = d["YOUR_GROUP"].str.replace(" ", "").str.capitalize()
    d.loc[d["YOUR_GROUP"].str.lower().str.startswith("unclass"), "YOUR_GROUP"] = "Unclassifiable"

    print("=" * 66)
    print("PRIMARY RESULT — random sample only")
    print("=" * 66)
    r = d[d["part"] == "random"]
    if len(r):
        k, po, pe, cm = kappa(list(r["YOUR_GROUP"]), list(r["algorithm_group"]))
        print(f"  n                    : {len(r)}")
        print(f"  Observed agreement   : {100*po:.1f}%")
        print(f"  Expected by chance   : {100*pe:.1f}%")
        print(f"  Cohen's kappa        : {k:.3f}")
        print("""
  Landis & Koch: above 0.81 is almost perfect agreement. Because the
  classifier is deterministic, anything materially below 1.0 points at a
  derivation problem rather than at ordinary rater variability.
""")

    print("=" * 66)
    print("EDGE CASES — reported separately, excluded from kappa")
    print("=" * 66)
    e = d[d["part"] == "edge"]
    if len(e):
        agree = (e["YOUR_GROUP"] == e["algorithm_group"]).sum()
        print(f"  {agree} of {len(e)} agree")

    print("\n" + "=" * 66)
    print("DISAGREEMENTS — the actual point of the exercise")
    print("=" * 66)
    bad = d[d["YOUR_GROUP"] != d["algorithm_group"]]
    if not len(bad):
        print("  None. The implementation reproduces manual classification exactly.")
        print("  Report this as complete agreement and move on.")
    else:
        for _, row in bad.iterrows():
            print(f"\n  {row['audit_id']} ({row['part']})")
            print(f"    you: {row['YOUR_GROUP']:<16} algorithm: {row['algorithm_group']}")
            print(f"    parity {row['previous_live_births']}+{row['previous_stillbirths']}"
                  f" | prev CS {row['previous_caesarean']}"
                  f" | induced {row['labour_induced']}"
                  f" | trial {row['trial_of_labour_attempted'][:3]}"
                  f" | route {row['route_of_delivery']}")
            print(f"    {row['fetal_presentation']} | {row['plurality']}"
                  f" | {row['gestational_age_weeks']} weeks")
            if str(row.get("YOUR_NOTES", "")).strip() not in ("", "nan"):
                print(f"    your note: {row['YOUR_NOTES']}")
        print("""
  Work through each one. Every disagreement resolves to either a bug in the
  code or an ambiguity in the definitions as they meet this data source.
  The first gets fixed; the second gets written into the limitations.
""")
        bad.to_csv("kappa_audit_disagreements.csv", index=False)
        print("  Saved kappa_audit_disagreements.csv")


GUIDE_TEXT = """
ROBSON CLASSIFICATION — WORKING SHEET
=====================================

Classify each record in kappa_audit_worksheet.csv WITHOUT looking at the
answer key. Write the group in the YOUR_GROUP column. Use the subgroup
labels where they apply: 2a, 2b, 4a, 4b, 5.1, 5.2. Write "Unclassifiable"
if a criterion needed for the decision is unknown.

DERIVED DEFINITIONS
  Parity        Nulliparous means no previous birth at 22 weeks or later,
                live or stillborn. Add previous live births and previous
                stillbirths; zero means nulliparous.
  Term          37 completed weeks or more.
  Onset         Induced if labour was induced.
                Otherwise spontaneous if the birth was vaginal, or if a
                caesarean followed a recorded trial of labour.
                Otherwise pre-labour caesarean.
                Note: trial of labour is recorded for caesareans only. A
                vaginal birth had labour by definition.

WORK THROUGH IN THIS ORDER. Stop at the first match.

  1.  Two or more fetuses ....................................... GROUP 8
  2.  Singleton, transverse or oblique lie ...................... GROUP 9
  3.  Singleton breech, nulliparous ............................. GROUP 6
  4.  Singleton breech, multiparous ............................. GROUP 7
  5.  Singleton cephalic, under 37 weeks ........................ GROUP 10
  6.  Singleton cephalic, term, one previous caesarean ......... GROUP 5.1
  7.  Singleton cephalic, term, two or more previous caesareans  GROUP 5.2
  8.  Singleton cephalic, term, nulliparous, no previous
      caesarean, spontaneous labour ............................ GROUP 1
  9.  ... same, induced labour ................................. GROUP 2a
  10. ... same, pre-labour caesarean ........................... GROUP 2b
  11. Singleton cephalic, term, multiparous, no previous
      caesarean, spontaneous labour ............................ GROUP 3
  12. ... same, induced labour ................................. GROUP 4a
  13. ... same, pre-labour caesarean ........................... GROUP 4b

THE ORDER MATTERS. Groups 6 to 10 all include women with a previous
caesarean, which is why the previous-caesarean question comes after them.
A multiparous breech with a previous caesarean is group 7, not group 5.

If a record makes you hesitate, say so in YOUR_NOTES. The hesitation is
data: it means the definition is ambiguous against this data source, and
that belongs in the limitations section.
"""


if __name__ == "__main__":
    if "--score" in sys.argv:
        score()
    else:
        build()
