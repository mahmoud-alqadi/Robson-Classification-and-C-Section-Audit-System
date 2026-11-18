"""
=============================================================================
STEPS 7, 8 and 9 — BUILD THE TRAINING SET
=============================================================================
    python step7_9_split.py

STEP 7 — WHICH BIRTHS THE MODEL LEARNS FROM

Three exclusions are not open to argument:

    2b  nulliparous, term, cephalic, PRE-LABOUR caesarean ..... 100% by definition
    4b  multiparous, same ..................................... 100% by definition
    9   abnormal lie .......................................... definitionally
                                                                contaminated in
                                                                this data source

Group membership in 2b and 4b is decided by the same fact the model is
asked to predict. A model trained on them reads the answer out of the
question and returns a high score that measures the classification, not
any clinical pattern. Group 9 is excluded for a different reason: CDC's
"other" presentation is broader than WHO's transverse lie, so the variable
itself is unreliable here.

The remaining question was left open on purpose, to be settled by evidence
rather than by preference: should the model also exclude groups 5, 6, 7, 8
and 10? Those are not leakage. A breech CAN deliver vaginally and a repeat
caesarean IS a decision — their high rates are clinical findings, not
definitions. But they may be so nearly determined in practice that they add
noise rather than signal.

So this script builds BOTH populations and carries them forward. Step 10
trains on each and compares calibration and within-group performance. The
decision is then made from the result and written into the record.

    NARROW   groups 1, 2a, 3, 4a
             term, singleton, cephalic, no uterine scar, labour occurred
    WIDE     everything classified except 2b, 4b and 9

STEP 8 — SPLITTING

Stratified on the target so the caesarean proportion is identical in every
part. Train 60 / validation 20 / test 20. The validation set tunes the
model and chooses the decision threshold. The test set is opened once, at
the very end, and never used to make a choice.

STEP 9 — IMBALANCE

Checked empirically on each restricted population, not assumed from the
national rate. This matters: the overall US rate is 32%, but groups 1 and 3
sit far below that, so the direction of imbalance in the narrow population
cannot be predicted from the headline figure.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

SRC = "model_ready.parquet"
if not Path(SRC).exists():
    raise SystemExit(f"{SRC} not found. Run step6_encode.py first.")

d = pd.read_parquet(SRC)
FEATURES = pd.read_csv("feature_list.csv")["feature"].tolist()
TARGET = "is_cesarean"
print(f"Loaded {len(d):,} rows, {len(FEATURES)} features\n")

# =============================================================================
print("=" * 74)
print("STEP 7 — TRAINING POPULATION")
print("=" * 74)

clf = d[d["Classification_status"] == "Classified"].copy()
print(f"  Classified births              : {len(clf):,}")
print(f"  Unclassifiable, excluded       : {len(d)-len(clf):,} "
      f"({100*(len(d)-len(clf))/len(d):.2f}%)")

LEAK = ["2b", "4b", "9"]
leak_rows = clf[clf["Robson_group"].isin(LEAK)]
print(f"\n  Excluded as leakage or contaminated:")
for g in LEAK:
    sub = clf[clf["Robson_group"] == g]
    if len(sub):
        print(f"    group {g:<3} {len(sub):>9,} births   caesarean rate {100*sub[TARGET].mean():>6.2f}%")
print(f"    total    {len(leak_rows):>9,} births   "
      f"({100*leak_rows[TARGET].sum()/clf[TARGET].sum():.1f}% of ALL caesareans in the data)")
print("""
  Read that last figure. A sizeable share of every positive case in the
  dataset sits inside groups whose outcome is fixed by definition. Left in,
  they would be free marks.
""")

NARROW = ["1", "2a", "3", "4a"]
pops = {
    "narrow": clf[clf["Robson_group"].isin(NARROW)].copy(),
    "wide":   clf[~clf["Robson_group"].isin(LEAK)].copy(),
}

for name, p in pops.items():
    print(f"\n  {name.upper()} population: {len(p):,} births, "
          f"caesarean rate {100*p[TARGET].mean():.2f}%")
    comp = (p.groupby(["Robson_group_sort_order", "Robson_group"], observed=True)
              .agg(n=(TARGET, "size"), cs=(TARGET, "mean"))
              .reset_index().sort_values("Robson_group_sort_order"))
    comp["share_%"] = (100 * comp["n"] / len(p)).round(2)
    comp["cs_rate_%"] = (100 * comp["cs"]).round(2)
    print(comp[["Robson_group", "n", "share_%", "cs_rate_%"]].to_string(index=False))

# =============================================================================
print("\n" + "=" * 74)
print("STEP 8 — STRATIFIED SPLIT  (60 / 20 / 20)")
print("=" * 74)

def split(p, seed=42):
    rng = np.random.default_rng(seed)
    p = p.copy()
    p["_part"] = ""
    for y in (0, 1):                      # stratify on the target
        idx = p.index[p[TARGET] == y].to_numpy().copy()   # to_numpy() on an Index is read-only
        rng.shuffle(idx)
        n = len(idx)
        a, b = int(0.60 * n), int(0.80 * n)
        p.loc[idx[:a],  "_part"] = "train"
        p.loc[idx[a:b], "_part"] = "valid"
        p.loc[idx[b:],  "_part"] = "test"
    return p

for name in pops:
    pops[name] = split(pops[name])
    p = pops[name]
    print(f"\n  {name.upper()}")
    t = p.groupby("_part").agg(n=(TARGET, "size"), cs_rate=(TARGET, "mean"))
    t["cs_rate_%"] = (100 * t["cs_rate"]).round(3)
    print(t[["n", "cs_rate_%"]].reindex(["train", "valid", "test"]).to_string())
    spread = t["cs_rate_%"].max() - t["cs_rate_%"].min()
    print(f"    caesarean rate spread across parts: {spread:.3f} pp "
          f"{'OK' if spread < 0.5 else '<-- CHECK'}")

# =============================================================================
print("\n" + "=" * 74)
print("STEP 9 — CLASS IMBALANCE, MEASURED NOT ASSUMED")
print("=" * 74)
print("""
  The national caesarean rate is 32%. That figure does not describe either
  training population, because the groups with fixed outcomes have been
  removed. This is exactly why imbalance is checked here rather than
  inherited from the headline.
""")
for name, p in pops.items():
    tr = p[p["_part"] == "train"]
    pos = tr[TARGET].sum()
    neg = len(tr) - pos
    rate = 100 * pos / len(tr)
    spw = neg / pos
    print(f"  {name.upper():<8} train n={len(tr):>9,}   caesarean {rate:>6.2f}%   "
          f"ratio 1:{spw:.2f}   scale_pos_weight={spw:.3f}")

print("""
  How to read the ratio. Below about 1:4 the imbalance is mild and standard
  training usually copes; beyond that, class weighting earns its place and
  AUPRC becomes the metric that matters rather than ROC-AUC.

  Note what is NOT being done: no resampling before the split. SMOTE or any
  oversampling applied to the whole dataset leaks synthetic neighbours of
  test cases into training and inflates every reported number. If resampling
  is used at all it goes inside the training fold only.
""")

for name, p in pops.items():
    cols = FEATURES + [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]
    p[cols].to_parquet(f"train_{name}.parquet", index=False)
    print(f"Saved train_{name}.parquet — {len(p):,} rows")

print("\nNext: step 10 — train XGBoost on both populations and compare.")
