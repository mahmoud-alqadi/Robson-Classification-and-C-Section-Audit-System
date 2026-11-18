"""
=============================================================================
STEP 12 — THE DECISION THRESHOLD
=============================================================================
    python step12_threshold.py

FIRST, WHERE THE THRESHOLD IS AND IS NOT USED

The project's central calculation — the expected caesarean rate for Egypt —
sums predicted PROBABILITIES. It never converts them to yes/no, so it uses
no threshold at all. That calculation depends on calibration, which step
11b established at a mean absolute error of 0.11 percentage points across
risk deciles.

A threshold is needed only for two secondary things: reporting precision,
recall, F1 and a confusion matrix, and any use of the model as an
individual-level flag.

A CORRECTION TO WHAT THE PROPOSAL SAYS

The proposal states the threshold will maximise F2, weighting recall on the
grounds that failing to anticipate a needed caesarean is worse than the
reverse. That reasoning is correct for a diagnostic test. It is not correct
here, and the difference matters.

This model does not predict whether a caesarean is NEEDED. It predicts
whether one will be PERFORMED — the project says so in its own methodology,
and that distinction is what makes the Egypt gap meaningful at all. So a
false positive is not a harmless over-caution. It is a tool telling an
Egyptian clinician that a woman is at high risk of caesarean, in a country
where 82.5% of caesarean decisions are already taken before labour begins
and the national rate is around 80%. A recall-maximising threshold in that
setting does not catch more genuine need; it manufactures more prediction,
and prediction here is capable of causing the outcome it predicts.

The project already holds this principle, from ACOG guidance on the VBAC
score: a predicted success probability must never become a barrier to
trial of labour. The same logic applies in reverse to this model. A
caesarean-risk score must never become a justification for booking one.

WHAT THIS SCRIPT DOES

Reports the full precision-recall curve and the operating points that
maximise F0.5, F1 and F2, plus the threshold at which predicted volume
matches observed volume. It recommends none of them silently: the choice
depends on what the number is used for, and each is shown with the
consequence spelled out in cases.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

import xgboost as xgb
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]

if not Path("xgb_final.json").exists():
    raise SystemExit("xgb_final.json not found. Run step11b_calibration.py first.")

wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")

val = wide[wide["Robson_group"].isin(NARROW) & (wide["_part"] == "valid")]
X, y = val[FEATURES], val[TARGET].to_numpy().astype(int)

bst = xgb.Booster(); bst.load_model("xgb_final.json")
p = bst.predict(xgb.DMatrix(X))

print(f"Validation set: {len(y):,} births, observed caesarean rate {100*y.mean():.2f}%")
print(f"Mean predicted probability: {100*p.mean():.2f}%")
print(f"ROC-AUC {roc_auc_score(y,p):.4f}   AUPRC {average_precision_score(y,p):.4f}\n")

prec, rec, thr = precision_recall_curve(y, p)
prec, rec = prec[:-1], rec[:-1]

def fbeta(pr, rc, b):
    return np.where((pr + rc) > 0, (1+b*b)*pr*rc / np.maximum(b*b*pr + rc, 1e-12), 0)

points = {}
for name, b in [("F0.5 (precision-weighted)", 0.5),
                ("F1 (balanced)", 1.0),
                ("F2 (recall-weighted)", 2.0)]:
    f = fbeta(prec, rec, b)
    i = int(np.argmax(f))
    points[name] = (float(thr[i]), float(prec[i]), float(rec[i]), float(f[i]))

# threshold where predicted volume equals observed volume
target_n = int(y.sum())
order = np.sort(p)[::-1]
vol_thr = float(order[target_n - 1])
pred = (p >= vol_thr).astype(int)
tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
fn = int(((pred == 0) & (y == 1)).sum())
points["Volume-matched"] = (vol_thr, tp/max(tp+fp,1), tp/max(tp+fn,1),
                            float(fbeta(np.array([tp/max(tp+fp,1)]),
                                        np.array([tp/max(tp+fn,1)]), 1.0)[0]))

print("=" * 88)
print("OPERATING POINTS")
print("=" * 88)
print(f"{'criterion':<28} {'threshold':>10} {'precision':>10} {'recall':>8} "
      f"{'flagged':>10} {'false alarms':>13}")
print("-" * 88)
for name, (t, pr, rc, f) in points.items():
    flag = int((p >= t).sum())
    fp_n = int(((p >= t) & (y == 0)).sum())
    print(f"{name:<28} {t:>10.3f} {pr:>10.3f} {rc:>8.3f} {flag:>10,} {fp_n:>13,}")

print(f"\n  For scale: {int(y.sum()):,} caesareans actually occurred among "
      f"{len(y):,} births.")

print("""
=========================================================================
READ THE LAST TWO COLUMNS TOGETHER
=========================================================================
  "Flagged" is how many women the tool would label as high risk.
  "False alarms" is how many of those did not have a caesarean.

  The recall-weighted point catches more true cases, and the price is
  visible above. In a health system where a prediction can become the
  reason for the intervention, those false alarms are not a statistical
  cost carried by the analyst. They are carried by women.

  RECOMMENDATION
  Do not adopt a single threshold as a project-wide default, and do not
  adopt F2. Report the curve, and state the operating point separately for
  each use:

    Egypt expected-rate calculation ... no threshold. Probabilities are
                                        summed directly. Calibration is
                                        what matters and it is documented.
    Quality-monitoring / audit ........ volume-matched. Predicting the
                                        same number of caesareans that
                                        occurred keeps aggregate
                                        comparisons honest.
    Individual counselling ............ F0.5 or higher precision, with the
                                        output framed as a probability
                                        rather than a recommendation, and
                                        never as a reason to book.

  The proposal's F2 commitment is withdrawn, with this reasoning recorded.
""")

# ---- the curve, at a readable resolution ----------------------------------
print("=" * 88)
print("PRECISION-RECALL CURVE")
print("=" * 88)
print(f"{'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>7} "
      f"{'flagged':>10} {'% of births':>12}")
print("-" * 88)
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
    pred = (p >= t)
    tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    pr = tp/max(tp+fp, 1); rc = tp/max(tp+fn, 1)
    f1 = 2*pr*rc/max(pr+rc, 1e-12)
    print(f"{t:>10.2f} {pr:>10.3f} {rc:>8.3f} {f1:>7.3f} {int(pred.sum()):>10,} "
          f"{100*pred.mean():>11.2f}%")

out = {"observed_rate": float(y.mean()), "mean_predicted": float(p.mean()),
       "roc_auc": float(roc_auc_score(y, p)),
       "auprc": float(average_precision_score(y, p)),
       "operating_points": {k: {"threshold": v[0], "precision": v[1],
                                "recall": v[2], "f": v[3]} for k, v in points.items()},
       "recommended": {"egypt_calculation": None,
                       "quality_monitoring": vol_thr,
                       "individual_counselling": points["F0.5 (precision-weighted)"][0]}}
with open("threshold_analysis.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved threshold_analysis.json")
print("\nNext: step 13 — full evaluation on the held-out test set, opened once.")
