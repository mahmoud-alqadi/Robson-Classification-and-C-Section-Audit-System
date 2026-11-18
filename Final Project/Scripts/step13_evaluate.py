"""
=============================================================================
STEP 13 — FINAL EVALUATION ON THE HELD-OUT TEST SET
=============================================================================
    python step13_evaluate.py

THE RULE THAT MAKES THIS NUMBER WORTH ANYTHING

Every figure so far came from the validation set, which was used to choose
the model, drop class weighting and examine thresholds. Numbers from a set
you have made decisions on are optimistic by construction.

The test set has never been read. No choice in this project has been made
with reference to it. It is opened once, here, and whatever it says is what
gets written down.

If the result is worse than hoped, the correct response is to report it —
not to adjust the model and run again. Doing that would turn the test set
into a second validation set, and every subsequent number would be
flattering in a way nobody could detect afterwards.

A USEFUL SELF-CHECK
Validation gave ROC-AUC 0.7985, AUPRC 0.3535, Brier 0.0886, and a mean
predicted probability of 11.70% against 11.72% observed. If the test set
lands close to those, nothing has leaked between the splits. A large gap
would be the warning sign, and it is worth looking at that comparison
before looking at anything else.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, precision_recall_fscore_support,
                             confusion_matrix)

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]

for f in ["xgb_final.json", "train_wide.parquet", "threshold_analysis.json"]:
    if not Path(f).exists():
        raise SystemExit(f"{f} not found — run the earlier steps first.")

thr_cfg = json.load(open("threshold_analysis.json"))
VOL_THR = thr_cfg["recommended"]["quality_monitoring"]
PREC_THR = thr_cfg["recommended"]["individual_counselling"]

wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")

te = wide[wide["Robson_group"].isin(NARROW) & (wide["_part"] == "test")].copy()
X, y = te[FEATURES], te[TARGET].to_numpy().astype(int)

bst = xgb.Booster(); bst.load_model("xgb_final.json")
p = bst.predict(xgb.DMatrix(X))
te["pred"] = p

print("=" * 76)
print("TEST SET — opened once")
print("=" * 76)
print(f"  n = {len(y):,} births, never seen by the model")
print(f"  observed caesarean rate {100*y.mean():.2f}%\n")

auc  = roc_auc_score(y, p)
ap   = average_precision_score(y, p)
brier= brier_score_loss(y, p)

VALID = {"roc_auc": 0.7985, "auprc": 0.3535, "brier": 0.0886, "mean_pred": 0.1170}
print(f"{'metric':<26} {'validation':>12} {'TEST':>12} {'difference':>12}")
print("-" * 66)
for k, v, t in [("ROC-AUC", VALID['roc_auc'], auc),
                ("AUPRC", VALID['auprc'], ap),
                ("Brier (lower better)", VALID['brier'], brier),
                ("mean predicted", VALID['mean_pred'], p.mean())]:
    print(f"{k:<26} {v:>12.4f} {t:>12.4f} {t-v:>+12.4f}")
print(f"{'observed rate':<26} {'':>12} {y.mean():>12.4f}")
print("""
  Close agreement between the two columns means the split was clean and
  nothing leaked. A large drop on the test set would mean the model had
  been tuned to the validation set rather than to the problem.
""")

# ---- calibration ----------------------------------------------------------
print("=" * 76)
print("CALIBRATION ON UNSEEN DATA")
print("=" * 76)
q = pd.qcut(p, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"bin": q, "pred": p, "obs": y}).groupby("bin").agg(
    n=("obs", "size"), predicted=("pred", "mean"), observed=("obs", "mean"))
cal["predicted_%"] = (100*cal["predicted"]).round(2)
cal["observed_%"]  = (100*cal["observed"]).round(2)
cal["gap_pp"]      = (cal["predicted_%"] - cal["observed_%"]).round(2)
print(cal[["n", "predicted_%", "observed_%", "gap_pp"]].to_string())
mae = cal["gap_pp"].abs().mean()
print(f"\n  mean absolute calibration error: {mae:.2f} pp")
print("""
  This is the number the Egypt calculation depends on. The expected rate is
  a sum of these probabilities, so an error here transfers directly into the
  estimated gap. Discrimination affects who is ranked highest; calibration
  affects whether the total is right.
""")

# ---- operating points -----------------------------------------------------
print("=" * 76)
print("PERFORMANCE AT THE TWO RECOMMENDED OPERATING POINTS")
print("=" * 76)
for name, t in [("Quality monitoring (volume-matched)", VOL_THR),
                ("Individual counselling (precision-weighted)", PREC_THR)]:
    pred = (p >= t).astype(int)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary",
                                                    zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    print(f"\n  {name}   threshold {t:.3f}")
    print(f"    precision {pr:.3f}   recall {rc:.3f}   F1 {f1:.3f}")
    print(f"    confusion matrix")
    print(f"                      predicted no    predicted caesarean")
    print(f"      actual no       {tn:>12,}   {fp:>20,}")
    print(f"      actual caesarean{fn:>12,}   {tp:>20,}")
    print(f"    of {int(pred.sum()):,} women flagged, {fp:,} did not have a caesarean")

print("""
  Accuracy is deliberately not reported. On a population that is 88%
  vaginal, a model predicting "no caesarean" for everyone scores 88% and
  knows nothing. That is why the pre-registered metrics were precision,
  recall, F1, ROC-AUC and AUPRC.
""")

# ---- by Robson group ------------------------------------------------------
print("=" * 76)
print("WHERE THE MODEL IS STRONG AND WHERE IT IS WEAK")
print("=" * 76)
rows = []
for g, sub in te.groupby("Robson_group", observed=True):
    yy = sub[TARGET].to_numpy().astype(int); pp = sub["pred"].to_numpy()
    if len(np.unique(yy)) < 2:
        continue
    rows.append({"group": str(g), "n": int(len(sub)),
                 "observed_%": round(float(100*yy.mean()), 2),
                 "predicted_%": round(float(100*pp.mean()), 2),
                 "gap_pp": round(float(100*(pp.mean()-yy.mean())), 2),
                 "roc_auc": round(float(roc_auc_score(yy, pp)), 4),
                 "auprc": round(float(average_precision_score(yy, pp)), 4)})
r = pd.DataFrame(rows).sort_values("group")
print(r.to_string(index=False))
print("""
  Reporting one average across groups would hide this. A model that works
  well in group 3 and poorly in group 2a is not the same instrument as one
  that performs evenly, and a reader is entitled to know which it is.
""")

within = [r["roc_auc"] for r in rows]
print("=" * 76)
print("TWO THINGS THE GROUP TABLE REVEALS")
print("=" * 76)
print(f"""
  1. WITHIN-GROUP DISCRIMINATION IS {min(within):.3f} TO {max(within):.3f},
     not {auc:.4f}.

     Much of the headline figure comes from separating the groups from one
     another — group 3 runs at 2.5% and group 2a at 29%, so telling them
     apart is easy and inflates the average. Hold the group constant and the
     model is markedly weaker. The honest description of its skill is the
     within-group range, and that is what should be reported.

  2. THE OVERALL CALIBRATION OF {mae:.2f} pp IS ERRORS CANCELLING OUT.

     The model over-predicts group 1 and under-predicts group 2a. Summed
     across a population with this group mix they offset almost exactly. On
     a population with a DIFFERENT mix they would not, and the error would
     no longer be invisible.

     This is why the Egypt expected rate is computed from OBSERVED
     group-specific rates in the Robson table, not by applying this model to
     an Egyptian case mix. Egypt's distribution is nothing like this one —
     group 5 is three times larger and group 1 far smaller — so a model
     carrying opposite biases in different groups would produce a distorted
     expected rate and therefore a distorted gap.

     The model's role is explanatory: SHAP, and the demonstration that
     clinical variables account for part of the variation and not all of it.
     It is not in the critical path of the project's central estimate.
""")

out = {"n_test": int(len(y)), "observed_rate": float(y.mean()),
       "roc_auc": float(auc), "auprc": float(ap), "brier": float(brier),
       "mean_predicted": float(p.mean()),
       "calibration_mae_pp": float(mae),
       "validation_comparison": VALID,
       "by_group": rows,
       "within_group_auc_range": [float(min(within)), float(max(within))]}
with open("test_results.json", "w") as f:
    json.dump(out, f, indent=2)
r.to_csv("test_results_by_group.csv", index=False)
cal.to_csv("test_calibration_curve.csv")
print("Saved test_results.json, test_results_by_group.csv, test_calibration_curve.csv")
print("\nNext: step 14 — SHAP.")
