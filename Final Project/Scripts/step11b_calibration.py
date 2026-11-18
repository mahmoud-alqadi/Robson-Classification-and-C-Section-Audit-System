"""
=============================================================================
STEP 11b — WHY THE BRIER SCORES DIFFER
=============================================================================
    python step11b_calibration.py

WHAT THE HEAD-TO-HEAD ACTUALLY SHOWED

On identical held-out rows the two models discriminated the same:
ROC-AUC 0.7985 against 0.7986, AUPRC 0.3538 against 0.3543. Nothing to
choose between them.

But Brier came out 0.1857 for narrow and 0.1035 for wide — the wide model
44% better. Brier combines discrimination and calibration. Discrimination
is identical here, so the whole of that difference is CALIBRATION: the
narrow model ranks births correctly but the probabilities it prints are
wrong.

The automatic verdict in step 10 looked only at AUPRC and therefore missed
this. The rule was too simple and is corrected here.

WHY CALIBRATION MATTERS MORE THAN DISCRIMINATION IN THIS PROJECT

Most clinical prediction papers care about ranking: who is at higher risk.
This project does something different. It sums predicted probabilities to
produce an EXPECTED caesarean rate, and subtracts that from the observed
rate to estimate excess. A model whose probabilities are inflated gives an
inflated expected rate and therefore an understated gap. Ranking correctly
is not enough — the numbers themselves have to be right.

THE HYPOTHESIS
scale_pos_weight was 7.561 for the narrow model and 2.465 for the wide one.
Class weighting corrects imbalance by making positives count for more,
which shifts every predicted probability upward. The larger the weight, the
larger the distortion. If that is the cause, the population is innocent and
the fix is to drop the weighting rather than to change who the model learns
from.

Three models, one question:
    narrow WITH weighting      the current model
    narrow WITHOUT weighting   isolates the weighting
    wide WITH weighting        the comparator
All scored on the same held-out rows.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import gc, json

import xgboost as xgb
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss)

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]

wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")
wide[TARGET] = wide[TARGET].astype("int8")
nmask = wide["Robson_group"].isin(NARROW)
print(f"Loaded {len(wide):,} births\n")


def xy(mask, part):
    s = wide[mask & (wide["_part"] == part)]
    return s[FEATURES], s[TARGET].to_numpy()


def fit(mask, weighted, label):
    Xtr, ytr = xy(mask, "train")
    Xva, yva = xy(mask, "valid")
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1)) if weighted else 1.0
    print(f"  {label:<28} n={len(Xtr):>9,}  scale_pos_weight={spw:.3f}")
    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=128)
    dva = xgb.QuantileDMatrix(Xva, label=yva, max_bin=128, ref=dtr)
    del Xtr, Xva; gc.collect()
    p = dict(objective="binary:logistic", eval_metric="aucpr", eta=0.08,
             max_depth=6, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=20, reg_lambda=1.0, scale_pos_weight=spw,
             tree_method="hist", max_bin=128, nthread=4, seed=42)
    b = xgb.train(p, dtr, num_boost_round=300, evals=[(dva, "valid")],
                  early_stopping_rounds=30, verbose_eval=False)
    del dtr, dva; gc.collect()
    return b


def predict(b, X):
    return b.predict(xgb.DMatrix(X), iteration_range=(0, b.best_iteration + 1))


print("Training three models ...")
models = {
    "narrow + weighting":    fit(nmask, True,  "narrow + weighting"),
    "narrow, NO weighting":  fit(nmask, False, "narrow, NO weighting"),
    "wide + weighting":      fit(pd.Series(True, index=wide.index), True,
                                 "wide + weighting"),
}

# ---- score all three on the SAME rows -------------------------------------
Xc, yc = xy(nmask, "valid")
print("\n" + "=" * 78)
print("ALL THREE ON THE SAME HELD-OUT ROWS")
print(f"n = {len(Xc):,}, observed caesarean rate {100*yc.mean():.2f}%")
print("=" * 78)
print(f"\n{'model':<24} {'ROC-AUC':>9} {'AUPRC':>8} {'Brier':>8} "
      f"{'mean predicted':>15} {'observed':>10}")
print("-" * 78)

preds, rows = {}, {}
for name, b in models.items():
    p = predict(b, Xc)
    preds[name] = p
    rows[name] = {"roc_auc": float(roc_auc_score(yc, p)),
                  "auprc": float(average_precision_score(yc, p)),
                  "brier": float(brier_score_loss(yc, p)),
                  "mean_pred": float(p.mean())}
    print(f"{name:<24} {rows[name]['roc_auc']:>9.4f} {rows[name]['auprc']:>8.4f} "
          f"{rows[name]['brier']:>8.4f} {100*p.mean():>14.2f}% {100*yc.mean():>9.2f}%")

print("""
  The last two columns are the calibration test in its simplest form. If a
  model is calibrated, the average probability it predicts across a
  population equals the rate actually observed in it. A model predicting a
  mean of 30% where 11.7% of births were caesarean is not slightly off — it
  would overstate Egypt's expected rate by a factor of two and shrink the
  estimated gap accordingly.
""")

# ---- calibration curve ----------------------------------------------------
print("=" * 78)
print("CALIBRATION CURVE — predicted against observed, by decile of risk")
print("=" * 78)
for name, p in preds.items():
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    t = pd.DataFrame({"bin": q, "pred": p, "obs": yc}).groupby("bin").agg(
        n=("obs", "size"), predicted=("pred", "mean"), observed=("obs", "mean"))
    t["predicted_%"] = (100 * t["predicted"]).round(2)
    t["observed_%"] = (100 * t["observed"]).round(2)
    t["gap_pp"] = (t["predicted_%"] - t["observed_%"]).round(2)
    print(f"\n  {name}")
    print(t[["n", "predicted_%", "observed_%", "gap_pp"]].to_string())
    print(f"    mean absolute gap: {t['gap_pp'].abs().mean():.2f} pp")

# ---- verdict --------------------------------------------------------------
nb_w  = rows["narrow + weighting"]["brier"]
nb_nw = rows["narrow, NO weighting"]["brier"]
wb    = rows["wide + weighting"]["brier"]

print("\n" + "=" * 78)
print("VERDICT")
print("=" * 78)
if nb_nw < nb_w * 0.75:
    print(f"""
  Removing class weighting cut the narrow model's Brier from {nb_w:.4f} to
  {nb_nw:.4f}. The calibration problem was the weighting, not the
  population. scale_pos_weight of 7.6 was inflating every probability.

  DECISION: train on the NARROW population WITHOUT class weighting.
  Discrimination was already identical between populations, so the narrow
  set — the births where a clinician faces genuine uncertainty — is the
  honest one to learn from, and dropping the weight fixes the probabilities.

  Imbalance is then handled where it belongs: in the choice of metric
  (AUPRC rather than accuracy) and in the decision threshold, not by
  distorting the model's output.
""")
    choice = "narrow_unweighted"
else:
    print(f"""
  Removing the weighting moved Brier from {nb_w:.4f} to {nb_nw:.4f}, which
  does not account for the gap against the wide model at {wb:.4f}. The
  weighting is not the whole story and the population itself is doing work.

  DECISION: carry both forward to step 12 and choose on calibrated
  performance after threshold selection.
""")
    choice = "undecided"

best = models["narrow, NO weighting"] if choice == "narrow_unweighted" \
       else models["wide + weighting"]
best.save_model("xgb_final.json")
with open("calibration_comparison.json", "w") as f:
    json.dump({"choice": choice, "results": rows,
               "observed_rate": float(yc.mean())}, f, indent=2)
print("Saved xgb_final.json and calibration_comparison.json")
print("\nNext: step 12 — decision threshold from the precision-recall curve.")
