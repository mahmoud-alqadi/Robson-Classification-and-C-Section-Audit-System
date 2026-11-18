"""
=============================================================================
STEP 10 + 11 — TRAIN  (memory-safe rewrite)
=============================================================================
    pip install xgboost scikit-learn
    python step10_train_v2.py

WHY THIS VERSION EXISTS

The first version died silently part-way through training the narrow model:
no error, no traceback, just a returned prompt. That is what a Windows
out-of-memory kill looks like. XGBoost with n_jobs=-1 makes a working copy
of the matrix per thread, and 1.5 million rows in float64 across every core
is more than a laptop has.

Four changes, none of which cost anything real:

  float32 instead of float64 ...... halves the matrix; birth weights and
                                    gestational ages do not need 15
                                    significant figures
  n_jobs=4 instead of -1 .......... caps the number of working copies
  max_bin=128 instead of 256 ...... halves the histogram, with no
                                    meaningful loss on continuous features
                                    of this resolution
  QuantileDMatrix ................. builds the histogram directly and
                                    releases the source frame, instead of
                                    holding both in memory at once

There is also a --sample switch. If the machine still cannot hold it, train
on a random subset. At this scale that costs very little: a model fitted on
500,000 births is not meaningfully worse than one fitted on 1.5 million,
because performance stopped improving with sample size long before either.

    python step10_train_v2.py --sample 500000

WHAT IS BEING DECIDED HERE
Whether the model should learn from groups 1, 2a, 3 and 4a only (narrow) or
from every group except the three excluded for leakage (wide). Both models
are scored on THE SAME held-out rows in the uncertain groups, so the
comparison is not confounded by the population being scored.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json, time, gc, sys

try:
    import xgboost as xgb
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
except ImportError as e:
    raise SystemExit(f"Missing package: {e}\n  pip install xgboost scikit-learn")

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]

SAMPLE = None
if "--sample" in sys.argv:
    SAMPLE = int(sys.argv[sys.argv.index("--sample") + 1])

if not Path("train_wide.parquet").exists():
    raise SystemExit("train_wide.parquet not found. Run step7_9_split.py first.")

print("Loading train_wide.parquet ...")
wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]

# cast once, up front — this is where most of the saving comes from
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")
wide[TARGET] = wide[TARGET].astype("int8")
print(f"  {len(wide):,} births, {len(FEATURES)} features, float32")
print(f"  memory: {wide.memory_usage(deep=True).sum()/1e9:.2f} GB\n")

if SAMPLE and SAMPLE < len(wide):
    keep = (wide.groupby("_part", group_keys=False)
                .apply(lambda g: g.sample(n=min(len(g), int(SAMPLE*len(g)/len(wide))),
                                          random_state=42)))
    wide = keep.reset_index(drop=True)
    print(f"  SAMPLED down to {len(wide):,} births\n")

narrow_mask = wide["Robson_group"].isin(NARROW)

print("=" * 74)
print("SHARED SPLIT — the narrow population inherits the wide split")
print("=" * 74)
for name, mask in [("wide", pd.Series(True, index=wide.index)), ("narrow", narrow_mask)]:
    t = wide[mask].groupby("_part").agg(n=(TARGET, "size"), r=(TARGET, "mean"))
    t["cs_%"] = (100 * t["r"]).round(3)
    print(f"\n  {name.upper()}")
    print(t[["n", "cs_%"]].reindex(["train", "valid", "test"]).to_string())


def xy(mask, part):
    s = wide[mask & (wide["_part"] == part)]
    return s[FEATURES], s[TARGET].to_numpy(), s


def train(mask, label):
    Xtr, ytr, _ = xy(mask, "train")
    Xva, yva, _ = xy(mask, "valid")
    spw = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    print(f"\n  Training {label}: {len(Xtr):,} rows, scale_pos_weight={spw:.3f}")

    dtr = xgb.QuantileDMatrix(Xtr, label=ytr, max_bin=128)
    dva = xgb.QuantileDMatrix(Xva, label=yva, max_bin=128, ref=dtr)
    del Xtr, Xva; gc.collect()

    params = dict(objective="binary:logistic", eval_metric="aucpr",
                  eta=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                  min_child_weight=20, reg_lambda=1.0,
                  scale_pos_weight=spw, tree_method="hist",
                  max_bin=128, nthread=4, seed=42)
    t0 = time.time()
    bst = xgb.train(params, dtr, num_boost_round=300,
                    evals=[(dva, "valid")], early_stopping_rounds=30,
                    verbose_eval=50)
    print(f"    done in {time.time()-t0:.0f}s, best iteration {bst.best_iteration}")
    del dtr, dva; gc.collect()
    return bst


def predict(bst, X):
    return bst.predict(xgb.DMatrix(X), iteration_range=(0, bst.best_iteration + 1))


def report(y, p, label, indent="    "):
    m = {"roc_auc": roc_auc_score(y, p),
         "auprc": average_precision_score(y, p),
         "brier": brier_score_loss(y, p),
         "positive_rate": float(np.mean(y))}
    print(f"{indent}{label:<26} ROC-AUC {m['roc_auc']:.4f}   "
          f"AUPRC {m['auprc']:.4f} (baseline {m['positive_rate']:.4f})   "
          f"Brier {m['brier']:.4f}")
    return m


print("\n" + "=" * 74)
print("STEP 10 — XGBOOST ON BOTH POPULATIONS")
print("=" * 74)
results, models = {}, {}
for name, mask in [("narrow", narrow_mask),
                   ("wide", pd.Series(True, index=wide.index))]:
    bst = train(mask, name)
    models[name] = bst
    Xva, yva, _ = xy(mask, "valid")
    results[f"xgb_{name}"] = report(yva, predict(bst, Xva), f"{name} on its own valid")
    del Xva; gc.collect()

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("THE HEAD-TO-HEAD — both models, identical rows")
print("=" * 74)
print("  Held-out births in groups 1, 2a, 3 and 4a only — where the")
print("  uncertainty actually lives.\n")

Xc, yc, _ = xy(narrow_mask, "valid")
print(f"  n = {len(Xc):,}, caesarean rate {100*np.mean(yc):.2f}%\n")
h2h = {n: report(yc, predict(models[n], Xc), f"{n}-trained model")
       for n in ("narrow", "wide")}

d_auc = h2h["wide"]["roc_auc"] - h2h["narrow"]["roc_auc"]
d_ap  = h2h["wide"]["auprc"]   - h2h["narrow"]["auprc"]
d_br  = h2h["wide"]["brier"]   - h2h["narrow"]["brier"]
print(f"\n  wide minus narrow:  ROC-AUC {d_auc:+.4f}   AUPRC {d_ap:+.4f}   "
      f"Brier {d_br:+.4f}  (lower Brier is better)")

best = "wide" if d_ap > 0.005 else "narrow"
print(f"""
  VERDICT — train on the {best.upper()} population.
  {"The extra groups improve prediction where it is genuinely uncertain."
    if best == "wide" else
   "The extra groups do not improve prediction where it is genuinely uncertain, so they are contributing high-certainty cases rather than information."}

  This is now settled on identical rows, and goes into the record as a
  result rather than a preference.
""")

# ---------------------------------------------------------------------------
print("=" * 74)
print("STEP 11 — LOGISTIC REGRESSION BASELINE")
print("=" * 74)
print("""  Median imputation and standardisation, because logistic regression
  cannot take missing values the way a tree can. That handicap is part of
  the comparison rather than a flaw in it.
""")
mask = narrow_mask if best == "narrow" else pd.Series(True, index=wide.index)
Xtr, ytr, _ = xy(mask, "train")
Xva, yva, _ = xy(mask, "valid")
sub = min(300_000, len(Xtr))
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight="balanced"))
t0 = time.time()
lr.fit(Xtr.iloc[:sub], ytr[:sub])
print(f"  fitted on {sub:,} rows in {time.time()-t0:.0f}s")
results["logreg"] = report(yva, lr.predict_proba(Xva)[:, 1], "logistic regression")
results["xgb_chosen"] = report(yva, predict(models[best], Xva), f"XGBoost ({best})")

gain = results["xgb_chosen"]["auprc"] - results["logreg"]["auprc"]
print(f"""
  XGBoost gains {gain:+.4f} AUPRC over logistic regression.
  {"A material gain: the relationships are non-linear and the tree ensemble earns its place."
    if gain > 0.01 else
   "A small gain. The relationships are close to linear — worth reporting in its own right, and it means the interpretable model is a legitimate alternative."}
""")

models[best].save_model(f"xgb_{best}.json")
pd.Series(FEATURES).to_csv("model_feature_order.csv", index=False, header=["feature"])
with open("model_metrics.json", "w") as f:
    json.dump({"chosen_population": best, "n_train": int(mask.sum()),
               "sampled": SAMPLE, "results": results, "head_to_head": h2h},
              f, indent=2)
print(f"Saved xgb_{best}.json, model_feature_order.csv, model_metrics.json")
print("\nNext: step 12 — choose the decision threshold from the precision-recall curve.")
