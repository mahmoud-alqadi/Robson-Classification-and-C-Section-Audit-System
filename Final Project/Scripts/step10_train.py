"""
=============================================================================
STEP 10 + 11 — TRAIN, AND SETTLE THE OPEN QUESTION
=============================================================================
    pip install xgboost scikit-learn
    python step10_train.py

A DESIGN FAULT FIXED BEFORE TRAINING

Step 8 split each population independently. The narrow population is a
subset of the wide one, so the same birth could land in narrow-test and
wide-train at once. Comparing the two models on the same rows would then
flatter the wide model, because it had already seen some of them.

This script takes the WIDE split as the single reference and lets the
narrow population inherit it. A birth now has the same part in both, and
the head-to-head is clean. The narrow file's own split assignment is
ignored.

THE QUESTION BEING SETTLED

Groups 5, 6, 7, 8 and 10 are not leakage. A breech CAN deliver vaginally; a
repeat caesarean IS a decision. But their rates run so high that they may
contribute noise rather than signal, and including them lets a model earn
most of its apparent skill on cases nobody needed a model for.

The test is deliberately narrow: both models are evaluated on the SAME
rows — the held-out births in groups 1, 2a, 3 and 4a, where genuine
uncertainty lives. If the wide model does not predict those better, the
extra groups are not helping, and the narrow population is the honest
choice.

STEP 11 — A BASELINE INSTEAD OF THREE MODELS

The proposal deferred a three-family comparison. On tabular data the winner
is close to certain in advance, and every extra model is another chance to
pick the best result, which quietly inflates whatever gets reported. One
logistic regression answers the question that actually matters — are the
relationships non-linear enough to need a tree ensemble — at a fraction of
the cost. If XGBoost cannot beat it by a clear margin, that is a real
finding about the data and it gets written down.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json, time

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

if not Path("train_wide.parquet").exists():
    raise SystemExit("train_wide.parquet not found. Run step7_9_split.py first.")

print("Loading train_wide.parquet (the reference split) ...")
wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
print(f"  {len(wide):,} births, {len(FEATURES)} features\n")

narrow = wide[wide["Robson_group"].isin(NARROW)].copy()

print("=" * 74)
print("SHARED SPLIT — inherited, so the comparison is clean")
print("=" * 74)
for name, p in [("wide", wide), ("narrow", narrow)]:
    t = p.groupby("_part").agg(n=(TARGET, "size"), r=(TARGET, "mean"))
    t["cs_%"] = (100 * t["r"]).round(3)
    print(f"\n  {name.upper()}")
    print(t[["n", "cs_%"]].reindex(["train", "valid", "test"]).to_string())
    sp = t["cs_%"].max() - t["cs_%"].min()
    print(f"    spread {sp:.3f} pp  {'OK' if sp < 0.5 else '<-- CHECK'}")

def parts(p):
    out = {}
    for k in ("train", "valid", "test"):
        s = p[p["_part"] == k]
        out[k] = (s[FEATURES].astype(float), s[TARGET].astype(int), s)
    return out


def train_xgb(p, label):
    D = parts(p)
    Xtr, ytr, _ = D["train"]; Xva, yva, _ = D["valid"]
    spw = (ytr == 0).sum() / (ytr == 1).sum()
    print(f"\n  Training {label}: {len(Xtr):,} rows, scale_pos_weight={spw:.3f}")
    m = xgb.XGBClassifier(
        n_estimators=600, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=20, reg_lambda=1.0,
        scale_pos_weight=spw, tree_method="hist",
        eval_metric="aucpr", early_stopping_rounds=30,
        n_jobs=-1, random_state=42)
    t0 = time.time()
    m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    print(f"    done in {time.time()-t0:.0f}s, best iteration {m.best_iteration}")
    return m, D


def report(y, p, label, indent="    "):
    auc  = roc_auc_score(y, p)
    ap   = average_precision_score(y, p)
    br   = brier_score_loss(y, p)
    base = y.mean()
    print(f"{indent}{label:<26} ROC-AUC {auc:.4f}   AUPRC {ap:.4f} "
          f"(baseline {base:.4f})   Brier {br:.4f}")
    return {"roc_auc": auc, "auprc": ap, "brier": br, "positive_rate": float(base)}


results = {}
models = {}
print("\n" + "=" * 74)
print("STEP 10 — XGBOOST ON BOTH POPULATIONS")
print("=" * 74)
for name, p in [("narrow", narrow), ("wide", wide)]:
    m, D = train_xgb(p, name)
    models[name] = (m, D)
    Xva, yva, _ = D["valid"]
    results[f"xgb_{name}"] = report(yva, m.predict_proba(Xva)[:, 1],
                                    f"{name} on its own valid")

# ---------------------------------------------------------------------------
print("\n" + "=" * 74)
print("THE HEAD-TO-HEAD — both models, identical rows")
print("=" * 74)
print("  Held-out births in groups 1, 2a, 3 and 4a only.\n")

_, _, nva = models["narrow"][1]["valid"]
common = nva[nva["Robson_group"].isin(NARROW)]
Xc = common[FEATURES].astype(float); yc = common[TARGET].astype(int)
print(f"  n = {len(common):,}, caesarean rate {100*yc.mean():.2f}%\n")

h2h = {}
for name in ("narrow", "wide"):
    m = models[name][0]
    h2h[name] = report(yc, m.predict_proba(Xc)[:, 1], f"{name}-trained model")

d_auc = h2h["wide"]["roc_auc"] - h2h["narrow"]["roc_auc"]
d_ap  = h2h["wide"]["auprc"]   - h2h["narrow"]["auprc"]
d_br  = h2h["wide"]["brier"]   - h2h["narrow"]["brier"]
print(f"\n  wide minus narrow:  ROC-AUC {d_auc:+.4f}   AUPRC {d_ap:+.4f}   "
       f"Brier {d_br:+.4f}  (lower Brier is better)")
print(f"""
  VERDICT
  {"The wider population helps on the uncertain groups. Keep it." if d_ap > 0.005 else
    "The wider population does not improve prediction where uncertainty actually lives. Train on the narrow population and say so."}

  Either way this is now a decision made from evidence, on identical rows,
  and it goes into the record as such rather than as a preference.
""")

# ---------------------------------------------------------------------------
print("=" * 74)
print("STEP 11 — LOGISTIC REGRESSION BASELINE")
print("=" * 74)
print("""  Median imputation and standardisation, because logistic regression
  cannot take missing values the way a tree can. That handicap is part of
  the comparison, not a flaw in it: handling missingness natively is one of
  the reasons to prefer the tree.
""")
best = "wide" if d_ap > 0.005 else "narrow"
p = wide if best == "wide" else narrow
D = parts(p)
Xtr, ytr, _ = D["train"]; Xva, yva, _ = D["valid"]

sub = min(400_000, len(Xtr))
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(max_iter=1000, class_weight="balanced"))
t0 = time.time()
lr.fit(Xtr.iloc[:sub], ytr.iloc[:sub])
print(f"  fitted on {sub:,} rows in {time.time()-t0:.0f}s")
results["logreg"] = report(yva, lr.predict_proba(Xva)[:, 1], "logistic regression")
results[f"xgb_{best}_same"] = report(yva, models[best][0].predict_proba(Xva)[:, 1],
                                     f"XGBoost ({best})")

gain = results[f"xgb_{best}_same"]["auprc"] - results["logreg"]["auprc"]
print(f"""
  XGBoost gains {gain:+.4f} AUPRC over logistic regression.

  {"That is a material gain: the relationships are non-linear and the tree ensemble earns its place." if gain > 0.01 else
    "That is a small gain. The relationships are close to linear, which is itself worth reporting — and it means the interpretable model is a legitimate alternative."}
""")

models[best][0].save_model(f"xgb_{best}.json")
with open("model_metrics.json", "w") as f:
    json.dump({"chosen_population": best, "results": results,
               "head_to_head": h2h}, f, indent=2)
print(f"Saved xgb_{best}.json and model_metrics.json")
print("\nNext: step 12 — choose the decision threshold from the precision-recall curve.")
