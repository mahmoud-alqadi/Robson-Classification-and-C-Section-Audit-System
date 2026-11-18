"""
=============================================================================
STEP 14 — SHAP
=============================================================================
    pip install shap matplotlib
    python step14_shap.py

WHAT SHAP DOES AND WHAT IT DOES NOT DO

SHAP explains THE MODEL. It says which inputs moved this model's output and
in which direction. It does not say what causes a caesarean. If the data
contain a pattern driven by practice rather than physiology, SHAP will
faithfully report that pattern as important, because it is important — to
the model. Reading it as a causal account is the most common way these
plots are misused, and the distinction belongs in the write-up.

TreeExplainer is used because it is exact for tree ensembles rather than an
approximation, and fast enough to run on a real sample.

TWO CHECKS WORTH MAKING BEFORE READING ANYTHING ELSE

  RF_CESARN should be almost inert. The training population is groups 1,
  2a, 3 and 4a, none of whom has a previous caesarean by definition. If the
  number of previous caesareans shows real influence, something has leaked.

  PRIORLIVE should be strong. The model cannot see onset of labour or the
  Robson group — both were excluded. Parity is the main thing separating
  groups 1 and 2a from 3 and 4a, and those pairs sit at very different
  caesarean rates. If parity dominates, the model is working through a
  mechanism we can name, which is a good sign rather than a surprising one.

OUTPUTS
    shap_summary.png            direction and magnitude per feature
    shap_importance.png         mean absolute contribution, ranked
    shap_importance.csv         the same, as numbers
    shap_by_group.csv           which features matter in which Robson group
    shap_dependence_*.png       how the top features behave across their range
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

import xgboost as xgb
try:
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as e:
    raise SystemExit(f"Missing package: {e}\n  pip install shap matplotlib")

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]
SAMPLE = 60_000          # enough for stable estimates, small enough to plot

wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")

te = wide[wide["Robson_group"].isin(NARROW) & (wide["_part"] == "test")]
s = te.sample(n=min(SAMPLE, len(te)), random_state=42)
X = s[FEATURES]
print(f"Explaining {len(X):,} held-out births ({len(FEATURES)} features)\n")

bst = xgb.Booster(); bst.load_model("xgb_final.json")
expl = shap.TreeExplainer(bst)
sv = expl.shap_values(X)
print("SHAP values computed.\n")

# ---- ranked importance ----------------------------------------------------
imp = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs_shap": np.abs(sv).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
imp["share_%"] = (100 * imp["mean_abs_shap"] / imp["mean_abs_shap"].sum()).round(2)

print("=" * 70)
print("RANKED CONTRIBUTION")
print("=" * 70)
print(imp.head(20).to_string(index=False))
imp.to_csv("shap_importance.csv", index=False)

# ---- the two sanity checks ------------------------------------------------
print("\n" + "=" * 70)
print("SANITY CHECKS")
print("=" * 70)
def rank_of(f):
    r = imp.index[imp.feature == f]
    return (int(r[0]) + 1, float(imp.loc[r[0], "share_%"])) if len(r) else (None, None)

r_cs, sh_cs = rank_of("RF_CESARN")
r_par, sh_par = rank_of("PRIORLIVE")
print(f"  RF_CESARN  rank {r_cs} of {len(imp)}, {sh_cs}% of total contribution")
print(f"    {'as expected — inert, since no woman here has a uterine scar' if sh_cs is not None and sh_cs < 2 else 'UNEXPECTED — investigate, this should be near zero'}")
print(f"  PRIORLIVE  rank {r_par} of {len(imp)}, {sh_par}% of total contribution")
print(f"    {'as expected — parity is the main thing separating the group pairs' if r_par and r_par <= 3 else 'lower than expected — worth understanding why'}")

# ---- plots ----------------------------------------------------------------
print("\nWriting plots ...")
plt.figure()
shap.summary_plot(sv, X, max_display=18, show=False)
plt.tight_layout(); plt.savefig("shap_summary.png", dpi=150); plt.close()

plt.figure()
shap.summary_plot(sv, X, plot_type="bar", max_display=18, show=False)
plt.tight_layout(); plt.savefig("shap_importance.png", dpi=150); plt.close()

for f in imp.head(4)["feature"]:
    plt.figure()
    shap.dependence_plot(f, sv, X, show=False)
    plt.tight_layout(); plt.savefig(f"shap_dependence_{f}.png", dpi=150); plt.close()
print("  shap_summary.png, shap_importance.png, 4 dependence plots")

# ---- per Robson group -----------------------------------------------------
print("\n" + "=" * 70)
print("WHICH FEATURES MATTER IN WHICH GROUP")
print("=" * 70)
rows = []
for g in NARROW:
    m = (s["Robson_group"] == g).to_numpy()
    if m.sum() < 500:
        continue
    gi = pd.Series(np.abs(sv[m]).mean(axis=0), index=FEATURES)
    gi = 100 * gi / gi.sum()
    for f, v in gi.sort_values(ascending=False).head(6).items():
        rows.append({"group": g, "feature": f, "share_%": round(float(v), 2)})
byg = pd.DataFrame(rows)
for g in NARROW:
    sub = byg[byg.group == g]
    if len(sub):
        parts = [f"{r['feature']} {r['share_%']}%" for _, r in sub.iterrows()]
        print(f"\n  Group {g}: " + " · ".join(parts))
byg.to_csv("shap_by_group.csv", index=False)

print("""

=====================================================================
HOW TO READ THIS IN THE WRITE-UP
=====================================================================
  A feature ranking high means the model relied on it. It does not mean
  that changing it would change the outcome. Two of the strongest signals
  in obstetric data — parity and gestational age — are also strongly
  associated with how clinicians manage a labour, so their importance
  here is a mixture of biology and practice that this data cannot
  separate.

  What the ranking CAN support is the negative claim the project rests
  on: if the measurable clinical picture accounted for the caesarean
  decision, a model given all of it would predict far better than 0.68 to
  0.70 within group. It does not. The residue is where practice lives,
  and that residue is what the Egypt comparison quantifies.
""")

json.dump({"top_features": imp.head(15).to_dict("records"),
           "n_explained": int(len(X))},
          open("shap_summary.json", "w"), indent=2)
print("Saved shap_importance.csv, shap_by_group.csv, shap_summary.json")
print("\nNext: step 15 — per-group performance is already in step 13; move to")
print("step 16, packaging the outputs for the dashboard.")
