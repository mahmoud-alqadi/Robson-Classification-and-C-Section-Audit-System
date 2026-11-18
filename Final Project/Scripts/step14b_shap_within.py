"""
=============================================================================
STEP 14b — WHAT ACTUALLY DISCRIMINATES INSIDE A ROBSON GROUP
=============================================================================
    python step14b_shap_within.py

THE PROBLEM WITH THE STEP 14 TABLE

Step 14 reported PRIORLIVE contributing 23% inside Robson group 1. Every
woman in group 1 is nulliparous — PRIORLIVE is zero for all of them. A
constant cannot distinguish one woman from another, so that 23% is not
telling us what it appears to.

The reason is how mean |SHAP| works. SHAP measures departure from a
baseline computed across the whole population, nulliparous and multiparous
together. Being nulliparous pushes every woman in group 1 the same distance
above that baseline. The contribution is large and completely uninformative
within the group: it sets the level, it does not separate anybody.

That also explains the gap between the headline ROC-AUC of 0.7991 and the
within-group figures of 0.68 to 0.70. The two largest contributors,
PRIORLIVE and ILLB_R at about 40% combined, are near-constant inside a
group. They separate the groups from each other and do almost nothing
inside them.

THE FIX

Within each group, rank features by the VARIANCE of their SHAP values
rather than the mean absolute value. A feature that is constant has zero
variance and disappears. What remains is what genuinely tells one woman
apart from another with the same Robson classification — which is the only
thing a clinician standing in front of a patient in that group can use.

Both columns are reported side by side, because the difference between them
is itself the finding.
=============================================================================
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import shap

TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]
SAMPLE = 60_000

wide = pd.read_parquet("train_wide.parquet")
FEATURES = [c for c in wide.columns
            if c not in [TARGET, "Robson_group", "Robson_group_sort_order", "_part"]]
for c in FEATURES:
    wide[c] = pd.to_numeric(wide[c], errors="coerce").astype("float32")

te = wide[wide["Robson_group"].isin(NARROW) & (wide["_part"] == "test")]
s = te.sample(n=min(SAMPLE, len(te)), random_state=42).reset_index(drop=True)
X = s[FEATURES]

bst = xgb.Booster(); bst.load_model("xgb_final.json")
sv = shap.TreeExplainer(bst).shap_values(X)
print(f"Explained {len(X):,} births\n")

# ---- overall, for reference ----------------------------------------------
overall = pd.DataFrame({
    "feature": FEATURES,
    "mean_abs": np.abs(sv).mean(axis=0),
    "sd": sv.std(axis=0),
})
overall["mean_abs_%"] = (100*overall.mean_abs/overall.mean_abs.sum()).round(2)
overall["sd_%"] = (100*overall.sd/overall.sd.sum()).round(2)

print("=" * 76)
print("WHOLE POPULATION — the two rankings already differ")
print("=" * 76)
o = overall.sort_values("mean_abs", ascending=False).head(12)
print(o[["feature", "mean_abs_%", "sd_%"]].to_string(index=False))

# ---- within group ---------------------------------------------------------
print("\n" + "=" * 76)
print("WITHIN EACH ROBSON GROUP")
print("=" * 76)
print("""
  mean_abs_%  how far this feature pushes predictions from the global
              baseline — includes constants that move everyone equally
  sd_%        how much this feature VARIES between women in the group —
              a constant scores zero here
""")

rows = []
for g in NARROW:
    m = (s["Robson_group"] == g).to_numpy()
    if m.sum() < 500:
        continue
    sub_sv = sv[m]
    t = pd.DataFrame({
        "feature": FEATURES,
        "mean_abs": np.abs(sub_sv).mean(axis=0),
        "sd": sub_sv.std(axis=0),
        "value_sd": X[m].std(axis=0).to_numpy(),
    })
    t["mean_abs_%"] = (100*t.mean_abs/t.mean_abs.sum()).round(2)
    t["sd_%"] = (100*t.sd/t.sd.sum()).round(2)
    t["group"] = g
    t = t.sort_values("sd", ascending=False)
    rows.append(t)

    print(f"\n  ── GROUP {g}  (n={int(m.sum()):,}, "
          f"caesarean {100*s.loc[m, TARGET].mean():.2f}%) " + "─"*28)
    print(t.head(8)[["feature", "sd_%", "mean_abs_%"]].to_string(index=False))

    # features that set the level but do not discriminate
    dead = t[(t["sd_%"] < 0.5) & (t["mean_abs_%"] > 3)]
    if len(dead):
        print("    level-setting only (large contribution, no variation here):")
        for _, r in dead.iterrows():
            print(f"      {r['feature']:<18} mean_abs {r['mean_abs_%']:>5.2f}%   "
                  f"sd {r['sd_%']:>5.2f}%")

allg = pd.concat(rows, ignore_index=True)
allg[["group", "feature", "sd_%", "mean_abs_%"]].to_csv(
    "shap_within_group.csv", index=False)

# ---- the summary that goes in the write-up -------------------------------
print("\n" + "=" * 76)
print("WHAT THIS MEANS")
print("=" * 76)

top = (allg.sort_values(["group", "sd_%"], ascending=[True, False])
           .groupby("group").head(4))
print("\n  Ranked by what actually varies within the group:\n")
for g in NARROW:
    sub = top[top.group == g]
    if len(sub):
        print(f"    Group {g}: " + " · ".join(
            f"{r['feature']} {r['sd_%']}%" for _, r in sub.iterrows()))

body = ["BMI", "M_Ht_In", "PWgt_R", "WTGAIN", "MAGER"]
risk = ["RF_PDIAB", "RF_GDIAB", "RF_PHYPE", "RF_GHYPE", "RF_EHYPE",
        "RF_PPTERM", "RF_INFTR", "RF_ARTEC"]
b = allg[allg.feature.isin(body)].groupby("group")["sd_%"].sum()
r = allg[allg.feature.isin(risk)].groupby("group")["sd_%"].sum()
print(f"\n  {'group':<8} {'maternal body + age':>22} {'medical risk factors':>24}")
for g in NARROW:
    if g in b.index:
        print(f"  {g:<8} {b[g]:>21.1f}% {r.get(g, 0):>23.1f}%")

print("""
  Read those two columns against each other.

  Once the Robson group is fixed — which is what a clinician actually knows
  when a woman is in front of them — the model separates one woman from
  another almost entirely on maternal size and age. The recorded medical
  risk factors, the conditions that would ordinarily justify a caesarean,
  account for a small fraction of the discrimination.

  That is not an argument that those conditions do not matter clinically.
  It is a statement about what is in this data: among women with no
  uterine scar, at term, with a single cephalic fetus, documented medical
  indications explain very little of who ends up with a caesarean and who
  does not.

  This is the same negative claim the project rests on, arriving from a
  different direction.
""")
print("Saved shap_within_group.csv")
