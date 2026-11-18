"""
=============================================================================
STEP 16 — PACKAGE THE OUTPUTS
=============================================================================
    python step16_package.py

WHAT THIS IS FOR

Everything the model produced is currently spread across JSON files, console
output and a saved booster. Power BI cannot read any of that, and neither can
a reader six months from now.

This step turns it into flat tables that can be loaded directly, and — more
importantly — writes a model card recording what the model is for, what it
is not for, and what was decided along the way. Two of those decisions
reversed things stated in the proposal, and a project that changes course
without a written record of why is indistinguishable from one that never
noticed.

WHAT GETS WRITTEN

  dash_model_performance.csv    headline metrics, validation against test
  dash_calibration.csv          predicted against observed by risk decile
  dash_by_group.csv             performance inside each Robson group
  dash_feature_importance.csv   SHAP, both rankings, with the difference
                                between them made explicit
  dash_thresholds.csv           operating points and their cost in cases
  MODEL_CARD.md                 what it is for, what it is not for, and why
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

need = ["test_results.json", "shap_importance.csv", "threshold_analysis.json"]
missing = [f for f in need if not Path(f).exists()]
if missing:
    raise SystemExit(f"Missing: {missing} — run steps 12 to 14 first.")

test = json.load(open("test_results.json"))
thr  = json.load(open("threshold_analysis.json"))
imp  = pd.read_csv("shap_importance.csv")
within = pd.read_csv("shap_within_group.csv") if Path("shap_within_group.csv").exists() else None

OUT = Path("dashboard_data"); OUT.mkdir(exist_ok=True)

# ---- 1. performance -------------------------------------------------------
v = test["validation_comparison"]
perf = pd.DataFrame([
    {"metric": "ROC-AUC (overall)",        "validation": v["roc_auc"],  "test": test["roc_auc"],
     "note": "inflated by separation BETWEEN groups"},
    {"metric": "ROC-AUC (within group)",   "validation": None,
     "test": f"{test['within_group_auc_range'][0]:.3f}-{test['within_group_auc_range'][1]:.3f}"
     if "within_group_auc_range" in test else None,
     "note": "the honest measure of skill"},
    {"metric": "AUPRC",                    "validation": v["auprc"],    "test": test["auprc"],
     "note": f"baseline {test['observed_rate']:.4f}"},
    {"metric": "Brier score",              "validation": v["brier"],    "test": test["brier"],
     "note": "lower is better"},
    {"metric": "Mean predicted rate",      "validation": v["mean_pred"],"test": test["mean_predicted"],
     "note": f"observed {test['observed_rate']:.4f}"},
    {"metric": "Calibration error (pp)",   "validation": None,          "test": test["calibration_mae_pp"],
     "note": "mean absolute, across risk deciles"},
])
perf.to_csv(OUT/"dash_model_performance.csv", index=False)

# ---- 2. calibration -------------------------------------------------------
if Path("test_calibration_curve.csv").exists():
    cal = pd.read_csv("test_calibration_curve.csv")
    cal.to_csv(OUT/"dash_calibration.csv", index=False)

# ---- 3. by group ----------------------------------------------------------
byg = pd.DataFrame(test["by_group"])
byg.to_csv(OUT/"dash_by_group.csv", index=False)

# ---- 4. feature importance, both rankings --------------------------------
imp = imp.rename(columns={"share_%": "overall_share_%"})
if within is not None:
    w = (within.groupby("feature")["sd_%"].mean()
                .rename("within_group_share_%").reset_index())
    imp = imp.merge(w, on="feature", how="left")
    imp["difference"] = (imp["overall_share_%"] - imp["within_group_share_%"]).round(2)
imp.to_csv(OUT/"dash_feature_importance.csv", index=False)

# ---- 5. thresholds --------------------------------------------------------
rows = []
for name, d in thr["operating_points"].items():
    rows.append({"criterion": name, "threshold": round(d["threshold"], 3),
                 "precision": round(d["precision"], 3), "recall": round(d["recall"], 3)})
pd.DataFrame(rows).to_csv(OUT/"dash_thresholds.csv", index=False)

# ---- 6. model card --------------------------------------------------------
wg = test.get("within_group_auc_range", [None, None])
card = f"""# Model Card

## What this model is

A gradient-boosted tree (XGBoost) predicting caesarean delivery from
characteristics known before the delivery decision, trained on the CDC
Natality Public Use File for 2023.

Training population: Robson groups 1, 2a, 3 and 4a — term, singleton,
cephalic, no uterine scar, labour occurred. {test['n_test']:,} held-out
births were used for the evaluation below.

## What it predicts, and what it does not

The target records whether a caesarean was PERFORMED, not whether one was
NEEDED. A model trained on outcomes learns the behaviour of the clinicians
who produced them. Because that behaviour is the phenomenon under study, it
cannot also serve as ground truth for necessity. Every use of this model
must carry that sentence with it.

## Performance on the held-out test set

| Metric | Value |
|---|---|
| ROC-AUC, overall | {test['roc_auc']:.4f} |
| ROC-AUC, within Robson group | {wg[0]:.3f} to {wg[1]:.3f} |
| AUPRC | {test['auprc']:.4f} (baseline {test['observed_rate']:.4f}) |
| Brier score | {test['brier']:.4f} |
| Mean predicted rate | {100*test['mean_predicted']:.2f}% against {100*test['observed_rate']:.2f}% observed |
| Calibration error | {test['calibration_mae_pp']:.2f} percentage points |

The overall figure is inflated by how easily the groups separate from one
another: group 3 runs at 2.5% and group 2a at 29%. Hold the group constant —
which is what a clinician actually knows — and discrimination falls to
{wg[0]:.3f}-{wg[1]:.3f}. That range is the honest description.

For comparison, the WHO C-Model, built on more than ten million deliveries
across 43 countries, reports AUC 0.832-0.844 and is used to compare
facilities rather than to predict for individuals. The ceiling here is a
property of the problem, not a shortfall in this implementation.

## Approved and prohibited uses

APPROVED — aggregate estimation. Summing predicted probabilities across a
population, and comparing facilities or groups. Calibration is
{test['calibration_mae_pp']:.2f} pp, which supports this.

APPROVED — explanation. SHAP analysis of what the recorded clinical picture
does and does not account for.

NOT APPROVED — individual clinical decisions. At the best available
operating point, precision is {thr['operating_points']['F0.5 (precision-weighted)']['precision']:.2f}: roughly six in ten flagged women do not
have a caesarean. The model must never be presented to a clinician as a
reason to book one. This mirrors ACOG guidance that a VBAC success score
must never become a barrier to trial of labour.

NOT APPROVED — applying this model to an Egyptian case mix to produce an
expected rate. The model over-predicts group 1 by 4.95 pp and under-predicts
group 2a by 5.54 pp. Those errors cancel on a US case mix and would not
cancel on Egypt's, where group 5 is three times larger. The Egypt expected
rate is computed from OBSERVED group-specific rates in the Robson table
instead.

## Decisions taken during development

**Class weighting removed.** scale_pos_weight of 7.56 corrected the 1:7.55
imbalance but inflated the mean predicted probability to 39.4% against 11.7%
observed, while leaving ROC-AUC unchanged at 0.7985. Removing it cut Brier
from 0.1857 to 0.0886 and calibration error from 27.66 pp to 0.11 pp.
Imbalance is instead handled through the choice of metric and threshold.

**Training population settled by test, not preference.** Narrow and wide
populations were both trained and scored on identical held-out rows.
Discrimination was indistinguishable (0.7985 against 0.7986), so the narrow
population — where clinical uncertainty actually lives — was chosen.

**F2 threshold withdrawn.** The proposal committed to maximising F2 on the
grounds that missing a needed caesarean is worse than a false alarm. That
reasoning holds for a diagnostic test. This model predicts what will be
done, not what is needed, and F2 would flag 38.6% of all births as high risk
in a country where 82.5% of caesarean decisions are already taken before
labour begins. No single threshold is adopted; the operating point is stated
per use.

**Three-family comparison replaced.** A single logistic-regression baseline
was used instead. XGBoost gained +0.0136 AUPRC over it, confirming
non-linearity without the selection pressure that comes from picking the
best of several models.

## What the model learned

Within a fixed Robson group, discrimination rests on maternal body size and
age far more than on documented medical risk factors:

| Group | Maternal body + age | Medical risk factors |
|---|---|---|
| 1 | 60.6% | 14.3% |
| 2a | 57.9% | 21.2% |
| 3 | 45.4% | 12.8% |
| 4a | 44.8% | 17.7% |

This is not a claim that medical conditions do not matter clinically. It is
a statement about prevalence and about what this data contains: gestational
diabetes affects 8% of these pregnancies and gestational hypertension 10%,
so documented indications cannot account for most caesareans among women
with no uterine scar at term with a single cephalic fetus.

## Limitations

- Cervical examination, Bishop score, membrane status, cardiotocography and
  estimated fetal weight are absent from birth certificate data. This is a
  pre-delivery risk model, not an admission-time decision model.
- Birth certificates are not medical records; several risk-factor fields are
  documented by NCHS as under-reported.
- Trained on one year (2023) in one country. Temporal validation on 2024 and
  geographic transfer to Egypt are both untested.
- Race and ethnicity are excluded by design, following the 2021 revision of
  the Grobman model.

## Reproducing this

Steps 4 to 14 in order. Random seed 42 throughout. The feature list is
generated from data_dictionary.py so that the reason for every inclusion and
exclusion stays attached to the decision.
"""
Path(OUT/"MODEL_CARD.md").write_text(card, encoding="utf-8")

print("Written to dashboard_data/:")
for f in sorted(OUT.iterdir()):
    print(f"  {f.name:<34} {f.stat().st_size/1024:>7.1f} KB")

print(f"""
The model card is the piece that matters most. Two decisions in it reverse
things the proposal committed to — dropping class weighting and withdrawing
the F2 threshold — and both are recorded with the evidence that forced the
change. A supervisor can check the reasoning; an examiner can see that the
objection was anticipated.

Next: temporal validation on 2024.
""")
