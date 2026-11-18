# Model Card

## What this model is

A gradient-boosted tree (XGBoost) predicting caesarean delivery from
characteristics known before the delivery decision, trained on the CDC
Natality Public Use File for 2023.

Training population: Robson groups 1, 2a, 3 and 4a — term, singleton,
cephalic, no uterine scar, labour occurred. 486,820 held-out
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
| ROC-AUC, overall | 0.7991 |
| ROC-AUC, within Robson group | 0.678 to 0.705 |
| AUPRC | 0.3542 (baseline 0.1172) |
| Brier score | 0.0886 |
| Mean predicted rate | 11.69% against 11.72% observed |
| Calibration error | 0.11 percentage points |

The overall figure is inflated by how easily the groups separate from one
another: group 3 runs at 2.5% and group 2a at 29%. Hold the group constant —
which is what a clinician actually knows — and discrimination falls to
0.678-0.705. That range is the honest description.

For comparison, the WHO C-Model, built on more than ten million deliveries
across 43 countries, reports AUC 0.832-0.844 and is used to compare
facilities rather than to predict for individuals. The ceiling here is a
property of the problem, not a shortfall in this implementation.

## Approved and prohibited uses

APPROVED — aggregate estimation. Summing predicted probabilities across a
population, and comparing facilities or groups. Calibration is
0.11 pp, which supports this.

APPROVED — explanation. SHAP analysis of what the recorded clinical picture
does and does not account for.

NOT APPROVED — individual clinical decisions. At the best available
operating point, precision is 0.41: roughly six in ten flagged women do not
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
