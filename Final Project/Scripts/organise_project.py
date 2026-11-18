"""
=============================================================================
ORGANISE THE PROJECT
=============================================================================
    python organise_project.py            # show what it would do
    python organise_project.py --apply    # actually move the files

WHY THIS MATTERS MORE THAN IT LOOKS

The scripts are not workings-out to be cleared away once the answer is
found. They ARE the answer: every figure in this project exists because a
named script produced it from a named input, and a reader who cannot follow
that chain has to take the numbers on trust. A folder with twenty-nine
scripts in no order is functionally the same as no scripts at all.

Nothing is deleted. Files are moved into folders that say what they are for,
and a README is written that names the order things run in.

STRUCTURE

  01_classifier/     the Robson rule engine and its tests
  02_reference/      building and validating the CDC reference population
  03_model/          the predictive model, in the order the steps run
  04_egypt/          Egyptian sources, standardisation, cost, governorates
  05_outputs/        tables and figures meant to be read
  06_data/           large intermediate files (not for sharing)
  archive/           superseded or one-off scripts, kept for the record
=============================================================================
"""

import shutil, sys
from pathlib import Path

APPLY = "--apply" in sys.argv

README = """# Robson Classification and Machine-Learning Prediction of Caesarean Delivery

Every figure in this project is produced by a named script from a named
input. This file gives the order.

## Setup

    pip install -r requirements.txt

xgboost is pinned to 2.1.4 deliberately: version 3.4.0 crashes silently on
Python 3.14 — it imports and then dies on the first training call with no
error message.

## Data you need

Neither file is included; both are public.

  CDC Natality Public Use File, 2023 and 2024 (~4.9 GB each)
      https://www.cdc.gov/nchs/data_access/VitalStatsOnline.htm
  Egypt DHS 2014 and Health Issues Survey 2015 Individual Recodes
      https://dhsprogram.com/data/  (free, registration and a stated
      research purpose required)

Put them in 06_data/.

## Order of execution

### 1. Classifier — 01_classifier/

    python test_robson.py

21 hand-written cases, one or more per Robson group, each annotated in
advance with the group it must produce. Run this before anything else; it
needs no data and it catches rule errors that would otherwise propagate
silently into every result.

kappa_audit.py generates a blinded worksheet for manual verification against
the WHO flowchart. Run it when you are ready to do that check by hand.

### 2. Reference population — 02_reference/

    python cdc_robson_pipeline.py Nat2023PublicUS.txt
    python robson_analysis.py
    python stage3_confounding.py
    python stage4_setting_and_wider_adjustment.py Nat2023PublicUS.txt
    python temporal_validation_2024.py Nat2024PublicUS.txt

The second script contains the stop gate: derived figures are checked
against statistics NCHS publishes independently from the same birth
certificates. Overall caesarean rate 32.35% against a published 32.3%, and
the low-risk NTSV rate 26.56% against roughly 26%. NCHS defines its low-risk
measure as exactly Robson groups 1 and 2 combined, so reproducing it
confirms the classification rather than merely agreeing with it.

Do not proceed past that check if it fails.

### 3. Model — 03_model/

    python data_dictionary.py
    python step4_5_clean.py
    python step6_encode.py
    python step7_9_split.py
    python step10_train_v2.py
    python step11b_calibration.py
    python step12_threshold.py
    python step13_evaluate.py
    python step14_shap.py
    python step14b_shap_within.py
    python step16_package.py

Two steps here reverse decisions the proposal committed to, and both are
documented with the evidence that forced the change:

  step11b  class weighting removed. scale_pos_weight of 7.56 left ROC-AUC
           unchanged but inflated mean predicted probability from 11.7% to
           39.4%. Calibration error fell from 27.66 to 0.11 percentage
           points once it was dropped.
  step12   the F2 threshold was withdrawn. F2 flags 38.6% of all births as
           high risk with 76% false alarms, which in Egypt would manufacture
           prediction rather than catch need.

step13 opens the test set. It is opened once. If the result disappoints, it
is reported — re-tuning after looking would turn it into a second
validation set and every later number would be quietly optimistic.

### 4. Egypt — 04_egypt/

    python dhs_egypt.py
    python ehis_2015.py
    python egypt_standardisation.py
    python egypt_gap.py
    python egypt_governorates.py
    python egypt_cost.py

egypt_standardisation.py imports its standardisation function from
stage3_confounding.py rather than reimplementing it. The two countries are
analysed by the same function, not by similar methods, and that is the point
of the comparison.

egypt_gap.py is a tool: add a case mix to the EGYPT dictionary and every
table recomputes.

## Where results live

  05_outputs/       tables, figures, MODEL_CARD.md
  06_data/          large intermediates, not for sharing

MODEL_CARD.md states what the model may and may not be used for. The
prohibition that matters most: it must not be applied to an Egyptian case
mix to produce an expected rate. It over-predicts group 1 by 4.95 points and
under-predicts group 2a by 5.54; those errors cancel on a US case mix and
would not cancel on Egypt's, where group 5 is three times larger. The Egypt
expected rate is computed from observed group-specific rates instead.

## Reproducibility

Random seed 42 throughout. The feature list is generated from
data_dictionary.py so the reason for every inclusion and exclusion stays
attached to the decision rather than living in someone's memory.
"""


PLAN = {
    "01_classifier": [
        "robson_classifier.py",
        "test_robson.py",
        "kappa_audit.py",
    ],
    "02_reference": [
        "cdc_robson_pipeline.py",
        "robson_analysis.py",
        "stage3_confounding.py",
        "stage4_setting_and_wider_adjustment.py",
        "temporal_validation_2024.py",
    ],
    "03_model": [
        "data_dictionary.py",
        "model_features.py",
        "step4_5_clean.py",
        "step6_encode.py",
        "step7_9_split.py",
        "step10_train_v2.py",
        "step11b_calibration.py",
        "step12_threshold.py",
        "step13_evaluate.py",
        "step14_shap.py",
        "step14b_shap_within.py",
        "step16_package.py",
    ],
    "04_egypt": [
        "dhs_egypt.py",
        "ehis_2015.py",
        "egypt_standardisation.py",
        "egypt_gap.py",
        "egypt_governorates.py",
        "egypt_cost.py",
    ],
    "archive": [
        "step10_train.py",        # superseded by v2 after the memory crash
        "find_cs_var.py",         # one-off: locating the EHIS caesarean variable
        "make_csv.py",            # superseded by step16_package.py
        "export_csvs.py",
    ],
}

# outputs by pattern rather than by name, since these accumulate
OUTPUT_PATTERNS = ["*.csv", "*.json", "*.png", "MODEL_CARD.md"]
DATA_PATTERNS   = ["*.parquet", "*.txt", "*.DTA", "*.dat"]
KEEP_AT_ROOT    = {"organise_project.py", "README.md", "requirements.txt"}

root = Path(".")
moves, missing = [], []

for folder, files in PLAN.items():
    for f in files:
        src = root / f
        if src.exists():
            moves.append((src, root / folder / f))
        else:
            missing.append(f)

for pat in OUTPUT_PATTERNS:
    for src in root.glob(pat):
        if src.name in KEEP_AT_ROOT or not src.is_file():
            continue
        moves.append((src, root / "05_outputs" / src.name))

for pat in DATA_PATTERNS:
    for src in root.glob(pat):
        if not src.is_file():
            continue
        moves.append((src, root / "06_data" / src.name))

print(f"{'MOVING' if APPLY else 'WOULD MOVE'} {len(moves)} files\n")
by_dest = {}
for s, d in moves:
    by_dest.setdefault(d.parent.name, []).append(s.name)
for folder in sorted(by_dest):
    print(f"  {folder}/  ({len(by_dest[folder])} files)")
    for n in sorted(by_dest[folder])[:6]:
        print(f"      {n}")
    if len(by_dest[folder]) > 6:
        print(f"      ... and {len(by_dest[folder])-6} more")
    print()

if missing:
    print(f"Not found (fine if you never downloaded them): {', '.join(missing)}\n")

if APPLY:
    for s, d in moves:
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(s), str(d))
    print("Moved.\n")

    Path("requirements.txt").write_text(
        "pandas>=2.0\nnumpy>=1.24\npyarrow>=14.0\nxgboost==2.1.4\n"
        "scikit-learn>=1.3\nshap>=0.44\nmatplotlib>=3.7\npyreadstat>=1.2\n",
        encoding="utf-8")
    Path("README.md").write_text(README, encoding="utf-8")
    print("Wrote README.md and requirements.txt")
else:
    print("Nothing moved. Re-run with --apply to do it.\n")



