# CaesarSight — Robson Classification & C-Section Audit System

An open, reproducible pipeline that classifies births with the **WHO Ten-Group Robson Classification**, benchmarks caesarean section (CS) rates against a large US reference population, and uses that benchmark to estimate how much of Egypt's caesarean rate is not explained by case mix.

> **Status:** capstone project (Digilians Digital Pioneers Initiative). Research and educational use only. **Not a clinical decision tool.**

---

## Why this project exists

Egypt performs well over a million caesareans a year, and national surveys and Ministry of Health releases put the rate far above the WHO-era benchmarks. In August 2025 the Ministry made monthly Robson-classification reporting mandatory for private facilities. Robson is the recommended way to audit caesarean rates, but a hospital needs a yardstick to compare itself against.

This project builds that yardstick from scratch on real data and tests it against Egyptian evidence.

## What it does

1. **Rule-based Robson classifier.** A deterministic implementation of the WHO flowchart (groups 1–10, with subgroups 2a/2b, 4a/4b, 5.1/5.2) for the CDC Natality Public Use File. It tracks `Classification_status` and missing variables per record. It is verified against a 21-case hand-written fixture (one case per group, expected group declared in advance), which caught two real bugs.
2. **US reference table.** CDC 2023 and 2024 births, validated against published national statistics.
3. **Descriptive and practice-pattern analysis** inside Robson group 1, with direct standardisation by payer and time of day.
4. **Predictive model** (XGBoost + SHAP) trained on the groups where the outcome is genuinely uncertain at admission. It is explanatory only, and is not used in the Egypt calculation.
5. **Egypt comparison layer.** Indirect standardisation of two Egyptian hospital Robson studies against the US reference, plus nationally representative DHS survey analysis.
6. **Power BI dashboard and an n8n demo** for a monthly hospital Robson report.

> Robson is a deterministic WHO rule set, not machine learning. The classifier and the predictive model are two separate layers on purpose.

---

## Key results

### The classifier reproduces published US statistics

| Check | This project | Published |
|---|---|---|
| Overall CS rate, US 2023 | 32.35% | 32.3% |
| NTSV (groups 1 + 2a + 2b) | 26.56% | ~25.6–26.6% |
| Induction, groups 1–4 | 40.58% | ~40% |
| Groups 2b and 4b CS rate | 100% | 100% by definition |

- 2023: 3,605,081 records read, 3,585,385 classified (0.55% unclassifiable).
- 2024 temporal validation (3,638,436 births): overall CS 32.37%, NTSV 26.56%. No field positions moved between years.

### Practice patterns in the US (within Robson group 1)

The crude payer gradient (private 14.25% vs Medicaid 11.97% vs self-pay 8.30%) **disappears** after restricting to hospital births and standardising for age, BMI, comorbidity, smoking and prenatal-care timing (13.26% / 13.90% / 13.56%). The self-pay anomaly was explained by home births. In the US, payer does not drive caesarean decisions in this population.

### Predictive model (CDC 2023 test set, n = 486,820)

| Metric | Value |
|---|---|
| ROC-AUC | 0.7991 |
| AUPRC | 0.3542 (baseline 0.1172) |
| Brier score | 0.0886 |
| Mean predicted vs observed | 11.69% vs 11.72% |
| Within-group ROC-AUC | 0.678–0.705 |
| 2024 transfer (no refit) | ROC-AUC 0.7975 |

Important design decisions, all documented in [`MODEL_CARD.md`](dashboard_data%20%2B%20MODEL_CARD.md/MODEL_CARD.md):

- **Class weighting was removed.** It left AUC unchanged but inflated mean predicted probability from 11.7% to 39.4%. Imbalance is handled through metric choice and thresholds instead.
- **The headline AUC is inflated by group separation.** The within-group range is the honest figure.
- **The model is never applied to Egyptian case mix.** Egypt's expected rate is computed from observed group-specific rates.
- **No high-recall threshold is recommended.** Precision tops out near 0.41, so individual clinical use is not approved.
- The target variable is *caesarean performed*, not *caesarean needed*.

### Egypt: how much excess is there?

Egyptian hospital Robson tables, standardised against the US reference on eight categories (the two decision-driven splits in groups 1/2 and 3/4 are collapsed so Egyptian practice is not treated as case mix):

| Source | n | Observed CS | Expected (US rates) | Gap | Gap if group 5 were at reference size |
|---|---|---|---|---|---|
| Benha University Hospital, 2018 | 850 | 54.71% | 50.25% | +4.46 pp | +22.01 pp |
| Qena University Hospitals, 2022–23 | 2,186 | 79.69% | 59.04% | +20.65 pp | +41.57 pp |

Both are single tertiary centres, so they are illustrations, not national estimates. Across every sensitivity scenario tested, the gap never closes. Group 5 (women with a previous caesarean) is the dominant assumption.

### Egypt: where the excess comes from

- Group 1 CS rates in both Egyptian hospitals are **lower** than the US reference. The excess is concentrated in groups defined by a decision not to labour.
- At Benha, 164 of 308 group 5 women arrived in spontaneous labour. None had a trial of labour.
- **DHS 2014** (21,762 women, 15,846 births): weighted CS rate 51.78%, reproducing the published figure. In facility births the private–public gap is +22.98 pp crude and **+21.07 pp after standardisation** (age, parity, wealth, residence, region). Case mix explains only about 8% of it. The identical procedure on US data erased the payer gap.
- **Correction to published statistics:** the 43.8% figure from the 2015 Egypt Health Issues Survey is the share of *mothers who ever had a caesarean*, not a caesarean rate per birth (lifetime per-birth rate in that survey: 26.25%). It should not share a trend line with EDHS 2014's 51.8%.
- 82.5% of caesarean decisions in the national survey were taken before labour began.

---

## Repository layout

```
.
├── Final Project/
│   ├── Scripts/                  # the full pipeline as runnable .py files
│   ├── Notebooks/                # 00_MASTER + five stage notebooks
│   ├── 9 Robson files + export_csvs.py/   # result tables (01–09 .csv)
│   ├── dashboard_data + MODEL_CARD.md/    # dashboard CSVs and the model card
│   ├── births/                   # Power BI export and .pbix dashboard
│   ├── N8N/                      # monthly hospital report demo
│   ├── Capstone_Final_Report_REVIEWED (1).docx
│   ├── CaesarSight_Final_Presentation .pptx
│   └── Capstone_Poster_Final (1) (2).pptx
├── .gitignore
└── .gitattributes
```

### Main scripts

| Script | Purpose |
|---|---|
| `robson_classifier.py`, `test_robson.py` | WHO rule tree and its 21-case verification |
| `cdc_robson_pipeline.py` | Fixed-width loader, derived flags, report table |
| `robson_analysis.py` | External validation, group 9 sensitivity, TOLAC/VBAC, safety by group |
| `stage3_confounding.py`, `stage4_setting_and_wider_adjustment.py` | Direct standardisation within group 1 |
| `step4_5_clean.py` → `step16_package.py` | Cleaning, encoding, split, training, calibration, thresholds, evaluation, SHAP, packaging |
| `temporal_validation_2024.py` | Out-of-time check on the 2024 file |
| `egypt_standardisation.py`, `egypt_gap_v2.py` | Indirect standardisation (v2 supersedes earlier versions) |
| `dhs_egypt.py`, `ehis_2015.py`, `egypt_governorates.py` | DHS / EHIS survey analysis and governorate data |
| `egypt_cost.py` | Cost differential model |
| `kappa_audit.py` | Agreement audit for classification logic |
| `export_for_powerbi_v2.py`, `export_csvs.py` | Dashboard and result exports |

---

## Reproducing the results

### 1. Environment

Python 3.10+ with, at least: `pandas`, `numpy`, `pyarrow`, `scikit-learn`, `xgboost`, `shap`, `matplotlib`, `jupyter`.

### 2. Get the data

The raw data is **not** included in this repository.

- **CDC Natality Public Use File (2023 and 2024):** free, no registration. Download the fixed-width `.txt` file and user guide from the [NCHS Vital Statistics Online data page](https://www.cdc.gov/nchs/data_access/vitalstatsonline.htm). The 2023 file is about 4.9 GB. Field positions are verified against the User Guides and are identical between 2023 and 2024.
- **DHS Egypt surveys (EDHS 2014, Egypt Health Issues Survey 2015):** DHS terms prohibit redistributing microdata. Register for access at [dhsprogram.com](https://dhsprogram.com) and download the Individual Recode files in Stata format yourself.
- **Egyptian Robson tables:** taken from the published Benha (Jadoon et al., *Women and Birth*, 2020) and Qena (*SVU-International Journal of Medical Sciences*, 2025, Table 3) papers. The aggregated tables are in `9 Robson files + export_csvs.py/`.

### 3. Run

Edit the file paths at the top of each script, then run the notebooks in order starting with `Notebooks/00_MASTER.ipynb`, or run the scripts from `Scripts/` in the order shown in the table above.

---

## Limitations

- **Egyptian case mix is hospital-level.** Both Egyptian Robson sources are single tertiary or university centres with referral bias. No nationally representative Egyptian Robson distribution was found. The results are reported as a range, and the direction of the unmeasured biases is stated in the final report.
- **No governorate-level excess estimate.** No Egyptian governorate Robson distribution exists, so the governorate dashboard page is descriptive only.
- **DHS lacks clinical detail** (indication, presentation, gestational age, comorbidity), so residual confounding by clinical risk cannot be excluded. The 2014 data are older than current rates.
- **Group 9** in the CDC file includes face, brow and compound presentations. It is reported as is, with a sensitivity analysis (effect on the overall rate: 0.12 pp).
- **Time-of-day finding is hypothesis-generating only.** CDC records time of birth, not time of decision.
- **Model and data scope.** The model is trained on US data and is explanatory only.
- **Cost figures** are 2014 differentials from the Population Council; no recent Egyptian costing study was found. 2026 figures are administrative, not economic costs.

## Responsible use

This is a research and audit tool. The model must not be used to deny or discourage a trial of labour, and a predicted probability must never be used as a barrier to vaginal birth after caesarean. Race and ethnicity are deliberately excluded from all predictive features. See `MODEL_CARD.md` for approved and prohibited uses.

## Key references

- WHO. *Robson Classification: Implementation Manual*. Geneva, 2017.
- NCHS. Natality Public Use Files and User Guides, 2023 and 2024.
- Jadoon B, et al. Robson classification at Benha University Hospital. *Women and Birth*, 2020.
- Qena University Hospitals Robson study. *SVU-International Journal of Medical Sciences*, 2025.
- Abdel-Tawab N, et al. *Caesarean section deliveries in Egypt: trends, practices, perceptions, and cost*. Population Council, 2018.
- Egypt Demographic and Health Survey 2014; Egypt Health Issues Survey 2015; Egypt Family Health Survey 2021.

## Author

**Mahmoud Alqadi**, physician (obstetrics and gynaecology) and healthcare data analyst.
Built as the capstone for the Digilians Digital Pioneers Initiative (data analytics).

## License

Code is released under the MIT License. Add a `LICENSE` file to the repository root. Published figures and third-party data remain subject to their original sources' terms.
