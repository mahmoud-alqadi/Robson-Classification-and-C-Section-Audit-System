"""
=============================================================================
TEMPORAL VALIDATION — 2024
=============================================================================
    python temporal_validation_2024.py Nat2024PublicUS.txt

WHY THIS MATTERS MORE THAN ANYTHING ELSE ADDED SO FAR

Every figure in the project comes from one year. The obvious question — does
this hold next year, or was it fitted to the peculiarities of 2023? — has
not been answered. Most student projects never answer it. It takes one run,
because the model and the classifier already exist and nothing needs
retraining.

Two separate things are being tested, and they can fail independently:

  1. THE CLASSIFIER. Run the same rules on 2024 and compare group sizes and
     group caesarean rates. This is a stability check on the data and on the
     derivation, especially onset of labour. Real shifts are expected and
     interesting; large ones would mean a coding change we missed.

  2. THE MODEL. The 2023-trained model is applied to 2024 without any
     refitting. Discrimination usually survives a year. CALIBRATION is the
     fragile part, and calibration is what this project depends on: the
     Egypt calculation adds up probabilities. A model whose probabilities
     drift a year later is a model whose expected rates cannot be trusted.

FIELD POSITIONS
All 48 positions used here were checked against the 2024 User Guide before
this was written. None moved. The same colspecs are reused unchanged, which
is why this can run at all.

HONESTY NOTE
This is a single run and its result is reported whatever it says. If
calibration degrades on 2024, that goes in the limitations as a finding
about how long a fitted model stays usable — which is worth knowing.
=============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys, json, gc

sys.path.insert(0, str(Path(__file__).parent))
from robson_classifier import classify, report_table
from cdc_robson_pipeline import (COLSPECS as BASE_COLSPECS,
                                 COLNAMES as BASE_COLNAMES,
                                 add_is_cesarean, add_practice_vars,
                                 add_near_miss_flag)

# The six columns added in stage 4 live in stage4_setting_and_wider_adjustment.py,
# not in the original pipeline. The 2023 run built them in a second pass over the
# raw file; here they are read in the same pass. Positions verified against the
# 2024 User Guide — identical to 2023.
EXTRA_COLSPECS = [
    (31, 32),     # BFACIL      pos 32
    (119, 120),   # DMAR        pos 120
    (123, 124),   # MEDUC       pos 124
    (226, 227),   # PRECARE5    pos 227
    (268, 269),   # CIG_REC     pos 269
    (317, 318),   # RF_PPTERM   pos 318
]
EXTRA_COLNAMES = ["BFACIL", "DMAR", "MEDUC", "PRECARE5", "CIG_REC", "RF_PPTERM"]

COLSPECS = BASE_COLSPECS + [c for c, n in zip(EXTRA_COLSPECS, EXTRA_COLNAMES)
                            if n not in BASE_COLNAMES]
COLNAMES = BASE_COLNAMES + [n for n in EXTRA_COLNAMES if n not in BASE_COLNAMES]

RAW = sys.argv[1] if len(sys.argv) > 1 else "Nat2024PublicUS.txt"
CACHE = "natality_2024_classified.parquet"
TARGET = "is_cesarean"
NARROW = ["1", "2a", "3", "4a"]

# =============================================================================
if Path(CACHE).exists():
    print(f"Loading cached {CACHE} ...")
    d24 = pd.read_parquet(CACHE)
else:
    if not Path(RAW).exists():
        raise SystemExit(f"{RAW} not found.")
    print(f"Reading {RAW} ...")
    print("  Field positions were verified identical to 2023 against the")
    print("  2024 User Guide — all 48, none moved.\n")
    frames = []
    for i, ch in enumerate(pd.read_fwf(RAW, colspecs=COLSPECS, names=COLNAMES,
                                       dtype=str, chunksize=500_000)):
        frames.append(ch)
        if (i + 1) % 3 == 0:
            print(f"  {sum(len(f) for f in frames):,} rows")
    d24 = pd.concat(frames, ignore_index=True)
    del frames; gc.collect()
    print(f"  {len(d24):,} births loaded\n")

    d24 = add_is_cesarean(d24)
    d24 = add_practice_vars(d24)
    d24 = add_near_miss_flag(d24)
    print("Classifying ...")
    d24 = classify(d24, subgroups=True)
    d24.to_parquet(CACHE, index=False)
    print(f"Cached to {CACHE}\n")

c24 = d24[d24["Classification_status"] == "Classified"]
print("=" * 76)
print("PART 1 — DOES THE CLASSIFIER REPRODUCE ITSELF ON A NEW YEAR?")
print("=" * 76)
print(f"  2024: {len(d24):,} births, {len(c24):,} classified "
      f"({100*(len(d24)-len(c24))/len(d24):.2f}% unclassifiable)")
print(f"  2023: 3,605,081 births, 3,585,385 classified (0.55% unclassifiable)")
print(f"  Overall caesarean rate 2024: {100*c24[TARGET].mean():.2f}%   (2023: 32.35%)\n")

R23 = {"1":(16.43,13.00),"2a":(15.01,29.30),"2b":(2.48,100.00),"3":(21.98,2.52),
       "4a":(14.44,5.89),"4b":(2.23,100.00),"5.1":(8.77,78.27),"5.2":(3.75,95.37),
       "6":(1.72,96.24),"7":(1.73,93.78),"8":(3.16,75.07),"9":(0.83,46.52),
       "10":(7.46,43.17)}

t = (c24.groupby(["Robson_group_sort_order","Robson_group"], observed=True)
        .agg(n=(TARGET,"size"), cs=(TARGET,"mean")).reset_index()
        .sort_values("Robson_group_sort_order"))
t["size_24"] = (100*t["n"]/len(c24)).round(2)
t["rate_24"] = (100*t["cs"]).round(2)

print(f"{'group':>6} {'size 23':>9} {'size 24':>9} {'Δ':>7}   "
      f"{'rate 23':>9} {'rate 24':>9} {'Δ':>7}")
print("-"*66)
rows=[]
for _, r in t.iterrows():
    g = r["Robson_group"]
    if g not in R23: continue
    s23, c23_ = R23[g]
    ds, dc = r["size_24"]-s23, r["rate_24"]-c23_
    flag = "  <-- shift" if abs(dc) > 3 or abs(ds) > 2 else ""
    print(f"{g:>6} {s23:>8.2f}% {r['size_24']:>8.2f}% {ds:>+7.2f}   "
          f"{c23_:>8.2f}% {r['rate_24']:>8.2f}% {dc:>+7.2f}{flag}")
    rows.append({"group":g,"size_2023":s23,"size_2024":float(r["size_24"]),
                 "rate_2023":c23_,"rate_2024":float(r["rate_24"]),
                 "size_delta":round(float(ds),2),"rate_delta":round(float(dc),2)})

ntsv = c24[c24["Robson_group"].isin(["1","2a","2b"])]
print(f"\n  NTSV 2024: {100*ntsv[TARGET].mean():.2f}%   (2023: 26.56%)")
print("""
  Groups 2b and 4b must stay at exactly 100% — they are defined that way.
  Anything else there would mean the onset-of-labour derivation behaves
  differently on this year's file.
""")

# =============================================================================
print("=" * 76)
print("PART 2 — DOES THE 2023 MODEL STILL WORK IN 2024?")
print("=" * 76)
print("  No refitting. The model trained on 2023 is applied unchanged.\n")

try:
    import xgboost as xgb
    from sklearn.metrics import (roc_auc_score, average_precision_score,
                                 brier_score_loss)
except ImportError as e:
    raise SystemExit(f"Missing package: {e}")

if not Path("xgb_final.json").exists():
    raise SystemExit("xgb_final.json not found — run step11b_calibration.py first.")

FEATURES = pd.read_csv("model_feature_order.csv")["feature"].tolist()

ENGINEERED = {"ILLB_first_birth", "ME_PRES_cephalic", "ME_PRES_breech", "ME_PRES_other"}
need_raw = [f for f in FEATURES if f not in ENGINEERED]
absent = [f for f in need_raw if f not in c24.columns]
if absent:
    raise SystemExit(
        f"These model features were not read from the raw file: {absent}\n"
        "Delete natality_2024_classified.parquet and re-run, so the cache is "
        "rebuilt with the full column set.")

# rebuild the engineered features exactly as steps 5, 6 did
n24 = c24[c24["Robson_group"].isin(NARROW)].copy()
SENT = {"M_Ht_In":[99],"PWgt_R":[999],"WTGAIN":[99],"BMI":[99.9],"MEDUC":[9],
        "DMAR":[9],"PRIORLIVE":[99],"PRIORDEAD":[99],"RF_CESARN":[99],
        "ILLB_R":[888,999],"OEGest_Comb":[99],"DPLURAL":[9]}
YN = ["CIG_REC","RF_PPTERM","RF_PDIAB","RF_GDIAB","RF_PHYPE","RF_GHYPE",
      "RF_EHYPE","RF_INFTR","RF_ARTEC"]

for c in YN:
    s = n24[c].astype(str).str.strip().str.upper()
    v = pd.Series(np.nan, index=n24.index)
    v[s=="Y"]=1.0; v[s=="N"]=0.0
    n24[c]=v
for c, codes in SENT.items():
    x = pd.to_numeric(n24[c], errors="coerce")
    n24[c] = x.where(~x.isin(codes))
pc = n24["PRECARE5"].astype(str).str.strip()
n24["PRECARE5"] = pd.to_numeric(pc.where(pc!="5"), errors="coerce")
# conditional recode, same rule as step 6
n24.loc[n24["RF_INFTR"]==0.0, "RF_ARTEC"] = 0.0
n24["ILLB_first_birth"] = (pd.to_numeric(n24["PRIORLIVE"],errors="coerce")==0).astype(float)
pres = n24["ME_PRES"].astype(str).str.strip()
for code,name in {"1":"cephalic","2":"breech","3":"other"}.items():
    n24[f"ME_PRES_{name}"] = (pres==code).astype(float)
    n24.loc[pres=="9", f"ME_PRES_{name}"] = np.nan

X = n24[FEATURES].apply(pd.to_numeric, errors="coerce").astype("float32")
y = n24[TARGET].to_numpy().astype(int)

bst = xgb.Booster(); bst.load_model("xgb_final.json")
p = bst.predict(xgb.DMatrix(X))

T23 = {"roc_auc":0.7991,"auprc":0.3542,"brier":0.0886,"mean_pred":0.1169,
       "observed":0.1172,"cal_mae":0.11}
auc, ap, br = roc_auc_score(y,p), average_precision_score(y,p), brier_score_loss(y,p)

print(f"  n = {len(y):,} births in groups 1, 2a, 3 and 4a\n")
print(f"{'metric':<24} {'2023 test':>11} {'2024':>11} {'change':>10}")
print("-"*60)
for k,v23,v24 in [("ROC-AUC",T23['roc_auc'],auc),("AUPRC",T23['auprc'],ap),
                  ("Brier",T23['brier'],br),
                  ("mean predicted",T23['mean_pred'],p.mean()),
                  ("observed rate",T23['observed'],y.mean())]:
    print(f"{k:<24} {v23:>11.4f} {v24:>11.4f} {v24-v23:>+10.4f}")

q = pd.qcut(p, 10, labels=False, duplicates="drop")
cal = pd.DataFrame({"b":q,"pred":p,"obs":y}).groupby("b").agg(
    n=("obs","size"), predicted=("pred","mean"), observed=("obs","mean"))
cal["predicted_%"]=(100*cal.predicted).round(2)
cal["observed_%"]=(100*cal.observed).round(2)
cal["gap_pp"]=(cal["predicted_%"]-cal["observed_%"]).round(2)
mae = cal["gap_pp"].abs().mean()

print(f"\n  Calibration on 2024, by risk decile")
print(cal[["n","predicted_%","observed_%","gap_pp"]].to_string())
print(f"\n  mean absolute calibration error: {mae:.2f} pp   (2023 test: {T23['cal_mae']:.2f} pp)")

print("\n" + "=" * 76)
print("VERDICT")
print("=" * 76)
disc_ok = abs(auc - T23["roc_auc"]) < 0.02
cal_ok  = mae < 1.0
print(f"""
  Discrimination {'HOLDS' if disc_ok else 'DEGRADED'} — ROC-AUC moved {auc-T23['roc_auc']:+.4f}.
  Calibration    {'HOLDS' if cal_ok else 'DEGRADED'} — mean error {mae:.2f} pp against {T23['cal_mae']:.2f} pp.
""")
if disc_ok and cal_ok:
    print("""  The model transfers to a year it was never shown, without refitting.
  That is the answer to the first question any reviewer asks, and most
  student projects cannot answer it. Calibration holding matters more than
  discrimination here, because the Egypt estimate is built from summed
  probabilities rather than from rankings.
""")
else:
    print("""  Report this as it stands. A model that discriminates but drifts in
  calibration is still usable for ranking and NOT usable for estimating
  expected rates — which would change what this model may be used for,
  and belongs in the model card rather than being quietly smoothed over.
""")

json.dump({"year":2024,"n_classified":int(len(c24)),
           "overall_cs_2024":float(c24[TARGET].mean()),
           "ntsv_2024":float(ntsv[TARGET].mean()),
           "groups":rows,
           "model":{"roc_auc":float(auc),"auprc":float(ap),"brier":float(br),
                    "mean_predicted":float(p.mean()),"observed":float(y.mean()),
                    "calibration_mae_pp":float(mae)},
           "comparison_2023":T23},
          open("temporal_validation_2024.json","w"), indent=2)
pd.DataFrame(rows).to_csv("robson_2023_vs_2024.csv", index=False)
print("Saved temporal_validation_2024.json and robson_2023_vs_2024.csv")
