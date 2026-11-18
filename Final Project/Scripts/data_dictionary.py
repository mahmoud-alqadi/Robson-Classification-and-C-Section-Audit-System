"""
=============================================================================
DATA DICTIONARY  +  MODEL FEATURE DECISIONS
=============================================================================
    python data_dictionary.py

Writes:
    data_dictionary.csv   every column, in English and Arabic, with its type,
                          whether it enters the predictive model, and why not
                          where it does not
    display_names.csv     raw name -> display name, for Power BI

WHY THE RAW NAMES STAY IN THE DATA

Each raw name corresponds to a documented field at a known character
position in the CDC User Guide. RF_PDIAB can be checked against the guide;
pre_pregnancy_diabetes cannot. The obscurity is the audit trail. Renaming in
place would also mean editing every script that already references them, and
would break against any other data year.

So: raw names in the data, clear names at the point of display. This file
is the bridge, and it doubles as the written record of which columns enter
the model and which do not.

WHY THE MODEL TAKES 22 OF 59 COLUMNS

The test is simple: was this fact known BEFORE the delivery decision was
taken? Birth weight, Apgar and NICU admission are known only afterwards. The
route of delivery IS the answer. Augmentation of labour happens after
admission. Feeding any of them to the model produces a high score built on
information a clinician standing in the room would not have had.

Practice variables (payer, attendant, birth place) are held out on different
grounds: they are not clinically wrong to know, but including them means the
model learns system behaviour alongside clinical risk, and the output stops
being "the rate expected for this clinical picture". They are analysed
separately, which is where they belong.

Robson_group is held out because it is DERIVED from six columns already in
the model. Including it would feed the same information twice and confuse
SHAP between the group and its own components. Its role is to split the
evaluation, not to predict.
=============================================================================
"""

import pandas as pd

# raw, english, arabic, type, in_model, note
D = [
# ---- maternal characteristics -------------------------------------------
("MAGER","Mother's age (years)","سن الأم","continuous",1,""),
("M_Ht_In","Mother's height (inches)","طول الأم (بوصة)","continuous",1,""),
("PWgt_R","Pre-pregnancy weight (lb)","وزن ما قبل الحمل","continuous",1,""),
("WTGAIN","Weight gain in pregnancy (lb)","الزيادة في الوزن","continuous",1,""),
("BMI","Body mass index","مؤشر كتلة الجسم","continuous",1,""),
("MEDUC","Mother's education","تعليم الأم","ordinal",1,""),
("DMAR","Marital status","الحالة الاجتماعية","binary",1,""),
("CIG_REC","Smoked during pregnancy","التدخين أثناء الحمل","binary",1,""),
("PRECARE5","Month antenatal care began","بداية متابعة الحمل","ordinal",1,""),
# ---- obstetric history ---------------------------------------------------
("PRIORLIVE","Previous live births","ولادات حية سابقة","continuous",1,""),
("PRIORDEAD","Previous stillbirths","مواليد موتى سابقون","continuous",1,""),
("PRIORTERM","Previous terminations","إنهاءات حمل سابقة","continuous",0,
 "Excluded from parity by WHO definition; kept in the data for reference"),
("RF_CESAR","Previous caesarean","قيصرية سابقة","binary",0,
 "Superseded by RF_CESARN, which carries the same information plus the count"),
("RF_CESARN","Number of previous caesareans","عدد القيصريات السابقة","continuous",1,""),
("ILLB_R","Interval since last live birth","الفترة منذ آخر ولادة","continuous",1,""),
("RF_PPTERM","Previous preterm birth","ولادة مبكرة سابقة","binary",1,""),
# ---- comorbidity ---------------------------------------------------------
("RF_PDIAB","Pre-pregnancy diabetes","سكري سابق للحمل","binary",1,""),
("RF_GDIAB","Gestational diabetes","سكري الحمل","binary",1,""),
("RF_PHYPE","Chronic hypertension","ضغط مزمن","binary",1,""),
("RF_GHYPE","Gestational hypertension","ضغط الحمل","binary",1,""),
("RF_EHYPE","Eclampsia","تشنج الحمل","binary",1,""),
("RF_INFTR","Infertility treatment","علاج عقم","binary",1,""),
("RF_FEDRG","Fertility-enhancing drugs","منشطات الإباضة","binary",0,
 "Subsumed by RF_INFTR and RF_ARTEC; keeping all three duplicates the signal"),
("RF_ARTEC","Assisted reproductive technology","حقن مجهري / أطفال أنابيب","binary",1,""),
# ---- current pregnancy ---------------------------------------------------
("ME_PRES","Fetal presentation","وضعية الجنين","nominal",1,""),
("DPLURAL","Number of fetuses","عدد الأجنة","ordinal",1,""),
("OEGest_Comb","Gestational age (weeks)","عمر الحمل بالأسابيع","continuous",1,""),
("OEGest_R3","Preterm / term recode","مبتسر أم تام","ordinal",0,
 "Derived from OEGest_Comb, which is already in the model"),
# ---- labour: excluded, occurs after admission ---------------------------
("LD_INDL","Labour induced","تحريض المخاض","binary",0,
 "Used for Robson classification. Not a pre-decision fact: induction IS an intervention"),
("LD_AUGM","Labour augmented","تعزيز المخاض","binary",0,
 "Occurs after admission — temporal leakage"),
("LD_CHOR","Chorioamnionitis","التهاب المشيمة والسلى","binary",0,
 "Diagnosed during labour, after the admission decision point"),
("ME_TRIAL","Trial of labour attempted","محاولة مخاض","binary",0,
 "Recorded for caesareans only, and downstream of the decision"),
# ---- the outcome and its aliases ----------------------------------------
("is_cesarean","Caesarean delivery (target)","قيصرية (المتغير الهدف)","binary",0,
 "TARGET VARIABLE"),
("ME_ROUT","Route of delivery","طريقة الولادة","nominal",0,
 "This IS the answer — direct leakage"),
("DMETH_REC","Delivery method (simple)","طريقة الولادة مبسطة","binary",0,
 "This IS the answer — direct leakage"),
("RDMETH_REC","Delivery method (expanded, 8 categories)","طريقة الولادة موسعة","nominal",0,
 "This IS the answer. Used for the descriptive VBAC analysis only"),
("delivery_known","Delivery method recorded","طريقة الولادة مسجلة","binary",0,
 "Data-quality flag, not a feature"),
# ---- newborn: all post-delivery -----------------------------------------
("DBWT","Birth weight (grams)","وزن المولود","continuous",0,
 "Known only after birth"),
("APGAR5","Five-minute Apgar","أبجار الخمس دقائق","ordinal",0,
 "Known only after birth"),
("AB_NICU","NICU admission","دخول الحضانة","binary",0,
 "Known only after birth"),
# ---- maternal morbidity: post-delivery ----------------------------------
("MM_MTR","Blood transfusion","نقل دم","binary",0,"Occurs after delivery"),
("MM_RUPT","Ruptured uterus","تمزق الرحم","binary",0,"Occurs after delivery"),
("MM_UHYST","Unplanned hysterectomy","استئصال رحم غير مخطط","binary",0,"Occurs after delivery"),
("MM_AICU","Maternal ICU admission","دخول الأم للعناية","binary",0,"Occurs after delivery"),
("near_miss_flag","Maternal near-miss (WHO 2012)","اقتراب الأم من الوفاة","binary",0,
 "Derived from the four above — an outcome, not a predictor"),
# ---- practice variables: held out on purpose ----------------------------
("PAY","Source of payment","جهة الدفع","nominal",0,
 "Practice variable — analysed separately, see the note at the top"),
("PAY_REC","Payment recode","جهة الدفع مختصرة","nominal",0,"Practice variable"),
("payer","Payer (readable)","جهة الدفع بالاسم","nominal",0,"Readable copy of PAY_REC"),
("ATTEND","Attendant at birth","مقدم الرعاية","nominal",0,
 "Practice variable, and NCHS documents it as unreliable for caesareans"),
("attendant","Attendant (readable)","مقدم الرعاية بالاسم","nominal",0,"Readable copy of ATTEND"),
("BFACIL","Birth place","مكان الولادة","nominal",0,"Practice variable"),
("DOB_WK","Day of week of birth","يوم الأسبوع","nominal",0,
 "Time of BIRTH, not known at the decision point"),
("DOB_TT","Time of birth","وقت الولادة","continuous",0,
 "Time of BIRTH, not known at the decision point"),
("weekend","Weekend birth","ولادة في عطلة","binary",0,"Derived from DOB_WK"),
("time_of_day","Time-of-day band","فترة اليوم","ordinal",0,"Derived from DOB_TT"),
# ---- classification and quality -----------------------------------------
("Robson_group","Robson group","مجموعة روبسون","nominal",0,
 "Derived from six columns already in the model. Used to split the evaluation"),
("Robson_group_sort_order","Robson sort order","ترتيب المجموعة","ordinal",0,
 "Display helper — prevents '10' sorting between '1' and '2'"),
("Classification_status","Classified / unclassifiable","حالة التصنيف","binary",0,
 "Data-quality measure"),
("Missing_variable","Which variable was missing","المتغير الناقص","nominal",0,
 "Data-quality measure"),
]

df = pd.DataFrame(D, columns=["raw_name","display_name_en","display_name_ar",
                              "type","in_model","reason_excluded"])
df["in_model"] = df["in_model"].map({1: "YES", 0: "no"})
df.to_csv("data_dictionary.csv", index=False, encoding="utf-8-sig")

df[["raw_name","display_name_en","display_name_ar"]].to_csv(
    "display_names.csv", index=False, encoding="utf-8-sig")

MODEL_FEATURES = df.loc[df.in_model == "YES", "raw_name"].tolist()
TARGET = "is_cesarean"

with open("model_features.py", "w", encoding="utf-8") as f:
    f.write('"""Feature list for the predictive model.\n\n'
            'Generated by data_dictionary.py. Do not edit by hand — change the\n'
            'dictionary and regenerate, so the reason for every decision stays\n'
            'attached to the decision.\n"""\n\n')
    f.write(f"TARGET = {TARGET!r}\n\nMODEL_FEATURES = [\n")
    for c in MODEL_FEATURES:
        row = df[df.raw_name == c].iloc[0]
        f.write(f"    {c!r},  # {row.display_name_en}  [{row.type}]\n")
    f.write("]\n")

print(f"{len(df)} columns documented")
print(f"{len(MODEL_FEATURES)} enter the model, {len(df)-len(MODEL_FEATURES)-1} held out, "
      f"1 is the target\n")
print("By type:")
print(df["type"].value_counts().to_string())
print("\nWritten: data_dictionary.csv, display_names.csv, model_features.py")
print("\nModel features:")
for c in MODEL_FEATURES:
    print("   ", c)
