"""
Unit test for the Robson classifier.

This fixture is NOT training data and NOT a statistical simulation. It is a
hand-written set of cases, one per Robson group, each with the group it must
land in written down in advance. Its only purpose is to prove the if/else
rules match the WHO flowchart. This is the verification the supervisor asked
for -- and it is exactly what should go in the appendix so a reviewer can
check the implementation without access to any data.
"""
import pandas as pd
from robson_classifier import classify, report_table, sanity_checks

# columns: prior_live, prior_dead, prev_cs, prev_cs_n, induction,
#          trial_labor, presentation, route, plurality, gest_age
CASES = [
    # (expected_group, description, values...)
    ("1",   "nullip, single cephalic, term, spontaneous labour, vaginal",
     0, 0, "N", 0, "N", "U", "1", "1", 1, 39),
    ("1",   "nullip, spontaneous labour, ended in caesarean (trial of labour = Y)",
     0, 0, "N", 0, "N", "Y", "1", "4", 1, 40),
    ("2a",  "nullip, term cephalic, INDUCED",
     0, 0, "N", 0, "Y", "U", "1", "1", 1, 39),
    ("2b",  "nullip, term cephalic, PRE-LABOUR caesarean (no induction, no trial)",
     0, 0, "N", 0, "N", "N", "1", "4", 1, 39),
    ("3",   "multip, no previous CS, term cephalic, spontaneous",
     2, 0, "N", 0, "N", "U", "1", "1", 1, 40),
    ("3",   "multip via previous STILLBIRTH only -- parity must count it",
     0, 1, "N", 0, "N", "U", "1", "1", 1, 39),
    ("4a",  "multip, no previous CS, term cephalic, induced",
     1, 0, "N", 0, "Y", "U", "1", "1", 1, 38),
    ("4b",  "multip, no previous CS, term cephalic, pre-labour caesarean",
     3, 0, "N", 0, "N", "N", "1", "4", 1, 39),
    ("5.1", "ONE previous caesarean, term cephalic singleton",
     1, 0, "Y", 1, "N", "N", "1", "4", 1, 39),
    ("5.2", "TWO previous caesareans, term cephalic singleton",
     2, 0, "Y", 2, "N", "N", "1", "4", 1, 38),
    ("5.1", "previous caesarean, successful VBAC (spontaneous, vaginal)",
     1, 0, "Y", 1, "N", "U", "1", "1", 1, 40),
    ("6",   "nulliparous breech",
     0, 0, "N", 0, "N", "N", "2", "4", 1, 39),
    ("7",   "multiparous breech",
     2, 0, "N", 0, "N", "N", "2", "4", 1, 39),
    ("7",   "multiparous breech WITH previous caesarean -- must stay in 7, not 5",
     1, 0, "Y", 1, "N", "N", "2", "4", 1, 39),
    ("8",   "twins, nulliparous",
     0, 0, "N", 0, "N", "N", "1", "4", 2, 36),
    ("8",   "twins WITH previous caesarean -- must stay in 8, not 5",
     1, 0, "Y", 1, "N", "N", "1", "4", 2, 37),
    ("9",   "transverse / oblique lie (CDC 'Other')",
     1, 0, "N", 0, "N", "N", "3", "4", 1, 39),
    ("10",  "single cephalic PRETERM (34 weeks) -- outranks parity and onset",
     0, 0, "N", 0, "Y", "U", "1", "1", 1, 34),
    ("10",  "single cephalic preterm WITH previous caesarean -- must stay in 10",
     1, 0, "Y", 1, "N", "N", "1", "4", 1, 33),
    ("Unclassifiable", "gestational age missing",
     0, 0, "N", 0, "N", "U", "1", "1", 1, 99),
    ("Unclassifiable", "presentation unknown",
     0, 0, "N", 0, "N", "U", "9", "1", 1, 39),
]

COLNAMES = ["PRIORLIVE", "PRIORDEAD", "RF_CESAR", "RF_CESARN", "LD_INDL",
            "ME_TRIAL", "ME_PRES", "ME_ROUT", "DPLURAL", "OEGEST_COMB"]

expected = [c[0] for c in CASES]
desc = [c[1] for c in CASES]
df = pd.DataFrame([c[2:] for c in CASES], columns=COLNAMES)

out = classify(df)
out["is_cesarean"] = (out["ME_ROUT"] == "4").astype(int)

print("=" * 78)
print("ROBSON CLASSIFIER -- RULE VERIFICATION")
print("=" * 78)
passed = 0
for i, (exp, d) in enumerate(zip(expected, desc)):
    got = out.loc[i, "Robson_group"]
    ok = got == exp
    passed += ok
    print(f"{'PASS' if ok else 'FAIL':4}  expected {exp:>14}  got {got:>14}   {d}")

print("-" * 78)
print(f"{passed}/{len(CASES)} cases passed")
print()
print("=" * 78)
print("STANDARD ROBSON REPORT TABLE (on the test fixture -- shape check only)")
print("=" * 78)
print(report_table(out).to_string(index=False))
print()
print("SANITY CHECKS")
for issue in sanity_checks(out):
    print(" -", issue)
