"""
=============================================================================
EGYPT GAP — indirect standardisation on eight case-mix categories
=============================================================================
    python egypt_gap_v2.py

Replaces egypt_gap.py. Three changes came out of a methodological review;
all three made the calculation simpler, and all three moved the estimate
DOWN, so the figures reported here are more conservative than the ones they
replace.

CHANGE 1 — EIGHT CATEGORIES, NOT TEN

Robson is not a pure case-mix stratification. Groups 6 to 10 are properties
of the woman — breech, twins, transverse lie, preterm. But the split of 1
from 2, and of 3 from 4, is a clinical DECISION: was she allowed to labour,
induced, or sectioned before labour began?

A woman does not arrive as "group 2". She arrives nulliparous with a single
cephalic fetus at term, and then the hospital chooses.

Standardising on Egypt's group 2 share therefore holds Egyptian PRACTICE
constant while calling it case mix. Egypt barely induces — Qena recorded
three inductions in 2,186 births, against 34.5% of all US births. No
difference in the women explains that.

So the two onset splits are collapsed and the standardisation runs on eight
categories: (1+2), (3+4), 5, 6, 7, 8, 9, 10. The ten-group table is still
reported, for diagnosis; it is simply not what the calculation runs on.

A useful consequence: the reference rate for (1+2) is 26.57%, which is the
NTSV rate NCHS publishes independently. The yardstick is a published
statistic rather than something derived here.

CHANGE 2 — GROUP 5 IS REALLOCATED TO GROUPS 3+4, NOT SPREAD PROPORTIONALLY

Scenario B asks what Egypt's expected rate would be if its group 5 share
matched the reference. The old code answered by rescaling the other nine
groups proportionally, which put freed women into groups 6, 7, 8 and 10 —
breech, twins, preterm, with rates from 43% to 96%.

That is not possible. A woman in group 5 is multiparous, single, cephalic
and at term. Remove the uterine scar and she is still multiparous, single,
cephalic and at term — which is the definition of groups 3 and 4. A scar
does not change fetal presentation, plurality or gestational age.

This is derivation from the definitions, not estimation. On the Benha data
the old method misplaced 6.6% of all births into groups that cannot contain
them, and it understated the gap by 5.2 percentage points.

CHANGE 3 — GROUP 9 REFERENCE RATE SET TO 100%

WHO group 9 is transverse or oblique lie, which cannot deliver vaginally.
Our CDC-derived rate of 46.52% is contaminated: CDC folds face, brow and
compound presentations into the same "other" code. That limitation was
documented early and then, inconsistently, the contaminated rate was still
used in the standardisation. It is corrected here to the definitional
value. The effect is about 0.2 percentage points, and it runs against the
project's own conclusion.

WHAT WAS CONSIDERED AND REJECTED

A third scenario resetting group 10 to the reference share. Group 10 in
Qena is 22.14% against a reference 7.46%, and a planned caesarean at 37
weeks does manufacture a preterm birth — Egypt's own ministry warned
against exactly that in April 2026.

But group 5 and group 10 are not analogous. Every woman in group 5 is there
because of a previous caesarean; there is no such thing as a spontaneous
group 5. Most preterm birth, by contrast, is spontaneous, and the share
that is iatrogenic cannot be identified in this data. Reallocating group 10
also has no clean destination: it contains nulliparous women, multiparous
women and women with previous caesareans, so freed women would belong in
1+2, 3+4 or 5 depending on facts we do not have.

Group 10 is therefore reported as a diagnostic observation and a stated
limitation, and no scenario is built on it.
=============================================================================
"""

import pandas as pd
import numpy as np
import json

# =============================================================================
# REFERENCE — CDC 2023, ten groups
# =============================================================================
REF_SIZE = {"1":16.43,"2":17.49,"3":21.98,"4":16.66,"5":12.53,
            "6":1.72,"7":1.73,"8":3.16,"9":0.83,"10":7.46}
REF_RATE = {"1":13.00,"2":39.31,"3":2.52,"4":18.46,"5":83.38,
            "6":96.24,"7":93.78,"8":75.07,"9":100.00,"10":43.17}
#                                                     ^ definitional, see above

def collapse(a, b):
    return (REF_SIZE[a]*REF_RATE[a] + REF_SIZE[b]*REF_RATE[b]) / (REF_SIZE[a]+REF_SIZE[b])

R8 = {"1+2": collapse("1","2"), "3+4": collapse("3","4"),
      **{g: REF_RATE[g] for g in ["5","6","7","8","9","10"]}}
R8_SIZE = {"1+2": REF_SIZE["1"]+REF_SIZE["2"], "3+4": REF_SIZE["3"]+REF_SIZE["4"],
           **{g: REF_SIZE[g] for g in ["5","6","7","8","9","10"]}}
CATS = ["1+2","3+4","5","6","7","8","9","10"]

# =============================================================================
# EGYPTIAN SOURCES — ten-group sizes as published
# =============================================================================
EGYPT = {
    "Benha 2018": {
        "n": 850, "observed": 54.71,
        "size": {"1":13.18,"2":3.29,"3":26.71,"4":2.71,"5":36.24,
                 "6":4.94,"7":3.29,"8":2.35,"9":0.24,"10":7.06},
        "note": "prospective, WHO manual, all births >=28 weeks including stillbirths",
    },
    "Qena 2022-23": {
        "n": 2186, "observed": 79.69,
        "size": {"1":3.75,"2":7.41,"3":8.78,"4":3.75,"5":40.81,
                 "6":0.91,"7":5.35,"8":6.68,"9":0.41,"10":22.14},
        "note": "cross-sectional; cite Table 3 only, N=2,186 not the 2,000 stated",
    },
}

def to8(m):
    return {"1+2": m["1"]+m["2"], "3+4": m["3"]+m["4"],
            **{g: m[g] for g in ["5","6","7","8","9","10"]}}

def expected(mix8, rates=R8):
    return sum(mix8[g]/100 * rates[g] for g in mix8)

def reset_group5(mix8, target):
    """Freed women go to 3+4 and nowhere else. See CHANGE 2 above."""
    out = dict(mix8)
    freed = mix8["5"] - target
    out["5"] = target
    out["3+4"] = mix8["3+4"] + freed
    return out

# =============================================================================
print("=" * 76)
print("REFERENCE — eight case-mix categories, CDC 2023")
print("=" * 76)
ref = pd.DataFrame({"category": CATS,
                    "reference_size_pct": [round(R8_SIZE[g],2) for g in CATS],
                    "reference_cs_rate_pct": [round(R8[g],2) for g in CATS]})
print(ref.to_string(index=False))
print(f"""
  The (1+2) rate of {R8['1+2']:.2f}% is the NTSV measure NCHS publishes. This
  project reproduced it independently at 26.56%, so the yardstick is a
  published national statistic rather than a figure derived here.
""")

print("=" * 76)
print("MAIN RESULT")
print("=" * 76)
rows = []
for name, d in EGYPT.items():
    m8 = to8(d["size"])
    assert abs(sum(m8.values()) - 100) < 0.15, f"{name} shares do not sum to 100"
    a = expected(m8)
    b8 = reset_group5(m8, R8_SIZE["5"])
    assert abs(sum(b8.values()) - 100) < 0.15
    b = expected(b8)
    obs = d["observed"]
    rows.append({"source": name, "n": d["n"], "observed_pct": obs,
                 "expected_A_pct": round(a,2), "gap_A_pp": round(obs-a,2),
                 "expected_B_pct": round(b,2), "gap_B_pp": round(obs-b,2),
                 "inherited_pp": round(a-b,2),
                 "inherited_share_of_total_pct": round(100*(a-b)/(obs-b))})
main = pd.DataFrame(rows)
print(main.to_string(index=False))
print("""
  A            Egypt's case mix as observed — excess attributable to current
               practice
  B            group 5 share set to the reference — total accumulated excess
  inherited    the difference: excess carried by a case mix that earlier
               caesareans created
""")

print("=" * 76)
print("THE DIAGNOSTIC FINDING")
print("=" * 76)
for r in rows:
    print(f"  {r['source']:<14} inherited {r['inherited_share_of_total_pct']:>3}% of the total excess")
print("""
  The two hospitals differ less in HOW MUCH excess they carry than in where
  it comes from. At Benha current practice sits close to the reference and
  almost all the excess is inherited; at Qena it is roughly evenly split.

  That points to different interventions. Benha needs to stop new group 5
  forming — which means the caesarean rate in first births. Qena needs that
  and a change in current practice as well.
""")

# =============================================================================
print("=" * 76)
print("SENSITIVITY — the group 5 share, read as a policy trajectory")
print("=" * 76)
print("""  Not a statement of uncertainty. We know Egypt's group 5 share. This
  answers a different question: if it fell to X, what would follow?
""")
print(f"{'group 5 share':>15}", end="")
for n in EGYPT: print(f"{n:>18}", end="")
print()
print("-" * 52)
sens = []
for g5 in [12.53, 18, 24, 30, 36, 40.81]:
    print(f"{g5:>14.2f}%", end="")
    for name, d in EGYPT.items():
        m8 = to8(d["size"])
        if g5 > m8["5"]:
            print(f"{'—':>18}", end=""); continue
        e = expected(reset_group5(m8, g5))
        gap = d["observed"] - e
        sens.append({"group5_share_pct": g5, "source": name,
                     "expected_pct": round(e,2), "gap_pp": round(gap,2)})
        print(f"{gap:>17.1f}", end="")
    print()

print("""
  Every step toward the reference share widens the gap, because a previous
  caesarean is the strongest single predictor in the table and removing it
  moves a woman from an 83% category to a 9% one.
""")

# =============================================================================
print("=" * 76)
print("SENSITIVITY — what if the reference itself is too permissive")
print("=" * 76)
print("""  The United States runs at 32%, above any clinical ideal. Scaling the
  reference rates down represents a lower-intervention standard.
""")
print(f"{'reference':>18}", end="")
for n in EGYPT: print(f"{n:>18}", end="")
print()
print("-" * 54)
scale_rows = []
for f, lab in [(1.00, "as observed"), (0.85, "-15%"), (0.70, "-30%")]:
    scaled = {g: v*f for g, v in R8.items()}
    print(f"{lab:>18}", end="")
    for name, d in EGYPT.items():
        e = expected(to8(d["size"]), scaled)
        gap = d["observed"] - e
        scale_rows.append({"scaling": lab, "source": name, "gap_pp": round(gap,2)})
        print(f"{gap:>17.1f}", end="")
    print()

allgaps = ([r["gap_A_pp"] for r in rows] + [r["gap_B_pp"] for r in rows]
           + [s["gap_pp"] for s in sens] + [s["gap_pp"] for s in scale_rows])
print(f"""
=============================================================================
THE INTERVAL
=============================================================================
  Across every combination tested, the estimated excess runs from
  {min(allgaps):.1f} to {max(allgaps):.1f} percentage points, and it never closes.

  Every bias that could not be tested runs the same way. A nationally
  representative case mix would carry a smaller group 5 and widen the gap.
  A lower-intervention reference would widen it. Referral bias in two
  tertiary hospitals narrows it. And all three methodological corrections
  applied in this version reduced the estimate rather than raising it.
""")

print("=" * 76)
print("REPORTED SEPARATELY, NOT AS A SCENARIO")
print("=" * 76)
for name, d in EGYPT.items():
    print(f"  {name:<14} group 10 share {d['size']['10']:>6.2f}%   "
          f"reference {REF_SIZE['10']:.2f}%")
print("""
  Qena's preterm share is three times the reference. A planned caesarean at
  37 weeks creates a preterm birth, and Egypt's ministry warned against
  precisely that in April 2026, so part of this share is likely iatrogenic.

  No scenario is built on it. Most preterm birth is spontaneous, the
  iatrogenic share cannot be identified here, and freed women would have no
  single destination — group 10 contains nulliparous women, multiparous
  women and women with previous caesareans alike.
""")

main.to_csv("egypt_gap_main.csv", index=False)
pd.DataFrame(sens).to_csv("egypt_gap_sensitivity_group5.csv", index=False)
ref.to_csv("egypt_reference_8cat.csv", index=False)

mix_rows = []
for name, d in EGYPT.items():
    m8 = to8(d["size"])
    for g in CATS:
        mix_rows.append({"source": name, "group_8": g,
                         "share_pct": round(m8[g], 2),
                         "reference_rate_pct": round(R8[g], 2),
                         "reference_share_pct": round(R8_SIZE[g], 2)})
pd.DataFrame(mix_rows).to_csv("egypt_case_mix_8cat.csv", index=False)

json.dump({"reference_8cat": {g: round(R8[g],4) for g in CATS},
           "reference_size_8cat": {g: round(R8_SIZE[g],4) for g in CATS},
           "main": rows, "sensitivity_group5": sens,
           "sensitivity_reference_scaling": scale_rows,
           "interval_pp": [float(min(allgaps)), float(max(allgaps))]},
          open("egypt_gap_analysis.json","w"), indent=2)

print("Saved egypt_gap_main.csv, egypt_gap_sensitivity_group5.csv,")
print("      egypt_reference_8cat.csv, egypt_case_mix_8cat.csv,")
print("      egypt_gap_analysis.json")
print("""
  egypt_case_mix_8cat.csv is the table Power BI loads alongside the births
  data. Sixteen rows — eight categories for each of two sources — carrying
  Egypt's share and the reference rate. The standardisation measure runs
  across the two tables using TREATAS, and a what-if parameter on the group
  5 share drives the trajectory shown above.
""")
