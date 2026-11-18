"""
=============================================================================
EGYPT COST MODEL
=============================================================================
    python egypt_cost.py

WHY THIS WAS DEFERRED, AND WHY IT IS NOW POSSIBLE

The supervisor's objection to the economic simulator was specific: the
proposal promised output calibrated on Egyptian cost data while citing no
cost source anywhere. That objection was correct. It is now answered — the
Population Council study the supervisor named by title has been located and
its figures are below, each with a source attached.

THE FIGURE THAT MATTERS MOST, AND ITS TRAP

Population Council (2018), costing 2014 practice:
    health-system cost per caesarean ....... EGP 1,076
    health-system cost per vaginal birth ... EGP   664
    difference ............................. EGP   412
    estimated cost of caesarean over-use in 2014 ... over EGP 900 million

Those are 2014 Egyptian pounds. Between 2014 and 2025 the pound fell from
roughly 7 to roughly 48 against the dollar. Multiplying a 2014 unit cost by
a 2025 birth count would produce a number that is wrong by a factor of
several, and wrong in a direction that understates the burden.

So this model does two separate things and never mixes them:

  1. REPRODUCES the 2014 estimate from 2014 costs and 2014 rates. This is
     the defensible published figure and it is reported as such.

  2. Rescales to 2025 using 2025 PRICES from a separate source, clearly
     labelled, with the caveat that these are prices charged rather than
     costs incurred — a different quantity that includes margin.

WHAT THIS MODEL DELIBERATELY DOES NOT CLAIM

It does not claim a saving. Reducing caesarean rates does not free the money
one-for-one: staff, theatres and buildings are largely fixed in the short
run. The honest quantity is the cost DIFFERENTIAL attached to caesareans in
excess of clinical need, which is what an economist would call the resource
burden rather than a recoverable saving. That distinction is stated in the
output rather than left for a reader to notice.
=============================================================================
"""

import pandas as pd
import json

# =============================================================================
# SOURCED FIGURES
# =============================================================================
COSTS = [
    # (label, caesarean, vaginal, currency/year, perspective, source)
    ("Health-system cost", 1076, 664, "EGP 2014", "provider cost: staff time, "
     "medicines, supplies, equipment",
     "Abdel-Tawab NG, Oraby D, Hassanein N, El-Nakib S. Caesarean section "
     "deliveries in Egypt: trends, practices, perceptions and cost. "
     "Population Council, 2018 (doi 10.31899/rh6.1004)"),
    ("Private facility charge (low)", 2000, 1000, "EGP 2018", "price charged",
     "Population Council 2018, as reported in The National, Sept 2022"),
    ("Private facility charge (high)", 5000, 2000, "EGP 2018", "price charged",
     "Population Council 2018, as reported in The National, Sept 2022"),
    ("Public hospital price (low)", 3000, None, "EGP 2025", "price charged",
     "Zawia3 report, citing recent Egyptian pricing"),
    ("Public hospital price (high)", 6000, None, "EGP 2025", "price charged",
     "Zawia3 report, citing recent Egyptian pricing"),
    ("Private facility price (low)", 15000, None, "EGP 2025", "price charged",
     "Zawia3 report, citing recent Egyptian pricing"),
    ("Private facility price (high)", 30000, None, "EGP 2025", "price charged",
     "Zawia3 report, citing recent Egyptian pricing"),
]

PUBLISHED_2014 = {
    "cs_rate": 52.0,
    "annual_births": 2_600_000,
    "overuse_cost_egp": 900_000_000,
    "source": "Population Council 2018",
}

# =============================================================================
print("=" * 78)
print("SOURCED EGYPTIAN COST FIGURES")
print("=" * 78)
c = pd.DataFrame(COSTS, columns=["item","caesarean_egp","vaginal_egp",
                                 "currency_year","perspective","source"])
c["difference_egp"] = c.caesarean_egp - c.vaginal_egp
print(c[["item","caesarean_egp","vaginal_egp","difference_egp",
         "currency_year","perspective"]].to_string(index=False))
print("""
  Note the perspective column. Provider COST and price CHARGED are different
  quantities: the second includes margin and, in Egyptian private practice,
  a separate payment to the doctor. Only the first row is a cost estimate;
  the rest are prices. They are not interchangeable and are not combined.
""")

# =============================================================================
print("=" * 78)
print("1. REPRODUCING THE PUBLISHED 2014 ESTIMATE")
print("=" * 78)

delta_2014 = 1076 - 664
births = PUBLISHED_2014["annual_births"]
cs_rate = PUBLISHED_2014["cs_rate"] / 100

print(f"""
  Annual births (2014, approximate)     : {births:,}
  Caesarean rate                        : {100*cs_rate:.0f}%
  Caesareans performed                  : {births*cs_rate:,.0f}
  Cost differential per caesarean       : EGP {delta_2014:,}
""")

print(f"  {'excess share':>14} {'excess caesareans':>20} {'differential cost':>22}")
print("  " + "-"*58)
rows_2014 = []
for pct in [10, 20, 30, 40, 50]:
    n_excess = births * cs_rate * pct/100
    cost = n_excess * delta_2014
    rows_2014.append({"excess_share_pct": pct, "n_excess": int(n_excess),
                      "cost_egp_2014": int(cost)})
    print(f"  {pct:>13}% {n_excess:>20,.0f} {'EGP ' + format(cost, ',.0f'):>22}")

print("""
  THE PUBLISHED FIGURE DOES NOT RECONCILE WITH THIS ARITHMETIC, AND THAT
  IS INFORMATIVE

  Population Council report "over EGP 900 million" for 2014. Dividing that
  by the EGP 412 differential implies 2.18 million excess caesareans — more
  than the 1.35 million caesareans performed in total that year. The figure
  therefore cannot be built from the cost differential.

  Reconstructing it: valuing excess caesareans at the FULL caesarean cost of
  EGP 1,076, and defining excess as everything above a WHO reference
  threshold, reproduces it closely:

      above 19% ..... 858,000 x EGP 1,076 = EGP 923 million
      above 15% ..... 962,000 x EGP 1,076 = EGP 1,035 million

  Valuing at full cost overstates the economic burden, because those women
  would have delivered either way. Had they delivered vaginally the system
  would still have spent EGP 664 each. The quantity attributable to the
  DECISION is the difference, not the whole procedure.

  This project therefore reports differential-based figures, which are
  roughly 2.6 times smaller than the published estimate. Being more
  conservative than the source is a deliberate choice and is stated as one.
""")

# =============================================================================
print("=" * 78)
print("2. THE SAME CALCULATION ON THE PROJECT'S OWN GAP ESTIMATE")
print("=" * 78)
print("""
  The standardisation put the excess between 5.0 and 38.4 percentage points
  of the caesarean rate, depending on assumptions. Converting that to a
  count rather than a share of caesareans:
""")
print(f"  {'gap (pp)':>10} {'excess caesareans':>20} {'2014 differential cost':>26}")
print("  " + "-"*58)
rows_gap = []
for gap in [5.0, 7.6, 12.8, 20.7, 21.0, 32.3]:
    n_excess = births * gap/100
    cost = n_excess * delta_2014
    rows_gap.append({"gap_pp": gap, "n_excess": int(n_excess),
                     "cost_egp_2014_prices": int(cost)})
    print(f"  {gap:>10.1f} {n_excess:>20,.0f} {'EGP ' + format(cost, ',.0f'):>26}")
print("""
  Read these as resource burden in 2014 prices, not as money that could be
  recovered. 12.8 pp is the figure the private-sector standardisation
  produced independently, and it sits inside the range the Robson
  standardisation produced from an entirely different method and data source.
""")

# =============================================================================
print("=" * 78)
print("3. WHY 2014 COSTS CANNOT SIMPLY BE UPRATED")
print("=" * 78)
print("""
  The Egyptian pound went from roughly 7 to the dollar in 2014 to roughly 48
  in 2025. Applying EGP 412 to a 2025 birth count would produce a figure
  that is nominally meaningless.

  The 2025 prices available are also prices, not costs, and they come from a
  press source rather than a costing study:

      public hospital caesarean ..... EGP 3,000-6,000
      private facility caesarean .... EGP 15,000-30,000

  No 2025 vaginal-delivery comparator was found, so no 2025 differential can
  be constructed. Reporting a 2025 saving figure would therefore require
  inventing the missing half of the subtraction.

  RECOMMENDATION: report the 2014 estimate as the sourced, defensible
  figure, present 2025 prices descriptively alongside it, and state plainly
  that a current differential cannot be computed until a current costing
  study exists. That absence is itself worth reporting — a country
  performing 1.35 million caesareans a year has not costed them since 2014.
""")

# =============================================================================
print("=" * 78)
print("4. WHAT THE COST STUDY FOUND BESIDES COSTS")
print("=" * 78)
print("""
  The same study asked physicians and key informants why caesarean use is
  high in Egypt. They unanimously agreed it is over-used, and named:

    financial incentive
    doctors' desire for control over their time
    fear of medical litigation
    vagueness of medical protocols
    limited opportunities for junior doctors to practise vaginal delivery
    shortage of pain-relief drugs in public hospitals
    shortage of anaesthetists trained in epidural administration

  Compare that list with the reasons Benha's authors gave independently for
  their 100% caesarean rate in Robson group 5: fear of litigation, no
  continuous fetal monitoring, no one-to-one care, no training in external
  cephalic version. Two separate sources, overlapping explanations, neither
  of them maternal preference.

  One further figure from the same study: only 6% of the women interviewed
  reported receiving any counselling on the disadvantages or health risks of
  a caesarean. Set that against the national survey finding that 80% of
  Egyptian women believe doctors prefer caesarean delivery.
""")

json.dump({"costs": COSTS, "published_2014": PUBLISHED_2014,
           "excess_share_scenarios": rows_2014,
           "gap_scenarios": rows_gap,
           "delta_2014_egp": delta_2014},
          open("egypt_cost_model.json","w"), indent=2)
c.to_csv("egypt_cost_sources.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(rows_gap).to_csv("egypt_cost_by_gap.csv", index=False)
print("Saved egypt_cost_model.json, egypt_cost_sources.csv, egypt_cost_by_gap.csv")
