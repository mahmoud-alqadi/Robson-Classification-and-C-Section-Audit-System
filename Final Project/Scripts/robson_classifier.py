"""
=============================================================================
Robson Ten-Group Classification System (TGCS)
Rule-based implementation per WHO Robson Classification Implementation Manual
(2017) -- Figure 2 flowchart & Table 3.

THIS MODULE CONTAINS NO MACHINE LEARNING.
Robson classification is a deterministic decision tree. Given the six
obstetric variables, group membership is fully determined. There is no
training, no fitting, and no accuracy to measure here.

Designed for: CDC Natality Public Use File (2016+ file layout)
              https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Dataset_Documentation/DVS/natality/

-----------------------------------------------------------------------------
!! BEFORE FIRST RUN !!
Open the User Guide PDF for the exact year you downloaded and verify every
field name and code value in COLS and CODES below. NCHS changes field
detail between years (e.g. several fields were coarsened in June 2022).
Everything year-specific is isolated in those two dicts on purpose --
you should never need to touch the classification logic itself.
-----------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

# =============================================================================
# 1. YEAR-SPECIFIC CONFIGURATION  --  verify against the CDC user guide
# =============================================================================

COLS = {
    "prior_live":   "PRIORLIVE",    # Prior births now living
    "prior_dead":   "PRIORDEAD",    # Prior births now dead (stillbirths)
    "prev_cs":      "RF_CESAR",     # Previous cesarean delivery  Y/N/U
    "prev_cs_n":    "RF_CESARN",    # Number of previous cesareans
    "induction":    "LD_INDL",      # Induction of labor          Y/N/U
    "trial_labor":  "ME_TRIAL",     # Trial of labor attempted (if cesarean)
    "presentation": "ME_PRES",      # 1 Cephalic  2 Breech  3 Other  9 Unknown
    "route":        "ME_ROUT",      # 1 Spont  2 Forceps  3 Vacuum  4 Cesarean
    "plurality":    "DPLURAL",      # 1 Single  2 Twin  3 Triplet+
    "gest_age":     "OEGest_Comb",  # Obstetric estimate, weeks (99 = unknown)
}

CODES = {
    "yes":            {"Y", "1"},
    "no":             {"N", "2"},
    "unknown":        {"U", "9", "99", ""},
    "presentation":   {"cephalic": {"1"}, "breech": {"2"}, "other": {"3"}},
    "route_cesarean": {"4"},
    "gest_unknown":   {99, 999},
    "count_unknown":  {99, 999},          # PRIORLIVE / PRIORDEAD / RF_CESARN
}

TERM_WEEKS = 37          # WHO cut-off: >= 37 weeks is term
PRETERM_LABEL = "10"


# =============================================================================
# 2. DERIVED VARIABLES  --  the six Robson criteria
# =============================================================================

def _yn(series, key):
    """Map a CDC Y/N/U field to True / False / NaN."""
    s = series.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=s.index, dtype="object")
    out[s.isin(CODES["yes"])] = True
    out[s.isin(CODES["no"])] = False
    return out


def derive_parity(df):
    """
    WHO parity = number of previous births at >= 22 weeks, live OR stillborn.
    Prior *abortions* (PRIORTERM) do NOT count toward parity -- a common
    implementation error that silently moves cases between groups 1/2 and 3/4.
    """
    live = pd.to_numeric(df[COLS["prior_live"]], errors="coerce")
    dead = pd.to_numeric(df[COLS["prior_dead"]], errors="coerce")
    live = live.where(~live.isin(CODES["count_unknown"]))
    dead = dead.where(~dead.isin(CODES["count_unknown"]))

    total = live.add(dead, fill_value=None)
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[total == 0] = "nulliparous"
    out[total > 0] = "multiparous"
    return out


def derive_onset_of_labour(df):
    """
    The single hardest derivation in the whole pipeline, and the one that
    decides whether groups 2a/2b and 4a/4b are correct.

        induced        -> labour was started artificially
        spontaneous    -> labour started on its own
        pre-labour CS  -> caesarean performed with NO labour at all

    ME_TRIAL is only recorded for caesareans. It must never be read as
    "no trial of labour" for a vaginal birth -- a vaginal birth by
    definition had labour.
    """
    induced = _yn(df[COLS["induction"]], "induction")
    trial = _yn(df[COLS["trial_labor"]], "trial_labor")
    is_cs = df[COLS["route"]].astype(str).str.strip().isin(CODES["route_cesarean"])

    onset = pd.Series(np.nan, index=df.index, dtype="object")

    # Vaginal births: labour happened by definition.
    onset[~is_cs & (induced == True)] = "induced"
    onset[~is_cs & (induced == False)] = "spontaneous"

    # Caesareans: induction wins, then trial of labour, else pre-labour CS.
    onset[is_cs & (induced == True)] = "induced"
    onset[is_cs & (induced == False) & (trial == True)] = "spontaneous"
    onset[is_cs & (induced == False) & (trial == False)] = "prelabour_cs"

    return onset


def derive_presentation(df):
    s = df[COLS["presentation"]].astype(str).str.strip()
    out = pd.Series(np.nan, index=df.index, dtype="object")
    out[s.isin(CODES["presentation"]["cephalic"])] = "cephalic"
    out[s.isin(CODES["presentation"]["breech"])] = "breech"
    # DOCUMENTED CAVEAT: CDC "Other" is a catch-all. WHO group 9 is strictly
    # transverse / oblique lie. Face and brow presentations are technically
    # cephalic but may be coded here. We assign "Other" -> group 9 and report
    # the count separately so the assumption is auditable, not hidden.
    out[s.isin(CODES["presentation"]["other"])] = "other_lie"
    return out


def derive_term(df):
    ga = pd.to_numeric(df[COLS["gest_age"]], errors="coerce")
    ga = ga.where(~ga.isin(CODES["gest_unknown"]))
    # NaN >= 37 evaluates to False, which would silently classify every
    # record with a missing gestational age as PRETERM (group 10).
    # Three-state result: True / False / NaN.
    is_term = pd.Series(np.nan, index=ga.index, dtype="object")
    is_term[ga.notna() & (ga >= TERM_WEEKS)] = True
    is_term[ga.notna() & (ga < TERM_WEEKS)] = False
    return ga, is_term


def derive_previous_cs(df):
    prev = _yn(df[COLS["prev_cs"]], "prev_cs")
    n = pd.to_numeric(df[COLS["prev_cs_n"]], errors="coerce")
    n = n.where(~n.isin(CODES["count_unknown"]))
    return prev, n


# =============================================================================
# 3. THE CLASSIFICATION  --  WHO Manual 2017, Figure 2
# =============================================================================

def classify(df, subgroups=True):
    """
    Returns the input frame with four added columns:
        Robson_group            display label ('1', '2a', '5.1', '10', ...)
        Robson_group_sort_order integer for correct ordering (text '10' sorts
                                between '1' and '2' otherwise -- a real bug)
        Classification_status   'Classified' / 'Unclassifiable'
        Missing_variable        comma-separated list of what was missing
    """
    df = df.copy()

    parity = derive_parity(df)
    onset = derive_onset_of_labour(df)
    pres = derive_presentation(df)
    ga, is_term = derive_term(df)
    prev_cs, prev_cs_n = derive_previous_cs(df)
    plural = pd.to_numeric(df[COLS["plurality"]], errors="coerce")

    # ---- data-quality tracking (WHO's own recommended completeness metric)
    missing = pd.DataFrame({
        "parity": parity.isna(),
        "onset_of_labour": onset.isna(),
        "presentation": pres.isna(),
        "gestational_age": ga.isna(),
        "previous_cs": prev_cs.isna(),
        "plurality": plural.isna(),
    }, index=df.index)

    df["Missing_variable"] = missing.apply(
        lambda r: ",".join(missing.columns[r.values]) if r.any() else "", axis=1
    )

    group = pd.Series(np.nan, index=df.index, dtype="object")

    # ---- ORDER MATTERS. This is the WHO flowchart sequence. --------------
    # Groups 8, 9, 6, 7 and 10 all INCLUDE women with a previous caesarean,
    # so the previous-CS test must come after them, not before.

    # Step 1 -- multiple pregnancy (any parity, any previous CS)
    m8 = plural >= 2
    group[m8] = "8"

    rest = ~m8 & plural.notna()

    # Step 2 -- abnormal lie (transverse / oblique), singletons only
    m9 = rest & (pres == "other_lie")
    group[m9] = "9"

    # Step 3 -- breech, singletons
    m6 = rest & ~m9 & (pres == "breech") & (parity == "nulliparous")
    m7 = rest & ~m9 & (pres == "breech") & (parity == "multiparous")
    group[m6] = "6"
    group[m7] = "7"

    # Step 4 -- single cephalic, preterm (< 37 weeks), incl. previous CS
    singleton_cephalic = rest & (pres == "cephalic")
    m10 = singleton_cephalic & (is_term == False)
    group[m10] = PRETERM_LABEL

    # Step 5 -- single cephalic, term, with previous caesarean
    term_cephalic = singleton_cephalic & (is_term == True)
    m5 = term_cephalic & (prev_cs == True)
    if subgroups:
        group[m5 & (prev_cs_n == 1)] = "5.1"
        group[m5 & (prev_cs_n >= 2)] = "5.2"
        group[m5 & prev_cs_n.isna()] = "5"       # count known, number not
    else:
        group[m5] = "5"

    # Step 6 -- single cephalic, term, NO previous caesarean
    no_prev = term_cephalic & (prev_cs == False)

    null_ = no_prev & (parity == "nulliparous")
    multi = no_prev & (parity == "multiparous")

    group[null_ & (onset == "spontaneous")] = "1"
    group[multi & (onset == "spontaneous")] = "3"

    if subgroups:
        group[null_ & (onset == "induced")] = "2a"
        group[null_ & (onset == "prelabour_cs")] = "2b"
        group[multi & (onset == "induced")] = "4a"
        group[multi & (onset == "prelabour_cs")] = "4b"
    else:
        group[null_ & onset.isin(["induced", "prelabour_cs"])] = "2"
        group[multi & onset.isin(["induced", "prelabour_cs"])] = "4"

    df["Robson_group"] = group
    df["Classification_status"] = np.where(group.isna(), "Unclassifiable", "Classified")
    df.loc[group.isna(), "Robson_group"] = "Unclassifiable"

    order = {"1": 100, "2": 200, "2a": 210, "2b": 220, "3": 300, "4": 400,
             "4a": 410, "4b": 420, "5": 500, "5.1": 510, "5.2": 520,
             "6": 600, "7": 700, "8": 800, "9": 900, "10": 1000,
             "Unclassifiable": 9999}
    df["Robson_group_sort_order"] = df["Robson_group"].map(order)

    return df


# =============================================================================
# 4. THE STANDARD ROBSON REPORT TABLE  (WHO Manual 2017, Table 5)
# =============================================================================

def report_table(df, cs_flag_col="is_cesarean"):
    """
    Produces the WHO standard report table. Every downstream analysis in the
    project depends on this table being correct -- do not move past it until
    the group distribution has been checked against published studies.

    Columns:
      n                       group size
      group_size_pct          group as % of all births
      cs_in_group             caesareans within the group
      cs_rate_pct             caesarean rate WITHIN the group
      absolute_contribution   group's caesareans / all births
      relative_contribution   absolute contribution / overall CS rate (share)
    """
    total_births = len(df)
    total_cs = df[cs_flag_col].sum()
    overall_rate = total_cs / total_births if total_births else np.nan

    g = df.groupby(["Robson_group_sort_order", "Robson_group"], dropna=False)
    t = g.agg(n=(cs_flag_col, "size"), cs_in_group=(cs_flag_col, "sum")).reset_index()

    t["group_size_pct"] = 100 * t["n"] / total_births
    t["cs_rate_pct"] = 100 * t["cs_in_group"] / t["n"]
    t["absolute_contribution"] = 100 * t["cs_in_group"] / total_births
    t["relative_contribution"] = 100 * (t["cs_in_group"] / total_cs)

    t = t.sort_values("Robson_group_sort_order").drop(columns="Robson_group_sort_order")

    total_row = pd.DataFrame([{
        "Robson_group": "TOTAL", "n": total_births, "cs_in_group": total_cs,
        "group_size_pct": 100.0, "cs_rate_pct": 100 * overall_rate,
        "absolute_contribution": 100 * overall_rate, "relative_contribution": 100.0,
    }])
    return pd.concat([t, total_row], ignore_index=True).round(2)


def sanity_checks(df):
    """Structural checks that must pass before any modelling begins."""
    issues = []
    d = df[df["Classification_status"] == "Classified"]

    for grp in ["2b", "4b", "9"]:
        sub = d[d["Robson_group"] == grp]
        if len(sub) and sub["is_cesarean"].mean() < 0.999:
            issues.append(
                f"Group {grp} caesarean rate is {100*sub['is_cesarean'].mean():.1f}% "
                f"-- it must be 100% by definition. Check the onset/presentation logic."
            )
    for grp in ["1", "3"]:
        sub = d[d["Robson_group"] == grp]
        if len(sub) and sub["is_cesarean"].mean() > 0.60:
            issues.append(f"Group {grp} caesarean rate implausibly high -- check parity logic.")

    empty = set(["1", "2a", "2b", "3", "4a", "4b", "5.1", "5.2", "6", "7", "8", "9", "10"]) \
            - set(d["Robson_group"].unique())
    if empty:
        issues.append(f"Empty groups: {sorted(empty)} -- a data problem to investigate, not a result.")

    unclass = 100 * (df["Classification_status"] == "Unclassifiable").mean()
    issues.append(f"[info] Unclassifiable: {unclass:.2f}% of records.")
    return issues


# =============================================================================
# 5. CHUNKED LOADER for the CDC fixed-width PUF (files are ~3-5 GB)
# =============================================================================

def load_cdc_natality(path, colspecs, names, chunksize=500_000, out_parquet=None):
    """
    Read the fixed-width PUF once, keep only the needed columns, write a
    compact Parquet file, and never touch the raw file again.

    colspecs / names come from the year's User Guide field-position table.
    """
    frames = []
    for chunk in pd.read_fwf(path, colspecs=colspecs, names=names,
                             dtype=str, chunksize=chunksize):
        frames.append(chunk)
    df = pd.concat(frames, ignore_index=True)
    if out_parquet:
        df.to_parquet(out_parquet, index=False)
    return df
