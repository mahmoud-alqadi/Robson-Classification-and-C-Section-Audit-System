"""
SHAP Within-Group Variation Fix
================================
Problem:  The original table uses mean(|SHAP|) per feature per group.
          This measures each feature's CONTRIBUTION TO THE PREDICTION LEVEL,
          not its contribution to VARIATION between patients within the group.

          Result: PRIORLIVE dominates Group 1 (23%) even though it's ZERO
          for every single patient — it can't explain why one patient gets
          a caesarean and another doesn't.

Fix:      Use std(SHAP) — the standard deviation of SHAP values within each group.
          A constant feature (same SHAP for everyone) → std ≈ 0 → disappears.
          A varying feature (different SHAP per patient) → std > 0 → ranks high.

This is the CORRECT metric for "within-group explained variation".
"""

import numpy as np
import pandas as pd


def compute_within_group_variation(shap_values, feature_names, group_labels,
                                   target_groups=None, top_n=6):
    """
    Compute true within-group variation using std(SHAP) instead of mean(|SHAP|).
    
    Parameters
    ----------
    shap_values : np.ndarray, shape (n_samples, n_features)
        SHAP values matrix from explainer.
    feature_names : list[str]
        Names corresponding to columns of shap_values.
    group_labels : array-like, shape (n_samples,)
        Robson group label for each sample (e.g. '1', '2a', '3', ...).
    target_groups : list[str], optional
        Which groups to include. Default: ['1', '2a', '3', '4a'].
    top_n : int
        How many top features to show per group.
    
    Returns
    -------
    pd.DataFrame
        Rows = top features, Columns = groups, Values = % of within-group variation.
    """
    if target_groups is None:
        target_groups = ['1', '2a', '3', '4a']
    
    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df['robson_group'] = group_labels
    
    # ──────────────────────────────────────────────────
    # OLD (WRONG for "within-group variation"):
    #   mean(|SHAP|) — measures level contribution, not variation
    #
    # old_result = shap_df.groupby('robson_group')[feature_names].apply(
    #     lambda x: x.abs().mean()
    # )
    # ──────────────────────────────────────────────────
    
    # NEW (CORRECT):
    #   std(SHAP) — measures how much each feature's SHAP value
    #   varies across patients WITHIN the group
    variation = shap_df.groupby('robson_group')[feature_names].std()
    
    # Normalise to percentages within each group
    variation_pct = variation.div(variation.sum(axis=1), axis=0) * 100
    
    # Build output table: top_n features per group
    results = {}
    all_top_features = set()
    
    for grp in target_groups:
        if grp not in variation_pct.index:
            continue
        row = variation_pct.loc[grp].sort_values(ascending=False)
        top_feats = row.head(top_n)
        results[f'Group {grp}'] = top_feats
        all_top_features.update(top_feats.index)
    
    # Create a unified table with the union of all top features
    all_top_features = sorted(all_top_features)
    
    table = pd.DataFrame(index=all_top_features)
    for grp_label, series in results.items():
        table[grp_label] = series.reindex(all_top_features)
    
    # Sort by mean rank across groups
    table['mean_pct'] = table.mean(axis=1)
    table = table.sort_values('mean_pct', ascending=False).drop(columns='mean_pct')
    
    # Format as percentages
    table = table.applymap(lambda x: f'{x:.2f}%' if pd.notna(x) else '—')
    
    return table


def diagnostic_comparison(shap_values, feature_names, group_labels, group='1'):
    """
    Show WHY mean(|SHAP|) is misleading for a specific group.
    Prints a side-by-side comparison of mean(|SHAP|) vs std(SHAP).
    
    This is useful for explaining to the committee.
    """
    mask = np.array(group_labels) == group
    shap_group = shap_values[mask]
    
    mean_abs = np.mean(np.abs(shap_group), axis=0)
    std_vals = np.std(shap_group, axis=0)
    
    comp = pd.DataFrame({
        'Feature': feature_names,
        'mean(|SHAP|)': mean_abs,
        'std(SHAP)': std_vals,
        'Ratio std/mean': std_vals / (mean_abs + 1e-10)
    }).sort_values('mean(|SHAP|)', ascending=False)
    
    print(f"\n{'='*70}")
    print(f"  Diagnostic for Group {group}")
    print(f"  Features with HIGH mean(|SHAP|) but LOW std(SHAP)")
    print(f"  are CONSTANT — they shift the group's baseline,")
    print(f"  not explain within-group variation.")
    print(f"{'='*70}\n")
    
    # Flag problematic features
    comp['Issue'] = comp.apply(
        lambda r: '⚠️ CONSTANT (baseline shift, not variation)'
        if r['Ratio std/mean'] < 0.1 else '✓ True variation driver',
        axis=1
    )
    
    print(comp.head(10).to_string(index=False))
    print()
    
    return comp


# ─────────────────────────────────────────────────
# Usage example (drop into your notebook):
# ─────────────────────────────────────────────────
#
# import shap
# explainer = shap.TreeExplainer(model)
# shap_values = explainer.shap_values(X_test)
#
# # 1. The corrected table
# table = compute_within_group_variation(
#     shap_values, 
#     feature_names=X_test.columns.tolist(),
#     group_labels=test_groups,  # your Robson group column
#     target_groups=['1', '2a', '3', '4a'],
#     top_n=6
# )
# print(table)
#
# # 2. Diagnostic showing why the old metric was wrong
# diagnostic_comparison(
#     shap_values,
#     feature_names=X_test.columns.tolist(), 
#     group_labels=test_groups,
#     group='1'
# )
