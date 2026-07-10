"""
demographics_statistics.py — Group demographics comparison for a specific task.
================================================================================

Compares age, height, weight, and BMI between groups using appropriate
statistical tests, and tests sex distribution with chi-square.

Only subjects who have kinectomes available for the specified task are included
— matching the groups used in the actual analysis.

Usage (in main.py):
    from src.data_utils.demographics_statistics import compare_group_demographics
    compare_group_demographics(DIAGNOSIS, TASK_NAMES, RESULT_BASE_PATH)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compare_group_demographics(diagnosis, task_names, result_base_path):
    """Compare demographics between groups for subjects with kinectomes.

    For each task, loads the matched groups (same subjects used in analysis),
    retrieves their demographics, runs appropriate statistical tests, and saves
    a summary CSV.

    Continuous variables (age, height, weight, BMI):
        Shapiro-Wilk normality test per group. If both normal → independent
        t-test + Cohen's d. Otherwise → Mann-Whitney U + rank-biserial r.

    Categorical variable (sex):
        Chi-square test (or Fisher's exact if any cell < 5).

    Parameters
    ----------
    diagnosis : list[str]
        Diagnosis column name(s) from the demographics file.
    task_names : list[str]
        Tasks to run demographics comparison for.
    result_base_path : str or Path
        Root results directory. CSV saved to
        ``result_base_path/demographics/demographics_<task>.csv``.
    """
    from config import DEMOGRAPHICS_PATH
    from src.data_utils.groups import get_matched_groups_for_task


    save_dir = Path(result_base_path) / 'demographics'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load demographics
    if DEMOGRAPHICS_PATH.suffix.lower() == '.csv':
        demo = pd.read_csv(DEMOGRAPHICS_PATH)
    else:
        demo = pd.read_excel(DEMOGRAPHICS_PATH)

    # Build pp-format ID column for matching
    demo['pp_id'] = demo['id'].apply(
        lambda x: f"pp{int(str(x).split('-')[0]):>03d}"
    )

    group_name = diagnosis[0][10:].capitalize()

    task_disease_ids, task_control_ids = get_matched_groups_for_task(
        diagnosis, task_names
    )

    for task in task_names:
        disease_ids = task_disease_ids.get(task, [])
        control_ids = task_control_ids.get(task, [])

        if not disease_ids or not control_ids:
            print(f"  No matched subjects for task {task} — skipping demographics.")
            continue

        g1 = demo[demo['pp_id'].isin(disease_ids)].copy()
        g2 = demo[demo['pp_id'].isin(control_ids)].copy()

        print(f"\n  Demographics for task: {task}")
        print(f"  {group_name}: n={len(g1)}, Control: n={len(g2)}")

        rows = []

        # ── Continuous variables ───────────────────────────────────────────
        continuous = {
            'Age (years)':      'age',
            'Height (cm)':      'height',
            'Weight (kg)':      'weight',
            'BMI (kg/m²)':      'bmi',
            'UPDRS III total':  'updrs_3_total',
        }

        for label, col in continuous.items():
            if col not in demo.columns:
                print(f"    Column '{col}' not found in demographics — skipping.")
                continue

            d1 = g1[col].dropna()
            d2 = g2[col].dropna()

            if len(d1) < 3 or len(d2) < 3:
                continue

            # Compute BMI on the fly if column missing
            if col == 'bmi' and 'bmi' not in demo.columns:
                if 'height' in demo.columns and 'weight' in demo.columns:
                    g1 = g1.copy()
                    g2 = g2.copy()
                    g1['bmi'] = g1['weight'] / (g1['height'] / 100) ** 2
                    g2['bmi'] = g2['weight'] / (g2['height'] / 100) ** 2
                else:
                    continue

            # Skip UPDRS for controls (they won't have scores — just skip silently)
            if col == 'updrs_3_total' and col not in g1.columns:
                continue

            _, p_norm1 = stats.shapiro(d1)
            _, p_norm2 = stats.shapiro(d2)
            normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)

            if normal:
                _, p_lev = stats.levene(d1, d2)
                stat, p = stats.ttest_ind(d1, d2, equal_var=(p_lev > 0.05))
                test = f"t-test (equal_var={p_lev > 0.05})"
                pooled = np.sqrt(
                    ((len(d1)-1)*d1.std(ddof=1)**2 + (len(d2)-1)*d2.std(ddof=1)**2)
                    / (len(d1)+len(d2)-2)
                )
                effect = (d1.mean() - d2.mean()) / pooled if pooled > 0 else np.nan
                effect_label = 'Cohen\'s d'
            else:
                stat, p = stats.mannwhitneyu(d1, d2, alternative='two-sided')
                test = "Mann-Whitney U"
                effect = 1 - (2 * stat) / (len(d1) * len(d2))
                effect_label = 'rank-biserial r'

            rows.append({
                'variable':         label,
                f'{group_name}_mean': round(d1.mean(), 2),
                f'{group_name}_sd':   round(d1.std(ddof=1), 2),
                f'{group_name}_n':    len(d1),
                'Control_mean':       round(d2.mean(), 2),
                'Control_sd':         round(d2.std(ddof=1), 2),
                'Control_n':          len(d2),
                'test':               test,
                'statistic':          round(stat, 3),
                'p_value':            round(p, 4),
                'significant':        p < 0.05,
                'p_level':            ('p<0.001' if p < 0.001 else
                                       'p<0.01' if p < 0.01 else
                                       'p<0.05' if p < 0.05 else 'n.s.'),
                'effect_size':        round(effect, 3),
                'effect_type':        effect_label,
                'normal_g1':          p_norm1 > 0.05,
                'normal_g2':          p_norm2 > 0.05,
            })

            sig = 'p<0.001' if p < 0.001 else 'p<0.01' if p < 0.01 else 'p<0.05' if p < 0.05 else 'n.s.'
            print(f"    {label}: {group_name} {d1.mean():.1f}±{d1.std(ddof=1):.1f} vs "
                  f"Control {d2.mean():.1f}±{d2.std(ddof=1):.1f} | "
                  f"{test}: {stat:.3f}, p={p:.4f} ({sig}), effect={effect:.3f}")

        # ── Sex distribution ───────────────────────────────────────────────
        for sex_col in ['sex', 'gender', 'Sex', 'Gender']:
            if sex_col not in demo.columns:
                continue

            s1 = g1[sex_col].dropna()
            s2 = g2[sex_col].dropna()

            if len(s1) == 0 or len(s2) == 0:
                continue

            # Build contingency table
            all_vals = sorted(set(s1) | set(s2))
            ct = pd.DataFrame(
                {group_name: s1.value_counts(), 'Control': s2.value_counts()},
                index=all_vals
            ).fillna(0).astype(int)

            # Use Fisher's exact if any cell < 5 and 2×2 table
            if ct.shape == (2, 2) and (ct.values < 5).any():
                stat, p = stats.fisher_exact(ct.values)
                test = "Fisher's exact"
            else:
                stat, p, _, _ = stats.chi2_contingency(ct.values)
                test = "Chi-square"

            # Describe as n (%)
            for group_df, gname in [(g1, group_name), (g2, 'Control')]:
                counts = group_df[sex_col].value_counts()
                desc = ', '.join(f"{v}: {counts.get(v,0)} ({100*counts.get(v,0)/len(group_df):.0f}%)"
                                 for v in all_vals)
                print(f"    Sex {gname}: {desc}")

            sig = 'p<0.001' if p < 0.001 else 'p<0.01' if p < 0.01 else 'p<0.05' if p < 0.05 else 'n.s.'
            print(f"    Sex: {test}: stat={stat:.3f}, p={p:.4f} ({sig})")

            # Add to rows
            for val in all_vals:
                n1 = int((s1 == val).sum())
                n2 = int((s2 == val).sum())
                rows.append({
                    'variable':           f'Sex: {val} (n)',
                    f'{group_name}_mean': n1,
                    f'{group_name}_sd':   f'{100*n1/len(s1):.0f}%',
                    f'{group_name}_n':    len(s1),
                    'Control_mean':       n2,
                    'Control_sd':         f'{100*n2/len(s2):.0f}%',
                    'Control_n':          len(s2),
                    'test':               test,
                    'statistic':          round(stat, 3),
                    'p_value':            round(p, 4),
                    'significant':        p < 0.05,
                    'p_level':            sig,
                    'effect_size':        '',
                    'effect_type':        '',
                    'normal_g1':          '',
                    'normal_g2':          '',
                })
            break  # only process first found sex column

        # ── Save CSV ───────────────────────────────────────────────────────
        if rows:
            df = pd.DataFrame(rows)
            csv_path = save_dir / f"demographics_{task}.csv"
            df.to_csv(csv_path, index=False)
            print(f"  Saved: {csv_path}")