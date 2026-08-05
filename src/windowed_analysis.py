"""
windowed_analysis.py — Time-resolved (windowed) kinectome analysis.
====================================================================

Motivation
----------
Every other module collapses a whole trial into ONE matrix per subject (mean
or std across all gait cycles). That discards how coordination *evolves over
the trial* — exactly where dynamic Parkinson's signatures (destabilisation,
drift, fatigue, dual-task accumulation) are expected to live.

This module keeps the temporal axis. For each subject it takes the ordered
per-gait-cycle kinectomes (already stored, already sorted by onset), slides a
window of consecutive cycles across them, and averages within each window to
get a *trajectory* of matrices through the trial. It then characterises that
trajectory with per-subject scalar metrics and compares them between groups.

Windowing (Option A — cycle blocks, no recomputation)
-----------------------------------------------------
A "window" is a block of consecutive gait cycles. Nothing is recomputed from
raw data — this reads the same .npy kinectomes as every other module, so it
fully respects the immutable-storage principle. Cycles arrive time-ordered
from load_kinectomes (sorted by onset).

Window size is chosen AUTOMATICALLY per subject to target a fixed *number* of
windows, so subjects with longer trials get proportionally larger windows and
everyone yields the same number of trajectory points (making reconfiguration
and drift metrics comparable). Window size is clamped to a minimum so a window
is never a single cycle.

Metrics (per subject, per direction)
------------------------------------
- **reconfiguration_rate** : mean over consecutive window pairs of
  (1 - Spearman rho between their upper triangles). How fast the whole
  coordination pattern reshuffles from window to window. Scalar.
- **mean_temporal_fluctuation** : mean over edges of the std of that edge
  across windows. How much individual links wander over the trial. Scalar.
- **drift_slope** : slope of (Spearman rho to the FIRST window) vs window
  index. Negative => coordination progressively departs from its initial
  configuration over the trial (systematic drift, e.g. fatigue / dual-task
  accumulation). Scalar.
- **temporal_fluctuation_matrix** : per-edge std across windows (for the
  exploratory edge-wise map; NOT the headline test).

Statistics
----------
Headline tests are the three per-subject SCALAR metrics — one Mann-Whitney U
per (task x kinematic x direction), so they are well powered and need no
multiple-comparison penalty beyond the small family of directions. Effect
size is rank-biserial (positive => higher in the clinical group). The
edge-wise fluctuation map is provided as EXPLORATORY only, FDR-corrected, with
the same p-value-distribution diagnostic as std_analysis (so an empty FDR
result is correctly read as a genuine null rather than over-correction).

Full vs directional
-------------------
Works transparently with both. It operates only on whatever matrices
load_kinectomes returns and infers directions from their dimensionality:
a 2D (n x n) array is a full kinectome (one 'full' direction); a 3D
(n x n x 3) array is directional (AP/ML/V). Marker labels are expanded to
per-direction form for full, exactly as elsewhere.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests

from src.data_utils.data_loader import load_kinectomes, exclude_markers_from_kinectome
from src.data_utils.groups import get_matched_groups_for_task


_DIRS3 = ['AP', 'ML', 'V']


# ── small helpers (mirror std_analysis conventions) ──────────────────────────

def _expand_markers(marker_list):
    return [f"{m}_{d}" for m in marker_list for d in _DIRS3]


def _rank_biserial(u_stat, n1, n2):
    """Positive => group 1 tends to have larger values (U from mannwhitneyu(g1,g2))."""
    if n1 == 0 or n2 == 0:
        return np.nan
    return (2.0 * u_stat) / (n1 * n2) - 1.0


def _effect_label(r):
    if r is None or np.isnan(r):
        return 'n/a'
    a = abs(r)
    if a < 0.1:
        return 'negligible'
    if a < 0.3:
        return 'small'
    if a < 0.5:
        return 'medium'
    return 'large'


def _mannwhitney(g1_vals, g2_vals):
    g1 = np.asarray(g1_vals, float); g1 = g1[~np.isnan(g1)]
    g2 = np.asarray(g2_vals, float); g2 = g2[~np.isnan(g2)]
    if len(g1) < 2 or len(g2) < 2:
        return np.nan, np.nan, np.nan
    try:
        u, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
    except ValueError:
        return np.nan, np.nan, np.nan
    return u, p, _rank_biserial(u, len(g1), len(g2))


def _upper(matrix):
    """Strict upper-triangle values of a square matrix, NaNs dropped in pairs
    handled by the caller."""
    n = matrix.shape[0]
    iu, ju = np.triu_indices(n, k=1)
    return matrix[iu, ju]


def _matrix_directions(kinectome):
    """Return list of (direction_label, 2D_matrix) for one stored kinectome,
    handling full (2D -> [('full', M)]) and directional (3D -> AP/ML/V)."""
    arr = np.asarray(kinectome, float)
    if arr.ndim == 2:
        return [('full', arr)]
    return [(_DIRS3[i], arr[:, :, i]) for i in range(arr.shape[2])]


# ── windowing ────────────────────────────────────────────────────────────────

def _auto_window_params(n_cycles, target_windows, min_window, overlap):
    """Choose (window_size, step) so ~target_windows windows tile n_cycles with
    the requested fractional overlap, with window_size >= min_window.

    Returns (window_size, step) or (None, None) if too few cycles.
    """
    if n_cycles < max(min_window * 2, min_window + 1):
        return None, None
    # Solve for window size that yields ~target_windows given the overlap.
    # With step = window*(1-overlap), n_windows ≈ (n_cycles - window)/step + 1.
    # Aim for target_windows; derive window from that, then clamp.
    frac = max(1e-6, 1.0 - overlap)
    # n_windows ≈ (n_cycles - w)/(w*frac) + 1  ->  solve for w
    # (target-1) = (n_cycles - w)/(w*frac)  ->  (target-1)*w*frac = n_cycles - w
    # w * ((target-1)*frac + 1) = n_cycles  ->  w = n_cycles / ((target-1)*frac + 1)
    w = int(round(n_cycles / ((target_windows - 1) * frac + 1)))
    w = max(min_window, min(w, n_cycles))          # clamp
    step = max(1, int(round(w * frac)))
    return w, step


def _windowed_matrices(cycle_matrices, window_size, step):
    """Average consecutive-cycle blocks into a trajectory of window matrices.
    cycle_matrices: list of 2D arrays (one direction). Returns list of 2D arrays."""
    n = len(cycle_matrices)
    windows = []
    start = 0
    while start + window_size <= n:
        block = np.stack(cycle_matrices[start:start + window_size], axis=0)
        windows.append(np.nanmean(block, axis=0))
        start += step
    # Ensure the tail is represented if the last window didn't reach the end
    if windows and (start - step + window_size) < n:
        block = np.stack(cycle_matrices[n - window_size:n], axis=0)
        windows.append(np.nanmean(block, axis=0))
    return windows


# ── trajectory metrics ───────────────────────────────────────────────────────

def _trajectory_metrics(window_mats):
    """Compute scalar trajectory metrics + per-edge fluctuation from a list of
    window matrices (all same shape). Returns dict."""
    if len(window_mats) < 2:
        return None

    uppers = [_upper(m) for m in window_mats]

    # reconfiguration rate: mean (1 - rho) between consecutive windows
    recon = []
    for a, b in zip(uppers[:-1], uppers[1:]):
        mask = ~(np.isnan(a) | np.isnan(b))
        if mask.sum() >= 3 and np.std(a[mask]) > 0 and np.std(b[mask]) > 0:
            rho, _ = stats.spearmanr(a[mask], b[mask])
            recon.append(1.0 - rho)
    reconfiguration_rate = float(np.mean(recon)) if recon else np.nan

    # drift: rho of each window to the first, regressed on window index
    ref = uppers[0]
    rhos, idxs = [], []
    for k, u in enumerate(uppers):
        mask = ~(np.isnan(ref) | np.isnan(u))
        if mask.sum() >= 3 and np.std(ref[mask]) > 0 and np.std(u[mask]) > 0:
            rho, _ = stats.spearmanr(ref[mask], u[mask])
            rhos.append(rho); idxs.append(k)
    if len(idxs) >= 3:
        drift_slope = float(np.polyfit(idxs, rhos, 1)[0])
    else:
        drift_slope = np.nan

    # per-edge temporal fluctuation: std of each edge across windows
    stack = np.stack(window_mats, axis=0)               # (n_win, n, n)
    fluct_matrix = np.nanstd(stack, axis=0)             # (n, n)
    mean_temporal_fluctuation = float(np.nanmean(_upper(fluct_matrix)))

    return {
        'reconfiguration_rate': reconfiguration_rate,
        'drift_slope': drift_slope,
        'mean_temporal_fluctuation': mean_temporal_fluctuation,
        'fluct_matrix': fluct_matrix,
        'n_windows': len(window_mats),
    }


# ── per-subject collection ───────────────────────────────────────────────────

def _collect_subject_trajectories(diagnosis, kinematics_list, task_names,
                                  tracking_systems, runs, pd_on, base_path,
                                  full, correlation_method, marker_list_affect,
                                  target_windows, min_window, overlap):
    """Load ordered per-cycle kinectomes for every matched subject, exclude
    markers, window them, and compute trajectory metrics.

    Returns:
      results[group][sub_id][task][kinematic][direction] = metrics dict
      marker_lists_per_task
      (pk_name, ctrl_name)
      skipped: list of (group, sub_id, task, reason)
    """
    from config import KINECTOME_SAVE_PATH, EXCLUDE_MARKERS_BY_TASK

    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    all_disease = sorted(set(s for ids in task_disease_ids.values() for s in ids))
    all_control = sorted(set(s for ids in task_control_ids.values() for s in ids))
    pk_name = f"{diagnosis[0][10:].capitalize()}"
    ctrl_name = "Control"

    # default per-task marker labels (expanded for full)
    if full:
        default_labels = _expand_markers(marker_list_affect)
    else:
        default_labels = list(marker_list_affect)
    marker_lists_per_task = {t: list(default_labels) for t in task_names}

    results = {pk_name: {}, ctrl_name: {}}
    skipped = []

    for kinematics in kinematics_list:
        for sub_id in all_disease + all_control:
            group = pk_name if sub_id in all_disease else ctrl_name
            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in all_disease:
                            run = None

                        # Per-task matched membership (not all subjects do all tasks)
                        if sub_id in all_disease and sub_id not in task_disease_ids.get(task_name, []):
                            continue
                        if sub_id in all_control and sub_id not in task_control_ids.get(task_name, []):
                            continue

                        kinectomes = load_kinectomes(
                            KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run,
                            kinematics, full, correlation_method
                        )
                        if not kinectomes:
                            continue

                        n_cycles = len(kinectomes)
                        window_size, step = _auto_window_params(
                            n_cycles, target_windows, min_window, overlap
                        )
                        if window_size is None:
                            skipped.append((group, sub_id, task_name,
                                            f"only {n_cycles} cycles (need >= {min_window*2})"))
                            continue

                        # Marker exclusion (full-aware): reduce every cycle matrix,
                        # track the reduced label list once.
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        if exclude:
                            if full:
                                labels = _expand_markers(marker_list_affect)
                                excl = [f"{m}_{d}" for m in exclude for d in _DIRS3]
                            else:
                                labels = list(marker_list_affect)
                                excl = list(exclude)
                            reduced_cycles, red_labels = [], labels
                            for k in kinectomes:
                                kr, red_labels = exclude_markers_from_kinectome(k, labels, excl)
                                reduced_cycles.append(kr)
                            kinectomes = reduced_cycles
                            marker_lists_per_task[task_name] = red_labels

                        # Split each cycle into its direction slices, window each
                        # direction's trajectory independently.
                        # Build: per direction, an ordered list of 2D cycle matrices.
                        per_dir_cycles = {}
                        for cyc in kinectomes:
                            for dlabel, dmat in _matrix_directions(cyc):
                                per_dir_cycles.setdefault(dlabel, []).append(dmat)

                        for dlabel, dcycles in per_dir_cycles.items():
                            window_mats = _windowed_matrices(dcycles, window_size, step)
                            metrics = _trajectory_metrics(window_mats)
                            if metrics is None:
                                skipped.append((group, sub_id, task_name,
                                                f"{dlabel}: <2 windows"))
                                continue
                            metrics['window_size'] = window_size
                            metrics['step'] = step
                            metrics['n_cycles'] = n_cycles
                            (results[group]
                                .setdefault(sub_id, {})
                                .setdefault(task_name, {})
                                .setdefault(kinematics, {})[dlabel]) = metrics

    return results, marker_lists_per_task, (pk_name, ctrl_name), skipped


# ── group comparison ─────────────────────────────────────────────────────────

def _infer_directions(results, groups):
    for g in groups:
        for sub in results.get(g, {}).values():
            for task in sub.values():
                for kin in task.values():
                    if kin:
                        return list(kin.keys())
    return ['full']


def _compare_scalars(results, task_names, kinematics_list, directions, groups, out_dir):
    """Group comparison of the three scalar trajectory metrics."""
    pk, ctrl = groups
    scalar_names = ['reconfiguration_rate', 'mean_temporal_fluctuation', 'drift_slope']
    rows = []

    for kinematic in kinematics_list:
        for task in task_names:
            for direction in directions:
                for metric in scalar_names:
                    g1 = [results[pk][s][task][kinematic][direction][metric]
                          for s in results.get(pk, {})
                          if direction in results[pk].get(s, {}).get(task, {}).get(kinematic, {})]
                    g2 = [results[ctrl][s][task][kinematic][direction][metric]
                          for s in results.get(ctrl, {})
                          if direction in results[ctrl].get(s, {}).get(task, {}).get(kinematic, {})]
                    u, p, r = _mannwhitney(g1, g2)
                    rows.append({
                        'task': task, 'kinematic': kinematic, 'direction': direction,
                        'metric': metric,
                        f'{pk}_median': np.nanmedian(g1) if g1 else np.nan,
                        f'{pk}_n': int(np.sum(~np.isnan(g1))) if g1 else 0,
                        f'{ctrl}_median': np.nanmedian(g2) if g2 else np.nan,
                        f'{ctrl}_n': int(np.sum(~np.isnan(g2))) if g2 else 0,
                        'U': u, 'p_value': p, 'rank_biserial': r,
                        'effect': _effect_label(r),
                        'higher_in': (pk if (not np.isnan(r) and r > 0) else
                                      (ctrl if not np.isnan(r) else 'n/a')),
                    })

    df = pd.DataFrame(rows)
    # FDR across the small family of tests within each kinematic (all
    # task x direction x metric combinations for that kinematic).
    if not df.empty:
        df['p_fdr'] = np.nan
        df['significant_fdr'] = False
        for kin in kinematics_list:
            sel = df[(df['kinematic'] == kin) & df['p_value'].notna()].index
            if len(sel):
                _, pc, _, _ = multipletests(df.loc[sel, 'p_value'], alpha=0.05, method='fdr_bh')
                df.loc[sel, 'p_fdr'] = pc
                df.loc[sel, 'significant_fdr'] = pc < 0.05
        path = out_dir / 'windowed_scalar_metrics.csv'
        df.to_csv(path, index=False)
        print(f"  Saved: {path}")
    return df


def _compare_edgewise_fluctuation(results, marker_lists_per_task, task_names,
                                  kinematics_list, directions, groups, out_dir):
    """EXPLORATORY per-edge temporal-fluctuation comparison, FDR-corrected,
    with the same p-value-distribution diagnostic as std_analysis."""
    pk, ctrl = groups

    for kinematic in kinematics_list:
        for task in task_names:
            markers = marker_lists_per_task.get(task, [])
            for direction in directions:
                pk_mats = [results[pk][s][task][kinematic][direction]['fluct_matrix']
                           for s in results.get(pk, {})
                           if direction in results[pk].get(s, {}).get(task, {}).get(kinematic, {})]
                ctrl_mats = [results[ctrl][s][task][kinematic][direction]['fluct_matrix']
                             for s in results.get(ctrl, {})
                             if direction in results[ctrl].get(s, {}).get(task, {}).get(kinematic, {})]
                if len(pk_mats) < 2 or len(ctrl_mats) < 2:
                    continue

                n = pk_mats[0].shape[0]
                iu, ju = np.triu_indices(n, k=1)
                pk_stack = np.stack(pk_mats, 0)
                ctrl_stack = np.stack(ctrl_mats, 0)

                rows, raw_p = [], []
                for i, j in zip(iu, ju):
                    u, p, r = _mannwhitney(pk_stack[:, i, j], ctrl_stack[:, i, j])
                    rows.append({
                        'task': task, 'kinematic': kinematic, 'direction': direction,
                        'node_i': markers[i] if i < len(markers) else str(i),
                        'node_j': markers[j] if j < len(markers) else str(j),
                        f'{pk}_mean': float(np.nanmean(pk_stack[:, i, j])),
                        f'{ctrl}_mean': float(np.nanmean(ctrl_stack[:, i, j])),
                        'U': u, 'p_value': p, 'rank_biserial': r,
                        'effect': _effect_label(r),
                    })
                    raw_p.append(p)

                finite = [k for k, p in enumerate(raw_p) if p is not None and not np.isnan(p)]
                p_fdr = [np.nan] * len(raw_p)
                if finite:
                    _, corr, _, _ = multipletests([raw_p[k] for k in finite],
                                                  alpha=0.05, method='fdr_bh')
                    for k, idx in enumerate(finite):
                        p_fdr[idx] = corr[k]
                for k, row in enumerate(rows):
                    row['p_fdr'] = float(p_fdr[k]) if not np.isnan(p_fdr[k]) else np.nan
                    row['significant_fdr'] = bool(not np.isnan(p_fdr[k]) and p_fdr[k] < 0.05)

                df = pd.DataFrame(rows).sort_values('p_value', na_position='last')
                path = out_dir / f'windowed_edgewise_fluctuation_{task}_{kinematic}_{direction}.csv'
                df.to_csv(path, index=False)
                n_sig = int(df['significant_fdr'].sum())
                print(f"  Saved: {path}  ({n_sig} edges significant after FDR)")

                valid_p = np.array([p for p in raw_p if p is not None and not np.isnan(p)])
                if valid_p.size:
                    obs = int((valid_p < 0.05).sum())
                    enrich = stats.binomtest(obs, valid_p.size, 0.05,
                                             alternative='greater').pvalue
                    print(f"    [diagnostic] {valid_p.size} edges | p<.05: {obs} "
                          f"(chance ~{0.05*valid_p.size:.0f}) | enrichment p={enrich:.3g}")
                    if enrich > 0.05:
                        print("    [diagnostic] Consistent with chance — empty FDR result is a "
                              "genuine null, not over-correction.")


# ── entry point ──────────────────────────────────────────────────────────────

def windowed_analysis_main(diagnosis, kinematics_list, task_names, tracking_systems,
                           runs, pd_on, base_path, marker_list_affect,
                           result_base_path, full, correlation_method,
                           target_windows=10, min_window=3, overlap=0.5):
    """Time-resolved (windowed) kinectome analysis; call signature mirrors the
    other *_main entry points.

    Parameters
    ----------
    target_windows : int
        Desired number of windows per subject (window size auto-adapts).
    min_window : int
        Minimum cycles per window (a window is never a single cycle).
    overlap : float
        Fractional overlap between consecutive windows (0.5 = 50%).
    """
    out_dir = Path(result_base_path) / 'windowed_analysis'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 64)
    print("TIME-RESOLVED (WINDOWED) KINECTOME ANALYSIS")
    print(f"Type: {'FULL' if full else 'DIRECTIONAL'} | target windows/subject: "
          f"{target_windows} | min window: {min_window} cycles | overlap: {overlap:.0%}")
    print(f"Output: {out_dir}")
    print("=" * 64)

    results, marker_lists_per_task, groups, skipped = _collect_subject_trajectories(
        diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on,
        base_path, full, correlation_method, marker_list_affect,
        target_windows, min_window, overlap
    )
    pk, ctrl = groups

    if skipped:
        print(f"\n  {len(skipped)} subject-task-direction(s) skipped (too few cycles/windows):")
        for g, s, t, why in skipped[:20]:
            print(f"    {g}/{s} {t}: {why}")
        if len(skipped) > 20:
            print(f"    ... and {len(skipped) - 20} more")

    directions = _infer_directions(results, groups)

    # Report how many subjects contributed per group/direction
    print("\n  Subjects with a valid trajectory:")
    for g in (pk, ctrl):
        for task in task_names:
            for kinematic in kinematics_list:
                for d in directions:
                    n = sum(1 for s in results.get(g, {})
                            if d in results[g].get(s, {}).get(task, {}).get(kinematic, {}))
                    print(f"    {g:>10} | {task} | {kinematic} | {d}: {n}")

    print("\n[1/2] Scalar trajectory metrics (headline tests) ...")
    _compare_scalars(results, task_names, kinematics_list, directions, groups, out_dir)

    print("\n[2/2] Edge-wise temporal fluctuation (exploratory) ...")
    _compare_edgewise_fluctuation(results, marker_lists_per_task, task_names,
                                  kinematics_list, directions, groups, out_dir)

    print("\nDone. Read windowed_scalar_metrics.csv first — reconfiguration_rate,")
    print("mean_temporal_fluctuation and drift_slope are the well-powered tests.")
    print("The edge-wise fluctuation CSVs are exploratory (see the diagnostic line).\n")
