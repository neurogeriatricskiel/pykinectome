"""
patterns.py — Dominant inter-segmental coordination pattern analysis.
======================================================================

This module identifies the strongest coordination patterns in kinectome
graphs and compares their strength between two groups.

Pipeline
--------
1. Load kinectomes for all subjects (once, before any looping).
2. For each (pattern_length, start_node) combination:
   a. Extract the strongest path subgraph per subject.
   b. Find the most common (modal) path in the reference group.
   c. Evaluate that pattern's strength on every subject in both groups.
3. Run statistical comparison between groups.
4. Save all results to a pickle file.

On subsequent runs, if the pickle file already exists it is loaded directly
and the computation is skipped. The summary CSV is always regenerated from
the pickle so you can adjust thresholds without recomputing.

Pickle filename convention
--------------------------
``patterns_<type>_<ref_group>_<task>_<direction>_<correlation>.pkl``

e.g. ``patterns_max_weight_Control_walkStroop_AP_pears.pkl``

This encodes the pattern type, reference group, task, direction, and
correlation method. Different combinations each get their own file.

Configuration (set in config.py)
---------------------------------
PATTERN_TYPE : str
    How the strongest path is selected. ``"max_weight"`` = maximum sum of
    edge weights. Other types can be added to this file.
PATTERN_REFERENCE_GROUP : str
    Which group's modal pattern is used as the template.
    ``"Control"`` → use control group patterns to compare both groups.
    ``"Parkinson"`` → use clinical group patterns.
PATTERN_TASK : str
    Task to use for pattern analysis (must be in TASK_NAMES).
PATTERN_DIRECTION : str
    Movement direction for statistical comparison: ``"AP"``, ``"ML"``, ``"V"``.
    Patterns are always extracted for all three directions; this controls
    which direction is compared statistically.
PATTERN_MIN_LENGTH : int
    Minimum path length to search (default 2).
PATTERN_MAX_LENGTH : int
    Maximum path length to search (default 20).
"""

from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
from src.graph_utils import kinectome2pattern
from src.graph_utils.graphs import build_graph
from src.kinectome_characteristics import calc_std_avg_matrices
import numpy as np
from collections import defaultdict, Counter
import networkx as nx
import pandas as pd
from pathlib import Path
import pickle
from scipy import stats


# ──────────────────────────────────────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────────────────────────────────────

def jaccard_similarity(edges1, edges2):
    """Jaccard similarity between two pattern edge sets.

    Patterns with Jaccard ≥ 0.8 are considered the same pattern
    (Sägner et al. 2024).

    Parameters
    ----------
    edges1, edges2 : list[tuple]
        Edge lists as ``(u, v)`` tuples.

    Returns
    -------
    float
        Jaccard similarity in [0, 1].
    """
    set1 = {(min(u, v), max(u, v)) for u, v in edges1}
    set2 = {(min(u, v), max(u, v)) for u, v in edges2}
    if not set1 and not set2:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ──────────────────────────────────────────────────────────────────────────────
# Core pattern functions
# ──────────────────────────────────────────────────────────────────────────────

def get_pattern_for_subject(all_kinectomes, marker_list, full,
                             pattern_length, start_node):
    """Extract the strongest coordination pattern per subject.

    For each subject, builds a weighted graph from their average kinectome
    and extracts the strongest path of ``pattern_length`` edges starting
    from ``start_node`` using a greedy maximum-weight traversal.

    Parameters
    ----------
    all_kinectomes : dict
        ``{group: {sub_id: {task: {kinematic: {direction: {'avg': matrix}}}}}}``.
    marker_list : list[str]
        Ordered marker names (after exclusion).
    full : bool
        Whether kinectomes are full combined-direction matrices.
    pattern_length : int
        Number of edges in the path.
    start_node : str
        Starting marker node for the path search.

    Returns
    -------
    dict
        ``{group: {key: [{'edges': ..., 'nodes': ..., 'graph': ...}]}}``
        where ``key`` = ``sub_id_task_kinematic_direction``.
    """
    subject_patterns = defaultdict(lambda: defaultdict(list))

    for group in all_kinectomes:
        for sub_id in all_kinectomes[group]:
            for task in all_kinectomes[group][sub_id]:
                for kinematic in all_kinectomes[group][sub_id][task]:
                    dirs = all_kinectomes[group][sub_id][task][kinematic]

                    if full and dirs.get('full') is not None:
                        graphs = build_graph(dirs['full']['avg'], marker_list)
                        direction_labels = ['full']
                    elif not full and all(
                        dirs.get(d) is not None for d in ['AP', 'ML', 'V']
                    ):
                        combined = np.stack(
                            [dirs['AP']['avg'], dirs['ML']['avg'], dirs['V']['avg']],
                            axis=-1
                        )
                        graphs = build_graph(combined, marker_list)
                        direction_labels = ['AP', 'ML', 'V']
                    else:
                        continue

                    for idx, G in enumerate(graphs):
                        direction = direction_labels[idx]
                        try:
                            pattern_graph = kinectome2pattern.strongest_pattern_subgraph(
                                G, length=pattern_length, start_node=start_node
                            )
                            key = f"{sub_id}_{task}_{kinematic}_{direction}"
                            subject_patterns[group][key].append({
                                'edges': list(pattern_graph.edges(data=True)),
                                'nodes': list(pattern_graph.nodes()),
                                'graph': pattern_graph,
                            })
                        except Exception:
                            continue

    return subject_patterns


def get_avg_group_patterns(subject_patterns, pattern_length, start_node):
    """Find the most common (modal) pattern across subjects in each group.

    Every subject always has a strongest path of any length because the greedy
    traversal always finds the next highest-weight edge. This function counts
    how many subjects share each unique path (by edge set) and returns the
    most frequently occurring one as the group's representative pattern.

    Parameters
    ----------
    subject_patterns : dict
        Output of :func:`get_pattern_for_subject`.
    pattern_length : int
        Number of edges in the path.
    start_node : str
        Starting marker node.

    Returns
    -------
    dict
        ``{group: {'edges': ..., 'nodes': ..., 'graph': ...,
        'n_subjects': int, 'frequency': int}}``.
        ``frequency`` = how many subjects had this exact path.
    """
    group_patterns = {}

    for group, subjects in subject_patterns.items():
        if not subjects:
            continue

        # Count occurrences per unique subject (key = sub_id_task_kin_direction,
        # so same subject appears 3× for AP/ML/V — count sub_id only once per path)
        path_subjects = {}   # canonical -> set of sub_ids
        path_store    = {}   # canonical -> one example pattern

        for key, pattern_list in subjects.items():
            sub_id = key.split('_')[0]
            for pattern in pattern_list:
                canonical = frozenset(
                    (min(u, v), max(u, v))
                    for u, v, _ in pattern['edges']
                )
                if canonical not in path_subjects:
                    path_subjects[canonical] = set()
                path_subjects[canonical].add(sub_id)
                path_store[canonical] = pattern

        if not path_subjects:
            continue

        # Most common path = shared by most unique subjects
        most_common_key = max(path_subjects, key=lambda k: len(path_subjects[k]))
        frequency  = len(path_subjects[most_common_key])
        n_subjects = len({key.split('_')[0] for key in subjects.keys()})
        best_pattern = path_store[most_common_key]

        group_patterns[group] = {
            'edges':      best_pattern['edges'],
            'nodes':      best_pattern['nodes'],
            'graph':      best_pattern['graph'],
            'n_subjects': n_subjects,
            'frequency':  frequency,
        }

    return group_patterns


def get_pattern_values_for_subjects(all_kinectomes, group_patterns, full,
                                     marker_list, result_base_path,
                                     pattern_length, start_node, save_csv=False):
    """Evaluate the group pattern strength on every subject.

    For each group-level reference pattern, builds each subject's kinectome
    graph and sums the weights of the edges that form the reference pattern
    path.  This gives one ``Path_Sum`` value per subject per direction,
    which is then compared statistically between groups.

    Parameters
    ----------
    all_kinectomes : dict
        Nested kinectome dict.
    group_patterns : dict
        Output of :func:`get_avg_group_patterns`.
    full : bool
        Whether kinectomes are full matrices.
    marker_list : list[str]
        Ordered marker names.
    result_base_path : str or Path
        Root results directory.
    pattern_length : int
        Number of edges in the path.
    start_node : str
        Starting marker node.
    save_csv : bool, optional
        Save per-combination CSV (default False).

    Returns
    -------
    pd.DataFrame
        Columns: ``Pattern_Group``, ``Subject_Group``, ``Subject``, ``Task``,
        ``Kinematics``, ``Direction``, ``Pattern``, ``Path_Sum``,
        ``Weakest_Link``.
    """
    pattern_values_data = []

    for pattern_group, group_pattern in group_patterns.items():
        group_pattern_edges = [(u, v) for u, v, _ in group_pattern['edges']]
        pattern_name = ' → '.join(group_pattern['nodes'])

        for subject_group in all_kinectomes:
            for sub_id in all_kinectomes[subject_group]:
                for task in all_kinectomes[subject_group][sub_id]:
                    for kinematic in all_kinectomes[subject_group][sub_id][task]:
                        dirs = all_kinectomes[subject_group][sub_id][task][kinematic]

                        direction_list = (
                            ['full'] if full and dirs.get('full')
                            else [d for d in ['AP', 'ML', 'V'] if dirs.get(d) is not None]
                        )

                        for direction in direction_list:
                            matrix = dirs[direction]['avg'] if dirs[direction] else None
                            if matrix is None:
                                continue

                            G = build_graph(
                                np.expand_dims(matrix, -1) if matrix.ndim == 2 else matrix,
                                marker_list
                            )[0]

                            edge_weights = []
                            edge_weights_with_nodes = []
                            missing_edges = []

                            for u, v in group_pattern_edges:
                                if G.has_edge(u, v):
                                    w = G[u][v].get('weight', 0)
                                    edge_weights.append(w)
                                    edge_weights_with_nodes.append((w, (u, v)))
                                else:
                                    missing_edges.append((u, v))

                            if not missing_edges and edge_weights:
                                path_sum = sum(edge_weights)
                                _, weakest_edge = min(
                                    edge_weights_with_nodes, key=lambda x: x[0]
                                )
                                weakest_link = f"{weakest_edge[0]}-{weakest_edge[1]}"
                            else:
                                path_sum = np.nan
                                weakest_link = None

                            pattern_values_data.append({
                                'Pattern_Group':  pattern_group,
                                'Subject_Group':  subject_group,
                                'Subject':        sub_id,
                                'Task':           task,
                                'Kinematics':     kinematic,
                                'Direction':      direction,
                                'Pattern':        pattern_name,
                                'Path_Sum':       path_sum,
                                'Weakest_Link':   weakest_link,
                            })

    df = pd.DataFrame(pattern_values_data)

    if save_csv and not df.empty:
        save_path = Path(result_base_path) / 'patterns'
        save_path.mkdir(parents=True, exist_ok=True)
        df.to_csv(
            save_path / f"path_values_length_{pattern_length}_start_{start_node}.csv",
            index=False
        )

    return df


def compare_groups_statistical(pattern_values_df, pattern_group,
                                subject_group1, subject_group2,
                                kinematics=None, direction=None, task=None):
    """Compare pattern strength between two groups statistically.

    Automatically selects t-test (normal data) or Mann-Whitney U (otherwise),
    checked per group with Shapiro-Wilk.  Effect size: Cohen's d (t-test) or
    rank-biserial correlation (Mann-Whitney U), interpreted as small/medium/
    large following Cohen (1988): 0.2 / 0.5 / 0.8.

    Parameters
    ----------
    pattern_values_df : pd.DataFrame
        Output of :func:`get_pattern_values_for_subjects`.
    pattern_group : str
        Reference group whose pattern is used (e.g. ``'Control'``).
    subject_group1 : str
        First group (e.g. ``'Control'``).
    subject_group2 : str
        Second group (e.g. ``'Parkinson'``).
    kinematics, direction, task : str, optional
        Filters applied before comparison.

    Returns
    -------
    dict or None
        Statistical results, or None if fewer than 3 subjects in either group.
    """
    def _filter(df, group):
        mask = ((df['Pattern_Group'] == pattern_group) &
                (df['Subject_Group'] == group))
        if kinematics:
            mask &= df['Kinematics'] == kinematics
        if direction:
            mask &= df['Direction'] == direction
        if task:
            mask &= df['Task'] == task
        return df[mask]

    data1 = _filter(pattern_values_df, subject_group1)['Path_Sum'].dropna()
    data2 = _filter(pattern_values_df, subject_group2)['Path_Sum'].dropna()

    if len(data1) < 3 or len(data2) < 3:
        return None

    raw1 = pattern_values_df[
        (pattern_values_df['Pattern_Group'] == pattern_group) &
        (pattern_values_df['Subject_Group'] == subject_group1)
    ]['Weakest_Link'].value_counts()
    raw2 = pattern_values_df[
        (pattern_values_df['Pattern_Group'] == pattern_group) &
        (pattern_values_df['Subject_Group'] == subject_group2)
    ]['Weakest_Link'].value_counts()

    weakest1 = raw1.index[0] if not raw1.empty else 'N/A'
    weakest2 = raw2.index[0] if not raw2.empty else 'N/A'

    pattern_name = (pattern_values_df[
        pattern_values_df['Pattern_Group'] == pattern_group
    ]['Pattern'].iloc[0] if not pattern_values_df.empty else 'N/A')

    _, p_norm1 = stats.shapiro(data1)
    _, p_norm2 = stats.shapiro(data2)
    normal = (p_norm1 > 0.05) and (p_norm2 > 0.05)

    if normal and len(data1) > 5 and len(data2) > 5:
        _, p_lev = stats.levene(data1, data2)
        stat, p = stats.ttest_ind(data1, data2, equal_var=(p_lev > 0.05))
        test_type = f"t-test (equal_var={p_lev > 0.05})"
        pooled_std = np.sqrt(
            ((len(data1) - 1) * data1.std(ddof=1) ** 2 +
             (len(data2) - 1) * data2.std(ddof=1) ** 2) /
            (len(data1) + len(data2) - 2)
        )
        effect = abs(data1.mean() - data2.mean()) / pooled_std if pooled_std > 0 else np.nan
    else:
        stat, p = stats.mannwhitneyu(data1, data2, alternative='two-sided')
        test_type = "Mann-Whitney U"
        effect = 1 - (2 * stat) / (len(data1) * len(data2))

    sig = ('p<0.001' if p < 0.001 else 'p<0.01' if p < 0.01
           else 'p<0.05' if p < 0.05 else 'n.s.')
    print(f"  {subject_group1} (n={len(data1)}) vs {subject_group2} (n={len(data2)}) | "
          f"{pattern_group} pattern | {task} {direction} {kinematics}: "
          f"stat={stat:.3f}, p={p:.4f} ({sig}), effect={effect:.3f}")

    return {
        'pattern_group':   pattern_group,
        'subject_group1':  subject_group1,
        'subject_group2':  subject_group2,
        'task':            task,
        'kinematics':      kinematics,
        'direction':       direction,
        'pattern':         pattern_name,
        'weakest_link_g1': weakest1,
        'weakest_link_g2': weakest2,
        'test_type':       test_type,
        'statistic':       stat,
        'p_value':         round(p, 4),
        'normal':          normal,
        'g1_mean':         round(data1.mean(), 3),
        'g1_std':          round(data1.std(ddof=1), 3),
        'g1_n':            len(data1),
        'g2_mean':         round(data2.mean(), 3),
        'g2_std':          round(data2.std(ddof=1), 3),
        'g2_n':            len(data2),
        'effect_size':     round(effect, 3),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────────────

def patterns_main(marker_list_affect, diagnosis, kinematics_list, task_names,
                  tracking_systems, run, pd_on, base_path, result_base_path,
                  full, correlation, interpol):
    """Run the full pattern analysis pipeline.

    Loads kinectomes once, then for every (pattern_length, start_node):
    1. Extracts each subject's strongest path (greedy max-weight traversal).
    2. Finds the modal (most common) path in the reference group.
    3. Scores that path on every subject in both groups.
    4. Compares scores statistically per direction.

    Results are saved to a pickle. If the pickle already exists, computation
    is skipped. The summary CSV is always regenerated from the pickle.

    Parameters
    ----------
    marker_list_affect : list[str]
        Marker names in affected-side order.
    diagnosis : list[str]
        Diagnosis column name(s) from the demographics file.
    kinematics_list : list[str]
        Kinematic types (e.g. ``['acc']``).
    task_names : list[str]
        Tasks to include.
    tracking_systems : list[str]
        Tracking systems (e.g. ``['omc']``).
    run : list[str]
        Medication states (e.g. ``['on']``).
    pd_on : list[str]
        PD subject IDs measured in ON state without run token in filename.
    base_path : str or Path
        Root data directory.
    result_base_path : str or Path
        Root results directory. Outputs go to ``result_base_path/patterns/``.
    full : bool
        Use full combined-direction kinectomes.
    correlation : str
        Correlation method (e.g. ``'pears'``).
    interpol : bool
        Kinectomes built from interpolated gait cycles.
    """
    from config import (
        PATTERN_REFERENCE_GROUP,
        PATTERN_TASK,
        PATTERN_DIRECTION,
        PATTERN_MIN_LENGTH,
        PATTERN_MAX_LENGTH,
        PATTERN_TYPE,
        KINECTOME_SAVE_PATH,
        EXCLUDE_MARKERS_BY_TASK,
    )

    save_dir = Path(result_base_path) / 'patterns'
    save_dir.mkdir(parents=True, exist_ok=True)

    pickle_path = save_dir / (
        f"patterns_{PATTERN_TYPE}_{PATTERN_REFERENCE_GROUP}_{PATTERN_TASK}_"
        f"{PATTERN_DIRECTION}_{correlation}.pkl"
    )

    # Effective markers (after task-specific exclusion) — needed in both branches
    exclude = EXCLUDE_MARKERS_BY_TASK.get(PATTERN_TASK, [])
    effective_markers = [m for m in marker_list_affect if m not in exclude]

    # ── Load from pickle if it exists ────────────────────────────────────────
    if pickle_path.exists():
        print(f"Loading existing patterns from: {pickle_path.name}")
        with open(pickle_path, 'rb') as fh:
            all_results = pickle.load(fh)
        print(f"  Loaded {len(all_results)} pattern combinations.")

    else:
        # ── Compute from scratch ──────────────────────────────────────────────
        print("Pattern pickle not found — computing patterns...")
        print(f"  Reference group : {PATTERN_REFERENCE_GROUP}")
        print(f"  Task            : {PATTERN_TASK}")
        print(f"  Direction       : {PATTERN_DIRECTION}")
        print(f"  Pattern type    : {PATTERN_TYPE}")
        print(f"  Pickle will be  : {pickle_path.name}")
        if exclude:
            print(f"  Excluding markers for {PATTERN_TASK}: {exclude}")

        # Load kinectomes once — reused for all (length, start_node) combinations.
        # calc_std_avg_matrices also re-matches group sizes per task based on
        # who actually has kinectomes available.
        print("  Loading kinectomes...")
        result = calc_std_avg_matrices(
                diagnosis, kinematics_list, task_names, tracking_systems, run,
                pd_on, base_path, full, correlation, interpol
            )
        # calc_std_avg_matrices returns 2 or 4 values depending on version
        all_kinectomes = result[0]
        marker_lists_per_task = result[1]

        # Filter to the chosen task, keeping all three directions.
        # Subjects without all three directions are excluded.
        task_kinectomes = {}
        for group in all_kinectomes:
            task_kinectomes[group] = {}
            for sub_id, sub_data in all_kinectomes[group].items():
                if PATTERN_TASK not in sub_data:
                    continue
                for kin in sub_data[PATTERN_TASK]:
                    dirs = sub_data[PATTERN_TASK][kin]
                    if all(dirs.get(d) is not None for d in ['AP', 'ML', 'V']):
                        task_kinectomes[group][sub_id] = {
                            PATTERN_TASK: {kin: dirs}
                        }

        n_per_group = {g: len(task_kinectomes[g]) for g in task_kinectomes}
        print(f"  Subjects with all directions: {n_per_group}")

        # Determine group names
        group_names  = list(all_kinectomes.keys())
        ref_group    = PATTERN_REFERENCE_GROUP
        other_groups = [g for g in group_names if g != ref_group]
        compare_group = other_groups[0] if other_groups else group_names[0]

        # Loop over all (pattern_length, start_node) combinations
        all_results = {}
        total = (PATTERN_MAX_LENGTH - PATTERN_MIN_LENGTH + 1) * len(effective_markers)
        done  = 0

        for pattern_length in range(PATTERN_MIN_LENGTH, PATTERN_MAX_LENGTH + 1):
            for start_node in effective_markers:
                done += 1
                if done % 50 == 0:
                    print(f"  Progress: {done}/{total} combinations...")

                subject_patterns = get_pattern_for_subject(
                    task_kinectomes, effective_markers, full,
                    pattern_length, start_node
                )
                group_patterns = get_avg_group_patterns(
                    subject_patterns, pattern_length, start_node
                )

                if not group_patterns:
                    continue

                pattern_values_df = get_pattern_values_for_subjects(
                    task_kinectomes, group_patterns, full,
                    effective_markers, result_base_path,
                    pattern_length, start_node, save_csv=False
                )

                if pattern_values_df.empty:
                    continue

                # Compare within each direction separately
                for direction in ['AP', 'ML', 'V']:
                    result = compare_groups_statistical(
                        pattern_values_df,
                        pattern_group=ref_group,
                        subject_group1=ref_group,
                        subject_group2=compare_group,
                        kinematics=kinematics_list[0],
                        direction=direction,
                        task=PATTERN_TASK,
                    )
                    if result is not None:
                        result['pattern_edges_raw'] = group_patterns.get(
                            ref_group, {}
                        ).get('edges', [])
                        result['pattern_frequency'] = group_patterns.get(
                            ref_group, {}
                        ).get('frequency', 0)
                        result['pattern_n_subjects'] = group_patterns.get(
                            ref_group, {}
                        ).get('n_subjects', 0)
                        all_results[(pattern_length, start_node, direction)] = result

        with open(pickle_path, 'wb') as fh:
            pickle.dump(all_results, fh)
        print(f"  Saved {len(all_results)} results to {pickle_path.name}")

    # ── Summary CSV (always regenerated) ─────────────────────────────────────
    if all_results:
        rows = []
        for key, r in all_results.items():
            pattern_length, start_node, direction = key
            effect = r.get('effect_size', float('nan'))
            try:
                effect_label = ('large'      if abs(effect) >= 0.8
                                else 'medium' if abs(effect) >= 0.5
                                else 'small'  if abs(effect) >= 0.2
                                else 'negligible')
            except Exception:
                effect_label = ''

            g1 = r.get('subject_group1', 'g1')
            g2 = r.get('subject_group2', 'g2')
            rows.append({
                'pattern_length':    pattern_length,
                'start_node':        start_node,
                'direction':         direction,
                'task':              r.get('task', ''),
                'kinematics':        r.get('kinematics', ''),
                'pattern_group':     r.get('pattern_group', ''),
                'pattern_nodes':     r.get('pattern', ''),
                'pattern_frequency': r.get('pattern_frequency', ''),
                'pattern_n_subjects':r.get('pattern_n_subjects', ''),
                'weakest_link_g1':   r.get('weakest_link_g1', ''),
                'weakest_link_g2':   r.get('weakest_link_g2', ''),
                'test_type':         r.get('test_type', ''),
                'statistic':         r.get('statistic', ''),
                'p_value_raw':       r.get('p_value', ''),
                'p_value_bonf':      '',
                'significant_raw':   r.get('p_value', 1.0) < 0.05,
                'significant_bonf':  False,
                'p_level_raw':       ('p<0.001' if r.get('p_value', 1) < 0.001
                                      else 'p<0.01'  if r.get('p_value', 1) < 0.01
                                      else 'p<0.05'  if r.get('p_value', 1) < 0.05
                                      else 'n.s.'),
                'p_level_bonf':      '',
                'effect_size':       effect,
                'effect_label':      effect_label,
                'normal_dist':       r.get('normal', ''),
                'jaccard_overlap':   '',
                'jaccard_partner':   '',
                f'{g1}_mean':        r.get('g1_mean', ''),
                f'{g1}_std':         r.get('g1_std', ''),
                f'{g1}_n':           r.get('g1_n', ''),
                f'{g2}_mean':        r.get('g2_mean', ''),
                f'{g2}_std':         r.get('g2_std', ''),
                f'{g2}_n':           r.get('g2_n', ''),
            })

        summary_df = pd.DataFrame(rows)

        # Bonferroni correction within same (pattern_length, direction)
        # n = number of start nodes tested (after marker exclusion)
        n_start_nodes = len(effective_markers)
        for (length, direction), grp_idx in summary_df.groupby(
                ['pattern_length', 'direction']).groups.items():
            p_raw  = summary_df.loc[grp_idx, 'p_value_raw'].astype(float)
            p_bonf = (p_raw * n_start_nodes).clip(upper=1.0)
            summary_df.loc[grp_idx, 'p_value_bonf']     = p_bonf.round(4)
            summary_df.loc[grp_idx, 'significant_bonf'] = p_bonf < 0.05
            summary_df.loc[grp_idx, 'p_level_bonf'] = p_bonf.apply(
                lambda p: ('p<0.001' if p < 0.001
                           else 'p<0.01'  if p < 0.01
                           else 'p<0.05'  if p < 0.05
                           else 'n.s.')
            )

        # Jaccard overlap between patterns of same (pattern_length, direction)
        edge_lookup = {
            (pl, sn, d): [(u, v) for u, v, _ in r.get('pattern_edges_raw', [])]
            for (pl, sn, d), r in all_results.items()
        }

        for i, row_i in summary_df.iterrows():
            pl, d, sn_i = row_i['pattern_length'], row_i['direction'], row_i['start_node']
            edges_i = edge_lookup.get((pl, sn_i, d), [])
            if not edges_i:
                continue
            partners = []
            for j, row_j in summary_df.iterrows():
                if (i == j or row_j['pattern_length'] != pl
                        or row_j['direction'] != d):
                    continue
                edges_j = edge_lookup.get((pl, row_j['start_node'], d), [])
                jac = jaccard_similarity(edges_i, edges_j)
                if jac >= 0.8:
                    partners.append(f"{row_j['start_node']}(J={jac:.2f})")
            summary_df.at[i, 'jaccard_overlap'] = bool(partners)
            summary_df.at[i, 'jaccard_partner'] = '; '.join(partners)

        summary_df = summary_df.sort_values(
            ['direction', 'pattern_length', 'p_value_raw']
        )

        csv_path = save_dir / (
            pickle_path.stem.replace('patterns_', 'summary_') + '.csv'
        )
        summary_df.to_csv(csv_path, index=False)

        n_sig_raw  = summary_df['significant_raw'].sum()
        n_sig_bonf = summary_df['significant_bonf'].sum()
        print(f"\n  Summary CSV : {csv_path.name}")
        print(f"  Total comparisons       : {len(summary_df)}")
        print(f"  Significant uncorrected : {n_sig_raw}")
        print(f"  Significant Bonferroni  : {n_sig_bonf}")
        if n_sig_bonf > 0:
            print("  Top results (Bonferroni):")
            for _, row in summary_df[summary_df['significant_bonf']].head(5).iterrows():
                print(f"    len={int(row['pattern_length'])}, "
                      f"start={row['start_node']}, dir={row['direction']}: "
                      f"p_bonf={row['p_value_bonf']:.4f}, "
                      f"effect={row['effect_size']:.3f} ({row['effect_label']}), "
                      f"pattern={row['pattern_nodes']}, "
                      f"freq={row['pattern_frequency']}/{row['pattern_n_subjects']} subjects")

    return all_results