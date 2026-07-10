import numpy as np
import pandas as pd
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
import pickle
from pathlib import Path
from src.data_utils.plotting import plot_community_nodal_strength, plot_community_nodal_strength_single_task
from src.data_utils.statistics import analyze_centrality_statistics_by_community
from src.graph_utils.graphs import all_graphs_for_subject


def weighted_degree_centrality(G):
    """Calculates weighted degree centrality for each node in the graph."""
    ## TODO: note that this may not be the best parameter. as weight can be negative and even it out. 
    ## TODO: THis is just as in the paper defined. We could think of different meaniful parameters. 
    
    return {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}

def export_centrality_to_csv(group_centrality_data, save_path=None):
    """
    Export centrality data to a single CSV file with all body segments.
    Each column is named as: {segment}_{task}_{direction}

    Walking-speed tasks are given short aliases (walkPreferred->pref,
    walkSlow->slow, walkFast->fast) to preserve the existing speed-based
    analysis. Any other task (e.g. walkStroop) is passed through using the
    task name itself with the leading 'walk' stripped, so its data is not
    silently dropped.
    """
    if save_path is None:
        save_path = Path("results") / "centrality" / "csv"
   
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
   
    # Get all unique body segments (markers/nodes) across all groups, subjects, tasks, and directions
    all_segments = set()
    all_tasks = set()
    for group in group_centrality_data:
        for sub_id in group_centrality_data[group]:
            for task_name in group_centrality_data[group][sub_id]:
                all_tasks.add(task_name)
                for direction in group_centrality_data[group][sub_id][task_name]:
                    all_segments.update(group_centrality_data[group][sub_id][task_name][direction].keys())
   
    # Walking-speed aliases (preserve existing speed-based analysis)
    speed_aliases = {
        'walkPreferred': 'pref',
        'walkSlow': 'slow',
        'walkFast': 'fast'
    }
    # Build a mapping for every task actually present: known speeds get their
    # alias, everything else keeps its own name (minus a leading 'walk').
    task_mapping = {}
    for task_name in sorted(all_tasks):
        if task_name in speed_aliases:
            task_mapping[task_name] = speed_aliases[task_name]
        else:
            task_mapping[task_name] = task_name.replace('walk', '', 1) if task_name.startswith('walk') else task_name

    # Column-order task prefixes, in a stable order (speeds first, then others)
    ordered_prefixes = [speed_aliases[t] for t in ['walkPreferred', 'walkSlow', 'walkFast'] if t in task_mapping]
    ordered_prefixes += [p for t, p in sorted(task_mapping.items()) if p not in ordered_prefixes]

    # Sort segments for consistent ordering
    all_segments = sorted(all_segments)
   
    rows = []
    
    # Iterate through each group and subject
    for group in group_centrality_data:
        for sub_id in group_centrality_data[group]:
            row = {'group': group, 'subject_id': sub_id}
           
            # For each segment, task, and direction combination
            for segment in all_segments:
                for task_name, task_prefix in task_mapping.items():
                    for direction in ['AP', 'ML', 'V']:
                        col_name = f"{segment}_{task_prefix}_{direction}"
                       
                        # Get the value if it exists, otherwise NaN
                        if (task_name in group_centrality_data[group][sub_id] and
                            direction in group_centrality_data[group][sub_id][task_name] and
                            segment in group_centrality_data[group][sub_id][task_name][direction]):
                           
                            value = group_centrality_data[group][sub_id][task_name][direction][segment]
                            row[col_name] = value
                        else:
                            row[col_name] = np.nan
           
            rows.append(row)
   
    # Create DataFrame
    df = pd.DataFrame(rows)
   
    # Order columns: group, subject_id, then all segment_task_direction combinations
    column_order = ['group', 'subject_id']
    for segment in all_segments:
        for task_prefix in ordered_prefixes:
            for direction in ['AP', 'ML', 'V']:
                column_order.append(f"{segment}_{task_prefix}_{direction}")
   
    df = df[column_order]
   
    # Save to CSV
    csv_filename = "all_segments_centrality.csv"
    df.to_csv(save_path / csv_filename, index=False)
    print(f"Saved {csv_filename}")
   
    print(f"CSV file saved to: {save_path}")
    return save_path, df

def community_weighted_degree_centrality(G, consensus_communities):
    """
    Calculates weighted degree centrality for each node considering only edges within their community.
    
    Parameters:
    G: NetworkX graph
    consensus_communities: List of sets, where each set contains nodes belonging to the same community
    
    Returns:
    Dictionary with node centrality values based on intra-community connections only
    """
    # Create a mapping from node to its community
    node_to_community = {}
    for i, community in enumerate(consensus_communities):
        for node in community:
            node_to_community[node] = i
    
    centrality = {}
    
    for node in G.nodes():
        if node not in node_to_community:
            # If node is not in any community, its centrality is 0
            centrality[node] = 0
            continue
            
        node_community = node_to_community[node]
        community_weight_sum = 0
        
        # Sum weights only for edges to nodes in the same community
        for _, neighbor, weight in G.edges(node, data='weight'):
            if neighbor in node_to_community and node_to_community[neighbor] == node_community:
                community_weight_sum += weight
                
        centrality[node] = community_weight_sum
    
    return centrality

def centrality_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                    correlation_method, interpol, consensus_communities, community_centrality = False):
    
    # Modify pickle path to distinguish between regular and community centrality
    centrality_type = 'community' if community_centrality else 'regular'
    centrality_pickle_path = Path(result_base_path, 'centrality', f'centrality_data_{correlation_method}_{centrality_type}.pkl')
    centrality_pickle_path.parent.mkdir(parents=True, exist_ok=True)

    if not centrality_pickle_path.exists():

        # -------------------------------------------------------------------
        # Subject selection — identical procedure to kinectome_characteristics.py
        # get_matched_groups_for_task is the single source of truth for group
        # matching. Matching is done PER TASK because not all subjects complete
        # all tasks (greedy nearest-neighbour age matching inside groups.py).
        # -------------------------------------------------------------------
        from src.data_utils.groups import get_matched_groups_for_task
        task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)

        # Union across tasks — used only to initialise the data structure.
        # Per-task membership is enforced inside the loop below.
        all_disease_ids = sorted(set(s for ids in task_disease_ids.values() for s in ids))
        all_control_ids = sorted(set(s for ids in task_control_ids.values() for s in ids))
        disease_sub_ids = all_disease_ids  # alias kept for the run-selection logic below

        group_name = f"{diagnosis[0][10:].capitalize()}"

        # Initialize the data structure with subject IDs
        group_centrality_data = {group_name: {}, "Control": {}}
        for sub_id in all_disease_ids:
            group_centrality_data[group_name][sub_id] = {}
        for sub_id in all_control_ids:
            group_centrality_data["Control"][sub_id] = {}

        # Config-driven marker exclusion (e.g. upper limb for dual tasks),
        # applied at analysis time only — never during kinectome computation.
        from config import KINECTOME_SAVE_PATH, EXCLUDE_MARKERS_BY_TASK, MARKER_LIST_AFFECT
        from src.data_utils.data_loader import exclude_markers_from_kinectome

        for kinematics in kinematics_list:
            for sub_id in all_disease_ids + all_control_ids:
                group = group_name if sub_id in all_disease_ids else "Control"

                for tracksys in tracking_systems:
                    for task_name in task_names:
                        # Per-task group membership: skip subjects not matched for this task
                        task_disease = task_disease_ids.get(task_name, [])
                        task_control = task_control_ids.get(task_name, [])
                        if group == group_name and sub_id not in task_disease:
                            continue
                        if group == "Control" and sub_id not in task_control:
                            continue

                        # Initialize task structure for this subject
                        if task_name not in group_centrality_data[group][sub_id]:
                            group_centrality_data[group][sub_id][task_name] = {}

                        for run in runs:
                            if sub_id in pd_on:
                                run = 'on'
                            elif sub_id not in disease_sub_ids:
                                run = None
                            else:
                                run = run

                            kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method, interpol)

                            if kinectomes is None:
                                continue

                            # Strip excluded markers (e.g. upper limb for dual tasks)
                            # and build the task-specific marker list to match.
                            current_markers = MARKER_LIST_AFFECT.copy()
                            exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                            if exclude:
                                reduced = []
                                for k in kinectomes:
                                    k_reduced, current_markers = exclude_markers_from_kinectome(k, current_markers, exclude)
                                    reduced.append(k_reduced)
                                kinectomes = reduced

                            # Build graphs with the marker list that matches the
                            # (possibly reduced) kinectome, not the global one.
                            graphs = all_graphs_for_subject(kinectomes, current_markers, bound_value=None)
                            subject_average_weights = {}

                            for direction in ['AP', 'ML', 'V']:
                                direction_graphs = graphs[direction]
                                if not direction_graphs:
                                    continue
                                total_weights = []
                                for current_graph in direction_graphs:
                                    # Choose centrality calculation method based on community_centrality flag
                                    if community_centrality:
                                        weights = community_weighted_degree_centrality(current_graph, consensus_communities)
                                    else:
                                        weights = weighted_degree_centrality(current_graph)
                                    total_weights.append(weights)
                                if not total_weights:
                                    continue
                                average_weights = {node: np.mean([w[node] for w in total_weights]) for node in total_weights[0]}
                                subject_average_weights[direction] = average_weights

                            # Store individual subject data for each node
                            for direction in ['AP', 'ML', 'V']:
                                if direction not in subject_average_weights:
                                    continue
                                if direction not in group_centrality_data[group][sub_id][task_name]:
                                    group_centrality_data[group][sub_id][task_name][direction] = {}
                                for node in subject_average_weights[direction]:
                                    group_centrality_data[group][sub_id][task_name][direction][node] = subject_average_weights[direction][node]

        # Diagnostic: count how many real (finite) centrality values were stored.
        _n_values = 0
        for grp in group_centrality_data:
            for sid in group_centrality_data[grp]:
                for tsk in group_centrality_data[grp][sid]:
                    for d in group_centrality_data[grp][sid][tsk]:
                        for node, v in group_centrality_data[grp][sid][tsk][d].items():
                            if v is not None and np.isfinite(v):
                                _n_values += 1
        print(f"[centrality] stored {_n_values} finite centrality values "
              f"across {len(group_centrality_data.get(group_name, {}))} {group_name} "
              f"+ {len(group_centrality_data.get('Control', {}))} Control subjects")
        if _n_values == 0:
            print("[centrality] WARNING: no centrality values were computed — "
                  "check that load_kinectomes is finding files and all_graphs_for_subject "
                  "returns non-empty graphs for these subjects/tasks.")

        # Drop subjects/tasks that ended up with no usable data so the CSV and
        # downstream stats only see real observations.
        for grp in list(group_centrality_data.keys()):
            for sid in list(group_centrality_data[grp].keys()):
                for tsk in list(group_centrality_data[grp][sid].keys()):
                    if not group_centrality_data[grp][sid][tsk]:
                        del group_centrality_data[grp][sid][tsk]
                if not group_centrality_data[grp][sid]:
                    del group_centrality_data[grp][sid]

        with open (centrality_pickle_path, 'wb') as centrality_file:
            pickle.dump(group_centrality_data, centrality_file)

    else:
        with open (centrality_pickle_path, 'rb') as centrality_file:
            group_centrality_data = pickle.load(centrality_file)

        # Guard against a stale pickle written by an older run (e.g. the old
        # NaN-filling code): if it holds no finite values, it is unusable and
        # would silently produce an all-NaN CSV and empty statistics.
        _has_values = any(
            v is not None and np.isfinite(v)
            for grp in group_centrality_data.values()
            for sid in grp.values()
            for tsk in sid.values()
            for d in tsk.values()
            for v in d.values()
        )
        if not _has_values:
            raise RuntimeError(
                f"Loaded centrality pickle contains no finite values: "
                f"{centrality_pickle_path}\n"
                f"This is a stale file from a previous run. Delete it and re-run "
                f"so the data is recomputed with the current code."
            )

    csv_save_path, centrality_df = export_centrality_to_csv(group_centrality_data)

    # Diagnostic: how much real data reached the DataFrame the stats will use.
    _value_cols = [c for c in centrality_df.columns if c not in ('group', 'subject_id')]
    _n_finite = int(centrality_df[_value_cols].notna().to_numpy().sum())
    print(f"[centrality] DataFrame: {len(centrality_df)} subjects, "
          f"{_n_finite} finite values across {len(_value_cols)} segment/speed/direction columns")
    if _n_finite == 0:
        print("[centrality] WARNING: DataFrame is all-NaN — no task data reached the "
              "DataFrame. Check that the pickle was rebuilt and that task_names are set.")

    # Determine which task columns are actually present in the DataFrame.
    speed_prefixes = ['pref', 'slow', 'fast']
    present_prefixes = set()
    for c in _value_cols:
        # column format: {segment}_{task_prefix}_{direction}
        parts = c.rsplit('_', 2)
        if len(parts) == 3:
            present_prefixes.add(parts[1])

    speed_present = [p for p in speed_prefixes if p in present_prefixes]
    other_present = sorted(p for p in present_prefixes if p not in speed_prefixes)

    # --- Speed-based analysis (walkPreferred / walkSlow / walkFast) ---
    # Only run the speed-oriented statistics + plots if speed columns exist.
    if speed_present:
        results = analyze_centrality_statistics_by_community(
            centrality_df, consensus_communities, alpha=0.05,
            correction_methods=['fdr_bh'], n_bootstrap=1000, use_bootstrap=False
        )
        # TO DO: save dir as global variable
        plot_community_nodal_strength(
            centrality_df, consensus_communities, results,
            save_dir=Path(result_base_path) / "community_plots"
        )

    # --- Single-condition tasks (e.g. walkStroop): no speed axis ---
    # These have one box per group per segment and only between-group tests.
    for task_prefix in other_present:
        plot_community_nodal_strength_single_task(
            centrality_df, consensus_communities, task_prefix,
            save_dir=Path(result_base_path) / "community_plots" / task_prefix
        )

    return