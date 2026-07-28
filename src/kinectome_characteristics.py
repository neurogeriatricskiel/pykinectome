from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups, permutation, plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests
from numpy.linalg import norm
import pandas as pd
import os
import random
from src.data_utils.permutation import bootstrap_permutation_test, permutation_test_one_p


def _infer_directions(variability_scores):
    """Return the direction keys present in the data: ['full'] for full
    kinectomes, ['AP','ML','V'] for directional ones. Falls back to the
    directional default if the structure can't be inspected."""
    for group in variability_scores.values():
        for sub_data in group.values():
            for task_data in sub_data.values():
                for kin_data in task_data.values():
                    if kin_data:
                        return list(kin_data.keys())
    return ['AP', 'ML', 'V']


def calc_std_avg_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method):
    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)

    # Use first task's groups as the baseline for variability_scores structure
    # (per-task filtering applied inside the loop below)
    all_disease_ids  = sorted(set(s for ids in task_disease_ids.values()  for s in ids))
    all_control_ids  = sorted(set(s for ids in task_control_ids.values()  for s in ids))

    # Keep these as aliases so inner loop references still work
    disease_sub_ids        = all_disease_ids
    matched_control_sub_ids = all_control_ids

    # Choose what to store based on the `full` flag
    direction_template = {"full": None} if full else {"AP": None, "ML": None, "V": None}

    # Store variability scores structured per subject, task, and direction
    variability_scores = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: direction_template.copy() for kinematics in kinematics_list}
                                                       for task in task_names}
                                                       for sub_id in all_disease_ids},
        "Control": {sub_id: {task: {kinematics: direction_template.copy() for kinematics in kinematics_list}
                             for task in task_names}
                             for sub_id in all_control_ids},
    }

    debug_ids = ['pp006', 'pp008']

    # Track per-task marker lists (may be reduced if EXCLUDE_MARKERS_BY_TASK is set).
    # For full kinectomes, labels span the three directions (marker_AP/ML/V) so
    # they match the (n_markers*3) matrix; for directional, they are the base markers.
    from config import MARKER_LIST_AFFECT as _default_markers
    if full:
        _default_labels = [f"{m}_{d}" for m in _default_markers for d in ['AP', 'ML', 'V']]
    else:
        _default_labels = _default_markers.copy()
    marker_lists_per_task = {task: _default_labels.copy() for task in task_names}

    # --- Kinectome accounting ---------------------------------------------
    # Report where kinectomes are loaded from and how many are used per
    # group/task, plus a grand total, so it's clear what feeds the analysis.
    from config import KINECTOME_SAVE_PATH as _kin_src
    print(f"\nLoading kinectomes from: {_kin_src}")
    print(f"Kinectome type: {'FULL' if full else 'DIRECTIONAL'} | correlation: {correlation_method}\n")
    # counts[group][task] = [n_subjects_with_data, n_kinectomes_total]
    kinectome_counts = {}

    for kinematics in kinematics_list:
        for sub_id in all_disease_ids + all_control_ids:
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in all_disease_ids else "Control"

            # for sub_id in debug_ids:
            #     group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"

            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on: # those sub ids which are measured in 'on' condition but there is no 'run-on' in the filename
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run
                        
                        from config import KINECTOME_SAVE_PATH, EXCLUDE_MARKERS_BY_TASK
                        kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)

                        # Tally how many kinectomes (gait cycles) this subject contributes
                        n_this = len(kinectomes) if kinectomes else 0
                        if n_this > 0:
                            kinectome_counts.setdefault(group, {}).setdefault(task_name, [0, 0])
                            kinectome_counts[group][task_name][0] += 1
                            kinectome_counts[group][task_name][1] += n_this
                            print(f"  {group:>10} | {sub_id} | {task_name} | run={run} | {n_this} kinectomes")

                        # Strip excluded markers (e.g. upper limb for dual tasks)
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        if kinectomes and exclude:
                            from config import MARKER_LIST_AFFECT
                            from src.data_utils.data_loader import exclude_markers_from_kinectome

                            if full:
                                # Full kinectome: rows/cols are markers expanded over the
                                # three directions (marker_AP, marker_ML, marker_V), so both
                                # the label list and the exclude list must be expanded to
                                # match the (n_markers*3) matrix dimension.
                                _dirs = ['AP', 'ML', 'V']
                                base_markers = MARKER_LIST_AFFECT.copy()
                                expanded_markers = [f"{m}_{d}" for m in base_markers for d in _dirs]
                                expanded_exclude = [f"{m}_{d}" for m in exclude for d in _dirs]
                                reduced = []
                                current_markers = expanded_markers
                                for k in kinectomes:
                                    k_reduced, current_markers = exclude_markers_from_kinectome(
                                        k, current_markers, expanded_exclude)
                                    reduced.append(k_reduced)
                                kinectomes = reduced
                                marker_lists_per_task[task_name] = current_markers
                            else:
                                reduced = []
                                current_markers = MARKER_LIST_AFFECT.copy()
                                for k in kinectomes:
                                    k_reduced, current_markers = exclude_markers_from_kinectome(k, current_markers, exclude)
                                    reduced.append(k_reduced)
                                kinectomes = reduced
                                marker_lists_per_task[task_name] = current_markers

                        try:
                            directions = ['full'] if full else ['AP', 'ML', 'V']

                            # Collect kinectomes by direction
                            all_kinectomes = {d: [] for d in directions}
                            for kinectome in kinectomes:
                                if full:
                                    all_kinectomes['full'].append(kinectome)
                                else:
                                    for idx, direction in enumerate(directions):
                                        all_kinectomes[direction].append(kinectome[:, :, idx])

                            # Calculate average and standard deviation kinectomes for each direction
                            for direction in directions:
                                if all_kinectomes[direction]:  # Check if the list is not empty
                                    # Stack the list of 2D arrays into a 3D array
                                    direction_stack = np.stack(all_kinectomes[direction], axis=0)
                                    
                                    # Calculate average kinectome for this direction
                                    avg_kinectome = np.mean(direction_stack, axis=0)
                                    
                                    # Calculate standard deviation kinectome for this direction
                                    std_kinectome = np.std(direction_stack, axis=0)
                                    
                                    # Mean absolute change between consecutive single-cycle kinectomes
                                    # (edge-wise stride-to-stride reconfiguration rate)
                                    if direction_stack.shape[0] > 1:
                                        reconfig_kinectome = np.mean(np.abs(np.diff(direction_stack, axis=0)), axis=0)
                                    else:
                                        reconfig_kinectome = np.full_like(avg_kinectome, np.nan)

                                    # Store the results in variability_scores - using explicit check for None
                                    # This avoids the numpy array comparison issue
                                    current_value = variability_scores[group][sub_id][task_name][kinematics][direction]
                                    if current_value is None:
                                        variability_scores[group][sub_id][task_name][kinematics][direction] = {
                                            "avg": avg_kinectome,
                                            "std": std_kinectome,
                                            "reconfig": reconfig_kinectome
                                        }
                                        
                                    else:
                                        # If you have multiple runs/tracking systems that should be combined,
                                        # you might need to implement a strategy here for combining them
                                        pass
                        except TypeError:
                            continue


    # --- Kinectome count summary ------------------------------------------
    print("\n" + "=" * 60)
    print("KINECTOME COUNT SUMMARY")
    print(f"Source folder: {_kin_src}")
    print(f"Type: {'FULL' if full else 'DIRECTIONAL'}")
    print("-" * 60)
    grand_subjects, grand_kinectomes = 0, 0
    for group in sorted(kinectome_counts):
        for task in sorted(kinectome_counts[group]):
            n_subj, n_kin = kinectome_counts[group][task]
            grand_subjects += n_subj
            grand_kinectomes += n_kin
            print(f"  {group:>10} | {task:<14} | {n_subj} subjects | {n_kin} kinectomes")
    print("-" * 60)
    print(f"  TOTAL: {grand_subjects} subject-sessions | {grand_kinectomes} kinectomes")
    print("=" * 60 + "\n")

    # Quick shape check — print matrix size for first available subject
    for group in variability_scores:
        for sub_id, sub_data in variability_scores[group].items():
            for task in sub_data:
                for kin in sub_data[task]:
                    for direction, val in sub_data[task][kin].items():
                        if val is not None and 'avg' in val:
                            print(f"  Matrix shape check: {group}/{sub_id}/{task}/{direction} = {val['avg'].shape}, markers={len(marker_lists_per_task[task])}")
                            break
                    break
                break
            break

    return variability_scores, marker_lists_per_task, task_disease_ids, task_control_ids

def sample_size_adequacy_analysis(variability_scores, task_names, kinematics_list, marker_list, 
                                 correlation_method, n_bootstraps=1000, n_permutations=5000, 
                                 matrix_type='std', random_seed=42):
    """
    Perform bootstrap analysis across different sample sizes to assess adequacy.
    
    Parameters:
    - variability_scores: nested dict with group -> subject -> task -> kinematic -> direction -> matrix_type structure
    - task_names: list of tasks to analyze
    - kinematics_list: list of kinematics to analyze  
    - marker_list: list of markers
    - correlation_method: correlation method to use
    - n_bootstraps: number of bootstrap iterations (default: 1000)
    - n_permutations: number of permutations per bootstrap (default: 5000)
    - matrix_type: type of matrix to analyze (default: 'std')
    - random_seed: seed for reproducibility (default: 42)
    
    Returns:
    - sample_size_results: dict with results for each subset fraction
    - observed_rhos: dict with observed rho values from full datasets
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    # Define subset fractions to test
    subset_fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    group_names = list(variability_scores.keys())
    if len(group_names) != 2:
        raise ValueError("This function currently supports comparisons between exactly 2 groups")
    
    group1, group2 = group_names

    # Derive direction keys from the data itself so this works for both
    # directional ({'AP','ML','V'}) and full ({'full'}) kinectomes.
    directions = _infer_directions(variability_scores)

    # Initialize results
    sample_size_results = {frac: {} for frac in subset_fractions}
    observed_rhos = {} 
    # First, compute observed rhos using full datasets
    print("Computing observed correlations using full datasets...")
    for task in task_names:
        observed_rhos[task] = {}
        
        for kinematic in kinematics_list:
            observed_rhos[task][kinematic] = {}
            
            for direction in directions:
                # Collect all available matrices for observed correlation
                group1_matrices = []
                group2_matrices = []
                
                for sub_id, sub_data in variability_scores[group1].items():
                    if (task in sub_data and kinematic in sub_data[task] and 
                        sub_data[task][kinematic] is not None and
                        direction in sub_data[task][kinematic] and
                        sub_data[task][kinematic][direction] is not None and
                        matrix_type in sub_data[task][kinematic][direction]):
                        group1_matrices.append(sub_data[task][kinematic][direction][matrix_type])
                
                for sub_id, sub_data in variability_scores[group2].items():
                    if (task in sub_data and kinematic in sub_data[task] and 
                        sub_data[task][kinematic] is not None and
                        direction in sub_data[task][kinematic] and
                        sub_data[task][kinematic][direction] is not None and
                        matrix_type in sub_data[task][kinematic][direction]):
                        group2_matrices.append(sub_data[task][kinematic][direction][matrix_type])
                
                if len(group1_matrices) == 0 or len(group2_matrices) == 0:
                    observed_rhos[task][kinematic][direction] = None
                    continue
                
                # Compute observed correlation
                group1_matrices = np.array(group1_matrices)
                group2_matrices = np.array(group2_matrices)
                avg_group1 = np.mean(group1_matrices, axis=0)
                avg_group2 = np.mean(group2_matrices, axis=0)
                
                # Get upper triangular parts and compute correlation
                upper_tri_mask = np.triu(np.ones(avg_group1.shape), k=1).astype(bool)
                rho_observed, _ = stats.spearmanr(avg_group1[upper_tri_mask], avg_group2[upper_tri_mask])
                observed_rhos[task][kinematic][direction] = rho_observed
                
                print(f"Observed rho = {rho_observed:.3f} for {task} {kinematic} {direction}")
    
    # Now perform bootstrap sampling for each subset fraction
    for subset_fraction in subset_fractions:
        print(f"\n{'='*60}")
        print(f"Analyzing subset fraction: {subset_fraction:.1%}")
        print(f"{'='*60}")
        
        # Initialize results for this fraction
        for task in task_names:
            sample_size_results[subset_fraction][task] = {}
            for kinematic in kinematics_list:
                sample_size_results[subset_fraction][task][kinematic] = {}
                for direction in directions:
                    sample_size_results[subset_fraction][task][kinematic][direction] = []
        
        # Perform bootstrap iterations for this subset fraction
        for bootstrap_iter in range(n_bootstraps):
            if (bootstrap_iter + 1) % 200 == 0:
                print(f"Bootstrap iteration {bootstrap_iter + 1}/{n_bootstraps} for {subset_fraction:.1%}")
            
            for task in task_names:
                for kinematic in kinematics_list:
                    for direction in directions:
                        # Collect subject IDs that have data for this condition
                        group1_subjects = []
                        group2_subjects = []
                        
                        for sub_id, sub_data in variability_scores[group1].items():
                            if (task in sub_data and kinematic in sub_data[task] and 
                                sub_data[task][kinematic] is not None and
                                direction in sub_data[task][kinematic] and
                                sub_data[task][kinematic][direction] is not None and
                                matrix_type in sub_data[task][kinematic][direction]):
                                group1_subjects.append(sub_id)
                        
                        for sub_id, sub_data in variability_scores[group2].items():
                            if (task in sub_data and kinematic in sub_data[task] and 
                                sub_data[task][kinematic] is not None and
                                direction in sub_data[task][kinematic] and
                                sub_data[task][kinematic][direction] is not None and
                                matrix_type in sub_data[task][kinematic][direction]):
                                group2_subjects.append(sub_id)
                        
                        if len(group1_subjects) == 0 or len(group2_subjects) == 0:
                            continue
                        
                        # Sample subset of subjects
                        n_group1_sample = max(1, int(len(group1_subjects) * subset_fraction))
                        n_group2_sample = max(1, int(len(group2_subjects) * subset_fraction))
                        
                        # Skip if sample size would be too small
                        if n_group1_sample < 2 or n_group2_sample < 2:
                            continue
                        
                        sampled_group1_subjects = random.sample(group1_subjects, n_group1_sample)
                        sampled_group2_subjects = random.sample(group2_subjects, n_group2_sample)
                        
                        # Collect matrices for sampled subjects
                        group1_matrices = []
                        group2_matrices = []
                        
                        for sub_id in sampled_group1_subjects:
                            group1_matrices.append(variability_scores[group1][sub_id][task][kinematic][direction][matrix_type])
                        
                        for sub_id in sampled_group2_subjects:
                            group2_matrices.append(variability_scores[group2][sub_id][task][kinematic][direction][matrix_type])
                        
                        # Compute average matrices and correlation
                        group1_matrices = np.array(group1_matrices)
                        group2_matrices = np.array(group2_matrices)
                        avg_group1 = np.mean(group1_matrices, axis=0)
                        avg_group2 = np.mean(group2_matrices, axis=0)
                        
                        # Get upper triangular parts and compute correlation
                        upper_tri_mask = np.triu(np.ones(avg_group1.shape), k=1).astype(bool)
                        rho_bootstrap, _ = stats.spearmanr(avg_group1[upper_tri_mask], avg_group2[upper_tri_mask])
                        
                        # Store bootstrap result
                        sample_size_results[subset_fraction][task][kinematic][direction].append(rho_bootstrap)
    
    # Create comprehensive plots
    print("\nCreating sample size adequacy plots...")
    create_sample_size_plots(sample_size_results, observed_rhos, task_names, kinematics_list, 
                           matrix_type, subset_fractions, group1, group2)
    
    return sample_size_results, observed_rhos

def create_summary_table(sample_size_results, observed_rhos, task_names, directions, 
                        kinematic, matrix_type, subset_fractions, output_dir):
    """
    Create a summary table showing key statistics for each condition and sample size.
    """
    
    summary_data = []
    
    for task in task_names:
        for direction in directions:
            observed_rho = observed_rhos[task][kinematic][direction]
            
            for frac in subset_fractions:
                bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                
                if len(bootstrap_rhos) > 0 and observed_rho is not None:
                    mean_rho = np.mean(bootstrap_rhos)
                    std_rho = np.std(bootstrap_rhos)
                    cv = (std_rho / mean_rho) * 100 if mean_rho != 0 else np.nan
                    bias = mean_rho - observed_rho
                    
                    summary_data.append({
                        'Task': task,
                        'Direction': direction,
                        'Sample_Size_Pct': int(frac * 100),
                        'Observed_Rho': observed_rho,
                        'Bootstrap_Mean': mean_rho,
                        'Bootstrap_STD': std_rho,
                        'CV_Percent': cv,
                        'Bias': bias,
                        'N_Bootstraps': len(bootstrap_rhos)
                    })
    
    # Convert to DataFrame and save
    if summary_data:
        df = pd.DataFrame(summary_data)
        df = df.round(4)
        
        filename = f'sample_size_summary_{kinematic}_{matrix_type}.csv'
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        
        print(f"  - Summary table: {filepath}")

def compare_between_groups(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, result_base_path, full, correlation_method):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices, marker_lists_per_task, task_disease_ids, task_control_ids = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method)

    bootstrap_avg, observed_avg = bootstrap_permutation_test(matrices, task_names, kinematics_list, marker_lists_per_task,
                                                                    result_base_path, correlation_method, n_bootstraps=5000,
                                                                    n_permutations=10000, matrix_type='avg', subset_fraction=0.8,
                                                                    random_seed=42)
    bootstrap_std, observed_std = bootstrap_permutation_test(matrices, task_names, kinematics_list, marker_lists_per_task,
                                                                    result_base_path, correlation_method, n_bootstraps=5000,
                                                                    n_permutations=10000, matrix_type='std', subset_fraction=0.8,
                                                                    random_seed=42)
    bootstrap_results = {'avg': bootstrap_avg, 'std': bootstrap_std}
    observed_rhos     = {'avg': observed_avg,  'std': observed_std}


    # sample_size_results, observed_rhos = sample_size_adequacy_analysis(matrices, task_names, kinematics_list, marker_list_affect, 
                                                                    # correlation_method, n_bootstraps=5000, n_permutations=5000, 
                                                                    # matrix_type='std', random_seed=42
                                                                                                    # )
    # Permutation testing of avg and std matrices
    # permutation_test_one_p always plots avg/std matrices.
    # When RUN_BOOTSTRAP=True it also runs bootstrap-wrapped permutation tests.
    # When RUN_BOOTSTRAP=False it runs a single fast permutation test.
    from config import N_PERMUTATIONS

    avg_p_values = permutation_test_one_p(
        matrices, task_names, kinematics_list, marker_lists_per_task,
        result_base_path, correlation_method,
        n_permutations=N_PERMUTATIONS, matrix_type='avg'
    )
    std_p_values = permutation_test_one_p(
        matrices, task_names, kinematics_list, marker_lists_per_task,
        result_base_path, correlation_method,
        n_permutations=N_PERMUTATIONS, matrix_type='std'
    )

    print()

    return matrices