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


def calc_std_avg_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method, interpol):
    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

    # Choose what to store based on the `full` flag
    direction_template = {"full": None} if full else {"AP": None, "ML": None, "V": None}

    # Store variability scores structured per subject, task, and direction
    variability_scores = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: direction_template.copy() for kinematics in kinematics_list} 
                                                       for task in task_names} 
                                                       for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: direction_template.copy() for kinematics in kinematics_list} 
                             for task in task_names} 
                             for sub_id in matched_control_sub_ids},
    }

    debug_ids = ['pp006', 'pp008']

    for kinematics in kinematics_list:
        for sub_id in disease_sub_ids + matched_control_sub_ids:
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
        
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
                        
                        kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method, interpol)
                        
                        try:
                            if not full:
                                # Initialize lists to store kinectomes for each direction
                                all_kinectomes = {"AP": [], "ML": [], "V": []}
                                
                                # Collect kinectomes by direction
                                for kinectome in kinectomes:
                                    for idx, direction in enumerate(['AP', 'ML', 'V']):
                                        all_kinectomes[direction].append(kinectome[:, :, idx])
                            elif full:
                                all_kinectomes = {'full' : []}
                                for kinectome in kinectomes:
                                    all_kinectomes['full'].append(kinectome)


                            # Calculate average and standard deviation kinectomes for each direction

                            for direction in ['AP', 'ML', 'V']:
                                if full:
                                    direction = 'full'
                                if all_kinectomes[direction]:  # Check if the list is not empty
                                    # Stack the list of 2D arrays into a 3D array
                                    direction_stack = np.stack(all_kinectomes[direction], axis=0)
                                    
                                    # Calculate average kinectome for this direction
                                    avg_kinectome = np.mean(direction_stack, axis=0)
                                    
                                    # Calculate standard deviation kinectome for this direction
                                    std_kinectome = np.std(direction_stack, axis=0)
                                    
                                    # Store the results in variability_scores - using explicit check for None
                                    # This avoids the numpy array comparison issue
                                    current_value = variability_scores[group][sub_id][task_name][kinematics][direction]
                                    if current_value is None:
                                        variability_scores[group][sub_id][task_name][kinematics][direction] = {
                                            "avg": avg_kinectome,
                                            "std": std_kinectome
                                        }
                                    else:
                                        # If you have multiple runs/tracking systems that should be combined,
                                        # you might need to implement a strategy here for combining them
                                        pass
                        except TypeError:
                            continue


    return variability_scores

def permutation_test_one_p(variability_scores, task_names, kinematics_list, marker_list, result_base_path, correlation_method, n_permutations, matrix_type):

    """
    Perform a permutation test comparing two sets of matrices and return a single p-value.
    
    Parameters:
    - group1_matrices (np.ndarray): Array of shape (N1, rows, cols) for group 1.
    - group2_matrices (np.ndarray): Array of shape (N2, rows, cols) for group 2.
    - n_permutations (int): Number of permutations (default: 10,000).
    
    Returns:
    - float: A single p-value for the group-level comparison.
    """
    results = {}
    
    group_names = list(variability_scores.keys())
    if len(group_names) != 2:
        raise ValueError("This function currently supports comparisons between exactly 2 groups")
    
    group1, group2 = group_names
    
    # # Get a sample subject from first group to extract task structure
    # sample_subject = next(iter(variability_scores[group1].values()))
    # tasks = sample_subject.keys()
    
    for task in task_names:
        results[task] = {}        
        for kinematic in kinematics_list:
            results[task][kinematic] = {}
            
            for direction in ['AP', 'ML', 'V']:
                # Collect matrices for each group
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
                
                # Skip if not enough data
                if len(group1_matrices) == 0 or len(group2_matrices) == 0:
                    results[task][kinematic][direction] = {
                        'p_values': None,
                        'significant_mask': None,
                        f'{group1}_n': len(group1_matrices),
                        f'{group2}_n': len(group2_matrices)
                    }
                    continue
                
                # Convert lists to numpy arrays
                group1_matrices = np.array(group1_matrices)
                group2_matrices = np.array(group2_matrices)    
    
    
                # Compute the average kinectome for each group
                avg_group1 = np.mean(group1_matrices, axis=0)
                avg_group2 = np.mean(group2_matrices, axis=0)


                # calculate the correlation between two matrices
                rho, p_value = stats.spearmanr(permutation.upper(avg_group1), permutation.upper(avg_group2))

                suptitle = (f'{"Standard deviation" if matrix_type == "std" else "Average"} kinectomes in {direction} direction during '
                            f'{"preferred walking speed" if task == "walkPreferred" else "fast walking speed" if task == "walkFast" else "slow walking speed"}')
                fig_name = f'{matrix_type}_matrices_{direction}_{task}.png'
                plotting.plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value, suptitle, fig_name)

                permutation.permute(avg_group1, avg_group2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method, n_permutations)

    return results


def calc_avg_group_matrix(matrices):
    """Calculates average matrix for each group per task, kinematic and direction (AP, ML, and V, or full for the full matrix)"""

    avg_matrices = {}

    for group, subjects in matrices.items():
        avg_matrices[group] = {}

        for subj_data in subjects.values():
            for task, task_data in subj_data.items():
                if task is None:
                    continue

                for kin, kin_data in task_data.items():
                    if kin is None:
                        continue

                    for direction, dir_data in kin_data.items():
                        if dir_data is None or 'avg' not in dir_data:
                            continue

                        # Initialize nested structure if needed
                        avg_matrices.setdefault(group, {}).setdefault(task, {}).setdefault(kin, {}).setdefault(direction, [])

                        # Add the matrix to the list
                        avg_matrices[group][task][kin][direction].append(dir_data['avg'])
    
    # Calculate mean for each group/task/kinematic/direction
    for group in avg_matrices:
        for task in avg_matrices[group]:
            for kin in avg_matrices[group][task]:
                for direction in avg_matrices[group][task][kin]:
                    matrices_list = avg_matrices[group][task][kin][direction]
                    avg_matrix = np.round(np.nanmean(matrices_list, axis=0), 3)
                    avg_matrices[group][task][kin][direction] = avg_matrix
    
    return avg_matrices

def average_symmetric_matrix(matrix, marker_list):
    paired = {}
    used = set()
    solo = []
    
    for i, marker in enumerate(marker_list):
        if marker in used:
            continue
        if marker.startswith("l_"):
            core = marker[2:]
            counterpart = "r_" + core
            if counterpart in marker_list:
                j = marker_list.index(counterpart)
                paired[core] = (i, j)
                used.update([marker, counterpart])
        elif marker.startswith("r_"):
            core = marker[2:]
            counterpart = "l_" + core
            if counterpart in marker_list:
                continue  # already handled by l_
            else:
                solo.append((i, marker))
        else:
            solo.append((i, marker))

    # New marker list
    new_marker_list = [f"avg_{core}" for core in paired.keys()] + [name for _, name in solo]
    
    # Build new averaged matrix
    indices = [*paired.values(), *[(i, i) for i, _ in solo]]
    new_size = len(indices)
    new_matrix = np.zeros((new_size, new_size))
    
    for m, (i1m, i2m) in enumerate(indices):
        for n, (i1n, i2n) in enumerate(indices):
            val = (matrix[i1m, i1n] + matrix[i1m, i2n] + matrix[i2m, i1n] + matrix[i2m, i2n]) / 4
            new_matrix[m, n] = val
    
    return new_matrix, new_marker_list

def summarize_by_anatomical_regions(diff_mat, marker_list):
    """
    Aggregates markers into anatomical regions and calculates average differences
    within and between these regions.
    
    Parameters:
    -----------
    diff_mat : numpy.ndarray
        The difference matrix (PD - Control)
    marker_list : list
        List of marker names corresponding to diff_mat rows/columns
        
    Returns:
    --------
    region_diff_matrix : numpy.ndarray
        Matrix of average differences between regions
    region_names : list
        List of region names
    """
    # Define anatomical regions based on marker prefixes
    region_mapping = {        
        'trunk': ['head', 'ster'],
        'pelvis': ['psis_la', 'psis_ma', 'asis_la', 'asis_ma'],
        'least affected arm': ['sho_la',  'elbl_la',  'wrist_la',  'hand_la', ],
        'most affected arm': ['sho_ma', 'elbl_ma', 'wrist_ma', 'hand_ma'],
        'least affected leg': ['th_la',  'sk_la',  'ank_la',  'toe_la' ],
        'most affected leg': ['th_ma', 'sk_ma', 'ank_ma', 'toe_ma']
    }
    
    # Create reverse mapping: marker -> region
    marker_to_region = {}
    for region, markers in region_mapping.items():
        for marker in markers:
            marker_to_region[marker] = region
    
    # Get unique regions
    regions = list(region_mapping.keys())
    num_regions = len(regions)
    
    # Initialize region difference matrix
    region_diff_matrix = np.zeros((num_regions, num_regions))
    region_counts = np.zeros((num_regions, num_regions))
    
    # Aggregate differences by region
    for i, marker_i in enumerate(marker_list):
        for j, marker_j in enumerate(marker_list):
            if marker_i in marker_to_region and marker_j in marker_to_region:
                region_i = regions.index(marker_to_region[marker_i])
                region_j = regions.index(marker_to_region[marker_j])
                region_diff_matrix[region_i, region_j] += diff_mat[i, j]
                region_counts[region_i, region_j] += 1
    
    # Compute averages (avoid division by zero)
    mask = region_counts > 0
    region_diff_matrix[mask] = region_diff_matrix[mask] / region_counts[mask]
    
    return region_diff_matrix, regions

def reorder_difference_matrix(matrices, marker_list, result_base_path, correlation_method, full):
    """Calculates the difference matrix between Control group and the group of interest
    and reorders it from biggest to smallest difference"""
    avg_matrices = calc_avg_group_matrix(matrices)
    groups = list(avg_matrices.keys())
    tasks = avg_matrices[groups[0]].keys()
    kinematics = avg_matrices[groups[0]][next(iter(tasks))].keys()
    directions = avg_matrices[groups[0]][next(iter(tasks))][next(iter(kinematics))].keys()
   
    # expand the marker list (add suffixes *AP, *ML, and _V) if analysing the full graph
    if 'full' in directions:
        marker_list = permutation.expand_marker_list(marker_list)
    
    for task in tasks:
        for kin in kinematics:
            for direction in directions:
                # Get matrices
                mat_group1 = avg_matrices[groups[0]][task][kin][direction]
                mat_group2 = avg_matrices[groups[1]][task][kin][direction]
                
                # Absolute difference matrix
                diff_mat = mat_group1 - mat_group2
                
                # Calculate regional differences
                region_diff_matrix, region_names = summarize_by_anatomical_regions(diff_mat, marker_list)
                region_figname = f"{task}_{kin}_{direction}_region_diffs_{correlation_method}.png"
                plotting.plot_region_difference_matrix(region_diff_matrix, region_names, task, kin, direction, 
                                                      groups[0], groups[1], result_base_path, region_figname)
                
                # Sort by highest total difference
                sort_order = np.argsort(-np.sum(np.abs(diff_mat), axis=1))
                diff_mtrx_sorted = diff_mat[np.ix_(sort_order, sort_order)]
                reordered_markers = [marker_list[i] for i in sort_order]
                
                # Plot
                figname = f"{task}_{kin}_{direction}_absdiff_affect_{correlation_method}.png"
                plotting.plot_difference_matrix(diff_mtrx_sorted, reordered_markers, task, kin, direction, groups[0], groups[1], result_base_path, figname)
    
    # NEW CODE: After processing all matrices, create distribution plots
    fig = plotting.plot_difference_distributions(avg_matrices, list(tasks), list(kinematics), list(directions))
    plt.savefig(f"{result_base_path}/difference_distributions_{correlation_method}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def bootstrap_permutation_test(variability_scores, task_names, kinematics_list, marker_list, 
                             correlation_method, n_bootstraps=1000, n_permutations=5000, 
                             matrix_type='std', subset_fraction=0.5, random_seed=42):
    """
    Perform bootstrap permutation testing by repeatedly sampling subsets of subjects
    and computing correlations between group-averaged matrices.
    
    Parameters:
    - variability_scores: nested dict with group -> subject -> task -> kinematic -> direction -> matrix_type structure
    - task_names: list of tasks to analyze
    - kinematics_list: list of kinematics to analyze  
    - marker_list: list of markers
    - correlation_method: correlation method to use
    - n_bootstraps: number of bootstrap iterations (default: 1000)
    - n_permutations: number of permutations per bootstrap (default: 5000)
    - matrix_type: type of matrix to analyze (default: 'std')
    - subset_fraction: fraction of subjects to sample from each group (default: 0.5)
    - random_seed: seed for reproducibility (default: 42)
    
    Returns:
    - bootstrap_results: nested dict with bootstrap rho values
    - observed_rhos: nested dict with observed rho values from full datasets
    """
    
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    group_names = list(variability_scores.keys())
    if len(group_names) != 2:
        raise ValueError("This function currently supports comparisons between exactly 2 groups")
    
    group1, group2 = group_names
    
    # Initialize results dictionaries
    bootstrap_results = {}
    observed_rhos = {}
    
    # First, compute observed rhos using full datasets
    print("Computing observed correlations using full datasets...")
    for task in task_names:
        observed_rhos[task] = {}
        bootstrap_results[task] = {}
        
        for kinematic in kinematics_list:
            observed_rhos[task][kinematic] = {}
            bootstrap_results[task][kinematic] = {}
            
            for direction in ['AP', 'ML', 'V']:
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
                    bootstrap_results[task][kinematic][direction] = []
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
    
    # Now perform bootstrap sampling
    print(f"\nPerforming {n_bootstraps} bootstrap iterations...")
    
    for bootstrap_iter in range(n_bootstraps):
        if (bootstrap_iter + 1) % 100 == 0:
            print(f"Bootstrap iteration {bootstrap_iter + 1}/{n_bootstraps}")
        
        for task in task_names:
            for kinematic in kinematics_list:
                for direction in ['AP', 'ML', 'V']:
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
                    if bootstrap_iter == 0:  # Initialize list on first iteration
                        bootstrap_results[task][kinematic][direction] = []
                    bootstrap_results[task][kinematic][direction].append(rho_bootstrap)
    
    # Create plots
    print("\nCreating bootstrap distribution plots...")
    create_bootstrap_plots(bootstrap_results, observed_rhos, task_names, kinematics_list, matrix_type)
    
    return bootstrap_results, observed_rhos


def create_bootstrap_plots(bootstrap_results, observed_rhos, task_names, kinematics_list, matrix_type):
    """
    Create 3x3 subplot showing bootstrap distributions with observed values marked.
    """
    
    # Create output directory
    output_dir = r"C:\Users\Karolina\Desktop\pykinectome\results\avg_std_matrices\bootstrapping"
    os.makedirs(output_dir, exist_ok=True)
    
    directions = ['AP', 'ML', 'V']
    
    for kinematic in kinematics_list:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        # fig.suptitle(f'Bootstrap Distribution of Correlations - {kinematic} ({matrix_type})', fontsize=16)
        fig.suptitle(f'Bootstrap Distribution of Correlations - Average Kinectomes', fontsize=16)

        x_min, x_max = 0.8, 1.0  # Adjust these values based on your data range

        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                
                # Get bootstrap results and observed value
                bootstrap_rhos = bootstrap_results[task][kinematic][direction]
                observed_rho = observed_rhos[task][kinematic][direction]
                
                if len(bootstrap_rhos) > 0 and observed_rho is not None:
                    # Plot histogram of bootstrap correlations
                    ax.hist(bootstrap_rhos, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Mark observed correlation with red line
                    ax.axvline(observed_rho, color='red', linestyle='--', linewidth=2, 
                              label=f'Observed ρ = {observed_rho:.3f}')
                    
                    # Add statistics
                    bootstrap_mean = np.mean(bootstrap_rhos)
                    bootstrap_std = np.std(bootstrap_rhos)
                    ax.axvline(bootstrap_mean, color='orange', linestyle=':', linewidth=2,
                              label=f'Bootstrap μ = {bootstrap_mean:.3f}')
                    
                    ax.legend(fontsize=8)
                    ax.set_xlabel('Correlation (ρ)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'bootstrap_correlations_{kinematic}_{matrix_type}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.show()
        
        print(f"Saved bootstrap plot: {filepath}")

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
    
    # Initialize results
    sample_size_results = {frac: {} for frac in subset_fractions}
    observed_rhos = {} 
    # First, compute observed rhos using full datasets
    print("Computing observed correlations using full datasets...")
    for task in task_names:
        observed_rhos[task] = {}
        
        for kinematic in kinematics_list:
            observed_rhos[task][kinematic] = {}
            
            for direction in ['AP', 'ML', 'V']:
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
                for direction in ['AP', 'ML', 'V']:
                    sample_size_results[subset_fraction][task][kinematic][direction] = []
        
        # Perform bootstrap iterations for this subset fraction
        for bootstrap_iter in range(n_bootstraps):
            if (bootstrap_iter + 1) % 200 == 0:
                print(f"Bootstrap iteration {bootstrap_iter + 1}/{n_bootstraps} for {subset_fraction:.1%}")
            
            for task in task_names:
                for kinematic in kinematics_list:
                    for direction in ['AP', 'ML', 'V']:
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


def create_sample_size_plots(sample_size_results, observed_rhos, task_names, kinematics_list, 
                           matrix_type, subset_fractions, group1, group2):
    """
    Create comprehensive plots showing how correlation stability changes with sample size.
    """
    
    # Create output directory
    output_dir = r"C:\Users\Karolina\Desktop\pykinectome\results\avg_std_matrices\bootstrapping\sample_size_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    directions = ['AP', 'ML', 'V']
    
    for kinematic in kinematics_list:
        # Plot 1: Mean and variability vs sample size (3x3 subplots)
        fig1, axes1 = plt.subplots(3, 3, figsize=(18, 12))
        fig1.suptitle(f'Correlation Stability vs Sample Size - {kinematic} ({matrix_type})', fontsize=16)
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes1[i, j]
                
                means = []
                stds = []
                sample_percentages = [int(f * 100) for f in subset_fractions]
                observed_rho = observed_rhos[task][kinematic][direction]
                
                for frac in subset_fractions:
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        means.append(np.mean(bootstrap_rhos))
                        stds.append(np.std(bootstrap_rhos))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                # Plot mean with error bars
                means = np.array(means)
                stds = np.array(stds)
                
                # Remove NaN values for plotting
                valid_idx = ~np.isnan(means)
                if np.any(valid_idx):
                    ax.errorbar(np.array(sample_percentages)[valid_idx], means[valid_idx], 
                              yerr=stds[valid_idx], marker='o', linestyle='-', linewidth=2, 
                              markersize=6, capsize=5, label='Bootstrap Mean ± SD')
                    
                    # Add observed value as horizontal line
                    if observed_rho is not None:
                        ax.axhline(observed_rho, color='red', linestyle='--', linewidth=2, 
                                  label=f'Observed ρ = {observed_rho:.3f}')
                    
                    ax.legend(fontsize=9)
                    ax.set_xlabel('Sample Size (%)', fontsize=10)
                    ax.set_ylabel('Correlation (ρ)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.0)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename1 = f'sample_size_stability_{kinematic}_{matrix_type}.png'
        filepath1 = os.path.join(output_dir, filename1)
        plt.savefig(filepath1, dpi=600, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Coefficient of variation (CV) vs sample size
        fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
        fig2.suptitle(f'Coefficient of Variation vs Sample Size - {kinematic} ({matrix_type})', fontsize=16)
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes2[i, j]
                
                cvs = []
                sample_percentages = [int(f * 100) for f in subset_fractions]
                
                for frac in subset_fractions:
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        mean_rho = np.mean(bootstrap_rhos)
                        std_rho = np.std(bootstrap_rhos)
                        cv = (std_rho / mean_rho) * 100 if mean_rho != 0 else np.nan
                        cvs.append(cv)
                    else:
                        cvs.append(np.nan)
                
                cvs = np.array(cvs)
                valid_idx = ~np.isnan(cvs)
                
                if np.any(valid_idx):
                    ax.plot(np.array(sample_percentages)[valid_idx], cvs[valid_idx], 
                           marker='o', linestyle='-', linewidth=2, markersize=6, color='purple')
                    ax.set_xlabel('Sample Size (%)', fontsize=10)
                    ax.set_ylabel('Coefficient of Variation (%)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add horizontal line at 5% CV (commonly used threshold for good stability)
                    ax.axhline(5, color='green', linestyle=':', linewidth=2, alpha=0.7, 
                              label='5% CV threshold')
                    ax.legend(fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename2 = f'coefficient_variation_{kinematic}_{matrix_type}.png'
        filepath2 = os.path.join(output_dir, filename2)
        plt.savefig(filepath2, dpi=600, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Distribution comparison at key sample sizes (10%, 50%, 90%)
        key_fractions = [0.1, 0.5, 0.9]
        fig3, axes3 = plt.subplots(3, 3, figsize=(18, 12))
        fig3.suptitle(f'Bootstrap Distributions at Key Sample Sizes - {kinematic} ({matrix_type})', fontsize=16)
        
        colors = ['lightcoral', 'skyblue', 'lightgreen']
        alphas = [0.6, 0.7, 0.8]
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes3[i, j]
                
                observed_rho = observed_rhos[task][kinematic][direction]
                
                for k, frac in enumerate(key_fractions):
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        ax.hist(bootstrap_rhos, bins=20, alpha=alphas[k], color=colors[k], 
                               edgecolor='black', linewidth=0.5, 
                               label=f'{int(frac*100)}% sample (μ={np.mean(bootstrap_rhos):.3f})')
                
                # Add observed value
                if observed_rho is not None:
                    ax.axvline(observed_rho, color='red', linestyle='--', linewidth=2, 
                              label=f'Observed ρ = {observed_rho:.3f}')
                
                ax.legend(fontsize=8)
                ax.set_xlabel('Correlation (ρ)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename3 = f'distribution_comparison_{kinematic}_{matrix_type}.png'
        filepath3 = os.path.join(output_dir, filename3)
        plt.savefig(filepath3, dpi=600, bbox_inches='tight')
        plt.show()
        
        print(f"Saved sample size analysis plots for {kinematic}:")
        print(f"  - Stability plot: {filepath1}")
        print(f"  - CV plot: {filepath2}")
        print(f"  - Distribution comparison: {filepath3}")
        
        # Create summary statistics table
        create_summary_table(sample_size_results, observed_rhos, task_names, directions, 
                           kinematic, matrix_type, subset_fractions, output_dir)


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

def compare_between_groups(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, result_base_path, full, correlation_method, interpol):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method, interpol)

    bootstrap_results, observed_rhos = bootstrap_permutation_test(matrices, task_names, kinematics_list, marker_list_affect, 
                                                                    correlation_method, n_bootstraps=5000, n_permutations=10000, 
                                                                    matrix_type='avg', subset_fraction=0.8, random_seed=42)


    # sample_size_results, observed_rhos = sample_size_adequacy_analysis(matrices, task_names, kinematics_list, marker_list_affect, 
                                                                    # correlation_method, n_bootstraps=5000, n_permutations=5000, 
                                                                    # matrix_type='std', random_seed=42
                                                                                                    # )
    # permutation testing of the average and std matrices
    std_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list_affect, result_base_path, correlation_method, n_permutations=10000, matrix_type='std' )

    # avg_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list_affect, result_base_path, correlation_method, n_permutations=10000, matrix_type='avg')
    
    # diff_p_values =  permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='avg', n_permutations=10000, diff=False)

    # reorder the difference matrix from biggest to smallest difference (row-wise)
    # reordered_difference_matrix = reorder_difference_matrix(matrices, marker_list_affect, result_base_path, correlation_method, full)

    print()

    return matrices

