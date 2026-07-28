import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
from src.data_utils import plotting
import random


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


def get_adaptive_subgroups(matrix, marker_list):
    """
    Return subgroups of body segments that adapt based on matrix dimensions
    and the actual markers present in marker_list.
    Markers excluded from the kinectome are automatically removed from subgroups.

    Works whether marker_list is a base list (e.g. 'head') or an already-expanded
    per-direction list (e.g. 'head_AP', 'head_ML', 'head_V'): subgroup membership
    is decided by the base marker name, so expanded labels are grouped correctly.
    """
    base_subgroups = {
        "upper_body": ['head', 'sternum', 'shoulder_las', 'shoulder_mas', 'asis_las', 'asis_mas', 'psis_las', 'psis_mas',
                    'elbow_las', 'wrist_las', 'hand_las', 'elbow_mas', 'wrist_mas', 'hand_mas'],
        "lower_body": ['thigh_las', 'shank_las', 'ankle_las', 'toe_las', 'thigh_mas', 'shank_mas', 'ankle_mas', 'toe_mas']
    }

    def _base_name(m):
        # Strip a trailing direction suffix (_AP/_ML/_V) if present.
        for d in ('_AP', '_ML', '_V'):
            if m.endswith(d):
                return m[:-len(d)]
        return m

    # Assign each entry in marker_list to a subgroup by its base marker name.
    # This keeps expanded labels ('head_AP') together with their group and
    # ensures the labels used for shuffling are exactly those in marker_list.
    base_to_group = {}
    for group, markers in base_subgroups.items():
        for m in markers:
            base_to_group[m] = group

    subgroups = {group: [] for group in base_subgroups}
    for m in marker_list:
        group = base_to_group.get(_base_name(m))
        if group is not None:
            subgroups[group].append(m)

    return subgroups

def permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method, n_iter):

    # Expand the marker list to per-direction labels FIRST if the matrix is a
    # full kinectome (nodes are head_AP, head_ML, ...). Subgroups must then be
    # built from this same (possibly expanded) list so the shuffle can locate
    # each member in shuffled_markers.
    if matrix1.shape != (len(marker_list), len(marker_list)):
        marker_list = expand_marker_list(marker_list)

    # Define subgroups of body segments (marker labels) for shuffling.
    # get_adaptive_subgroups matches by base marker name, so it groups the
    # expanded labels correctly and returns them exactly as they appear here.
    subgroups = get_adaptive_subgroups(matrix1, marker_list)


    # Convert avg_group1 (numpy array) into a DataFrame
    df_group1 = pd.DataFrame(matrix1, index=marker_list, columns=marker_list)
    df_group2 = pd.DataFrame(matrix2, index=marker_list, columns=marker_list)
    
    # observed_diff = np.abs(df_group1) - np.abs(np.group2)

    # Now lets measure the similarity 
    rho, p_value = stats.spearmanr(upper(df_group1), upper(df_group2))
    print(f'rho = {np.round(rho, 2)} p_value = {p_value}  during {task} ({matrix_type}) for {kinematic} in {direction} direction')


    # """Nonparametric permutation testing Monte Carlo"""
    np.random.seed(0)
    rhos = []
    # n_iter = 5000
    true_rho, _ = stats.spearmanr(upper(df_group1), upper(df_group2))

    # upper triangle of the matrix 
    m2_v = upper(df_group2)

    for _ in range(n_iter):
        shuffled_markers = marker_list.copy()

        # Shuffle **within** each broader subgroup
        for group in subgroups.values():
            shuffled_group = np.random.permutation(group)
            for original, shuffled in zip(group, shuffled_group):
                shuffled_markers[shuffled_markers.index(original)] = shuffled

        # Apply shuffled marker order
        shuffled_df1 = df_group1.loc[shuffled_markers, shuffled_markers]

        r, _ = stats.spearmanr(upper(shuffled_df1), m2_v)
        rhos.append(r)

    # Compute two-tailed p-value
    perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos))) + 1) / (n_iter + 1)
    
    plotting.plot_permutation_histogram(rhos, true_rho, perm_p, result_base_path, task, kinematic, direction, matrix_type, correlation_method)
    return true_rho, perm_p

def permute_difference_matrix(matrix1, matrix2, group1, group2, marker_list, task, kinematic, direction, result_base_path, matrix_type = 'diff', n_permutations=5000):
    """
    Perform a permutation test on the difference matrix (matrix1 - matrix2)
    to determine whether the observed differences are significantly different from chance.

    Parameters:
    - matrix1, matrix2: (numpy arrays) Symmetric matrices to compare.
    - marker_list: (list) List of marker names corresponding to rows/columns.
    - task, matrix_type, kinematic, direction: (str) Metadata for reporting.
    - result_base_path: (str) Path to save the histogram.
    - n_iter: (int) Number of permutation iterations.

    Returns:
    - perm_p: (float) Two-tailed p-value for the permutation test.
    """
 
 # Convert matrices to DataFrames
    df_group1 = pd.DataFrame(matrix1, index=marker_list, columns=marker_list)
    df_group2 = pd.DataFrame(matrix2, index=marker_list, columns=marker_list)

    # Compute the **true** difference matrix and extract its upper triangle
    true_diff_matrix = df_group1 - df_group2
    true_diffs = upper(true_diff_matrix)

    # Permutation setup
    np.random.seed(0)
    perm_diffs = []
    n_iter = 5000

    # Define marker subgroups for within-group shuffling
    subgroups = {
        "upper_body": ['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'r_asis', 'l_psis', 'r_psis', 
                    'l_elbl', 'l_wrist', 'l_hand', 'r_elbl', 'r_wrist', 'r_hand'],
        "lower_body": ['l_th', 'l_sk', 'l_ank', 'l_toe', 'r_th', 'r_sk', 'r_ank', 'r_toe']
    }

    for _ in range(n_iter):
        shuffled_markers = marker_list.copy()  # Copy original marker order

        # Shuffle within subgroups
        for group in subgroups.values():
            shuffled_group = np.random.permutation(group)  # Shuffle the subgroup
            indices = [shuffled_markers.index(m) for m in group]  # Find original indices
            for idx, shuffled_m in zip(indices, shuffled_group):
                shuffled_markers[idx] = shuffled_m  # Replace in original list

        # Debugging: Check if markers are actually shuffled
        print(f"Iteration {_}: {shuffled_markers[:5]}")  # Print first few markers

        # Apply shuffled order
        shuffled_df1 = df_group1.loc[shuffled_markers, shuffled_markers]

        # Compute shuffled difference matrix
        shuffled_diff_matrix = upper(shuffled_df1 - df_group2)

        # Debugging: Check if shuffled matrices are changing
        print(f"Iteration {_}, Sum of shuffled_diff_matrix: {shuffled_diff_matrix.sum()}")

        perm_diffs.append(shuffled_diff_matrix)
        
        # perm_diff = true_diffs - shuffled_diff_matrix

        # perm_diffs.append(perm_diff)

    perm_p = (np.sum(np.mean(np.abs(true_diffs)) <= np.mean(np.abs(perm_diffs)), axis=0) + 1) / (n_permutations + 1)
    
    # Convert perm_diffs list into a NumPy array for vectorized operations
    perm_diffs = np.array(perm_diffs)  # Shape: (5000, num_elements)

    # Compute two-tailed p-values for each matrix element
    p_values = np.mean(np.abs(perm_diffs) >= np.abs(true_diffs), axis=0)

    # Optionally, compute an overall significance level (average across all elements)
    mean_p_value = np.mean(p_values)
    


    # Plot histogram
    plot_diff_permutation_histogram(perm_diffs, true_diffs, perm_p, result_base_path, task, kinematic, direction, matrix_type='diff')

def plot_diff_permutation_histogram(perm_diffs, true_diffs, perm_p, results_path, task, kinematic, direction, matrix_type='diff'):
    plt.figure(figsize=(8, 6))

    # Compute mean absolute difference across all upper-triangle elements for each permutation
    perm_dist = np.mean(np.abs(perm_diffs), axis=1)  

    # Compute the true mean absolute difference
    true_mean_diff = np.mean(np.abs(true_diffs))

    # Plot histogram
    plt.hist(perm_dist, bins=50, color='gray', alpha=0.7, edgecolor='black', label="Permutation Distribution")
    plt.axvline(true_mean_diff, color='red', linestyle='dashed', linewidth=2, label="True Difference")

    # Labels and legend
    plt.xlabel("Mean Absolute Difference")
    plt.ylabel("Frequency")
    plt.title("Permutation Test Distribution")
    plt.legend()

    save_dir = Path(results_path) / "kinectome_characteristics"
    save_dir.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_dir / f'permutation_{task}_{kinematic}_{direction}_{matrix_type}.png', dpi=600)

def upper(df):
    '''Returns the upper triangle of a correlation matrix.
    You can use scipy.spatial.distance.squareform to recreate matrix from upper triangle.
    Args:
      df: pandas or numpy correlation matrix
    Returns:
      list of values from upper triangle
    '''
    try:
        assert(type(df)==np.ndarray)
    except:
        if type(df)==pd.DataFrame:
            df = df.values
        else:
            raise TypeError('Must be np.ndarray or pd.DataFrame')
    mask = np.triu_indices(df.shape[0], k=1)
    return df[mask]

def expand_marker_list(marker_list):

    """
    input - a list of 22 marker names

    output - a list of 66 marker names where each marker gets AP, ML, and V direction suffix
    
    """
    expanded_list = []

    directions = ['AP', 'ML', 'V']

    for marker in marker_list:
        for direction in directions:
            new_marker_name = f'{marker}_{direction}'

            expanded_list.append(new_marker_name)

    return expanded_list

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

    # Direction keys derived from the data: ['full'] or ['AP','ML','V'].
    directions = _infer_directions(variability_scores)

    # # Get a sample subject from first group to extract task structure
    # sample_subject = next(iter(variability_scores[group1].values()))
    # tasks = sample_subject.keys()
    
    for task in task_names:
        results[task] = {}        
        for kinematic in kinematics_list:
            results[task][kinematic] = {}
            
            for direction in directions:
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
                
                # Resolve per-task marker list (may be a dict if markers were excluded)
                task_markers = marker_list[task] if isinstance(marker_list, dict) else marker_list

                avg_group1 = np.mean(np.array(group1_matrices), axis=0)
                avg_group2 = np.mean(np.array(group2_matrices), axis=0)

                from config import RUN_BOOTSTRAP

                if RUN_BOOTSTRAP:
                    # Bootstrap-wrapped permutation test
                    from config import N_BOOTSTRAPS, BOOTSTRAP_SUBSET_FRACTION, N_BOOTSTRAP_PERMUTATIONS
                    bootstrap_rhos, bootstrap_pvalues, observed_rho, observed_p = bootstrap_permute(
                        list(group1_matrices), list(group2_matrices), task_markers,
                        task, matrix_type, kinematic, direction, result_base_path,
                        correlation_method, n_permutations=N_BOOTSTRAP_PERMUTATIONS,
                        n_bootstraps=N_BOOTSTRAPS,
                        subset_fraction=BOOTSTRAP_SUBSET_FRACTION
                    )
                    # Plot bootstrap distribution
                    plotting.plot_permutation_histogram(bootstrap_rhos, observed_rho, observed_p,
                                                        result_base_path, task, kinematic,
                                                        direction, matrix_type, correlation_method)
                else:
                    # Single fast permutation test
                    bootstrap_rhos, bootstrap_pvalues = [], []
                    observed_rho, observed_p = permute(
                        avg_group1, avg_group2, task_markers, task, matrix_type,
                        kinematic, direction, result_base_path, correlation_method,
                        n_permutations
                    )

                # Always plot avg/std matrices
                matrix_label = "Standard deviation" if matrix_type == "std" else "Average"
                suptitle = f'{matrix_label} kinectomes — {direction} direction — {task}'
                fig_name = f'{matrix_type}_matrices_{direction}_{task}.png'
                plotting.plot_avg_matrices(avg_group1, avg_group2, group1, group2, task_markers,
                                           task, direction, matrix_type, result_base_path,
                                           observed_rho, observed_p, suptitle, fig_name)

                results[task][kinematic][direction] = {
                    'observed_rho': observed_rho,
                    'observed_p': observed_p,
                    'bootstrap_rhos': bootstrap_rhos,
                    'bootstrap_pvalues': bootstrap_pvalues,
                    f'{group1}_n': len(group1_matrices),
                    f'{group2}_n': len(group2_matrices)
                }

    return results



def bootstrap_permute(group1_matrices, group2_matrices, marker_list, task,
                      matrix_type, kinematic, direction, result_base_path,
                      correlation_method, n_permutations=500,
                      n_bootstraps=100, subset_fraction=0.8, random_seed=42):
    """Bootstrap-wrapped permutation test for small sample sizes.

    For each bootstrap iteration, random subsets of subjects are drawn from
    each group, group averages are computed, and a permutation test is run.
    This produces a distribution of p-values reflecting both the group
    difference and the uncertainty from small sample size.

    Parameters
    ----------
    group1_matrices : list[np.ndarray]
        Per-subject average or std kinectome matrices for group 1.
    group2_matrices : list[np.ndarray]
        Per-subject average or std kinectome matrices for group 2.
    marker_list : list[str]
        Marker names corresponding to matrix rows/columns.
    n_permutations : int
        Number of marker-shuffle permutations per bootstrap iteration.
    n_bootstraps : int
        Number of bootstrap iterations.
    subset_fraction : float
        Fraction of subjects to sample per iteration (default 0.8).

    Returns
    -------
    bootstrap_rhos : list[float]
        Observed Spearman ρ from each bootstrap iteration.
    bootstrap_pvalues : list[float]
        Permutation p-value from each bootstrap iteration.
    observed_rho : float
        Spearman ρ on the full (non-bootstrapped) group averages.
    observed_p : float
        Permutation p-value on the full group averages.
    """
    np.random.seed(random_seed)

    subgroups = get_adaptive_subgroups(group1_matrices[0], marker_list)

    def run_permutation(mat1, mat2):
        """Single permutation test between two matrices."""
        df1 = pd.DataFrame(mat1, index=marker_list, columns=marker_list)
        df2 = pd.DataFrame(mat2, index=marker_list, columns=marker_list)
        true_rho, _ = stats.spearmanr(upper(df1), upper(df2))
        m2_v = upper(df2)
        rhos = []
        for _ in range(n_permutations):
            shuffled = marker_list.copy()
            for group in subgroups.values():
                perm = np.random.permutation(group)
                for orig, shuf in zip(group, perm):
                    shuffled[shuffled.index(orig)] = shuf
            r, _ = stats.spearmanr(upper(df1.loc[shuffled, shuffled]), m2_v)
            rhos.append(r)
        perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos))) + 1) / (n_permutations + 1)
        return true_rho, perm_p

    # Observed values on full group averages
    avg1_full = np.mean(np.array(group1_matrices), axis=0)
    avg2_full = np.mean(np.array(group2_matrices), axis=0)
    observed_rho, observed_p = run_permutation(avg1_full, avg2_full)

    print(f"  Observed rho={observed_rho:.3f}, p={observed_p:.4f} "
          f"[{task} {kinematic} {direction} {matrix_type}]")

    # Bootstrap iterations
    n1 = max(2, int(len(group1_matrices) * subset_fraction))
    n2 = max(2, int(len(group2_matrices) * subset_fraction))

    bootstrap_rhos = []
    bootstrap_pvalues = []

    for i in range(n_bootstraps):
        if (i + 1) % 200 == 0:
            print(f"    Bootstrap {i+1}/{n_bootstraps}...")
        idx1 = np.random.choice(len(group1_matrices), n1, replace=True)
        idx2 = np.random.choice(len(group2_matrices), n2, replace=True)
        avg1 = np.mean(np.array(group1_matrices)[idx1], axis=0)
        avg2 = np.mean(np.array(group2_matrices)[idx2], axis=0)
        rho, p = run_permutation(avg1, avg2)
        bootstrap_rhos.append(rho)
        bootstrap_pvalues.append(p)

    return bootstrap_rhos, bootstrap_pvalues, observed_rho, observed_p

def bootstrap_permutation_test(variability_scores, task_names, kinematics_list, marker_list,
                             result_base_path, correlation_method, n_bootstraps=1000,
                             n_permutations=5000, matrix_type='std', subset_fraction=0.5,
                             random_seed=42):
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

    # Direction keys derived from the data: ['full'] or ['AP','ML','V'].
    directions = _infer_directions(variability_scores)

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
    plotting.create_bootstrap_plots(bootstrap_results, observed_rhos, task_names, kinematics_list, matrix_type, result_base_path)

    return bootstrap_results, observed_rhos