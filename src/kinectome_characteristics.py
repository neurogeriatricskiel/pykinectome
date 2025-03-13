import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
import seaborn as sns
import csv
import pickle
from scipy import stats
from pathlib import Path
from statsmodels.stats.multitest import multipletests
from numpy.linalg import norm
import random
import scipy.stats
import pandas as pd


def calc_std_avg_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path):
    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

    # Store variability scores structured per subject, task, and direction
    variability_scores = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} for kinematics in kinematics_list} 
                                                       for task in task_names} 
                                                       for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} for kinematics in kinematics_list} 
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
                        
                        kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics)
                        
                        try:
                            # Initialize lists to store kinectomes for each direction
                            all_kinectomes = {"AP": [], "ML": [], "V": []}
                            
                            # Collect kinectomes by direction
                            for kinectome in kinectomes:
                                for idx, direction in enumerate(['AP', 'ML', 'V']):
                                    all_kinectomes[direction].append(kinectome[:, :, idx])


                            # Calculate average and standard deviation kinectomes for each direction
                            for direction in ['AP', 'ML', 'V']:
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

def permutation_test_one_p(variability_scores, task_names, kinematics_list, marker_list, result_base_path, matrix_type, n_permutations):
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
                
                # # upper triangle of the matrices (since they are mirrored)
                # upper1 = upper(avg_group1)
                # upper2 = upper(avg_group2)

                

                # Convert avg_group1 (numpy array) into a DataFrame
                df_group1 = pd.DataFrame(avg_group1, index=marker_list, columns=marker_list)
                df_group2 = pd.DataFrame(avg_group2, index=marker_list, columns=marker_list)
               
                
                # Now lets measure the similarity 
                rho, p_value = stats.spearmanr(upper(df_group1), upper(df_group2))
                print(f'rho = {np.round(rho, 2)} p_value = {p_value}  during {task} ({matrix_type}) for {kinematic} in {direction} direction')
                plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value)

                """Nonparametric permutation testing Monte Carlo"""
                np.random.seed(0)
                rhos = []
                n_iter = 5000
                true_rho, _ = stats.spearmanr(upper(df_group1), upper(df_group2))
                # matrix permutation, shuffle the groups
                m_ids = list(df_group1.columns)
                m2_v = upper(df_group2)
                for iter in range(n_iter):
                    np.random.shuffle(m_ids) # shuffle list 
                    r, _ = stats.spearmanr(upper(df_group1.loc[m_ids, m_ids]), m2_v)  
                    rhos.append(r)
                perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos)))+1)/(n_iter+1) # two-tailed test
                
                # """Nonparametric permutation testing Monte Carlo"""
                # np.random.seed(0)
                # rhos = []
                # n_iter = n_permutations

                # # Spearman correlation coefficient between the two group matrices (upper triangle)
                # true_rho, _ = stats.spearmanr(upper(avg_group1), upper(avg_group2))

                # # Permutation test
                # m2_v = upper(avg_group2)  # Keep one group fixed
                # for _ in range(n_iter):
                #     permuted_matrix = avg_group1.copy()
                #     np.random.shuffle(permuted_matrix)  # Shuffle rows randomly
                #     permuted_matrix = (permuted_matrix + permuted_matrix.T) / 2  # Ensure symmetry
                #     r, _ = stats.spearmanr(upper(permuted_matrix), m2_v)
                #     rhos.append(r)

                # # Compute two-tailed p-value
                # perm_p = ((np.sum(np.abs(true_rho) <= np.abs(rhos))) + 1) / (n_iter + 1)

                plot_permutation_histogram(rhos, true_rho, perm_p, result_base_path, task, kinematic, direction, matrix_type)
            
    
    return results

def plot_permutation_histogram(rhos, true_rho, perm_p, results_path, task, kinematic, direction, matrix_type):
    f,ax = plt.subplots()
    plt.hist(rhos,bins=20)
    ax.axvline(true_rho,  color = 'r', linestyle='--')
    ax.set(title=f"Permuted p: {perm_p:.3f}", ylabel="counts", xlabel="rho")
    os.chdir(Path(results_path, "avg_std_matrices"))
    plt.savefig(f'permutation_{task}_{kinematic}_{direction}_{matrix_type}.png', dpi=600)

# def upper(matrix):
#     """returns the upper triangle part of the mirrored matrix
#     """

#     upper_triangle = matrix[np.triu_indices(matrix.shape[0], k=1)]

#     return upper_triangle

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

def plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value):
    " Plots the average or std of the kinectomes based on task and direction"

    # Define marker ordering
    left_markers = [m for m in marker_list if m.startswith('l_')]
    right_markers = [m for m in marker_list if m.startswith('r_')]
    middle_markers = ['head', 'ster']
    ordered_marker_list = middle_markers + left_markers + right_markers

    # Get new indices based on the ordered list
    index_map = {marker: i for i, marker in enumerate(marker_list)}
    new_order = [index_map[m] for m in ordered_marker_list]

    # Reorder matrices
    reordered_group1 = avg_group1[np.ix_(new_order, new_order)]
    reordered_group2 = avg_group2[np.ix_(new_order, new_order)]

    # Find global min/max for consistent color scaling
    global_min = min(np.min(reordered_group1), np.min(reordered_group2))
    global_max = max(np.max(reordered_group1), np.max(reordered_group2))

    # define the limits of the colour bars
    norm = plt.Normalize(vmin=global_min, vmax=global_max)

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 7))

    # Plot Parkinson's group
    ax1 = axes[0]
    sns.heatmap(reordered_group2, cmap='viridis', norm = norm, center=0, xticklabels=ordered_marker_list, yticklabels=ordered_marker_list, ax=ax1)
    ax1.set_title(f'{group2} {matrix_type} matrix\nTask: {task}, Direction: {direction}')
    ax1.set_xticklabels(ordered_marker_list, rotation=90)
    ax1.set_yticklabels(ordered_marker_list, rotation=0)

    # Plot Control group
    ax2 = axes[1]
    sns.heatmap(reordered_group1, cmap='viridis', norm = norm, center=0, xticklabels=ordered_marker_list, yticklabels=ordered_marker_list, ax=ax2)
    ax2.set_title(f'{group1} {matrix_type} matrix\nTask: {task}, Direction: {direction}')
    ax2.set_xticklabels(ordered_marker_list, rotation=90)
    ax2.set_yticklabels(ordered_marker_list, rotation=0)

    plt.tight_layout()
    plt.suptitle(f"Spearman's rho = {np.round(rho, 3)}, p_value = {np.round(p_value, 3)}")

    # Define result path
    result_folder = Path(result_base_path) / "avg_std_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"avg_matrices_{task}_{direction}_{matrix_type}.png"

    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')



def compare_kinectomes(matrices, group1, group2, task, kinematic, direction, matrix_type, alpha, correction_method):
    """
    Statistically compare kinectomes between two groups.
    
    Parameters:
    -----------
    matrices : dict
        Nested dictionary containing the matrices
    group1, group2 : str
        Names of the groups to compare
    task, kinematic, direction : str
        Specific conditions to compare
    matrix_type : str
        'avg' for average kinectomes or 'std' for standard deviation kinectomes
    alpha : float
        Significance level for statistical tests
    correction_method : str
        Multiple comparison correction method ('fdr_bh', 'bonferroni', etc.)
        
    Returns:
    --------
    results : dict
        Dictionary containing p-values, corrected p-values, and significant mask
    """
    # Extract matrices for each group
    group1_matrices = []
    group1_subjects = []
    for subject in matrices[group1]:
        try:
            # Handle the case where there is no matrix (no kinectome)
            if matrices[group1][subject][task][kinematic][direction] is None:
                print(f"Missing data for subject {subject} in {group1} (None value)")
                continue

            matrix = matrices[group1][subject][task][kinematic][direction][matrix_type]
            group1_matrices.append(matrix)
            group1_subjects.append(subject)
        except KeyError or TypeError:
            print(f"Missing data for subject {subject} in {group1}")
            continue
    
    group2_matrices = []
    group2_subjects = []
    for subject in matrices[group2]:
        try:
            if matrices[group2][subject][task][kinematic][direction] is None:
                print(f"Missing data for subject {subject} in {group1} (None value)")
                continue
            
            matrix = matrices[group2][subject][task][kinematic][direction][matrix_type]
            group2_matrices.append(matrix)
            group2_subjects.append(subject)
        except KeyError:
            print(f"Missing data for subject {subject} in {group2}")
            continue
    
    # Convert to numpy arrays
    group1_matrices = np.array(group1_matrices)
    group2_matrices = np.array(group2_matrices)
    
    print(f"Comparing {len(group1_matrices)} subjects in {group1} vs {len(group2_matrices)} subjects in {group2}")
    
    # Get matrix dimensions
    n_rows, n_cols = group1_matrices[0].shape
    
    # Initialize arrays for results
    t_values = np.zeros((n_rows, n_cols))
    p_values = np.zeros((n_rows, n_cols))
    
    # Perform element-wise t-tests
    for i in range(n_rows):
        for j in range(n_cols):
            # Skip diagonal elements (self-correlations)
            if i == j:
                t_values[i, j] = 0
                p_values[i, j] = 1.0
                continue
            
            # Apply Fisher's z-transformation for correlation coefficients if analyzing 'avg' matrices
            if matrix_type == 'avg':
                # Fisher's z-transformation: z = 0.5 * ln((1+r)/(1-r))
                group1_z = 0.5 * np.log((1 + group1_matrices[:, i, j]) / (1 - group1_matrices[:, i, j]))
                group2_z = 0.5 * np.log((1 + group2_matrices[:, i, j]) / (1 - group2_matrices[:, i, j]))
                t_stat, p_val = stats.ttest_ind(group1_z, group2_z, equal_var=False)
            else:
                # For standard deviations, use raw values
                t_stat, p_val = stats.ttest_ind(group1_matrices[:, i, j], group2_matrices[:, i, j], equal_var=False)
            
            t_values[i, j] = t_stat
            p_values[i, j] = p_val
    
    # Correct for multiple comparisons
    # Flatten p-values for correction (excluding diagonal)
    p_values_flat = p_values[~np.eye(n_rows, dtype=bool)]
    reject, p_corrected_flat, _, _ = multipletests(p_values_flat, alpha=alpha, method=correction_method)
    
    # Reconstruct corrected p-value matrix
    p_corrected = np.ones((n_rows, n_cols))
    p_corrected[~np.eye(n_rows, dtype=bool)] = p_corrected_flat
    
    # Create significant difference mask
    significant_mask = (p_corrected < alpha)
    
    # Calculate average matrices for visualization
    avg_group1 = np.mean(group1_matrices, axis=0)
    avg_group2 = np.mean(group2_matrices, axis=0)
    difference = avg_group1 - avg_group2
    
    # Prepare results dictionary
    results = {
        'group1_mean': avg_group1,
        'group2_mean': avg_group2,
        'difference': difference,
        't_values': t_values,
        'p_values': p_values,
        'p_corrected': p_corrected,
        'significant_mask': significant_mask,
        'group1_subjects': group1_subjects,
        'group2_subjects': group2_subjects
    }
    
    return results

def visualize_kinectome_comparison(results, results_path, segment_labels=None, group1_name='Parkinson', group2_name='Control', 
                                 matrix_type='avg', figure_size=(18, 12), ):
    """
    Visualize the kinectome comparison results.
    
    Parameters:
    -----------
    results : dict
        Output from compare_kinectomes function
    segment_labels : list
        Labels for the body segments
    group1_name, group2_name : str
        Names of the groups for plot labels
    matrix_type : str
        'avg' or 'std' to label plots appropriately
    figure_size : tuple
        Size of the figure
    """
    if segment_labels is None:
        n = results['group1_mean'].shape[0]
        segment_labels = [f'Segment {i+1}' for i in range(n)]
    
    fig, axes = plt.subplots(2, 3, figsize=figure_size)
    
    # Plot average matrices
    sns.heatmap(results['group1_mean'], ax=axes[0, 0], cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=segment_labels, yticklabels=segment_labels)
    axes[0, 0].set_title(f'{group1_name} Group {matrix_type.upper()} Kinectome')
    
    sns.heatmap(results['group2_mean'], ax=axes[0, 1], cmap='RdBu_r', vmin=-1, vmax=1,
                xticklabels=segment_labels, yticklabels=segment_labels)
    axes[0, 1].set_title(f'{group2_name} Group {matrix_type.upper()} Kinectome')
    
    # Plot difference
    vmax_diff = max(abs(np.min(results['difference'])), abs(np.max(results['difference'])))
    sns.heatmap(results['difference'], ax=axes[0, 2], cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                xticklabels=segment_labels, yticklabels=segment_labels)
    axes[0, 2].set_title(f'Difference ({group1_name} - {group2_name})')
    
    # Plot t-values
    vmax_t = max(abs(np.min(results['t_values'])), abs(np.max(results['t_values'])))
    sns.heatmap(results['t_values'], ax=axes[1, 0], cmap='RdBu_r', vmin=-vmax_t, vmax=vmax_t,
                xticklabels=segment_labels, yticklabels=segment_labels)
    axes[1, 0].set_title('T-values')
    
    # Plot p-values (log scale for better visualization)
    sns.heatmap(-np.log10(results['p_values']), ax=axes[1, 1], cmap='viridis',
                xticklabels=segment_labels, yticklabels=segment_labels)
    axes[1, 1].set_title('-log10(p-values)')
    
    # Plot significant differences
    mask = ~results['significant_mask']  # Invert mask for seaborn
    sns.heatmap(results['difference'], ax=axes[1, 2], cmap='RdBu_r', vmin=-vmax_diff, vmax=vmax_diff,
                mask=mask, xticklabels=segment_labels, yticklabels=segment_labels)
    axes[1, 2].set_title('Significant Differences')
    
    plt.tight_layout()

    os.chdir(Path(results_path, "avg_std_matrices"))
    plt.savefig('test_plot1.png', dpi=600)



def perform_global_comparison(matrices, group1='Parkinson', group2='Control', 
                             task='walkPreferred', kinematic='acc', direction='AP',
                             matrix_type='avg'):
    """
    Perform global comparison of kinectomes using permutation testing.
    
    Parameters:
    -----------
    matrices : dict
        Nested dictionary containing the matrices
    group1, group2 : str
        Names of the groups to compare
    task, kinematic, direction : str
        Specific conditions to compare
    matrix_type : str
        'avg' for average kinectomes or 'std' for standard deviation kinectomes
        
    Returns:
    --------
    p_value : float
        Global p-value from permutation test
    """

    
    # Extract matrices
    group1_matrices = []
    for subject in matrices[group1]:
        try:
            matrix = matrices[group1][subject][task][kinematic][direction][matrix_type]
            group1_matrices.append(matrix)
        except KeyError:
            continue
    
    group2_matrices = []
    for subject in matrices[group2]:
        try:
            matrix = matrices[group2][subject][task][kinematic][direction][matrix_type]
            group2_matrices.append(matrix)
        except KeyError:
            continue
    
    # Convert to numpy arrays
    group1_matrices = np.array(group1_matrices)
    group2_matrices = np.array(group2_matrices)
    
    # Calculate mean matrices
    mean1 = np.mean(group1_matrices, axis=0)
    mean2 = np.mean(group2_matrices, axis=0)
    
    # Calculate observed distance between means (Frobenius norm)
    observed_distance = norm(mean1 - mean2, 'fro')
    
    # Combine all matrices for permutation
    all_matrices = np.vstack([group1_matrices, group2_matrices])
    n1 = len(group1_matrices)
    n2 = len(group2_matrices)
    n_total = n1 + n2
    
    # Permutation test
    n_permutations = 1000
    permutation_distances = []
    
    for _ in range(n_permutations):
        # Shuffle indices
        indices = list(range(n_total))
        random.shuffle(indices)
        
        # Split into new groups
        perm_group1 = all_matrices[indices[:n1]]
        perm_group2 = all_matrices[indices[n1:]]
        
        # Calculate means
        perm_mean1 = np.mean(perm_group1, axis=0)
        perm_mean2 = np.mean(perm_group2, axis=0)
        
        # Calculate distance
        perm_distance = norm(perm_mean1 - perm_mean2, 'fro')
        permutation_distances.append(perm_distance)
    
    # Calculate p-value
    p_value = np.mean([d >= observed_distance for d in permutation_distances])
    
    return {
        'observed_distance': observed_distance,
        'permutation_distances': permutation_distances,
        'p_value': p_value,
        'n_permutations': n_permutations
    }


def compare_between_groups(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path)

    # control_group_name = 'Control'

    # for diagnosis in diagnosis_list:
    #     diagnosis_group_name = diagnosis[10:].capitalize()
    #     for task in task_names:
    #         for kinematic in kinematics_list:
    #             for direction in ['AP', 'ML', 'V']:
    #                 for matrix_type in ['avg', 'std']:
    #                     avg_results = compare_kinectomes(matrices, diagnosis_group_name, control_group_name, 
    #                                                      task, kinematic, direction, matrix_type,
    #                                                      alpha = 0.05, correction_method='bonferroni')
    #                     visualize_kinectome_comparison(avg_results, result_base_path, marker_list, 'Parkinson', 'Control', 'Average')


    # permutation testing of the matrices
    # compare_matrices(matrices, result_path, matrix_type='std', n_permutations=10000)

    std_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='std', n_permutations=10000)

    avg_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='avg', n_permutations=10000)
    
    