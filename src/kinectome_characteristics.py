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

def permutation_test_one_p(variability_scores, marker_list, result_base_path, matrix_type, n_permutations):
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
    
    # Get a sample subject from first group to extract task structure
    sample_subject = next(iter(variability_scores[group1].values()))
    tasks = sample_subject.keys()
    
    for task in tasks:
        results[task] = {}
        
        sample_task = sample_subject[task]
        kinematics = sample_task.keys()
        
        for kinematic in kinematics:
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
                
                plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path)
                
                # Compute the observed difference (single number)
                observed_diff = np.mean(avg_group2 - avg_group1)
                
                # Combine all samples into one pool
                combined = np.vstack([group1_matrices, group2_matrices])
                n1 = group1_matrices.shape[0]
                n_total = combined.shape[0]

                perm_diffs = np.zeros(n_permutations)

                for i in range(n_permutations):
                    np.random.shuffle(combined)
                    new_group1 = combined[:n1]
                    new_group2 = combined[n1:]
                    perm_diffs[i] = np.mean(np.mean(new_group2, axis=0) - np.mean(new_group1, axis=0))

                # Compute p-value
                p_value = (np.sum(np.abs(perm_diffs) >= np.abs(observed_diff)) + 1) / (n_permutations + 1)  # Avoid zero p-values
                
                # # Debug: Print stats
                # print(f"Observed Diff: {observed_diff}")
                # print(f"Permutation Mean: {np.mean(perm_diffs)}, Std: {np.std(perm_diffs)}")
                # print(f"Min Perm Diff: {np.min(perm_diffs)}, Max Perm Diff: {np.max(perm_diffs)}")

                results[task][kinematic][direction] = np.round(p_value, 3)
    
    return results


def plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path):
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
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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

    # Define result path
    result_folder = Path(result_base_path) / "avg_std_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"avg_matrices_{task}_{direction}_{matrix_type}.png"

    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')




def compare_between_groups(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices = calc_std_avg_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path)

    # permutation testing of the matrices
    # compare_matrices(matrices, result_path, matrix_type='std', n_permutations=10000)

    # std_p_values = permutation_test_one_p(matrices, marker_list, result_base_path, matrix_type='std', n_permutations=10000)

    # avg_p_values = permutation_test_one_p(matrices, marker_list, result_base_path, matrix_type='avg', n_permutations=10000)
    
    std_p_value_matrix = compare_matrices(matrices, matrix_type='std', n_permutations=10000)

    avg_p_value_matrix = compare_matrices(matrices, matrix_type='avg', n_permutations=10000)
   
    print()


















############################
def permutation_test(group1_matrices, group2_matrices, n_permutations=10000):
    """
    Perform permutation testing to compare two sets of matrices element-wise.
    
    Parameters:
    group1_matrices (np.ndarray): Array of shape (N1, rows, cols) for group 1.
    group2_matrices (np.ndarray): Array of shape (N2, rows, cols) for group 2.
    n_permutations (int): Number of permutations to perform (default: 10,000).
    
    Returns:
    np.ndarray: Matrix of p-values with shape (rows, cols).
    """
    # Get matrix dimensions
    n1, rows, cols = group1_matrices.shape
    n2 = group2_matrices.shape[0]
    
    # Compute observed difference in means
    observed_diff = np.mean(group2_matrices, axis=0) - np.mean(group1_matrices, axis=0)
    
    # Combine all samples into one pool
    combined = np.vstack([group1_matrices, group2_matrices])
    
    # Number of total samples
    n_total = n1 + n2
    
    # Initialize p-value matrix
    p_values = np.ones((rows, cols))
    
    # Perform permutation testing
    count_extreme = np.zeros((rows, cols))
    for _ in range(n_permutations):
        # Shuffle indices and split into new groups
        np.random.shuffle(combined)
        new_group1 = combined[:n1]
        new_group2 = combined[n1:]
        
        # Compute new difference in means
        perm_diff = np.mean(new_group2, axis=0) - np.mean(new_group1, axis=0)
        
        # Count cases where permuted difference is at least as extreme as observed
        count_extreme += np.abs(perm_diff) >= np.abs(observed_diff)
    
    # Compute p-values
    p_values = (count_extreme + 1) / (n_permutations + 1)  # Avoid zero p-values
    
    return p_values

def compare_matrices(variability_scores, matrix_type, n_permutations):
    """
    Compare variability matrices (e.g., standard deviation) between two groups
    using permutation testing.
    
    Parameters:
    variability_scores (dict): Dictionary containing variability matrices for each subject.
    matrix_type (str): Type of matrix to analyze ('avg' for mean or 'std' for standard deviation).
    n_permutations (int): Number of permutations for statistical testing.
    
    Returns:
    dict: Dictionary containing p-values and significance masks for each task, kinematic type, and direction.
    """
    results = {}
    
    group_names = list(variability_scores.keys())
    if len(group_names) != 2:
        raise ValueError("This function currently supports comparisons between exactly 2 groups")
    
    group1, group2 = group_names
    
    # Get a sample subject from first group to extract task structure
    sample_subject = next(iter(variability_scores[group1].values()))
    tasks = sample_subject.keys()
    
    for task in tasks:
        results[task] = {}
        
        sample_task = sample_subject[task]
        kinematics = sample_task.keys()
        
        for kinematic in kinematics:
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
                
                # Compute p-values using permutation testing
                p_values = permutation_test(group1_matrices, group2_matrices, n_permutations)
                
                
                # Store results
                results[task][kinematic][direction] = p_values
    
    return results