import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
from src.data_utils import permutation
from src.data_utils import plotting
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


def calc_std_avg_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method):
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
                        
                        kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)
                        
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

def permutation_test_one_p(variability_scores, task_names, kinematics_list, marker_list, result_base_path, matrix_type, n_permutations, diff):
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
                plotting.plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value)

                if diff:
                    permutation.permute_difference_matrix(avg_group1, avg_group2, group1, group2, marker_list, task, kinematic, direction, result_base_path, matrix_type)
                permutation.permute(avg_group1, avg_group2, marker_list, task, matrix_type, kinematic, direction, result_base_path)

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



def reorder_difference_matrix(matrices, marker_list, result_base_path, correlation_method):
    """Calculates the difference nmatrix between Control group and the group of ineterest
    and reorders it from biggest to smallest difference"""

    avg_matrices = calc_avg_group_matrix(matrices)

    groups = list(avg_matrices.keys())
    tasks = avg_matrices[groups[0]].keys()
    kinematics = avg_matrices[groups[0]][next(iter(tasks))].keys()
    directions = avg_matrices[groups[0]][next(iter(tasks))][next(iter(kinematics))].keys()

    for task in tasks:
        for kin in kinematics:
            for direction in directions:
                # Get matrices
                mat_group1 = avg_matrices[groups[0]][task][kin][direction]
                mat_group2 = avg_matrices[groups[1]][task][kin][direction]

                # # Average left/right sides
                # mat1_avg, markers_avg = average_symmetric_matrix(mat_group1, marker_list)   # for single-body matrix
                # mat2_avg, _ = average_symmetric_matrix(mat_group2, marker_list)   # for single-body matrix

                # Absolute difference matrix
                # diff_mat = mat1_avg - mat2_avg   # for single-body matrix
                diff_mat = mat_group1 - mat_group2

                # Sort by highest total difference
                sort_order = np.argsort(-np.sum(diff_mat, axis=1))
                diff_mtrx_sorted = diff_mat[np.ix_(sort_order, sort_order)]
                # reordered_markers = [markers_avg[i] for i in sort_order]   # for single-body matrix
                reordered_markers = [marker_list[i] for i in sort_order]

                # Plot
                figname = f"{task}_{kin}_{direction}_absdiff_affect_{correlation_method}.png"
                plotting.plot_difference_matrix(diff_mtrx_sorted, reordered_markers, task, kin, direction, groups[0], groups[1], result_base_path, figname)

    print()


def compare_between_groups(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, result_base_path, full, correlation_method):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full, correlation_method)

    # permutation testing of the average and std matrices
    # std_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='std', n_permutations=10000)

    # avg_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='avg', n_permutations=10000)
    
    # diff_p_values =  permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='avg', n_permutations=10000, diff=False)

    reordered_difference_matrix = reorder_difference_matrix(matrices, marker_list_affect, result_base_path, correlation_method)

    print()


