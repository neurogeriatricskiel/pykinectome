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


                # calculate the correlation between two matrices
                rho, p_value = stats.spearmanr(permutation.upper(avg_group1), permutation.upper(avg_group1))
                plotting.plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value)

                permutation.permute(avg_group1, avg_group2, marker_list, task, matrix_type, kinematic, direction, result_base_path)
             
    return results



def compare_between_groups(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path):

    # calculate the matrices of mean and standard deviation of the kinectomes (mean and sd matrix for each subject-task-kinematics-direction)
    matrices = calc_std_avg_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path)

    # permutation testing of the average and std matrices
    std_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='std', n_permutations=10000)

    avg_p_values = permutation_test_one_p(matrices, task_names, kinematics_list, marker_list, result_base_path, matrix_type='avg', n_permutations=10000)
    
    