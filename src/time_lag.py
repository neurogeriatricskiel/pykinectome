import numpy as np
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups, permutation, plotting
from scipy import stats
from statsmodels.stats.multitest import multipletests
from numpy.linalg import norm
import pandas as pd
from collections import defaultdict

def get_all_time_lag_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full):
    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

    # Choose what to store based on the `full` flag
    direction_template = {"full": None} if full else {"AP": None, "ML": None, "V": None}

    # Store variability scores structured per subject, task, and direction
    time_lag = {
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

                        # time lag matrices for the subject 
                        time_lag_matrices = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, 'time_lag')

                        # kinectomes made with cross correlation for the subject
                        cross_kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, 'cross')


                        try:
                            if not full:
                                # Initialize lists to store time_lag_matrices for each direction
                                all_time_lag_matrices = {"AP": [], "ML": [], "V": []}
                                
                                # Collect time lag matrices by direction
                                for time_lag_matrix in time_lag_matrices:
                                    for idx, direction in enumerate(['AP', 'ML', 'V']):
                                        all_time_lag_matrices[direction].append(time_lag_matrix[:, :, idx])
                            elif full:
                                all_time_lag_matrices = {'full' : []}
                                for time_lag_matrix in time_lag_matrices:
                                    all_time_lag_matrices['full'].append(time_lag_matrix)


                            # Calculate average and standard deviation kinectomes for each direction
                            for direction in ['AP', 'ML', 'V']:
                                if full:
                                    direction = 'full'
                                if all_time_lag_matrices[direction]:  # Check if the list is not empty
                                    # Stack the list of 2D arrays into a 3D array
                                    direction_stack = np.stack(all_time_lag_matrices[direction], axis=0)
                                    
                                    # Calculate average kinectome for this direction
                                    avg_time_lag_matrix= np.mean(direction_stack, axis=0)
                                    
                                    # Calculate standard deviation kinectome for this direction
                                    std_time_lag_matrix = np.std(direction_stack, axis=0)
                                    
                                    # Store the results in variability_scores - using explicit check for None
                                    # This avoids the numpy array comparison issue
                                    current_value = time_lag[group][sub_id][task_name][kinematics][direction]
                                    if current_value is None:
                                        time_lag[group][sub_id][task_name][kinematics][direction] = {
                                            "avg": avg_time_lag_matrix,
                                            "std": std_time_lag_matrix
                                        }
                                    else:
                                        # If you have multiple runs/tracking systems that should be combined,
                                        # you might need to implement a strategy here for combining them
                                        pass
                        except TypeError:
                            continue
    
    return time_lag


def time_lag_between_groups(time_lag, marker_list, group1, group2, 
                            task, kinematics, direction, apply_correction = True):
    """
    Compare time lags between two groups (e.g., Parkinson's vs Control) to identify 
    which marker pairs show significant differences.
    
    Parameters:
    -----------
    time_lag : dict
        Nested dictionary containing time lag matrices
    marker_list : list
        List of markers used in the time lag matrices
    group1, group2 : str
        Keys for the groups to compare (default: 'Parkinson' and 'Control')
    task : str
        Name of the task to analyze (default: 'walkPreferred')
    kinematics : str
        Type of kinematics data (default: 'acc')
    direction : str
        Direction of movement (default: 'AP')
    apply_correction : bool
        Whether to apply multiple comparison correction (default: False)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with results of statistical tests for each marker pair
    pandas.DataFrame
        DataFrame with significant marker pairs based on chosen threshold
    dict
        Dictionary of arrays organized by significance
    """
    if direction == 'full':
        marker_list = permutation.expand_marker_list(marker_list)

    # Get subject IDs for each group
    subjects_group1 = list(time_lag[group1].keys())
    subjects_group2 = list(time_lag[group2].keys())
    
    num_markers = len(marker_list)
    
    # Create empty lists to store values for each marker pair
    group1_values = {}
    group2_values = {}
    
    results = []

    # Store results by base marker for group-wise correction
    results_by_segment = defaultdict(list)

    for i in range(num_markers):
        base_marker = marker_list[i]
        for j in range(i+1, num_markers):
            key = f"{marker_list[i]}_to_{marker_list[j]}"
            
            # Collect data
            g1 = [time_lag[group1][s][task][kinematics][direction]['avg'][i, j]
                  for s in subjects_group1
                  if time_lag[group1][s][task][kinematics][direction] is not None]

            g2 = [time_lag[group2][s][task][kinematics][direction]['avg'][i, j]
                  for s in subjects_group2
                  if time_lag[group2][s][task][kinematics][direction] is not None]

            if g1 and g2:
                t_stat, p_val = stats.ttest_ind(g1, g2, equal_var=False)
                mean1, mean2 = np.mean(g1), np.mean(g2)
                pooled_std = np.sqrt((np.std(g1, ddof=1)**2 + np.std(g2, ddof=1)**2) / 2)
                cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0

                result = {
                    'marker_pair': key,
                    'base_marker': base_marker,
                    'p_value': p_val,
                    'cohens_d': cohens_d,
                    'mean_diff': mean1 - mean2,
                    f'mean_{group1}': mean1,
                    f'mean_{group2}': mean2,
                }

                results.append(result)
                results_by_segment[base_marker].append(result)

    results_df = pd.DataFrame(results)

    # Apply per-segment correction
    if apply_correction and not results_df.empty:
        from statsmodels.stats.multitest import multipletests
        corrected_results = []
        
        for base_marker, res_list in results_by_segment.items():
            p_vals = [r['p_value'] for r in res_list]
            reject, corrected_p, _, _ = multipletests(p_vals, method='fdr_bh')
            for idx, r in enumerate(res_list):
                r['corrected_p_value'] = corrected_p[idx]
                r['significant'] = reject[idx]
                corrected_results.append(r)

        results_df = pd.DataFrame(corrected_results)
        sig_threshold = 'corrected_p_value'
    else:
        results_df['significant'] = results_df['p_value'] < 0.05
        sig_threshold = 'p_value'

    # Organize into significance groups
    significance_dict = {
        'high_significance': results_df[results_df[sig_threshold] < 0.01],
        'medium_significance': results_df[(results_df[sig_threshold] >= 0.01) & 
                                          (results_df[sig_threshold] < 0.05)],
        'no_significance': results_df[results_df[sig_threshold] >= 0.05]
    }

    sig_results = results_df[results_df['significant']]

    return results_df.sort_values(sig_threshold), sig_results, significance_dict

def calculate_average_group_timelag_matrix(time_lag_data):

    """
    Calculate average matrices for each group, task, and kinematic direction.
    
    Args:
        time_lag_data: Dictionary containing time lag data organized by group, subject, task, kinematic
        
    Returns:
        Dictionary of group averages organized by group, task, kinematic direction
    """
    # Initialize dictionaries to store group data
    group_data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    group_averages = {}
    
    # Extract all unique groups, tasks, and kinematics from the data
    all_groups = set()
    all_tasks = set()
    all_kinematics = set()
    all_directions = set()
    
    # First pass: Identify all unique values
    for group in time_lag_data:
        all_groups.add(group)
        for subject in time_lag_data[group]:
            for task in time_lag_data[group][subject]:
                all_tasks.add(task)
                for kinematic in time_lag_data[group][subject][task]:
                    all_kinematics.add(kinematic)
                    for direction in time_lag_data[group][subject][task][kinematic]:
                        all_directions.add(direction)
    
    # Second pass: Organize data by group, task, kinematic, direction
    for group in all_groups:
        group_averages[group] = {}
        
        for task in all_tasks:
            group_averages[group][task] = {}
            
            for kinematic in all_kinematics:
                group_averages[group][task][kinematic] = {}
                
                for direction in all_directions:
                    # Collect all 'avg' matrices for this combination
                    matrices = []
                    
                     # Process each subject individually and handle exceptions at the subject level
                    for subject in time_lag_data.get(group, {}):
                        try:
                            # Check if this subject has data for this combination
                            if (task in time_lag_data[group][subject] and 
                                kinematic in time_lag_data[group][subject][task] and
                                direction in time_lag_data[group][subject][task][kinematic] and
                                'avg' in time_lag_data[group][subject][task][kinematic][direction]):
                                
                                avg_matrix = time_lag_data[group][subject][task][kinematic][direction]['avg']
                                matrices.append(avg_matrix)
                        except (TypeError, KeyError, AttributeError):
                            # If any error occurs for this subject, just skip it and continue with others
                            continue

                    if matrices:
                        # If matrices have different shapes, use element-wise mean
                        group_avg = np.mean(matrices, axis=0)
                        group_std = np.std(matrices, axis=0)
                            
                        group_averages[group][task][kinematic][direction] = {
                            'avg': group_avg,
                            'std': group_std
                        }
    
    return group_averages

def main(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list_affect, result_base_path, full):

    # a dictionary containing all time lag matrices 
    all_time_lag_matrices = get_all_time_lag_matrices(diagnosis_list, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, full)

    average_time_lag_matrices = calculate_average_group_timelag_matrix(all_time_lag_matrices)

    # PD vs. controls (Do pwPD have more time lag than controls)?

    # plotting.plot_lag_heatmap(average_time_lag_matrices['Parkinson']['walkPreferred']['acc']['AP']['avg'], marker_list_affect)

    results_df, sig_results, significance_dict = time_lag_between_groups(all_time_lag_matrices, marker_list_affect,group1='Parkinson', group2='Control', 
                                task='walkFast', kinematics='acc', direction='full')
    
    # matrix1 = average_time_lag_matrices['Parkinson']['walkPreferred']['acc']['AP']['avg']
    # matrix2 = average_time_lag_matrices['Control']['walkPreferred']['acc']['AP']['avg']
    # marker_list = marker_list_affect
    # matrix_type = 'avg'
    # task = 'walkPreferred'
    # kinematic = 'acc'
    # direction = 'AP'
    # correlation_method = 'time_lag'


    # permutation.permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method)


    # more affected vs. less affected (does more affected side lag more than the less affected?)

    print()


