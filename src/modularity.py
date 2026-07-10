import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import networkx as nx
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
from src.data_utils import plotting
import seaborn as sns
import csv
import pickle
from pathlib import Path
from src.data_utils import permutation
from src.data_utils.plotting import draw_graph_with_selected_weights, draw_graph_with_weights
from collections import defaultdict
from scipy import stats
from sklearn.metrics import adjusted_rand_score
from statsmodels.stats.multitest import multipletests
import warnings
warnings.filterwarnings('ignore')
from itertools import combinations
from src.graph_utils.graphs import build_graph, jaccard_complete_communities
from config import KINECTOME_SAVE_PATH


def run_louvain(G, clustering_method, num_iterations=100, resolution=1.0):
    """Runs Louvain community detection multiple times and returns all partitions."""
    partitions= []

    for _ in range(num_iterations):
        if clustering_method == 'louvain':
            partition= nx.community.louvain_communities(G, weight='weight', resolution=resolution)
        elif clustering_method == 'leiden':
            partition= nx.community.leiden_communities(G, weight='weight', resolution=resolution)
        partitions.append(partition)

    return partitions

#used
def compute_allegiance_matrix(partitions, marker_list, num_nodes):
    """Constructs an allegiance matrix from Louvain community partitions."""
    allegiance_matrix = np.zeros((num_nodes, num_nodes))

    for partition in partitions:
        node_to_community = {}
        for comm_idx, community in enumerate(partition):
            for node in community:
                node_to_community[node] = comm_idx  # Map each marker name to its community index

        for i, marker_i in enumerate(marker_list):
            for j, marker_j in enumerate(marker_list):
                if node_to_community.get(marker_i) == node_to_community.get(marker_j):  
                    allegiance_matrix[i, j] += 1

    allegiance_matrix /= len(partitions)  # Normalize by number of iterations
    
    return allegiance_matrix

#used
def all_allegiance_matrices_for_subject(kinectomes, marker_list, clustering_method, resolution=1.0):
    """ A function which saves allegiance matrices built from the kinectomes
    note:
        it is not computed per group, so all allegiance matrices (from one subject per trial and per direction) are put into all_allegiance_matrices dict
    """
    all_allegiance_matrices = {"AP": [], "ML": [], "V": []}
   
    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list)
        
        from config import LOUVAIN_ITERATIONS
        if len(graphs) == 1:
            # If only one (full) graph, assign it to AP direction (idx 0)
            G = graphs[0]
            partitions = run_louvain(G, clustering_method, num_iterations=LOUVAIN_ITERATIONS, resolution=resolution)
            marker_list_exp = permutation.expand_marker_list(marker_list)
            allegiance_matrix = compute_allegiance_matrix(partitions, marker_list_exp, num_nodes=G.number_of_nodes())
            all_allegiance_matrices["AP"].append(np.array(allegiance_matrix))
        else:
            # If multiple graphs (3 directions), process each one
            for idx, direction in enumerate(["AP", "ML", "V"]):
                G = graphs[idx]
                partitions = run_louvain(G, clustering_method, num_iterations=LOUVAIN_ITERATIONS, resolution=resolution)
                allegiance_matrix = compute_allegiance_matrix(partitions, marker_list, num_nodes=G.number_of_nodes())
                all_allegiance_matrices[direction].append(np.array(allegiance_matrix))
           
        # Visualize one of the graphs (for debugging purposes)
        # draw_graph_with_weights(G)
   
    return all_allegiance_matrices

#used
def modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method, clustering_method, resolution=1.0):

    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])


    # Store variability scores structured per subject, task, and direction
    all_avg_allegiance = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                              for kinematics in kinematics_list} 
                                                              for task in task_names} 
                                                              for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                    for kinematics in kinematics_list}
                                    for task in task_names} 
                                    for sub_id in matched_control_sub_ids},
    }

    all_std_allegiance = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                              for kinematics in kinematics_list} 
                                                              for task in task_names} 
                                                              for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                    for kinematics in kinematics_list}
                                    for task in task_names} 
                                    for sub_id in matched_control_sub_ids},
    }

    debug_ids = ['pp006', 'pp008', 'pp021']


    _all_subs = disease_sub_ids + matched_control_sub_ids
    print(f"  modularity_analysis: Louvain on {len(_all_subs)} subjects × gait cycles...")
    for kinematics in kinematics_list:
        for _sub_idx, sub_id in enumerate(_all_subs):
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            print(f"  [{_sub_idx+1}/{len(_all_subs)}] {sub_id} ({group})", flush=True)

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

                        if kinectomes is None:
                            continue

                        # Strip excluded markers (e.g. upper limb for dual tasks) BEFORE
                        # community detection, so they never influence the allegiance matrices
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        effective_marker_list = marker_list
                        if exclude:
                            from src.data_utils.data_loader import exclude_markers_from_kinectome
                            reduced_kinectomes = []
                            current_markers = marker_list
                            for k in kinectomes:
                                k_reduced, current_markers = exclude_markers_from_kinectome(k, current_markers, exclude)
                                reduced_kinectomes.append(k_reduced)
                            kinectomes = reduced_kinectomes
                            effective_marker_list = current_markers

                        allegiance_matrices = all_allegiance_matrices_for_subject(kinectomes, effective_marker_list, clustering_method, resolution)
                        
                        avg_sub_allegiance_matrices = {}
                        std_sub_allegiance_matrices = {}

                        for direction in allegiance_matrices.keys():

                            # compute average allegiance matrix for one subject
                            avg_sub_allegiance_matrices[direction] = np.mean(allegiance_matrices[direction], axis=0)

                            # calculate variability (as std) of allegiance matrices 
                            # the resulting varibility matrix shows which body segments consistently belong to the same community (low std) and which fluctuate more (high std)
                            std_sub_allegiance_matrices[direction] = np.std(allegiance_matrices[direction], axis=0)
                            

                            # add the avg and std allegiance matrices to the dictionary 
                            all_avg_allegiance[group][sub_id][task_name][kinematics][direction] = avg_sub_allegiance_matrices[direction]

                            all_std_allegiance[group][sub_id][task_name][kinematics][direction] = std_sub_allegiance_matrices[direction]




                            # if visualise:
                            #     visualise_allegiance_matrix(avg_sub_allegiance_matrices[direction], marker_list, sub_id, task_name, direction,
                            #                                 figname=f'allegiance_matrix_{sub_id}_{task_name}_{direction}.png')
                            #     visualise_allegiance_matrix(std_sub_allegiance_matrices[direction], marker_list, sub_id, task_name, direction,
                                                            # figname=f'std_allegiance_matrix_{sub_id}_{task_name}_{direction}.png')
                            


    # Define result path
    result_folder = Path(result_path) / "modularity" / "allegiance_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save paths for the pickle files
    # Resolution is included so different MODULARITY_RESOLUTION_LIST values don't
    # overwrite each other's cached Louvain results
    avg_save_path = result_folder / f"avg_allegiance_matrices_{'full_' if full else ''}{correlation_method}_{clustering_method}_reso_{resolution}.pkl"
    std_save_path = result_folder / f"std_allegiance_matrices_{'full_' if full else ''}{correlation_method}_{clustering_method}_reso_{resolution}.pkl"

    # Save dictionaries as pickle files
    with open(avg_save_path, "wb") as f:
        pickle.dump(all_avg_allegiance, f)

    with open(std_save_path, "wb") as f:
        pickle.dump(all_std_allegiance, f)

#used
def load_allegiance_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method, clustering_method, resolution=1.0):
    """ checks if the allegiance matrices are calculated and saved as a pickle file (and loads them). 
    otherwise calculates them and saves as a pickle file

    returns:
    a dict containing average allegiance matrices per group, subject, task, kinematics, and direction
    a dict containing std (as a matrix) of allegiance matrices per group, subject, task, kinematics, and direction
    """
        # Define result path
    result_folder = Path(result_path) / "modularity" / "allegiance_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Resolution is included so different MODULARITY_RESOLUTION_LIST values don't
    # overwrite/reuse each other's cached Louvain results
    avg_save_path = result_folder / f"avg_allegiance_matrices_{'full_' if full else ''}{correlation_method}_{clustering_method}_reso_{resolution}.pkl"
    std_save_path = result_folder / f"std_allegiance_matrices_{'full_' if full else ''}{correlation_method}_{clustering_method}_reso_{resolution}.pkl"
    
    # if allegiance matrices are not calculated
    if not avg_save_path.exists() and not std_save_path.exists():
        modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method, clustering_method, resolution)
        # load the allegiane matrices once they are calculated
        with open (avg_save_path, 'rb') as avg_file:
            avg_allegience_matrices = pickle.load(avg_file)
        with open (std_save_path, 'rb') as std_file:
            std_allegience_matrices = pickle.load(std_file)
    
    # load pickle files if they already exist
    else:
        with open (avg_save_path, 'rb') as avg_file:
            avg_allegience_matrices = pickle.load(avg_file)
        with open (std_save_path, 'rb') as std_file:
            std_allegience_matrices = pickle.load(std_file)

    return avg_allegience_matrices, std_allegience_matrices

#used
def calculate_avg_allg_mtrx(avg_allegiance_matrices, full):

    """
    Compute average matrices for each group, task, kinematic type, and direction.
    
    Parameters:
    - avg_allegiance_matrices: Dictionary with nested structure containing all allegiance matrices
    
    Returns:
    - Dictionary with structure {group: {task: {kinematic: {direction: avg_matrix}}}}
    """
    group_avg_matrices = {}
    
    # Iterate through each group
    for group, participants in avg_allegiance_matrices.items():
        group_avg_matrices[group] = {}
        
        # Find all unique tasks across all participants
        all_tasks = set()
        for participant_data in participants.values():
            all_tasks.update(participant_data.keys())
        
        # Initialize task dictionaries
        for task in all_tasks:
            group_avg_matrices[group][task] = {}
        
        # Find all unique kinematic types and directions
        all_kinematics = {}
        for participant_id, participant_data in participants.items():
            for task in participant_data:
                if task not in all_kinematics:
                    all_kinematics[task] = {}
                
                for kinematic, direction_data in participant_data[task].items():
                    if kinematic not in all_kinematics[task]:
                        all_kinematics[task][kinematic] = set()
                    
                    all_kinematics[task][kinematic].update(direction_data.keys())


# Compute averages for each task, kinematic, direction combination
        for task in all_tasks:
            for kinematic in all_kinematics.get(task, {}):
                group_avg_matrices[group][task][kinematic] = {}
                
                if full:
                    # Handle 66x66 matrices - check if AP key has a 66x66 matrix
                    valid_matrices = []
                    
                    for participant_id, participant_data in participants.items():
                        if (task in participant_data and
                            kinematic in participant_data[task] and
                            'AP' in participant_data[task][kinematic]):
                            
                            matrix = participant_data[task][kinematic]['AP']
                            
                            # Check if matrix is valid (square, non-empty)
                            if (matrix is not None and 
                                hasattr(matrix, 'shape') and 
                                matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and
                                matrix.shape[0] > 0 and
                                not np.isnan(matrix).all()):
                                valid_matrices.append(matrix)
                    
                    # Compute average if we have valid matrices
                    if valid_matrices:
                        avg_matrix = np.nanmean(valid_matrices, axis=0)
                        group_avg_matrices[group][task][kinematic]['full'] = avg_matrix
                
                else:
                    # Original code for 22x22 matrices with directions
                    for direction in all_kinematics[task][kinematic]:
                        # Collect matrices for this combination
                        valid_matrices = []
                       
                        for participant_id, participant_data in participants.items():
                            if (task in participant_data and
                                kinematic in participant_data[task] and
                                direction in participant_data[task][kinematic]):
                                # Get the matrix
                                matrix = participant_data[task][kinematic][direction]
                               
                                # Only include non-None matrices with actual content and a valid square shape
                                # (shape varies by task once EXCLUDE_MARKERS_BY_TASK removes markers)
                                if (matrix is not None and 
                                    hasattr(matrix, 'shape') and 
                                    matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1] and
                                    matrix.shape[0] > 0 and
                                    not np.isnan(matrix).all()):
                                    valid_matrices.append(matrix)
                       
                        # Compute average if we have valid matrices
                        if valid_matrices:
                            # All matrices should be numpy arrays with shape (22, 22)
                            avg_matrix = np.nanmean(valid_matrices, axis=0)
                            group_avg_matrices[group][task][kinematic][direction] = avg_matrix
   
    return group_avg_matrices

# def plot_all_allegiance_matrices(allegiance_matrices, marker_list, result_base_path, correlation_method, full):
#     """ visualise and save all group allegiance matrices as .png
#     """
    
#     for group in allegiance_matrices.keys():
#         for task in allegiance_matrices[group].keys():
#             for kinematic in allegiance_matrices[group][task].keys():
#                 for direction in allegiance_matrices[group][task][kinematic].keys():
#                     matrix = allegiance_matrices[group][task][kinematic][direction]
#                     plotting.visualise_allegiance_matrix(matrix, marker_list, group, task, kinematic, direction, result_base_path, correlation_method, full)

#used
def plot_all_allegiance_matrices_with_communities(allegiance_matrices, group_communities, marker_list, result_base_path, correlation_method, full, resolution=None):
    """ visualise and save all group allegiance matrices as .png with community-based ordering

    marker_list is reduced per-task to match EXCLUDE_MARKERS_BY_TASK, so the matrix shape
    always matches the number of labels passed to the plotting function (otherwise a
    downstream shape-mismatch fallback can silently mislabel axes with expanded ×3
    direction-suffixed names, as if this were a combined "full" kinectome).

    resolution, if given, is folded into the correlation_method string passed to the
    plotting function purely so the saved filename (which is built from group/task/
    kinematic/correlation_method/direction, with no resolution component of its own)
    doesn't get overwritten by every resolution in a MODULARITY_RESOLUTION_LIST sweep.
    """
    from config import EXCLUDE_MARKERS_BY_TASK

    for group in allegiance_matrices.keys():
        for task in allegiance_matrices[group].keys():
            exclude = EXCLUDE_MARKERS_BY_TASK.get(task, [])
            effective_marker_list = [m for m in marker_list if m not in exclude] if exclude else marker_list
            for kinematic in allegiance_matrices[group][task].keys():
                for direction in allegiance_matrices[group][task][kinematic].keys():
                    matrix = allegiance_matrices[group][task][kinematic][direction]
                    communities = group_communities[group][task][kinematic][direction]
                    if matrix is None or not communities:
                        continue
                    correlation_method_label = (
                        f"{correlation_method}_reso{resolution}" if resolution is not None else correlation_method
                    )
                    plotting.visualise_allegiance_matrix_with_communities(matrix, communities, effective_marker_list, group, task, kinematic, direction, result_base_path, correlation_method_label, full)

#used
def extract_communities_threshold(allegiance_matrix, threshold):
    """
    Extract communities using threshold method
    """
    n_nodes = allegiance_matrix.shape[0]
    
    # Create adjacency matrix based on threshold
    adj_matrix = (allegiance_matrix >= threshold).astype(int)
    np.fill_diagonal(adj_matrix, 0)  # Remove self-loops
    
    # Create graph and find connected components
    G = nx.from_numpy_array(adj_matrix)
    communities = list(nx.connected_components(G))
    
    return communities

#used
def calc_subject_communities(avg_subject_allegience_matrices, threshold):
    """ returns a dictionary with the functional communities based on the threshold method for each subject separately"""
    all_subject_communities = {}

    for group in avg_subject_allegience_matrices.keys():
        all_subject_communities[group] = {}

        for sub_id in avg_subject_allegience_matrices[group].keys():
            all_subject_communities[group][sub_id] = {}

            for task in avg_subject_allegience_matrices[group][sub_id].keys():
                all_subject_communities[group][sub_id][task] = {}

                for kinematics in avg_subject_allegience_matrices[group][sub_id][task].keys():
                    all_subject_communities[group][sub_id][task][kinematics] = {}

                    for direction in avg_subject_allegience_matrices[group][sub_id][task][kinematics].keys():
                        all_subject_communities[group][sub_id][task][kinematics][direction] = {}

                        allegiance_matrix = avg_subject_allegience_matrices[group][sub_id][task][kinematics][direction]

                        if allegiance_matrix is None:
                            continue
                        else:
                            communities = extract_communities_threshold(allegiance_matrix, threshold)
                            all_subject_communities[group][sub_id][task][kinematics][direction] = communities
    
    return all_subject_communities

#used
def calc_group_communities(average_group_allegiance_matrices, threshold):
    """ returns a dictionary with the functional communities based on the threshold method - the community structure is for the group """

    group_communities = {}

    for group in average_group_allegiance_matrices.keys():
        group_communities[group] = {}
    
        for task in average_group_allegiance_matrices[group].keys():
            group_communities[group][task] = {}

            for kinematics in average_group_allegiance_matrices[group][task]:
                group_communities[group][task][kinematics] = {}

                for direction in average_group_allegiance_matrices[group][task][kinematics].keys():
                    group_communities[group][task][kinematics][direction] = {}

                    allegiance_matrix = average_group_allegiance_matrices[group][task][kinematics][direction]
                    communities = extract_communities_threshold(allegiance_matrix, threshold)

                    group_communities[group][task][kinematics][direction] = communities
    
    return group_communities

# def calc_community_fit(subject_communities, group_communities):
#     """Calculate fit between individual and group community structures using ARI."""
#     fits = {}
    
#     for group in subject_communities:
#         fits[group] = {}
#         for subject in subject_communities[group]:
#             fits[group][subject] = {}
#             for task in subject_communities[group][subject]:
#                 fits[group][subject][task] = {}
#                 for kinematic in subject_communities[group][subject][task]:
#                     fits[group][subject][task][kinematic] = {}
#                     for direction in subject_communities[group][subject][task][kinematic]:
#                         # Get individual and group communities
#                         ind_comm = subject_communities[group][subject][task][kinematic][direction]
#                         grp_comm = group_communities[group][task][kinematic][direction]
#                         if not bool(ind_comm): # check if the dict is empty for subjects with no data
#                             continue
                        
#                         # Convert to node labels for ARI calculation
#                         max_node = max(max(comm) for comm in ind_comm + grp_comm)
#                         ind_labels = np.zeros(max_node + 1)
#                         grp_labels = np.zeros(max_node + 1)
                        
#                         for i, comm in enumerate(ind_comm):
#                             for node in comm:
#                                 ind_labels[node] = i
                        
#                         for i, comm in enumerate(grp_comm):
#                             for node in comm:
#                                 grp_labels[node] = i
                        
#                         # Calculate ARI as fit measure
#                         fits[group][subject][task][kinematic][direction] = adjusted_rand_score(ind_labels, grp_labels)
    
#     return fits

# def calc_community_fit_stats(fits):
#     """Compare fits between two groups using appropriate statistical test."""
#     results = {}

#     group1 = list(fits.keys())[0]
#     group2 = list(fits.keys())[1]

#     # Get all combinations of task/kinematic/direction
#     all_combinations = set()
#     for group in fits:
#         for subject in fits[group]:
#             for task in fits[group][subject]:
#                 for kinematic in fits[group][subject][task]:
#                     for direction in fits[group][subject][task][kinematic]:
#                         all_combinations.add((task, kinematic, direction))
    
#     for task, kinematic, direction in all_combinations:
#         # Extract fits for both groups
#         g1_fits = [fits[group1][subj][task][kinematic][direction] 
#                    for subj in fits[group1] 
#                    if task in fits[group1][subj] and kinematic in fits[group1][subj][task] 
#                    and direction in fits[group1][subj][task][kinematic]]
        
#         g2_fits = [fits[group2][subj][task][kinematic][direction] 
#                    for subj in fits[group2] 
#                    if task in fits[group2][subj] and kinematic in fits[group2][subj][task] 
#                    and direction in fits[group2][subj][task][kinematic]]
        
#         if len(g1_fits) < 3 or len(g2_fits) < 3:
#             continue
            
#         # Test normality
#         _, p1 = stats.shapiro(g1_fits)
#         _, p2 = stats.shapiro(g2_fits)
        
#         # Choose appropriate test
#         if p1 > 0.05 and p2 > 0.05:
#             # Both normal - use t-test
#             stat, p = stats.ttest_ind(g1_fits, g2_fits, alternative='less')  # Test if g1 < g2
#             test_used = 'ttest'
#         else:
#             # Non-normal - use Mann-Whitney U
#             stat, p = stats.mannwhitneyu(g1_fits, g2_fits, alternative='less')
#             test_used = 'mannwhitney'
        
#         results[(task, kinematic, direction)] = {
#             'group1_mean': np.mean(g1_fits),
#             'group2_mean': np.mean(g2_fits),
#             'statistic': stat,
#             'p_value': np.round(p, 3),
#             'test_used': test_used,
#             'n1': len(g1_fits),
#             'n2': len(g2_fits)
#         }
    
#     return pd.DataFrame(results)

# def calculate_modularity_scores(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities):
#     """Calculate modularity scores for group communities and individual subjects."""
#     group_modularity = {}
#     subject_modularity = {}
    
#     for group in group_communities:
#         group_modularity[group] = {}
#         subject_modularity[group] = {}
        
#         for task in group_communities[group]:
#             group_modularity[group][task] = {}
#             subject_modularity[group][task] = {}
            
#             for kinematic in group_communities[group][task]:
#                 group_modularity[group][task][kinematic] = {}
#                 subject_modularity[group][task][kinematic] = {}
                
#                 for direction in group_communities[group][task][kinematic]:
#                     communities = group_communities[group][task][kinematic][direction]
                    
#                     # Group modularity using group allegiance matrix
#                     group_matrix = average_group_allegiance_matrices[group][task][kinematic][direction]
#                     G_group = nx.from_numpy_array(group_matrix)
#                     group_modularity[group][task][kinematic][direction] = nx.community.modularity(G_group, communities, weight='weight')
                    
#                     # Subject modularity scores
#                     subject_modularity[group][task][kinematic][direction] = {}
#                     for subject in avg_subject_allegiance_matrices[group]:
#                         if task in avg_subject_allegiance_matrices[group][subject] and \
#                            kinematic in avg_subject_allegiance_matrices[group][subject][task] and \
#                            direction in avg_subject_allegiance_matrices[group][subject][task][kinematic]:
                            
#                             subj_matrix = avg_subject_allegiance_matrices[group][subject][task][kinematic][direction]
#                             if subj_matrix is None:
#                                 continue 
#                             elif len(communities) <= 1:
#                                 subject_modularity[group][task][kinematic][direction][subject] = np.nan
#                             else:    
#                                 G_subj = nx.from_numpy_array(subj_matrix)
#                                 subject_modularity[group][task][kinematic][direction][subject] = nx.community.modularity(G_subj, communities, weight='weight')
          
#     return group_modularity, subject_modularity

# def compare_modularity_between_groups(subject_modularity):
#     """Compare modularity scores between groups using appropriate statistical test."""
#     results = {}
    
#     group1 = list(subject_modularity.keys())[0]
#     group2 = list(subject_modularity.keys())[1]

#     # Get all combinations
#     all_combinations = set()
#     for group in subject_modularity:
#         for task in subject_modularity[group]:
#             for kinematic in subject_modularity[group][task]:
#                 for direction in subject_modularity[group][task][kinematic]:
#                     all_combinations.add((task, kinematic, direction))
    
#     for task, kinematic, direction in all_combinations:
#         # Extract modularity scores
#         g1_scores = list(subject_modularity[group1][task][kinematic][direction].values()) if \
#                    task in subject_modularity[group1] and kinematic in subject_modularity[group1][task] and \
#                    direction in subject_modularity[group1][task][kinematic] else []
        
#         g2_scores = list(subject_modularity[group2][task][kinematic][direction].values()) if \
#                    task in subject_modularity[group2] and kinematic in subject_modularity[group2][task] and \
#                    direction in subject_modularity[group2][task][kinematic] else []
        
#         if len(g1_scores) < 3 or len(g2_scores) < 3:
#             continue
        
#         # Test normality and choose appropriate test
#         _, p1 = stats.shapiro(g1_scores)
#         _, p2 = stats.shapiro(g2_scores)
        
#         if p1 > 0.05 and p2 > 0.05:
#             stat, p = stats.ttest_ind(g1_scores, g2_scores)
#             test_used = 'ttest'
#         else:
#             stat, p = stats.mannwhitneyu(g1_scores, g2_scores)
#             test_used = 'mannwhitney'
        
#         results[(task, kinematic, direction)] = {
#             'group1_mean': np.mean(g1_scores),
#             'group2_mean': np.mean(g2_scores),
#             'group1_std': np.std(g1_scores),
#             'group2_std': np.std(g2_scores),
#             'statistic': stat,
#             'p_value': np.round(p, 3),
#             'test_used': test_used,
#             'n1': len(g1_scores),
#             'n2': len(g2_scores)
#         }
    
#     return pd.DataFrame(results)

# def calculate_within_community_density(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities):
#     """Calculate within-community density for each community with community identification."""
#     group_densities = {}
#     subject_densities = {}
    
#     for group in group_communities:
#         group_densities[group] = {}
#         subject_densities[group] = {}
        
#         for task in group_communities[group]:
#             group_densities[group][task] = {}
#             subject_densities[group][task] = {}
            
#             for kinematic in group_communities[group][task]:
#                 group_densities[group][task][kinematic] = {}
#                 subject_densities[group][task][kinematic] = {}
                
#                 for direction in group_communities[group][task][kinematic]:
#                     communities = group_communities[group][task][kinematic][direction]
                    
#                     # Group within-community densities
#                     group_matrix = average_group_allegiance_matrices[group][task][kinematic][direction]
#                     group_densities[group][task][kinematic][direction] = {}
                    
#                     for i, community in enumerate(communities):
#                         nodes = list(community)
#                         community_key = f"community_{i}_nodes_{sorted(nodes)}"
                        
#                         if len(nodes) > 1:
#                             submatrix = group_matrix[np.ix_(nodes, nodes)]
#                             mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
#                             density = np.mean(submatrix[mask]) if mask.sum() > 0 else 0
#                         else:
#                             density = 0
#                         group_densities[group][task][kinematic][direction][community_key] = density
                    
#                     # Subject within-community densities
#                     subject_densities[group][task][kinematic][direction] = {}
#                     for subject in avg_subject_allegiance_matrices[group]:
#                         if task in avg_subject_allegiance_matrices[group][subject] and \
#                            kinematic in avg_subject_allegiance_matrices[group][subject][task] and \
#                            direction in avg_subject_allegiance_matrices[group][subject][task][kinematic]:
                            
#                             subj_matrix = avg_subject_allegiance_matrices[group][subject][task][kinematic][direction]

#                             if subj_matrix is None:
#                                 continue
#                             else:
#                                 subject_densities[group][task][kinematic][direction][subject] = {}
                            
#                             for i, community in enumerate(communities):
#                                 nodes = list(community)
#                                 community_key = f"community_{i}_nodes_{sorted(nodes)}"
                                
#                                 if len(nodes) > 1:
#                                     submatrix = subj_matrix[np.ix_(nodes, nodes)]
#                                     mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
#                                     density = np.mean(submatrix[mask]) if mask.sum() > 0 else 0
#                                 else:
#                                     density = 0
#                                 subject_densities[group][task][kinematic][direction][subject][community_key] = density
    
#     return group_densities, subject_densities

# def plot_weight_distributions(weight_distributions, task_names):
#     """
#     Generates a 3x3 grid of histograms showing the distribution of graph edge weights.

#     Args:
#         weight_distributions (dict): A dictionary containing the edge weights, structured by 
#                                      group, task, and direction.
#         task_names (list): A list of task names (e.g., ['slow', 'normal', 'fast'])
#                            which will be used for row labels.
#     """
#     groups = list(weight_distributions.keys())
#     directions = ["AP", "ML", "V"]
#     # Use matplotlib's default 'C0', 'C1', etc. for consistent coloring
#     colors = {group: f'C{i}' for i, group in enumerate(groups)} 
    
#     if len(task_names) != 3:
#         raise ValueError("This function is designed for exactly 3 tasks to create a 3x3 grid.")

#     fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True, sharey=True)
#     fig.suptitle('Distribution of Graph Edge Weights by Task and Direction', fontsize=16)

#     for row, task in enumerate(task_names):
#         for col, direction in enumerate(directions):
#             ax = axes[row, col]

#             # Plot histogram for each group on the same subplot
#             for group in groups:
#                 weights = weight_distributions[group][task][direction]
#                 if weights: # Check if the list is not empty
#                     ax.hist(weights, bins=30, alpha=0.7, label=group, density=True, color=colors[group])

#                     # Calculate and plot quartiles
#                     q1, median, q3 = np.percentile(weights, [25, 50, 75])
                    
#                     # Plot median as a solid vertical line
#                     ax.axvline(median, color=colors[group], linestyle='-', linewidth=2, alpha=0.9)
                    
#                     # Plot 25th and 75th percentiles as dashed lines
#                     ax.axvline(q1, color=colors[group], linestyle='--', linewidth=1.5, alpha=0.9)
#                     ax.axvline(q3, color=colors[group], linestyle='--', linewidth=1.5, alpha=0.9)

#             # Set titles for the top row
#             if row == 0:
#                 ax.set_title(direction, fontsize=12)

#             # Set Y-axis labels for the first column
#             if col == 0:
#                 ax.set_ylabel(f"{task.capitalize()}\nDensity", fontsize=12)

#             # Set X-axis labels for the bottom row
#             if row == 2:
#                 ax.set_xlabel("Edge Weight", fontsize=12)

#     # Create a single legend for the entire figure
#     handles, labels = ax.get_legend_handles_labels()
#     fig.legend(handles, labels, loc='upper right', fontsize=12)
    
#     plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make room for suptitle

#used 
def calculate_modularity(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                            correlation_method, consensus_communities, resolution):

    """description
    """ 

    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])

    # Store variability scores structured per subject, task, and direction
    modularity_scores = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                              for kinematics in kinematics_list} 
                                                              for task in task_names} 
                                                              for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                    for kinematics in kinematics_list}
                                    for task in task_names} 
                                    for sub_id in matched_control_sub_ids},
    }

    weight_distributions = {
        f"{diagnosis[0][10:].capitalize()}": {task: {"AP": [], "ML": [], "V": []} for task in task_names},
        "Control": {task: {"AP": [], "ML": [], "V": []} for task in task_names},
    }


    debug_ids = ['pp006', 'pp008', 'pp021']


    all_sub_ids = disease_sub_ids + matched_control_sub_ids
    n_total = len(all_sub_ids)
    print(f"  modularity_analysis: processing {n_total} subjects...")

    for kinematics in kinematics_list:
        for sub_idx, sub_id in enumerate(all_sub_ids):
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            print(f"  [{sub_idx+1}/{n_total}] {sub_id} ({group})", flush=True)

            # for sub_id in debug_ids:
            #     group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"

            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run

                        kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)

                        if kinectomes is None:
                            print(f"    No kinectomes for {sub_id}/{task_name}. Skipping.")
                            continue
                        
                        # average (between gait cycles) kinectomes in each movement direction
                        avg_kinectomes = np.mean(kinectomes, axis=0)

                        # Apply marker exclusion for dual tasks
                        from config import EXCLUDE_MARKERS_BY_TASK
                        from src.data_utils.data_loader import exclude_markers_from_kinectome
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        effective_marker_list = marker_list
                        if exclude:
                            avg_kinectomes, effective_marker_list = exclude_markers_from_kinectome(
                                avg_kinectomes, marker_list, exclude
                            )

                        # make graphs from the kinectomes (returns a list of three graph objects for AP, ML, and V directions)
                        graphs = build_graph(avg_kinectomes, effective_marker_list)  # include all edges (no threshold)

                        # calculate modularity for each subject (per direction)
                        modularity_per_subject = []

                        for G in graphs:
                            modularity_per_dir = nx.community.modularity(G, consensus_communities, weight = 'weight', resolution = resolution)
                            modularity_per_subject.append(np.round(modularity_per_dir, 2))

                        # Store modularity scores in the main dictionary
                        directions = ["AP", "ML", "V"]
                        for i, direction in enumerate(directions):
                            modularity_scores[group][sub_id][task_name][kinematics][direction] = modularity_per_subject[i]

                            current_graph = graphs[i]
                            weights = [data['weight'] for u, v, data in current_graph.edges(data=True)]
                            weight_distributions[group][task_name][direction].extend(weights)
                            


    # generates the distribution of the graph weights (for each group, walking speed, and direction) to determine the threshold values to build the graphs
    # plot_weight_distributions(weight_distributions, task_names)

    return modularity_scores

# def convert_modularity_to_dataframe(modularity_scores):
#     """
#     Convert nested modularity dictionary to long-format DataFrame for analysis.
    
#     Parameters:
#     -----------
#     modularity_scores : dict
#         Nested dictionary with structure: {group: {subject: {task: {kinematics: {direction: value}}}}}
    
#     Returns:
#     --------
#     pd.DataFrame
#         Long-format DataFrame with columns: group, subject_id, task, kinematics, direction, modularity
#     """
#     data_rows = []
    
#     for group, subjects in modularity_scores.items():
#         for subject_id, tasks in subjects.items():
#             for task, kinematics_dict in tasks.items():
#                 for kinematics, directions in kinematics_dict.items():
#                     for direction, modularity_value in directions.items():
#                         if modularity_value is not None:  # Only include non-None values
#                             data_rows.append({
#                                 'group': group,
#                                 'subject_id': subject_id,
#                                 'task': task,
#                                 'kinematics': kinematics,
#                                 'direction': direction,
#                                 'modularity': float(modularity_value)
#                             })
    
#     return pd.DataFrame(data_rows)

# def perform_modularity_statistical_tests(df, alpha=0.05, correction_method='fdr_bh'):
#     """
#     Perform statistical tests comparing groups within each task and direction.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Long-format DataFrame from convert_modularity_to_dataframe
#     alpha : float
#         Significance level
#     correction_method : str
#         Multiple comparison correction method ('bonferroni', 'fdr_bh', etc.)
    
#     Returns:
#     --------
#     dict : Statistical test results
#     """
    
#     results = {
#         'test_results': [],
#         'summary_table': None
#     }
    
#     tasks = df['task'].unique()
#     directions = df['direction'].unique()
#     groups = df['group'].unique()
    
#     print(f"Statistical Analysis: Modularity Comparison")
#     print(f"Groups: {list(groups)}")
#     print(f"Tasks: {list(tasks)}")
#     print(f"Directions: {list(directions)}")
#     print(f"Multiple comparison correction: {correction_method.upper()}")
#     print("=" * 60)
    
#     # Perform tests for each task-direction combination
#     test_data = []
    
#     for task in tasks:
#         for direction in directions:
#             # Filter data for this specific task and direction
#             subset = df[(df['task'] == task) & (df['direction'] == direction)]
            
#             if len(subset) == 0:
#                 continue
            
#             # Get data for each group
#             group_data = {}
#             for group in groups:
#                 group_subset = subset[subset['group'] == group]['modularity'].values
#                 if len(group_subset) > 0:
#                     group_data[group] = group_subset
            
#             if len(group_data) < 2:
#                 continue
            
#             # Perform statistical test
#             group_names = list(group_data.keys())
#             data1 = group_data[group_names[0]]
#             data2 = group_data[group_names[1]]
            
#             # Check for normality and equal variances
#             if len(data1) >= 3 and len(data2) >= 3:
#                 _, p_norm1 = stats.shapiro(data1)
#                 _, p_norm2 = stats.shapiro(data2)
#                 _, p_levene = stats.levene(data1, data2)
                
#                 # Use parametric test if both groups are normal and have equal variances
#                 if p_norm1 > 0.05 and p_norm2 > 0.05 and p_levene > 0.05:
#                     statistic, p_value = stats.ttest_ind(data1, data2)
#                     test_name = "Independent t-test"
#                 else:
#                     statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
#                     test_name = "Mann-Whitney U test"
#             else:
#                 # Use non-parametric test for small samples
#                 statistic, p_value = stats.mannwhitneyu(data1, data2, alternative='two-sided')
#                 test_name = "Mann-Whitney U test"
            
#             # Store results
#             test_info = {
#                 'task': task,
#                 'direction': direction,
#                 'test_name': test_name,
#                 'statistic': statistic,
#                 'p_value': p_value,
#                 'n1': len(data1),
#                 'n2': len(data2),
#                 'mean1': np.mean(data1),
#                 'mean2': np.mean(data2),
#                 'std1': np.std(data1),
#                 'std2': np.std(data2),
#                 'median1': np.median(data1),
#                 'median2': np.median(data2),
#                 'groups': group_names
#             }
            
#             test_data.append(test_info)
#             results['test_results'].append(test_info)
    
#     # Apply multiple comparison correction
#     if test_data:
#         p_values = [test['p_value'] for test in test_data]
#         rejected, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method=correction_method)
        
#         print(f"\nResults with {correction_method.upper()} correction (α = {alpha}):")
#         print(f"{'Task':<12} {'Direction':<10} {'Test':<20} {'Statistic':<12} {'p-value':<12} {'p-corrected':<12} {'Significant':<12}")
#         print("-" * 100)
        
#         for i, test in enumerate(test_data):
#             sig = "Yes" if rejected[i] else "No"
#             print(f"{test['task']:<12} {test['direction']:<10} {test['test_name']:<20} "
#                   f"{test['statistic']:<12.3f} {test['p_value']:<12.4f} {p_corrected[i]:<12.4f} {sig:<12}")
            
#             # Add corrected results
#             test['p_corrected'] = p_corrected[i]
#             test['significant'] = rejected[i]
        
#         # Create summary DataFrame
#         summary_data = []
#         for test in test_data:
#             summary_data.append({
#                 'Task': test['task'],
#                 'Direction': test['direction'],
#                 'Test': test['test_name'],
#                 'N1': test['n1'],
#                 'N2': test['n2'],
#                 'Mean1': f"{test['mean1']:.3f}",
#                 'Mean2': f"{test['mean2']:.3f}",
#                 'Statistic': f"{test['statistic']:.3f}",
#                 'p-value': f"{test['p_value']:.4f}",
#                 'p-corrected': f"{test['p_corrected']:.4f}",
#                 'Significant': 'Yes' if test['significant'] else 'No'
#             })
        
#         results['summary_table'] = pd.DataFrame(summary_data)
        
#         print(f"\nSummary:")
#         print(f"Total tests performed: {len(test_data)}")
#         print(f"Significant results (after correction): {sum(rejected)}/{len(test_data)}")
    
#     return results

# def create_modularity_violin_plot(df, stats_results, resolution, save_path=None, figsize=(15, 12)):
#     """
#     Create violin plot similar to the example image with 3x3 subplots.
    
#     Parameters:
#     -----------
#     df : pd.DataFrame
#         Long-format DataFrame from convert_modularity_to_dataframe
#     stats_results : dict
#         Results from perform_modularity_statistical_tests
#     save_path : str, optional
#         Path to save the figure
#     figsize : tuple
#         Figure size (width, height)
#     """
    
#     tasks = sorted(df['task'].unique())
#     directions = sorted(df['direction'].unique())
#     groups = sorted(df['group'].unique())
    
#     fig, axes = plt.subplots(len(tasks), len(directions), figsize=figsize, sharex=True, sharey=True)
    
#     # Color palette for groups
#     colors = ['lightblue', 'lightcoral']  # Adjust colors as needed
#     group_colors = {group: colors[i] for i, group in enumerate(groups)}
    
#     for i, task in enumerate(tasks):
#         for j, direction in enumerate(directions):
#             ax = axes[i, j]
            
#             # Filter data for this subplot
#             subset = df[(df['task'] == task) & (df['direction'] == direction)]
            
#             if len(subset) == 0:
#                 ax.set_title(f'{direction}\n{task}')
#                 ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
#                 continue
            
#             # Create violin plot
#             parts = ax.violinplot([subset[subset['group'] == group]['modularity'].values 
#                                  for group in groups], 
#                                 positions=range(len(groups)), widths=0.6)
            
#             # Color the violin plots
#             for k, pc in enumerate(parts['bodies']):
#                 pc.set_facecolor(colors[k])
#                 pc.set_alpha(0.7)
            
#             # Add individual data points
#             for k, group in enumerate(groups):
#                 group_data = subset[subset['group'] == group]['modularity'].values
#                 if len(group_data) > 0:
#                     # Add jitter to x-coordinates
#                     x_jitter = np.random.normal(k, 0.05, len(group_data))
#                     ax.scatter(x_jitter, group_data, alpha=0.6, s=30, 
#                              color='darkblue' if k == 0 else 'darkred', zorder=3)
                    
#                     # Add mean marker
#                     ax.scatter(k, np.mean(group_data), marker='D', s=50, 
#                              color='white', edgecolors='black', linewidth=2, zorder=4)
            
#             # Find statistical significance for this combination
#             sig_symbol = ''
#             for test in stats_results['test_results']:
#                 if test['task'] == task and test['direction'] == direction:
#                     if test.get('significant', False):
#                         if test['p_corrected'] < 0.001:
#                             sig_symbol = '***'
#                         elif test['p_corrected'] < 0.01:
#                             sig_symbol = '**'
#                         elif test['p_corrected'] < 0.05:
#                             sig_symbol = '*'
#                     break
            
#             # Add significance annotation
#             if sig_symbol:
#                 y_max = subset['modularity'].max()
#                 y_min = subset['modularity'].min()
#                 y_range = y_max - y_min
#                 y_sig = y_max + 0.1 * y_range
                
#                 ax.plot([0, 1], [y_sig, y_sig], 'k-', linewidth=1)
#                 ax.plot([0, 0], [y_sig - 0.02 * y_range, y_sig], 'k-', linewidth=1)
#                 ax.plot([1, 1], [y_sig - 0.02 * y_range, y_sig], 'k-', linewidth=1)
#                 ax.text(0.5, y_sig + 0.02 * y_range, sig_symbol, ha='center', va='bottom', fontsize=12, fontweight='bold')
            
#             # Formatting
#             ax.set_xticks(range(len(groups)))
#             ax.set_xticklabels(groups, rotation=45, ha='right')
#             ax.grid(True, alpha=0.3, axis='y')
            
#             # Set title (direction on top row, task on left column)
#             if i == 0:
#                 ax.set_title(f'{direction}', fontsize=12, fontweight='bold')
#             if j == 0:
#                 ax.text(-0.3, 0.5, f'{task}', rotation=90, ha='center', va='center', 
#                        transform=ax.transAxes, fontsize=12, fontweight='bold')
    
#     # Set common y-label
#     fig.text(0.04, 0.5, 'Modularity', va='center', rotation='vertical', fontsize=14, fontweight='bold')
    
#     # Add main title
#     plt.suptitle(f'Modularity Comparison Between Groups (resolution = {str(resolution)})', fontsize=16, fontweight='bold', y=1.01)
    
#     # Add legend
#     legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.7, label=group) 
#                       for i, group in enumerate(groups)]
#     fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.95))
    
#     plt.tight_layout()
    
#     # Save if path provided
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         print(f"Figure saved to: {save_path}")
    
#     plt.show()

# def analyze_modularity_data(modularity_scores, resolution, save_path=None, correction_method='fdr_bh'):
#     """
#     Complete analysis pipeline for modularity data.
    
#     Parameters:
#     -----------
#     modularity_scores : dict
#         Your nested modularity dictionary
#     save_path : str, optional
#         Path to save the figure
#     correction_method : str
#         Multiple comparison correction method
    
#     Returns:
#     --------
#     tuple : (DataFrame, statistical_results)
#     """
    
#     # Convert to DataFrame
#     df = convert_modularity_to_dataframe(modularity_scores)
#     print(f"Data converted to DataFrame: {len(df)} observations")
#     print(f"Groups: {df['group'].unique()}")
#     print(f"Tasks: {df['task'].unique()}")
#     print(f"Directions: {df['direction'].unique()}")
    
#     # Perform statistical tests
#     stats_results = perform_modularity_statistical_tests(df, correction_method=correction_method)
    
#     # Create visualization
#     create_modularity_violin_plot(df, stats_results, resolution, save_path=save_path)
    
#     return df, stats_results

#used 
def calc_modularity_per_subject(subject_communities, marker_list, diagnosis, kinematics_list, task_names, base_path, tracking_systems, runs, pd_on, full, correlation_method, resolution):
    """ Calculates modularity scores for each subject using his own community structure (as identifying using louvain algorithm)
    """

    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])

    # Store variability scores structured per subject, task, and direction
    modularity_subject_scores = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                              for kinematics in kinematics_list} 
                                                              for task in task_names} 
                                                              for sub_id in disease_sub_ids},

        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                    for kinematics in kinematics_list}
                                    for task in task_names} 
                                    for sub_id in matched_control_sub_ids},
    }



    debug_ids = ['pp006', 'pp008', 'pp021']


    all_sub_ids = disease_sub_ids + matched_control_sub_ids
    n_total = len(all_sub_ids)
    print(f"  modularity_analysis: processing {n_total} subjects...")

    for kinematics in kinematics_list:
        for sub_idx, sub_id in enumerate(all_sub_ids):
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            print(f"  [{sub_idx+1}/{n_total}] {sub_id} ({group})", flush=True)

            # for sub_id in debug_ids:
            #     group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"

            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run

                        kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)

                        if kinectomes is None:
                            print(f"    No kinectomes for {sub_id}/{task_name}. Skipping.")
                            continue
                        
                        # average (between gait cycles) kinectomes in each movement direction
                        avg_kinectomes = np.mean(kinectomes, axis=0)

                        # Strip excluded markers so the graph matches the (already-reduced)
                        # community structure computed for this task
                        from config import EXCLUDE_MARKERS_BY_TASK
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        effective_marker_list = marker_list
                        if exclude:
                            from src.data_utils.data_loader import exclude_markers_from_kinectome
                            avg_kinectomes, effective_marker_list = exclude_markers_from_kinectome(
                                avg_kinectomes, marker_list, exclude
                            )

                        # make graphs from the kinectomes (returns a list of three graph objects for AP, ML, and V directions)
                        graphs = build_graph(avg_kinectomes, effective_marker_list) # bound_value = 0.4 - 25%, ~0.95 - median, 1.25 - 75%


                        for i, direction in enumerate(subject_communities[group][sub_id][task_name][kinematics].keys()):
                                G = graphs[i]
                                subject_community_structure = subject_communities[group][sub_id][task_name][kinematics][direction]
                                
                                # subject's community structure with marker names
                                mapped_structure = [{effective_marker_list[i] for i in community} for community in subject_community_structure]

                                subject_modularity = np.round(nx.community.modularity(G, mapped_structure, weight='weight', resolution=resolution), 2)
                                modularity_subject_scores[group][sub_id][task_name][kinematics][direction] = subject_modularity
                        

    return modularity_subject_scores

#used
#used
def plot_modularity_vs_resolution(q_by_resolution, task_names, kinematics_list, result_base_path, label):
    """
    Plot mean modularity (Q) vs resolution — one figure per task/kinematic, with
    AP/ML/V as subplots and one line per group. Mirrors the kind of resolution-
    sweep analysis in Pmc13246872 (Q rising as resolution drops from 1.5 to 0.1).

    Parameters
    ----------
    q_by_resolution : dict
        {resolution: {group: {sub_id: {task: {kinematics: {direction: Q}}}}}}
        i.e. one entry per resolution value tested, each holding the full
        modularity_scores dict returned by calculate_modularity() or
        calc_modularity_per_subject() for that resolution.
    label : str
        'CONSENSUS' or 'LOUVAIN' — used only in the title/filename, to keep the
        two modularity-scoring pathways in separate files.
    """
    resolutions = sorted(q_by_resolution.keys())
    if not resolutions:
        return

    groups = list(q_by_resolution[resolutions[0]].keys())
    directions = ["AP", "ML", "V"]

    result_folder = Path(result_base_path) / "modularity"
    result_folder.mkdir(parents=True, exist_ok=True)

    for task in task_names:
        for kinematics in kinematics_list:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            any_data = False

            for ax, direction in zip(axes, directions):
                for group in groups:
                    color = 'tab:blue' if group == 'Control' else 'tab:red'
                    xs, means, sems = [], [], []
                    for resolution in resolutions:
                        vals = []
                        subs = q_by_resolution[resolution].get(group, {})
                        for sub_id in subs:
                            q = subs[sub_id].get(task, {}).get(kinematics, {}).get(direction)
                            if q is not None:
                                vals.append(q)
                        if vals:
                            xs.append(resolution)
                            means.append(np.mean(vals))
                            sems.append(np.std(vals) / np.sqrt(len(vals)) if len(vals) > 1 else 0)
                    if xs:
                        any_data = True
                        ax.errorbar(xs, means, yerr=sems, marker='o', label=group, color=color, capsize=3)

                ax.set_title(f'{task} - {direction}')
                ax.set_xlabel('Resolution')
                ax.set_ylabel('Modularity (Q)')
                ax.axhline(0, color='gray', linewidth=0.5, linestyle='--')
                ax.legend()
                ax.grid(alpha=0.3)

            if not any_data:
                plt.close(fig)
                continue

            fig.suptitle(f'Modularity vs Resolution ({label} communities) — {task} ({kinematics})',
                         fontsize=14, fontweight='bold')
            plt.tight_layout()

            save_path = result_folder / f"modularity_vs_resolution_{label}_{task}_{kinematics}.png"
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved: {save_path}")


def calculate_between_community_ratio(G, communities):
    """
    Calculate ratio of inter-community to intra-community connectivity strength
    Higher ratio = stronger between vs within community connections
    For PD: Higher ratio could indicate compensatory inter-community coupling
    """
    # Create node to community mapping
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    intra_weights = []
    inter_weights = []

    for edge in G.edges(data=True):
        node1, node2, data = edge
        weight = abs(data['weight'])  # Use absolute value for correlation strength

        comm1 = node_to_community.get(node1, -1)
        comm2 = node_to_community.get(node2, -1)

        if comm1 == comm2 and comm1 != -1:
            intra_weights.append(weight)
        elif comm1 != -1 and comm2 != -1:
            inter_weights.append(weight)

    mean_intra = np.mean(intra_weights) if intra_weights else 0
    mean_inter = np.mean(inter_weights) if inter_weights else 0

    # Return ratio: higher values = stronger inter relative to intra
    return mean_inter / mean_intra if mean_intra > 0 else 0

def calculate_cross_community_coupling(G, communities):
    """
    Calculate absolute strength of connectivity between communities
    Higher coupling = stronger absolute inter-community connections
    For PD: Higher coupling could indicate rigidity/compensatory mechanisms
    """
    # Create node to community mapping
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    inter_weights = []

    for edge in G.edges(data=True):
        node1, node2, data = edge
        weight = abs(data['weight'])  # Use absolute correlation strength

        comm1 = node_to_community.get(node1, -1)
        comm2 = node_to_community.get(node2, -1)

        # Only count inter-community connections
        if comm1 != comm2 and comm1 != -1 and comm2 != -1:
            inter_weights.append(weight)

    # Return absolute inter-community coupling strength
    return np.mean(inter_weights) if inter_weights else 0

def calculate_intra_community_strength_per_community(G, communities):
    """
    Calculate absolute strength of connectivity within each community separately
    Returns a list of strength values, one for each community
    """
    # Create node to community mapping
    node_to_community = {}
    for i, community in enumerate(communities):
        for node in community:
            node_to_community[node] = i

    # Initialize lists for each community's intra-weights
    community_weights = [[] for _ in range(len(communities))]

    for edge in G.edges(data=True):
        node1, node2, data = edge
        weight = abs(data['weight'])  # Use absolute correlation strength

        comm1 = node_to_community.get(node1, -1)
        comm2 = node_to_community.get(node2, -1)

        # Only count intra-community connections
        if comm1 == comm2 and comm1 != -1:
            community_weights[comm1].append(weight)

    # Calculate mean strength for each community
    community_strengths = []
    for weights in community_weights:
        if weights:
            community_strengths.append(np.mean(weights))
        else:
            community_strengths.append(0)

    return community_strengths


#used
def calculate_community_strength_metrics(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                            correlation_method, consensus_communities, resolution):

    """
    Calculate Between-Community Ratio and Cross-Community Coupling metrics
    Specifically designed for fully connected graphs to investigate:
    1. Whether communities are weaker in pwPD (lower intra-community strength)
    2. Whether there are stronger inter-community connections (rigidity/compensation)

    NOTE: this version scores subjects against the FIXED, literature-based
    ``consensus_communities`` (same partition for every subject). Because the
    partition never changes, these metrics are mathematically independent of
    ``resolution`` — resolution only affects the *value* of a modularity score
    computed elsewhere (see calculate_modularity), not a plain mean-edge-weight
    metric evaluated on a fixed grouping. Use
    calculate_community_strength_metrics_own() for the Louvain-detected,
    resolution-sensitive counterpart.
    """ 

    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])


    # Initialize dictionaries for the two key metrics
    metrics = {
        'between_community_ratio': {},
        'cross_community_coupling': {}
    }
    
    # Initialize structure for each metric
    for metric_name in metrics.keys():
        metrics[metric_name] = {
            f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                                  for kinematics in kinematics_list} 
                                                                  for task in task_names} 
                                                                  for sub_id in disease_sub_ids},

            "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                        for kinematics in kinematics_list}
                                        for task in task_names} 
                                        for sub_id in matched_control_sub_ids},
        }

    # Add intra-community strength as a bonus metric
    metrics['intra_community_strength'] = {
        f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                                              for kinematics in kinematics_list} 
                                                              for task in task_names} 
                                                              for sub_id in disease_sub_ids},
        "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                    for kinematics in kinematics_list}
                                    for task in task_names} 
                                    for sub_id in matched_control_sub_ids},
    }

    # Main calculation loop
    _csm_ids = disease_sub_ids + matched_control_sub_ids
    print(f"  calculate_community_strength_metrics: processing {len(_csm_ids)} subjects...")
    for kinematics in kinematics_list:
        for _csm_idx, sub_id in enumerate(_csm_ids):
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            print(f"  [{_csm_idx+1}/{len(_csm_ids)}] {sub_id} ({group})", flush=True)

            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run
                        
                        # Load kinectomes
                        from config import EXCLUDE_MARKERS_BY_TASK
                        kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)

                        if kinectomes is None:
                            continue
                        
                        # Average kinectomes across gait cycles
                        avg_kinectomes = np.mean(kinectomes, axis=0)

                        # Strip excluded markers (e.g. upper limb for dual tasks) before
                        # building the graph, so they never enter the strength calculation
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        effective_marker_list = marker_list
                        if exclude:
                            from src.data_utils.data_loader import exclude_markers_from_kinectome
                            avg_kinectomes, effective_marker_list = exclude_markers_from_kinectome(
                                avg_kinectomes, marker_list, exclude
                            )

                        # Build graphs
                        graphs = build_graph(avg_kinectomes, effective_marker_list)

                        # Calculate metrics for each direction
                        directions = ["AP", "ML", "V"]
                        for i, direction in enumerate(directions):
                            G = graphs[i]
                            
                            # Calculate the three key metrics
                            between_ratio = calculate_between_community_ratio(G, consensus_communities)
                            cross_coupling = calculate_cross_community_coupling(G, consensus_communities)
                            intra_strength = calculate_intra_community_strength_per_community(G, consensus_communities)
                            
                            # Store results
                            metrics['between_community_ratio'][group][sub_id][task_name][kinematics][direction] = np.round(between_ratio, 4)
                            metrics['cross_community_coupling'][group][sub_id][task_name][kinematics][direction] = np.round(cross_coupling, 4)
                            metrics['intra_community_strength'][group][sub_id][task_name][kinematics][direction] = np.round(intra_strength, 4)

    return metrics


def align_communities_across_groups(reference_communities, other_communities):
    """
    Align two independently-Louvain-detected community structures so that the
    same index in both refers to the best-corresponding (most similar) grouping,
    rather than whatever order each group's own Louvain run happened to produce.

    Without this, "community 0" for one group and "community 0" for the other
    are not guaranteed to contain the same (or even similar) markers — comparing
    them directly conflates two unrelated groupings under the same label.

    Parameters
    ----------
    reference_communities : list[set]
        Community structure to use as the reference ordering (e.g. Control's).
        Sorted largest-first internally for a stable, reproducible ordering.
    other_communities : list[set]
        Community structure to align to the reference (e.g. the disease group's).

    Returns
    -------
    aligned_reference, aligned_other : list[set], list[set]
        Same-length lists where index i is the best-matched pair. A reference
        community with no good match in `other` (or vice versa) is paired with
        an empty set at that index, rather than force-matching unrelated groups.
    valid_mask : list[bool]
        True at index i only when BOTH aligned_reference[i] and aligned_other[i]
        are real (non-empty) communities — i.e. a genuine two-sided match.
        False marks a one-sided padding slot (a community that exists in only
        one group, with no counterpart in the other). Callers should EXCLUDE
        False slots from any between-group comparison: comparing a group's
        real distribution against the other group's empty-placeholder "0 for
        every subject" is not a biological finding, it's an artifact of the
        two groups having different numbers/compositions of communities.
    """
    ref_sorted = sorted(reference_communities, key=len, reverse=True)
    used = set()
    aligned_other = []

    for ref_comm in ref_sorted:
        best_j, best_score = None, -1
        for j, comm in enumerate(other_communities):
            if j in used:
                continue
            score = jaccard_complete_communities(ref_comm, comm)
            if score > best_score:
                best_score, best_j = score, j
        if best_j is not None and best_score > 0:
            aligned_other.append(other_communities[best_j])
            used.add(best_j)
        else:
            aligned_other.append(set())  # no overlapping community found in `other`

    # Any of `other`'s communities that never matched anything get appended as
    # new trailing indices, with the reference padded by an empty set there
    leftovers = [c for j, c in enumerate(other_communities) if j not in used]
    aligned_reference = list(ref_sorted) + [set() for _ in leftovers]
    aligned_other = aligned_other + leftovers

    valid_mask = [bool(r) and bool(o) for r, o in zip(aligned_reference, aligned_other)]

    return aligned_reference, aligned_other, valid_mask


#used
def calculate_community_strength_metrics_own(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full,
                            correlation_method, group_communities, resolution):
    """
    Data-driven counterpart to calculate_community_strength_metrics().

    Instead of scoring every subject against the fixed, literature-based
    ``consensus_communities``, each subject is scored against their OWN
    group's Louvain-detected community structure (``group_communities``,
    from calc_group_communities on that resolution's allegiance matrices).
    This makes the resulting strength values genuinely sensitive to
    ``resolution``, since it changes which markers Louvain groups together.

    Because Control and the disease group are Louvain-detected independently,
    "community 0" for one is not automatically the same set of markers as
    "community 0" for the other. Before comparing groups, this function aligns
    the disease group's communities to Control's via best-Jaccard-similarity
    matching (align_communities_across_groups) — so a given community index
    means the same (or best-corresponding) body-segment grouping in both
    groups. Any community with no real counterpart in the other group (a
    one-sided match) is DROPPED from the comparison entirely, rather than
    padded with an empty placeholder — comparing real data against a fake
    "0 for every subject" group is not a biological finding.
    """

    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])

    # Align each group's independently-detected communities to Control's ordering
    # (Control is the reference; sorted largest-first for a stable ordering)
    disease_group_label = f"{diagnosis[0][10:].capitalize()}"
    aligned_group_communities = {"Control": {}, disease_group_label: {}}
    # Record (control_size, disease_size) per surviving matched community, so the
    # plot can distinguish "genuine singleton — mathematically 0 by definition"
    # from anything that would otherwise look like an unexplained 0.
    community_sizes = {}
    for task in task_names:
        aligned_group_communities["Control"][task] = {}
        aligned_group_communities[disease_group_label][task] = {}
        community_sizes[task] = {}
        for kinematics in kinematics_list:
            aligned_group_communities["Control"][task][kinematics] = {}
            aligned_group_communities[disease_group_label][task][kinematics] = {}
            for direction in ["AP", "ML", "V"]:
                control_comms = group_communities.get("Control", {}).get(task, {}).get(kinematics, {}).get(direction)
                disease_comms = group_communities.get(disease_group_label, {}).get(task, {}).get(kinematics, {}).get(direction)
                if not control_comms or not disease_comms:
                    aligned_group_communities["Control"][task][kinematics][direction] = control_comms
                    aligned_group_communities[disease_group_label][task][kinematics][direction] = disease_comms
                    continue
                aligned_ref, aligned_dis, valid_mask = align_communities_across_groups(control_comms, disease_comms)
                # Drop one-sided slots entirely (a community with no counterpart in
                # the other group) rather than storing them as empty placeholders —
                # comparing a group's real distribution against the other group's
                # "0 for every subject because there's no such community" isn't a
                # real finding, it's an artifact of unequal community counts.
                n_dropped = sum(1 for v in valid_mask if not v)
                if n_dropped:
                    print(f"    [{task}/{kinematics}/{direction}] dropped {n_dropped} community(ies) "
                          f"with no counterpart in the other group (one-sided match)")
                kept_ref = [c for c, v in zip(aligned_ref, valid_mask) if v]
                kept_dis = [c for c, v in zip(aligned_dis, valid_mask) if v]
                aligned_group_communities["Control"][task][kinematics][direction] = kept_ref
                aligned_group_communities[disease_group_label][task][kinematics][direction] = kept_dis
                # direction is unique across kinematics in practice (single kinematics
                # value is the norm here); keyed by direction only, matching how
                # create_community_strength_plot groups its subplots
                community_sizes[task][direction] = [(len(r), len(d)) for r, d in zip(kept_ref, kept_dis)]

    metrics = {
        'between_community_ratio': {},
        'cross_community_coupling': {},
        'intra_community_strength': {},
    }
    for metric_name in metrics.keys():
        metrics[metric_name] = {
            f"{diagnosis[0][10:].capitalize()}": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None}
                                                                  for kinematics in kinematics_list}
                                                                  for task in task_names}
                                                                  for sub_id in disease_sub_ids},
            "Control": {sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None}
                                        for kinematics in kinematics_list}
                                        for task in task_names}
                                        for sub_id in matched_control_sub_ids},
        }

    _csm_ids = disease_sub_ids + matched_control_sub_ids
    print(f"  calculate_community_strength_metrics_own: processing {len(_csm_ids)} subjects...")
    for kinematics in kinematics_list:
        for _csm_idx, sub_id in enumerate(_csm_ids):
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            print(f"  [{_csm_idx+1}/{len(_csm_ids)}] {sub_id} ({group})", flush=True)

            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run

                        # Load kinectomes
                        from config import EXCLUDE_MARKERS_BY_TASK
                        kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)

                        if kinectomes is None:
                            continue

                        # Average kinectomes across gait cycles
                        avg_kinectomes = np.mean(kinectomes, axis=0)

                        # Strip excluded markers (e.g. upper limb for dual tasks) before
                        # building the graph, so they never enter the strength calculation
                        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                        effective_marker_list = marker_list
                        if exclude:
                            from src.data_utils.data_loader import exclude_markers_from_kinectome
                            avg_kinectomes, effective_marker_list = exclude_markers_from_kinectome(
                                avg_kinectomes, marker_list, exclude
                            )

                        # Build graphs
                        graphs = build_graph(avg_kinectomes, effective_marker_list)

                        # Calculate metrics for each direction, using THIS subject's
                        # own group's Louvain-detected community structure (indices
                        # into effective_marker_list -> mapped to marker names)
                        directions = ["AP", "ML", "V"]
                        for i, direction in enumerate(directions):
                            G = graphs[i]

                            own_structure = aligned_group_communities.get(group, {}).get(task_name, {}).get(kinematics, {}).get(direction)
                            if not own_structure:
                                continue
                            own_communities = [{effective_marker_list[idx] for idx in community} for community in own_structure]

                            between_ratio = calculate_between_community_ratio(G, own_communities)
                            cross_coupling = calculate_cross_community_coupling(G, own_communities)
                            intra_strength = calculate_intra_community_strength_per_community(G, own_communities)

                            metrics['between_community_ratio'][group][sub_id][task_name][kinematics][direction] = np.round(between_ratio, 4)
                            metrics['cross_community_coupling'][group][sub_id][task_name][kinematics][direction] = np.round(cross_coupling, 4)
                            metrics['intra_community_strength'][group][sub_id][task_name][kinematics][direction] = np.round(intra_strength, 4)

    return metrics, community_sizes


#     """Initialize metrics structure for per-community analysis"""
#     metrics = {}
#     metrics['intra_community_strength'] = {
#         f"{diagnosis[0][10:].capitalize()}": {
#             sub_id: {
#                 task: {
#                     kinematics: {
#                         direction: [None] * num_communities  # List for each community
#                         for direction in ["AP", "ML", "V"]
#                     }
#                     for kinematics in kinematics_list
#                 }
#                 for task in task_names
#             }
#             for sub_id in disease_sub_ids
#         },
#         "Control": {
#             sub_id: {
#                 task: {
#                     kinematics: {
#                         direction: [None] * num_communities  # List for each community
#                         for direction in ["AP", "ML", "V"]
#                     }
#                     for kinematics in kinematics_list
#                 }
#                 for task in task_names
#             }
#             for sub_id in matched_control_sub_ids
#         },
#     }
#     return metrics

#used
def analyze_community_strength_data(metrics_dict, resolution, save_path=None, correction_method='fdr_bh', community_sizes=None):
    """
    Analyze community strength data and create visualization
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with structure: {group: {subject: {task: {kinematics: {direction: array}}}}}
    resolution : float
        Resolution parameter used for modularity detection
    save_path : str, optional
        Path to save the figure
    correction_method : str
        Method for multiple comparison correction ('fdr_bh' or 'bonferroni')
    community_sizes : dict, optional
        {task: {direction: [(control_size, disease_size), ...]}} — when provided
        (only meaningful for Louvain-detected/"own" communities, not the fixed
        consensus ones), lets the plot label which side of a community is a
        genuine singleton (size 1, mathematically 0 intra-community strength by
        definition) instead of leaving a 0 value looking unexplained.
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing all the data in long format
    stats_results : dict
        Dictionary containing statistical test results
    """
    
    # Convert data to DataFrame
    df = convert_to_dataframe(metrics_dict)
    
    # Perform statistical analysis
    stats_results = perform_statistical_tests(df, correction_method)
    
    # Create visualization
    fig = create_community_strength_plot(df, stats_results, resolution, save_path, community_sizes)
    
    return df, stats_results

#used
def convert_to_dataframe(metrics_dict):
    """Convert nested metrics dictionary to pandas DataFrame"""
    
    data_rows = []
    
    for group, subjects in metrics_dict.items():
        for subject_id, tasks in subjects.items():
            for task, kinematics_dict in tasks.items():
                for kinematics, directions in kinematics_dict.items():
                    for direction, community_values in directions.items():
                        if community_values is not None:
                            # Handle both array and list inputs
                            if hasattr(community_values, '__len__'):
                                for community_idx, value in enumerate(community_values):
                                    if value is not None:
                                        data_rows.append({
                                            'group': group,
                                            'subject_id': subject_id,
                                            'task': task,
                                            'kinematics': kinematics,
                                            'direction': direction,
                                            'community': f'Community_{community_idx + 1}',
                                            'community_idx': community_idx,
                                            'strength': float(value)
                                        })
    
    return pd.DataFrame(data_rows)

#used 
def perform_statistical_tests(df, correction_method='fdr_bh'):
    """Perform statistical tests comparing groups for each condition"""
    
    if df.empty:
        return {}
    
    # Get unique conditions
    tasks = df['task'].unique()
    directions = df['direction'].unique()
    communities = df['community'].unique()
    groups = df['group'].unique()
    
    if len(groups) != 2:
        print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
        return {}
    
    group1, group2 = groups
    
    results = {}
    all_p_values = []
    comparison_info = []
    
    for task in tasks:
        results[task] = {}
        for direction in directions:
            results[task][direction] = {}
            for community in communities:
                # Filter data for this specific condition
                mask = (df['task'] == task) & (df['direction'] == direction) & (df['community'] == community)
                condition_data = df[mask]
                
                if condition_data.empty:
                    continue
                
                # Split by group
                group1_data = condition_data[condition_data['group'] == group1]['strength'].values
                group2_data = condition_data[condition_data['group'] == group2]['strength'].values
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    # Perform Mann-Whitney U test
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            group1_data, group2_data, alternative='two-sided'
                        )
                        
                        # Calculate effect size (Cohen's d approximation)
                        pooled_std = np.sqrt((np.var(group1_data) + np.var(group2_data)) / 2)
                        if pooled_std > 0:
                            cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
                        else:
                            cohens_d = 0
                        
                        results[task][direction][community] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            f'{group1}_mean': np.mean(group1_data),
                            f'{group1}_std': np.std(group1_data),
                            f'{group1}_n': len(group1_data),
                            f'{group2}_mean': np.mean(group2_data),
                            f'{group2}_std': np.std(group2_data),
                            f'{group2}_n': len(group2_data),
                            'cohens_d': cohens_d,
                            'groups': (group1, group2)
                        }
                        
                        all_p_values.append(p_value)
                        comparison_info.append({
                            'task': task,
                            'direction': direction,
                            'community': community
                        })
                        
                    except Exception as e:
                        print(f"Statistical test failed for {task}-{direction}-{community}: {e}")
    
    # Multiple comparison correction
    if all_p_values:
        if correction_method == 'fdr_bh':
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
        elif correction_method == 'bonferroni':
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='bonferroni')
        else:
            # No correction
            rejected = [p < 0.05 for p in all_p_values]
            p_corrected = all_p_values
        
        # Update results with corrected p-values
        for i, (p_corr, is_significant) in enumerate(zip(p_corrected, rejected)):
            info = comparison_info[i]
            task, direction, community = info['task'], info['direction'], info['community']
            results[task][direction][community]['p_corrected'] = p_corr
            results[task][direction][community]['significant'] = is_significant
    
    return results

#used
def create_community_strength_plot(df, stats_results, resolution, save_path=None, community_sizes=None):
    """Create the community strength visualization.

    community_sizes, if given: {task: {direction: [(control_size, disease_size), ...]}}
    indexed by the same 0-based community_idx stored in df — used to label a
    community's x-tick when one side is a genuine singleton (size 1), so a 0
    value there reads as "mathematically expected" rather than "unexplained".
    """
    
    if df.empty:
        print("No data to plot")
        return None
    
    # Get unique values
    tasks = sorted(df['task'].unique())
    directions = ['AP', 'ML', 'V']  # Fixed order
    groups = sorted(df['group'].unique())
    
    # Filter directions that exist in data
    existing_directions = [d for d in directions if d in df['direction'].unique()]
    
    if len(existing_directions) == 0:
        print("No valid directions found in data")
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(tasks), len(existing_directions), 
                            figsize=(5 * len(existing_directions), 4 * len(tasks)))
    
    if len(tasks) == 1:
        axes = axes.reshape(1, -1)
    if len(existing_directions) == 1:
        axes = axes.reshape(-1, 1)
    if len(tasks) == 1 and len(existing_directions) == 1:
        axes = np.array([[axes]])
    
    # Color scheme
    group_colors = {'Parkinson': '#E74C3C', 'Control': '#3498DB'}
    if len(groups) == 2:
        group_colors = {'Parkinson': '#E74C3C', 'Control': '#3498DB'}
        if set(groups) != {'Parkinson', 'Control'}:
            group_colors = {groups[0]: '#E74C3C', groups[1]: '#3498DB'}
    
    community_colors = ['#E74C3C', '#3498DB', '#2ECC71']  # Red, Blue, Green for communities
    
    fig.suptitle(f'Intra-Community Strength Analysis (Resolution = {resolution})', 
                 fontsize=16, fontweight='bold')
    
    for task_idx, task in enumerate(tasks):
        for dir_idx, direction in enumerate(existing_directions):
            ax = axes[task_idx, dir_idx]
            
            # Filter data for this subplot
            subset = df[(df['task'] == task) & (df['direction'] == direction)]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task} - {direction}')
                continue
            
            # Filter out communities where every value (both groups) is exactly 0.
            # These are singleton communities (a single marker on its own) — they
            # have no possible intra-community edges, so their "strength" is always
            # trivially 0 for every subject. That's mathematically correct, not
            # missing data, but it clutters the plot with an uninformative flat box.
            _all_communities = sorted(subset['community'].unique())
            _informative_communities = [
                c for c in _all_communities
                if not np.allclose(subset[subset['community'] == c]['strength'].values, 0)
            ]
            n_dropped = len(_all_communities) - len(_informative_communities)
            subset = subset[subset['community'].isin(_informative_communities)]

            if subset.empty:
                ax.text(0.5, 0.5, 'All communities are singletons\n(no intra-community edges)',
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task} - {direction}')
                continue

            # Create boxplot
            communities = sorted(subset['community'].unique())
            positions = []
            data_for_box = []
            colors_for_box = []
            
            pos_offset = 0
            for comm_idx, community in enumerate(communities):
                for group_idx, group in enumerate(groups):
                    group_data = subset[(subset['community'] == community) & 
                                      (subset['group'] == group)]['strength'].values
                    
                    if len(group_data) > 0:
                        data_for_box.append(group_data)
                        positions.append(pos_offset + group_idx * 0.4)
                        colors_for_box.append(group_colors.get(group, f'C{group_idx}'))
                
                pos_offset += 1.2  # Space between communities
            
            if data_for_box:
                bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                               widths=0.3, showfliers=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors_for_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Add significance markers — star and p-value combined into a single
            # compact text element (rather than two separately-placed pieces) so
            # they can't visually collide with each other
            y_max = ax.get_ylim()[1]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            
            n_sig_shown = 0
            for comm_idx, community in enumerate(communities):
                if (task in stats_results and direction in stats_results[task] and 
                    community in stats_results[task][direction]):
                    
                    result = stats_results[task][direction][community]
                    if result.get('significant', False):
                        x_pos = comm_idx * 1.2 + 0.2  # Center between the two group positions
                        label_y = y_max + 0.06 * y_range

                        p_val = result['p_corrected']
                        p_text = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'

                        ax.text(x_pos, label_y, f'* {p_text}', ha='center', va='bottom',
                               fontsize=9, fontweight='bold', color='red')
                        n_sig_shown += 1
            
            # Give the annotations room so they don't collide with the subplot title
            if n_sig_shown > 0:
                ax.set_ylim(top=y_max + 0.22 * y_range)
            
            # Customize subplot
            ax.set_title(f'{task} - {direction}'
                         + (f'  ({n_dropped} singleton community skipped)' if n_dropped == 1
                            else f'  ({n_dropped} singleton communities skipped)' if n_dropped > 1
                            else ''),
                         pad=16 if n_sig_shown > 0 else 6, fontsize=11)
            ax.set_ylabel('Intra-Community Strength')
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels — annotate singleton sides using community_sizes
            # (Louvain/"own" path only) so a 0 there reads as expected, not a bug
            community_labels = []
            for i, community in enumerate(communities):
                label = f'C{i+1}'
                if community_sizes is not None:
                    idx_vals = subset[subset['community'] == community]['community_idx'].values
                    if len(idx_vals) > 0:
                        orig_idx = int(idx_vals[0])
                        sizes_for_dir = community_sizes.get(task, {}).get(direction, [])
                        if 0 <= orig_idx < len(sizes_for_dir):
                            control_size, disease_size = sizes_for_dir[orig_idx]
                            singleton_note = []
                            if control_size == 1:
                                singleton_note.append('Ctrl')
                            if disease_size == 1:
                                singleton_note.append(list(groups)[1] if len(groups) > 1 and groups[0] == 'Control' else 'PD')
                            if singleton_note:
                                label += f"\n({'+'.join(singleton_note)} singleton)"
                community_labels.append(label)
            ax.set_xticks([i * 1.2 + 0.2 for i in range(len(communities))])
            ax.set_xticklabels(community_labels, fontsize=8)
            
            # Add legend only to the first subplot
            if task_idx == 0 and dir_idx == 0:
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_colors.get(group, f'C{i}'), 
                                                alpha=0.7, label=group) 
                                 for i, group in enumerate(groups)]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig

# def print_significant_results(stats_results, alpha=0.05):
#     """Print summary of statistically significant comparisons"""
#     print(f"\nSignificant Results (corrected p < {alpha}):")
#     print("=" * 70)
    
#     significant_count = 0
#     total_count = 0
    
#     for task in stats_results:
#         for direction in stats_results[task]:
#             for community, result in stats_results[task][direction].items():
#                 total_count += 1
#                 if result.get('significant', False):
#                     significant_count += 1
#                     groups = result['groups']
#                     print(f"\n{task} - {direction} - {community}:")
#                     print(f"  p-value: {result['p_value']:.6f}")
#                     print(f"  p-corrected: {result['p_corrected']:.6f}")
#                     print(f"  Cohen's d: {result['cohens_d']:.4f}")
#                     print(f"  {groups[0]} mean ± SD: {result[f'{groups[0]}_mean']:.4f} ± {result[f'{groups[0]}_std']:.4f} (n={result[f'{groups[0]}_n']})")
#                     print(f"  {groups[1]} mean ± SD: {result[f'{groups[1]}_mean']:.4f} ± {result[f'{groups[1]}_std']:.4f} (n={result[f'{groups[1]}_n']})")
    
#     print(f"\nTotal significant comparisons: {significant_count}/{total_count}")
    
#     if significant_count > 0:
#         print(f"Proportion significant: {significant_count/total_count:.2%}")

# def get_data_summary(df):
#     """Get summary statistics of the data"""
#     if df.empty:
#         print("No data available")
#         return
    
#     print("Data Summary:")
#     print("=" * 50)
#     print(f"Groups: {list(df['group'].unique())}")
#     print(f"Tasks: {list(df['task'].unique())}")
#     print(f"Directions: {list(df['direction'].unique())}")
#     print(f"Communities: {list(df['community'].unique())}")
#     print(f"Total subjects: {df['subject_id'].nunique()}")
#     print(f"Total observations: {len(df)}")
    
#     # Group-wise summary
#     print(f"\nSubjects per group:")
#     for group in df['group'].unique():
#         n_subjects = df[df['group'] == group]['subject_id'].nunique()
#         print(f"  {group}: {n_subjects}")
    
#     # Strength statistics
#     print(f"\nStrength statistics:")
#     print(f"  Mean: {df['strength'].mean():.4f}")
#     print(f"  Std: {df['strength'].std():.4f}")
#     print(f"  Range: {df['strength'].min():.4f} - {df['strength'].max():.4f}")

# def calculate_pairwise_inter_community_strength(G, communities):
#     """
#     Calculate connectivity strength between each pair of communities
#     Returns a list of inter-community strengths for each community pair
    
#     For 3 communities, returns:
#     [comm1_to_comm2_strength, comm1_to_comm3_strength, comm2_to_comm3_strength]
#     """
#     # Create node to community mapping
#     node_to_community = {}
#     for i, community in enumerate(communities):
#         for node in community:
#             node_to_community[node] = i
    
#     num_communities = len(communities)
    
#     # Initialize weights for each community pair
#     pair_weights = {}
#     for i in range(num_communities):
#         for j in range(i + 1, num_communities):
#             pair_weights[(i, j)] = []
    
#     # Collect inter-community weights
#     for edge in G.edges(data=True):
#         node1, node2, data = edge
#         weight = abs(data['weight'])  # Use absolute correlation strength
        
#         comm1 = node_to_community.get(node1, -1)
#         comm2 = node_to_community.get(node2, -1)
        
#         # Only consider inter-community connections
#         if comm1 != comm2 and comm1 != -1 and comm2 != -1:
#             # Ensure consistent ordering (smaller index first)
#             pair = tuple(sorted([comm1, comm2]))
#             if pair in pair_weights:
#                 pair_weights[pair].append(weight)
    
#     # Calculate mean strength for each pair
#     pairwise_strengths = []
#     for i in range(num_communities):
#         for j in range(i + 1, num_communities):
#             pair = (i, j)
#             if pair in pair_weights and pair_weights[pair]:
#                 pairwise_strengths.append(np.mean(pair_weights[pair]))
#             else:
#                 pairwise_strengths.append(0)
    
#     return pairwise_strengths

# def calculate_pairwise_inter_to_intra_ratios(G, communities):
#     """
#     Calculate ratio of inter-community to intra-community connectivity for each pair
#     Returns ratios for each community pair relative to their respective intra-community strengths
#     """
#     # Get intra-community strengths for each community
#     intra_strengths = calculate_intra_community_strength_per_community(G, communities)
    
#     # Get inter-community strengths for each pair
#     inter_strengths = calculate_pairwise_inter_community_strength(G, communities)
    
#     num_communities = len(communities)
#     ratios = []
    
#     pair_idx = 0
#     for i in range(num_communities):
#         for j in range(i + 1, num_communities):
#             inter_strength = inter_strengths[pair_idx]
            
#             # Use average of the two communities' intra-community strengths as denominator
#             avg_intra = (intra_strengths[i] + intra_strengths[j]) / 2
            
#             if avg_intra > 0:
#                 ratio = inter_strength / avg_intra
#             else:
#                 ratio = 0
            
#             ratios.append(ratio)
#             pair_idx += 1
    
#     return ratios

# def calculate_intra_community_strength_per_community(G, communities):
#     """
#     Calculate absolute strength of connectivity within each community separately
#     Returns a list of strength values, one for each community
#     """
#     # Create node to community mapping
#     node_to_community = {}
#     for i, community in enumerate(communities):
#         for node in community:
#             node_to_community[node] = i
    
#     # Initialize lists for each community's intra-weights
#     community_weights = [[] for _ in range(len(communities))]
    
#     for edge in G.edges(data=True):
#         node1, node2, data = edge
#         weight = abs(data['weight'])  # Use absolute correlation strength
        
#         comm1 = node_to_community.get(node1, -1)
#         comm2 = node_to_community.get(node2, -1)
        
#         # Only count intra-community connections
#         if comm1 == comm2 and comm1 != -1:
#             community_weights[comm1].append(weight)
    
#     # Calculate mean strength for each community
#     community_strengths = []
#     for weights in community_weights:
#         if weights:
#             community_strengths.append(np.mean(weights))
#         else:
#             community_strengths.append(0)
    
#     return community_strengths

#used
def analyze_pairwise_inter_community_data(metrics_dict, resolution, metric_type='strength', 
                                        save_path=None, correction_method='fdr_bh'):
    """
    Analyze pairwise inter-community connectivity data
    
    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with pairwise inter-community data
    resolution : float
        Resolution parameter used for modularity detection
    metric_type : str
        Type of metric ('strength' or 'ratio')
    save_path : str, optional
        Path to save the figure
    correction_method : str
        Method for multiple comparison correction
    
    Returns:
    --------
    df : pd.DataFrame
        DataFrame containing all the data
    stats_results : dict
        Dictionary containing statistical test results
    """
    # Convert data to DataFrame
    df = convert_pairwise_to_dataframe(metrics_dict)
    
    # Perform statistical analysis
    stats_results = perform_pairwise_statistical_tests(df, correction_method)
    
    # Create visualization
    fig = create_pairwise_community_plot(df, stats_results, resolution, metric_type, save_path)
    
    return df, stats_results

#used
def convert_pairwise_to_dataframe(metrics_dict):
    """Convert pairwise inter-community metrics to DataFrame"""
    
    data_rows = []
    
    # Define pair labels
    pair_labels = ['Comm1_to_Comm2', 'Comm1_to_Comm3', 'Comm2_to_Comm3']
    
    for group, subjects in metrics_dict.items():
        for subject_id, tasks in subjects.items():
            for task, kinematics_dict in tasks.items():
                for kinematics, directions in kinematics_dict.items():
                    for direction, pair_values in directions.items():
                        if pair_values is not None:
                            # Handle both array and list inputs
                            if hasattr(pair_values, '__len__'):
                                for pair_idx, value in enumerate(pair_values):
                                    if value is not None and pair_idx < len(pair_labels):
                                        data_rows.append({
                                            'group': group,
                                            'subject_id': subject_id,
                                            'task': task,
                                            'kinematics': kinematics,
                                            'direction': direction,
                                            'pair': pair_labels[pair_idx],
                                            'pair_idx': pair_idx,
                                            'value': float(value)
                                        })
    
    return pd.DataFrame(data_rows)

#used
def perform_pairwise_statistical_tests(df, correction_method='fdr_bh'):
    """Perform statistical tests for pairwise inter-community data"""
    
    if df.empty:
        return {}
    
    # Get unique conditions
    tasks = df['task'].unique()
    directions = df['direction'].unique()
    pairs = df['pair'].unique()
    groups = df['group'].unique()
    
    if len(groups) != 2:
        print(f"Warning: Expected 2 groups, found {len(groups)}: {groups}")
        return {}
    
    group1, group2 = groups
    
    results = {}
    all_p_values = []
    comparison_info = []
    
    for task in tasks:
        results[task] = {}
        for direction in directions:
            results[task][direction] = {}
            for pair in pairs:
                # Filter data for this specific condition
                mask = (df['task'] == task) & (df['direction'] == direction) & (df['pair'] == pair)
                condition_data = df[mask]
                
                if condition_data.empty:
                    continue
                
                # Split by group
                group1_data = condition_data[condition_data['group'] == group1]['value'].values
                group2_data = condition_data[condition_data['group'] == group2]['value'].values
                
                if len(group1_data) > 0 and len(group2_data) > 0:
                    # Perform Mann-Whitney U test
                    try:
                        statistic, p_value = stats.mannwhitneyu(
                            group1_data, group2_data, alternative='two-sided'
                        )
                        
                        # Calculate effect size
                        pooled_std = np.sqrt((np.var(group1_data) + np.var(group2_data)) / 2)
                        if pooled_std > 0:
                            cohens_d = (np.mean(group1_data) - np.mean(group2_data)) / pooled_std
                        else:
                            cohens_d = 0
                        
                        results[task][direction][pair] = {
                            'statistic': statistic,
                            'p_value': p_value,
                            f'{group1}_mean': np.mean(group1_data),
                            f'{group1}_std': np.std(group1_data),
                            f'{group1}_n': len(group1_data),
                            f'{group2}_mean': np.mean(group2_data),
                            f'{group2}_std': np.std(group2_data),
                            f'{group2}_n': len(group2_data),
                            'cohens_d': cohens_d,
                            'groups': (group1, group2)
                        }
                        
                        all_p_values.append(p_value)
                        comparison_info.append({
                            'task': task,
                            'direction': direction,
                            'pair': pair
                        })
                        
                    except Exception as e:
                        print(f"Statistical test failed for {task}-{direction}-{pair}: {e}")
    
    # Multiple comparison correction
    if all_p_values:
        if correction_method == 'fdr_bh':
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='fdr_bh')
        elif correction_method == 'bonferroni':
            rejected, p_corrected, _, _ = multipletests(all_p_values, alpha=0.05, method='bonferroni')
        else:
            rejected = [p < 0.05 for p in all_p_values]
            p_corrected = all_p_values
        
        # Update results with corrected p-values
        for i, (p_corr, is_significant) in enumerate(zip(p_corrected, rejected)):
            info = comparison_info[i]
            task, direction, pair = info['task'], info['direction'], info['pair']
            results[task][direction][pair]['p_corrected'] = p_corr
            results[task][direction][pair]['significant'] = is_significant
    
    return results

#used
def create_pairwise_community_plot(df, stats_results, resolution, metric_type='strength', save_path=None):
    """Create visualization for pairwise inter-community analysis"""
    
    if df.empty:
        print("No data to plot")
        return None
    
    # Get unique values
    tasks = sorted(df['task'].unique())
    directions = ['AP', 'ML', 'V']
    groups = sorted(df['group'].unique())
    
    # Filter directions that exist in data
    existing_directions = [d for d in directions if d in df['direction'].unique()]
    
    if len(existing_directions) == 0:
        print("No valid directions found in data")
        return None
    
    # Create figure
    fig, axes = plt.subplots(len(tasks), len(existing_directions), 
                            figsize=(5 * len(existing_directions), 4 * len(tasks)))
    
    if len(tasks) == 1:
        axes = axes.reshape(1, -1)
    if len(existing_directions) == 1:
        axes = axes.reshape(-1, 1)
    if len(tasks) == 1 and len(existing_directions) == 1:
        axes = np.array([[axes]])
    
    # Color scheme
    group_colors = {'Parkinson': '#E74C3C', 'Control': '#3498DB'} if len(groups) == 2 else {}
    pair_colors = ['#E74C3C', '#9B59B6', '#F39C12']  # Different colors for each pair
    
    title_text = f'Inter-Community {metric_type.capitalize()} Analysis (Resolution = {resolution})'
    fig.suptitle(title_text, fontsize=16, fontweight='bold')
    
    for task_idx, task in enumerate(tasks):
        for dir_idx, direction in enumerate(existing_directions):
            ax = axes[task_idx, dir_idx]
            
            # Filter data for this subplot
            subset = df[(df['task'] == task) & (df['direction'] == direction)]
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{task} - {direction}')
                continue
            
            # Create boxplot
            pairs = sorted(subset['pair'].unique())
            positions = []
            data_for_box = []
            colors_for_box = []
            
            pos_offset = 0
            for pair_idx, pair in enumerate(pairs):
                for group_idx, group in enumerate(groups):
                    group_data = subset[(subset['pair'] == pair) & 
                                      (subset['group'] == group)]['value'].values
                    
                    if len(group_data) > 0:
                        data_for_box.append(group_data)
                        positions.append(pos_offset + group_idx * 0.4)
                        colors_for_box.append(group_colors.get(group, f'C{group_idx}'))
                
                pos_offset += 1.2  # Space between pairs
            
            if data_for_box:
                bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                               widths=0.3, showfliers=True)
                
                # Color the boxes
                for patch, color in zip(bp['boxes'], colors_for_box):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            # Add significance markers
            y_max = ax.get_ylim()[1]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            
            for pair_idx, pair in enumerate(pairs):
                if (task in stats_results and direction in stats_results[task] and 
                    pair in stats_results[task][direction]):
                    
                    result = stats_results[task][direction][pair]
                    if result.get('significant', False):
                        # Position significance marker
                        x_pos = pair_idx * 1.2 + 0.2
                        y_pos = y_max + 0.02 * y_range
                        
                        ax.text(x_pos, y_pos, '*', ha='center', va='bottom', 
                               fontsize=16, fontweight='bold', color='red')
                        
                        # Add p-value
                        p_val = result['p_corrected']
                        if p_val < 0.001:
                            p_text = 'p<0.001'
                        else:
                            p_text = f'p={p_val:.3f}'
                        
                        ax.text(x_pos, y_pos + 0.03 * y_range, p_text, ha='center', va='bottom', 
                               fontsize=8, color='red')
            
            # Customize subplot
            ax.set_title(f'{task} - {direction}')
            ylabel = f'Inter-Community {metric_type.capitalize()}'
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            
            # Set x-axis labels
            pair_labels = ['1↔2', '1↔3', '2↔3']  # Shorter labels
            ax.set_xticks([i * 1.2 + 0.2 for i in range(len(pairs))])
            ax.set_xticklabels([pair_labels[i] for i in range(len(pairs))])
            
            # Add legend only to the first subplot
            if task_idx == 0 and dir_idx == 0:
                legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_colors.get(group, f'C{i}'), 
                                                alpha=0.7, label=group) 
                                 for i, group in enumerate(groups)]
                ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {save_path}")
    
    plt.show()
    return fig

#used
def analyze_inter_community_strength(metrics_dict, resolution, save_path=None):
    """Analyze inter-community strength between specific pairs"""
    return analyze_pairwise_inter_community_data(
        metrics_dict, resolution, 'strength', save_path
    )

#used
def analyze_inter_to_intra_ratios(metrics_dict, resolution, save_path=None):
    """Analyze inter-to-intra community strength ratios between specific pairs"""
    return analyze_pairwise_inter_community_data(
        metrics_dict, resolution, 'ratio', save_path
    )

# used
def analyze_threshold_ratios(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, 
                           base_path, marker_list, full, correlation_method, consensus_communities,
                           threshold_list=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Analyze ratio of edges above threshold for each community
    """
    
    from src.data_utils.groups import get_matched_groups_for_task
    task_disease_ids, task_control_ids = get_matched_groups_for_task(diagnosis, task_names)
    disease_sub_ids        = task_disease_ids.get(task_names[0], [])
    matched_control_sub_ids = task_control_ids.get(task_names[0], [])

    # Filter to subjects with kinectomes for these tasks
    def _has_k(sub_id, task):
        from pathlib import Path
        from config import KINECTOME_SAVE_PATH
        d = Path(KINECTOME_SAVE_PATH) / f"sub-{sub_id}"
        return d.exists() and any(task in f.name for f in d.iterdir() if f.suffix == '.npy')

    avail_disease  = [s for s in disease_sub_ids        if any(_has_k(s, t) for t in task_names)]
    avail_controls = [s for s in matched_control_sub_ids if any(_has_k(s, t) for t in task_names)]
    n = len(avail_disease)
    disease_sub_ids        = avail_disease
    matched_control_sub_ids = avail_controls[:n]
    print(f"  analyze_threshold_ratios: {len(disease_sub_ids)} disease, {len(matched_control_sub_ids)} controls")

    def calculate_community_threshold_ratios(G, communities, threshold):
        """Calculate ratio of edges above threshold for each community"""
        node_to_community = {}
        for i, community in enumerate(communities):
            for node in community:
                node_to_community[node] = i
        
        community_ratios = []
        
        for comm_idx in range(len(communities)):
            above_threshold = 0
            total_edges = 0
            
            for edge in G.edges(data=True):
                node1, node2, data = edge
                comm1 = node_to_community.get(node1, -1)
                comm2 = node_to_community.get(node2, -1)
                
                # Only count intra-community edges
                if comm1 == comm2 == comm_idx:
                    total_edges += 1
                    if abs(data['weight']) >= threshold:
                        above_threshold += 1
            
            ratio = above_threshold / total_edges if total_edges > 0 else 0
            community_ratios.append(ratio)
        
        return community_ratios
    
    # Store results for all thresholds
    all_results = {}
    
    for threshold in threshold_list:
        print(f"Processing threshold: {threshold}")
        # Debug: check graph nodes vs community nodes on first subject
        _debug_done = False
        _ctrl_debug_done = False

        # Initialize metrics for this threshold
        metrics = {
            f"{diagnosis[0][10:].capitalize()}": {
                sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                               for kinematics in kinematics_list} 
                               for task in task_names} 
                               for sub_id in disease_sub_ids},
            "Control": {
                sub_id: {task: {kinematics: {"AP": None, "ML": None, "V": None} 
                                for kinematics in kinematics_list}
                                for task in task_names} 
                                for sub_id in matched_control_sub_ids},
        }
        
        # Main calculation loop
        for kinematics in kinematics_list:
            for sub_id in disease_sub_ids + matched_control_sub_ids:
                group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"

                for tracksys in tracking_systems:
                    for task_name in task_names:
                        for run in runs:
                            if sub_id in pd_on:
                                run = 'on'
                            elif sub_id not in disease_sub_ids:
                                run = None
                            else:
                                run = run
                            
                            # Load kinectomes
                            from config import EXCLUDE_MARKERS_BY_TASK
                            kinectomes = load_kinectomes(KINECTOME_SAVE_PATH, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)
                            if kinectomes is None:
                                print(f"    No kinectomes: {sub_id}/{task_name}/run={run}")
                                continue
                            
                            # Average kinectomes
                            avg_kinectomes = np.mean(kinectomes, axis=0)

                            # Strip excluded markers (e.g. upper limb for dual tasks)
                            exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
                            effective_marker_list = marker_list
                            if exclude:
                                from src.data_utils.data_loader import exclude_markers_from_kinectome
                                avg_kinectomes, effective_marker_list = exclude_markers_from_kinectome(
                                    avg_kinectomes, marker_list, exclude
                                )

                            # Build graphs
                            graphs = build_graph(avg_kinectomes, effective_marker_list)
                            
                            # Calculate for each direction
                            directions = ["AP", "ML", "V"]
                            for i, direction in enumerate(directions):
                                G = graphs[i]
                                ratios = calculate_community_threshold_ratios(G, consensus_communities, threshold)
                                metrics[group][sub_id][task_name][kinematics][direction] = np.array(ratios)
                                if not _debug_done or (group == "Control" and not _ctrl_debug_done):
                                    all_w = [d['weight'] for _,_,d in G.edges(data=True)]
                                    print(f"  [debug] {group}/{sub_id} {direction}: "
                                          f"w_range=[{min(all_w):.3f},{max(all_w):.3f}], "
                                          f"ratios={[round(r,3) for r in ratios]}")
                                    if direction == 'V':
                                        if group == "Control":
                                            _ctrl_debug_done = True
                                        else:
                                            _debug_done = True
        
        all_results[threshold] = metrics
    
    return all_results

# used
def perform_threshold_statistics(all_results, correction_method='fdr_bh'):
    """Perform statistical analysis across all thresholds"""
    
    stats_results = {}
    
    for threshold, metrics in all_results.items():
        # Convert to DataFrame
        df = convert_to_dataframe(metrics)
        
        # Perform tests
        results = {}
        
        tasks = df['task'].unique()
        directions = df['direction'].unique()
        communities = df['community'].unique()
        groups = df['group'].unique()
        
        if len(groups) != 2:
            continue
        
        group1, group2 = groups
        
        for task in tasks:
            results[task] = {}
            for direction in directions:
                results[task][direction] = {}
                
                # Collect p-values for this specific task-direction combination
                task_dir_p_values = []
                task_dir_results = []
                
                for community in communities:
                    mask = (df['task'] == task) & (df['direction'] == direction) & (df['community'] == community)
                    condition_data = df[mask]
                    
                    if condition_data.empty:
                        continue
                    
                    group1_data = condition_data[condition_data['group'] == group1]['strength'].values
                    group2_data = condition_data[condition_data['group'] == group2]['strength'].values
                    
                    if len(group1_data) > 0 and len(group2_data) > 0:
                        statistic, p_value = stats.mannwhitneyu(group1_data, group2_data, alternative='two-sided')
                        
                        result = {
                            'p_value': p_value,
                            f'{group1}_mean': np.mean(group1_data),
                            f'{group1}_std': np.std(group1_data),
                            f'{group2}_mean': np.mean(group2_data),
                            f'{group2}_std': np.std(group2_data),
                            'groups': (group1, group2),
                            'community': community
                        }
                        
                        task_dir_p_values.append(p_value)
                        task_dir_results.append(result)
                
                # Apply FDR correction within this task-direction combination only
                if task_dir_p_values:
                    rejected, p_corrected, _, _ = multipletests(task_dir_p_values, alpha=0.05, method=correction_method)
                    
                    # Store corrected results
                    for i, result in enumerate(task_dir_results):
                        community = result['community']
                        result['p_corrected'] = p_corrected[i]
                        result['significant'] = rejected[i]
                        # Remove the temporary community key
                        del result['community']
                        results[task][direction][community] = result
        
        stats_results[threshold] = results
    
    return stats_results

#used
def convert_to_dataframe_threshold(metrics):
    """Convert threshold-sweep metrics to DataFrame (used by plot_threshold_results only).

    Renamed from convert_to_dataframe to fix a name collision: this function and
    the community-strength convert_to_dataframe() had the same name, so this
    later definition was silently shadowing the other one at module level for
    every caller in the file — including analyze_community_strength_data(),
    which needs the other version's 'community_idx' column and was getting
    this one's (missing) columns instead.
    """
    data_rows = []
    
    for group, subjects in metrics.items():
        for subject_id, tasks in subjects.items():
            for task, kinematics_dict in tasks.items():
                for kinematics, directions in kinematics_dict.items():
                    for direction, community_values in directions.items():
                        if community_values is not None:
                            for community_idx, value in enumerate(community_values):
                                if value is not None:
                                    data_rows.append({
                                        'group': group,
                                        'subject_id': subject_id,
                                        'task': task,
                                        'direction': direction,
                                        'community': f'Community_{community_idx + 1}',
                                        'strength': float(value)
                                    })
    
    return pd.DataFrame(data_rows)

#used
def plot_threshold_results(all_results, stats_results, save_dir):
    """Create plots for each threshold"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    for threshold, metrics in all_results.items():
        df = convert_to_dataframe_threshold(metrics)
        
        if df.empty:
            continue
        
        # Create plot
        tasks = sorted(df['task'].unique())
        directions = ['AP', 'ML', 'V']
        existing_directions = [d for d in directions if d in df['direction'].unique()]
        
        fig, axes = plt.subplots(len(tasks), len(existing_directions), 
                                figsize=(5 * len(existing_directions), 4 * len(tasks)))
        
        if len(tasks) == 1:
            axes = axes.reshape(1, -1)
        if len(existing_directions) == 1:
            axes = axes.reshape(-1, 1)
        if len(tasks) == 1 and len(existing_directions) == 1:
            axes = np.array([[axes]])
        
        groups = sorted(df['group'].unique())
        group_colors = {'Parkinson': '#E74C3C', 'Control': '#3498DB'}
        if set(groups) != {'Parkinson', 'Control'}:
            group_colors = {groups[0]: '#E74C3C', groups[1]: '#3498DB'} if len(groups) == 2 else {}
        
        fig.suptitle(f'Community Threshold Ratio Analysis (Threshold = {threshold})', 
                     fontsize=16, fontweight='bold')
        
        for task_idx, task in enumerate(tasks):
            for dir_idx, direction in enumerate(existing_directions):
                ax = axes[task_idx, dir_idx]
                
                subset = df[(df['task'] == task) & (df['direction'] == direction)]
                
                if subset.empty:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{task} - {direction}')
                    continue
                
                communities = sorted(subset['community'].unique())
                positions = []
                data_for_box = []
                colors_for_box = []
                
                pos_offset = 0
                for comm_idx, community in enumerate(communities):
                    for group_idx, group in enumerate(groups):
                        group_data = subset[(subset['community'] == community) & 
                                          (subset['group'] == group)]['strength'].values
                        
                        if len(group_data) > 0:
                            data_for_box.append(group_data)
                            positions.append(pos_offset + group_idx * 0.4)
                            colors_for_box.append(group_colors.get(group, f'C{group_idx}'))
                    
                    pos_offset += 1.2
                
                if data_for_box:
                    bp = ax.boxplot(data_for_box, positions=positions, patch_artist=True,
                                   widths=0.3, showfliers=True)
                    
                    for patch, color in zip(bp['boxes'], colors_for_box):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.7)
                
                # Add significance markers
                if threshold in stats_results:
                    y_max = ax.get_ylim()[1]
                    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
                    
                    for comm_idx, community in enumerate(communities):
                        if (task in stats_results[threshold] and 
                            direction in stats_results[threshold][task] and 
                            community in stats_results[threshold][task][direction]):
                            
                            result = stats_results[threshold][task][direction][community]
                            if result.get('significant', False):
                                x_pos = comm_idx * 1.2 + 0.2
                                y_pos = y_max + 0.02 * y_range
                                
                                ax.text(x_pos, y_pos, '*', ha='center', va='bottom', 
                                       fontsize=16, fontweight='bold', color='red')
                                
                                p_val = result['p_corrected']
                                p_text = 'p<0.001' if p_val < 0.001 else f'p={p_val:.3f}'
                                ax.text(x_pos, y_pos + 0.03 * y_range, p_text, ha='center', va='bottom', 
                                       fontsize=8, color='red')
                
                ax.set_title(f'{task} - {direction}')
                ax.set_ylabel('Threshold Ratio')
                ax.grid(True, alpha=0.3)
                
                community_labels = [f'C{i+1}' for i in range(len(communities))]
                ax.set_xticks([i * 1.2 + 0.2 for i in range(len(communities))])
                ax.set_xticklabels(community_labels)
                
                if task_idx == 0 and dir_idx == 0:
                    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=group_colors.get(group, f'C{i}'), 
                                                    alpha=0.7, label=group) 
                                     for i, group in enumerate(groups)]
                    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, f'threshold_ratio_{threshold}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved plot for threshold {threshold}")

# used
def check_community_consistency(consensus_communities, marker_list):
    """Check which body segments belong to each community"""
    print("Community Composition:")
    print("=" * 50)
    
    for i, community in enumerate(consensus_communities):
        print(f"\nCommunity {i+1}:")
        community_markers = [marker for marker in marker_list if marker in community]
        for marker in sorted(community_markers):
            print(f"  - {marker}")
    
    print(f"\nTotal markers in communities: {sum(len(comm) for comm in consensus_communities)}")
    print(f"Total markers in marker_list: {len(marker_list)}")
    
    # Check for markers not in any community
    all_community_markers = set()
    for community in consensus_communities:
        all_community_markers.update(community)
    
    missing_markers = set(marker_list) - all_community_markers
    if missing_markers:
        print(f"\nMarkers not in any community: {sorted(missing_markers)}")
    
    return consensus_communities

#used
# Main execution function
def run_threshold_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on,
                          base_path, marker_list, full, correlation_method, consensus_communities,
                          threshold_list=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], result_base_path=None):
    """
    Complete threshold analysis pipeline
    """
    
    # Check community composition
    print("Checking community composition...")
    check_community_consistency(consensus_communities, marker_list)
    
    # Analyze threshold ratios
    print("\nAnalyzing threshold ratios...")
    all_results = analyze_threshold_ratios(diagnosis, kinematics_list, task_names, tracking_systems, 
                                         runs, pd_on, base_path, marker_list, full, correlation_method, 
                                         consensus_communities, threshold_list)
    
    # Perform statistics
    print("Performing statistical analysis...")
    stats_results = perform_threshold_statistics(all_results)
    
    # Create plots
    print("Creating plots...")
    from config import RESULT_BASE_PATH
    _base = Path(result_base_path) if result_base_path else RESULT_BASE_PATH
    save_dir = _base / "modularity" / "threshold_ratio_plots"
    plot_threshold_results(all_results, stats_results, save_dir)
    
    # Print summary
    print(f"\nAnalysis complete! Results saved to: {save_dir}")
    
    return all_results, stats_results

def modularity_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                    correlation_method, threshold, clustering_method, consensus_communities):
    

    # Threshold list comes from config — set MODULARITY_THRESHOLD_LIST there
    from config import MODULARITY_THRESHOLD_LIST
    threshold_list = MODULARITY_THRESHOLD_LIST

    # Filter consensus communities to remove excluded markers per task
    from config import EXCLUDE_MARKERS_BY_TASK
    effective_communities = []
    for task_name in task_names:
        exclude = EXCLUDE_MARKERS_BY_TASK.get(task_name, [])
        if exclude:
            filtered = [
                {m for m in community if m not in exclude}
                for community in consensus_communities
            ]
            # Remove empty communities
            effective_communities = [c for c in filtered if c]
            print(f"  Modularity: excluding {exclude} for {task_name}")
            print(f"  Effective communities: {[sorted(c) for c in effective_communities]}")
        else:
            effective_communities = list(consensus_communities)
    consensus_communities = effective_communities if effective_communities else list(consensus_communities)

    # Run the complete analysis
    all_results, stats_results = run_threshold_analysis(
        diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on,
        base_path, marker_list, full, correlation_method, consensus_communities,
        threshold_list, result_base_path=result_base_path
    )

    def _print_significant(stats_results, label):
        """Print all significant results from a stats_results dict of
        {task: {direction: {community: {..., 'significant': bool, 'p_corrected': float}}}}."""
        print(f"\nSignificant results ({label}):")
        found = False
        for task in stats_results:
            for direction in stats_results[task]:
                for community, result in stats_results[task][direction].items():
                    if result.get('significant'):
                        found = True
                        print(f"  {task} - {direction} - community {community}: "
                              f"p_corrected = {result['p_corrected']:.4f}")
        if not found:
            print("  none")

    # Resolution values for modularity/Louvain (nx.community.louvain_communities(...,
    # resolution=...) and nx.community.modularity(..., resolution=...)).
    # Comes from config — set MODULARITY_RESOLUTION_LIST there. The full resolution-dependent
    # analysis (Louvain community detection + allegiance matrices, community strength metrics
    # for BOTH the fixed literature communities and the Louvain-detected ones, consensus/
    # per-subject modularity scores, and all associated plots) runs once per value in the
    # list, in a single call to modularity_main.
    from config import MODULARITY_RESOLUTION_LIST
    resolution_list = MODULARITY_RESOLUTION_LIST

    # Collect modularity (Q) scores across the whole resolution sweep, for the
    # Q-vs-resolution plot at the end (mirrors the Pmc13246872 style analysis).
    q_consensus_by_resolution = {}
    q_own_by_resolution = {}

    for resolution in resolution_list:
        print(f"\n=== Modularity analysis at resolution = {resolution} ===")

        # average (between gait cycles) allegiance matrices for each subject (for each walking
        # speed and direction). Louvain detection itself now uses this resolution, so this has
        # to be recomputed (or reloaded from its resolution-specific cache) every iteration.
        print("  Step: load_allegiance_matrices (slow — runs Louvain per gait cycle)...")
        avg_subject_allegiance_matrices, std_subject_allegience_matrices = load_allegiance_matrices(diagnosis, kinematics_list, task_names, 
                                                                                    tracking_systems, runs, pd_on, base_path,
                                                                                    marker_list, result_base_path, full, correlation_method, clustering_method, resolution)

        # average (between the subjects) group allegiance matrices for each group, walking speed, and direction
        print("  Step: calculate_avg_allg_mtrx...")
        average_group_allegiance_matrices = calculate_avg_allg_mtrx(avg_subject_allegiance_matrices, full)

        # returns a dict with communities for each subject (Louvain, at this resolution)
        # e.g. subject_communities['Parkinson']['pp102']['walkPreferred']['acc']['AP'] is [{0, 1, 2, 4, 6, 8, 10, 11, 12, 13, 15, 17, 19, 21}, {3, 5, 7, 9, 14, 16, 18, 20}]
        print("  Step: calc_subject_communities...")
        subject_communities = calc_subject_communities(avg_subject_allegiance_matrices, threshold)

        # returns a dict with communities for each group (Louvain, at this resolution; speed and direction specific)
        # e.g. group_communities['Parkinson']['walkPreferred']['acc']['AP'] is [{0, 1, 2, 3, 10, 11, 12, 13}, {4, 6, 8, 15, 17, 19, 21}, {5, 7, 9, 14, 16, 18, 20}]
        print("  Step: calc_group_communities...")
        group_communities = calc_group_communities(average_group_allegiance_matrices, threshold)

        # Diagnostic: show how many communities Louvain actually found per group/task/
        # kinematic/direction at this resolution, so degenerate cases (1 giant community,
        # or N singletons) are visible up front instead of only showing up as a NaN-histogram
        # crash three functions later.
        print(f"  Community counts at resolution={resolution} (threshold={threshold}):")
        for _grp in group_communities:
            for _task in group_communities[_grp]:
                for _kin in group_communities[_grp][_task]:
                    for _dir, _comms in group_communities[_grp][_task][_kin].items():
                        if _comms:
                            _sizes = sorted(len(c) for c in _comms)
                            print(f"    {_grp}/{_task}/{_kin}/{_dir}: {len(_comms)} communities, sizes={_sizes}")

        # --- Path A: fixed, literature-based consensus communities ---
        # (Same partition applied to everyone. Intra/inter-community strength here is
        # mathematically independent of resolution by construction — resolution only
        # changes the modularity SCORE computed below, not this mean-edge-weight metric.
        # Still run every iteration for consistent output naming across the sweep.)
        print("  Step: calculate_community_strength_metrics (literature/consensus communities)...")
        metrics_consensus = calculate_community_strength_metrics(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                                correlation_method, consensus_communities, resolution)

        # --- Path B: Louvain-detected, data-driven communities (own group's structure) ---
        # This one genuinely varies with resolution, since resolution changes what Louvain finds.
        print("  Step: calculate_community_strength_metrics_own (Louvain-detected communities)...")
        metrics_own, community_sizes_own = calculate_community_strength_metrics_own(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full,
                                correlation_method, group_communities, resolution)

        # uses nx.modularity, graphs from every subject and community structure from every subject
        print("  Step: calc_modularity_per_subject...")
        own_modularity_per_subject = calc_modularity_per_subject(subject_communities, marker_list, diagnosis, kinematics_list, 
                                                         task_names, base_path, tracking_systems, runs, pd_on, full, correlation_method, resolution)
        q_own_by_resolution[resolution] = own_modularity_per_subject

        # uses nx.modularity, graphs from every subject and consensus community structure (defined as global variable)
        print("  Step: calculate_modularity (consensus)...")
        consensus_modularity_per_subject = calculate_modularity(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                        correlation_method, consensus_communities, resolution)
        q_consensus_by_resolution[resolution] = consensus_modularity_per_subject

        # --- Plots + stats for both community sources, kept in clearly separate files ---
        community_sizes_by_label = {"CONSENSUS": None, "LOUVAIN": community_sizes_own}
        for label, metrics in [("CONSENSUS", metrics_consensus), ("LOUVAIN", metrics_own)]:

            df, stats_results = analyze_community_strength_data(
                metrics['intra_community_strength'], 
                resolution,
                save_path=Path(result_base_path) / "modularity" / f"intra_community_strength_{label}_reso_{str(resolution)}.png",
                correction_method='fdr_bh',
                community_sizes=community_sizes_by_label[label]
                )
            _print_significant(stats_results, f"intra-community strength ({label}), resolution={resolution}")

            # For inter-community strength analysis (only if computed)
            if 'inter_community_strength' in metrics:
                df, stats_results = analyze_inter_community_strength(
                    metrics['inter_community_strength'],
                    resolution,
                    save_path=Path(result_base_path) / "modularity" / f"inter_community_strength_{label}_reso_{str(resolution)}.png"
                )
                _print_significant(stats_results, f"inter-community strength ({label}), resolution={resolution}")

            # For inter-to-intra ratio analysis (only if computed)
            if 'inter_to_intra_ratios' in metrics:
                df, stats_results = analyze_inter_to_intra_ratios(
                    metrics['inter_to_intra_ratios'],
                    resolution,
                    save_path=Path(result_base_path) / "modularity" / f"inter_intra_ratios_inter_community_strength_{label}_reso_{str(resolution)}.png"
                )
                _print_significant(stats_results, f"inter-to-intra ratio ({label}), resolution={resolution}")

        # NOTE: plot_all_allegiance_matrices_with_communities() and permutation.permute()'s
        # underlying plotting.py filenames are NOT resolution-aware (I don't have plotting.py
        # to confirm/patch this safely) — the allegiance-matrix visualisations below will
        # overwrite each other across resolutions. The permutation histograms are protected
        # from that below by folding resolution into matrix_type, which IS part of their
        # filename. If you want the allegiance-matrix plots kept per-resolution too, share
        # plotting.py and I'll wire it through properly.
        plot_all_allegiance_matrices_with_communities(average_group_allegiance_matrices, group_communities, marker_list, result_base_path, correlation_method, full, resolution)

        #TO DO: put the below in a separate function
        from config import EXCLUDE_MARKERS_BY_TASK
        for task in task_names:
            exclude = EXCLUDE_MARKERS_BY_TASK.get(task, [])
            effective_marker_list = [m for m in marker_list if m not in exclude] if exclude else marker_list
            for kinematic in kinematics_list:
                pd_data = average_group_allegiance_matrices.get('Parkinson', {}).get(task, {}).get(kinematic, {})
                hc_data = average_group_allegiance_matrices.get('Control', {}).get(task, {}).get(kinematic, {})
                if not pd_data or not hc_data:
                    print(f"  Skipping permutation for {task}-{kinematic}: no data for one or both groups")
                    continue
                # only run for directions present in BOTH groups
                directions = sorted(set(pd_data.keys()) & set(hc_data.keys()))
                for direction in directions:
                    matrix1 = pd_data[direction]
                    matrix2 = hc_data[direction]

                    # Guard against degenerate matrices — e.g. Louvain collapsing everyone
                    # into one giant community (or all singletons) at an extreme resolution
                    # value makes the allegiance matrix constant, which makes Spearman
                    # correlation undefined (NaN) for every permutation, crashing the
                    # downstream histogram. Skip with a clear message instead.
                    if np.isnan(matrix1).any() or np.isnan(matrix2).any():
                        print(f"  Skipping permutation for {task}-{kinematic}-{direction} "
                              f"(resolution={resolution}): NaN values in allegiance matrix")
                        continue
                    if np.nanstd(matrix1) == 0 or np.nanstd(matrix2) == 0:
                        print(f"  Skipping permutation for {task}-{kinematic}-{direction} "
                              f"(resolution={resolution}): constant allegiance matrix — Louvain "
                              f"likely collapsed to a single community (or all singletons) at this resolution")
                        continue

                    # resolution folded into matrix_type so per-resolution permutation
                    # histograms don't overwrite each other (matrix_type IS part of the
                    # filename built inside plotting.plot_permutation_histogram)
                    matrix_type = f'allegiance_avg_reso_{resolution}'
                    print(f"  Running permutation: {task} - {kinematic} - {direction} (resolution={resolution})")
                    permutation.permute(matrix1, matrix2, effective_marker_list, task, matrix_type, kinematic, direction,
                                         result_base_path, correlation_method, n_iter=5000)

    # After the full resolution sweep: plot mean modularity (Q) vs resolution,
    # for both the fixed literature communities and the Louvain-detected ones.
    # This is the metric that should trace a smooth curve across resolution
    # (per Pmc13246872), even at resolutions where the partition itself degenerates.
    print("\n=== Plotting modularity (Q) vs resolution ===")
    plot_modularity_vs_resolution(q_consensus_by_resolution, task_names, kinematics_list, result_base_path, "CONSENSUS")
    plot_modularity_vs_resolution(q_own_by_resolution, task_names, kinematics_list, result_base_path, "LOUVAIN")

    return None