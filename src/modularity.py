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

def build_graph(kinectome, marker_list, bound_value=None):

    """Builds weighted graphs for AP, ML, V directions if ndim==2, 
    else builds one graph for the full kinectome (containins all directions)     
    while preserving meaningful negative correlations."""
    
    # np.expand_dims
    # kinectome = kinectome[..., None] if kinectome.ndim == 2 else kinectome
    directions = ['AP', 'ML', 'V']
    marker_list = (
                    [f"{m}_{d}" for m in marker_list for d in directions] if kinectome.ndim == 2 else marker_list
                    )
    kinectome = np.expand_dims(kinectome, axis=-1) if kinectome.ndim == 2 else kinectome
     
    
    graphs = []
    for direction in range(kinectome.shape[-1]):
        G = nx.Graph()
        num_nodes = kinectome.shape[0]
        min_weight = np.min(kinectome[:, :, direction])
        shift = -min_weight if min_weight < 0 else 0  # Shift weights to be non-negative
        
        # Add nodes with marker labels
        for i in range(num_nodes):
            G.add_node(marker_list[i])  # Assign marker name as node label

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = kinectome[i, j, direction] + shift  # Apply shift
                if bound_value is None:
                    if not np.isnan(weight):
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
                else: # Apply threshold
                    if not np.isnan(weight) and weight >= bound_value:
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
        
        graphs.append(G)
    
    return graphs

def run_louvain(G, num_iterations=100):
    """Runs Louvain community detection multiple times and returns all partitions."""
    partitions= []

    for _ in range(num_iterations):
        partition= nx.community.louvain_communities(G, weight='weight')
        partitions.append(partition)

    return partitions


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


def all_allegiance_matrices_for_subject(kinectomes, marker_list):
    """ A function which saves allegiance matrices built from the kinectomes
    note:
        it is not computed per group, so all allegiance matrices (from one subject per trial and per direction) are put into all_allegiance_matrices dict
    """
    all_allegiance_matrices = {"AP": [], "ML": [], "V": []}
   
    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list)
        
        if len(graphs) == 1:
            # If only one (full) graph, assign it to AP direction (idx 0)
            G = graphs[0]
            partitions = run_louvain(G, num_iterations=100)
            marker_list_exp = permutation.expand_marker_list(marker_list)
            allegiance_matrix = compute_allegiance_matrix(partitions, marker_list_exp, num_nodes=G.number_of_nodes())
            all_allegiance_matrices["AP"].append(np.array(allegiance_matrix))
        else:
            # If multiple graphs (3 directions), process each one
            for idx, direction in enumerate(["AP", "ML", "V"]):
                G = graphs[idx]
                partitions = run_louvain(G, num_iterations=100)
                allegiance_matrix = compute_allegiance_matrix(partitions, marker_list, num_nodes=G.number_of_nodes())
                all_allegiance_matrices[direction].append(np.array(allegiance_matrix))
           
        # Visualize one of the graphs (for debugging purposes)
        # draw_graph_with_weights(G)
   
    return all_allegiance_matrices



def modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method):

    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)

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

                        if kinectomes is None:
                            continue

                        allegiance_matrices = all_allegiance_matrices_for_subject(kinectomes, marker_list)
                        
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
    result_folder = Path(result_path) / "allegiance_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    
    # Define the save paths for the pickle files
    avg_save_path = result_folder / f"avg_allegiance_matrices_{correlation_method}.pkl"
    std_save_path = result_folder / f"std_allegiance_matrices_{correlation_method}.pkl"

    # Save dictionaries as pickle files
    with open(avg_save_path, "wb") as f:
        pickle.dump(all_avg_allegiance, f)

    with open(std_save_path, "wb") as f:
        pickle.dump(all_std_allegiance, f)

def load_allegiance_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method):
    """ checks if the allegiance matrices are calculated and saved as a pickle file (and loads them). 
    otherwise calculates them and saves as a pickle file

    returns:
    a dict containing average allegiance matrices per group, subject, task, kinematics, and direction
    a dict containing std (as a matrix) of allegiance matrices per group, subject, task, kinematics, and direction
    """
        # Define result path
    result_folder = Path(result_path) / "allegiance_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    avg_save_path = result_folder / f"avg_allegiance_matrices_{'full_' if full else ''}{correlation_method}.pkl"
    std_save_path = result_folder / f"std_allegiance_matrices_{'full_' if full else ''}{correlation_method}.pkl"
    
    # if allegiance matrices are not calculated
    if not avg_save_path.exists() and not std_save_path.exists():
        modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path, full, correlation_method)
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
                            
                            # Check if matrix is valid and 66x66
                            if (matrix is not None and 
                                hasattr(matrix, 'shape') and 
                                matrix.shape == (66, 66) and
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
                               
                                # Only include non-None matrices with actual content and correct shape
                                if (matrix is not None and 
                                    hasattr(matrix, 'shape') and 
                                    matrix.shape == (22, 22) and
                                    not np.isnan(matrix).all()):
                                    valid_matrices.append(matrix)
                       
                        # Compute average if we have valid matrices
                        if valid_matrices:
                            # All matrices should be numpy arrays with shape (22, 22)
                            avg_matrix = np.nanmean(valid_matrices, axis=0)
                            group_avg_matrices[group][task][kinematic][direction] = avg_matrix
   
    return group_avg_matrices

def plot_all_allegiance_matrices(allegiance_matrices, marker_list, result_base_path, correlation_method, full):
    """ visualise and save all group allegiance matrices as .png
    """
    
    for group in allegiance_matrices.keys():
        for task in allegiance_matrices[group].keys():
            for kinematic in allegiance_matrices[group][task].keys():
                for direction in allegiance_matrices[group][task][kinematic].keys():
                    matrix = allegiance_matrices[group][task][kinematic][direction]
                    plotting.visualise_allegiance_matrix(matrix, marker_list, group, task, kinematic, direction, result_base_path, correlation_method, full)

def plot_all_allegiance_matrices_with_communities(allegiance_matrices, group_communities, marker_list, result_base_path, correlation_method, full):
    """ visualise and save all group allegiance matrices as .png with community-based ordering
    """
   
    for group in allegiance_matrices.keys():
        for task in allegiance_matrices[group].keys():
            for kinematic in allegiance_matrices[group][task].keys():
                for direction in allegiance_matrices[group][task][kinematic].keys():
                    matrix = allegiance_matrices[group][task][kinematic][direction]
                    communities = group_communities[group][task][kinematic][direction]
                    plotting.visualise_allegiance_matrix_with_communities(matrix, communities, marker_list, group, task, kinematic, direction, result_base_path, correlation_method, full)

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


def calc_community_fit(subject_communities, group_communities):
    """Calculate fit between individual and group community structures using ARI."""
    fits = {}
    
    for group in subject_communities:
        fits[group] = {}
        for subject in subject_communities[group]:
            fits[group][subject] = {}
            for task in subject_communities[group][subject]:
                fits[group][subject][task] = {}
                for kinematic in subject_communities[group][subject][task]:
                    fits[group][subject][task][kinematic] = {}
                    for direction in subject_communities[group][subject][task][kinematic]:
                        # Get individual and group communities
                        ind_comm = subject_communities[group][subject][task][kinematic][direction]
                        grp_comm = group_communities[group][task][kinematic][direction]
                        if not bool(ind_comm): # check if the dict is empty for subjects with no data
                            continue
                        
                        # Convert to node labels for ARI calculation
                        max_node = max(max(comm) for comm in ind_comm + grp_comm)
                        ind_labels = np.zeros(max_node + 1)
                        grp_labels = np.zeros(max_node + 1)
                        
                        for i, comm in enumerate(ind_comm):
                            for node in comm:
                                ind_labels[node] = i
                        
                        for i, comm in enumerate(grp_comm):
                            for node in comm:
                                grp_labels[node] = i
                        
                        # Calculate ARI as fit measure
                        fits[group][subject][task][kinematic][direction] = adjusted_rand_score(ind_labels, grp_labels)
    
    return fits

def calc_community_fit_stats(fits):
    """Compare fits between two groups using appropriate statistical test."""
    results = {}

    group1 = list(fits.keys())[0]
    group2 = list(fits.keys())[1]

    # Get all combinations of task/kinematic/direction
    all_combinations = set()
    for group in fits:
        for subject in fits[group]:
            for task in fits[group][subject]:
                for kinematic in fits[group][subject][task]:
                    for direction in fits[group][subject][task][kinematic]:
                        all_combinations.add((task, kinematic, direction))
    
    for task, kinematic, direction in all_combinations:
        # Extract fits for both groups
        g1_fits = [fits[group1][subj][task][kinematic][direction] 
                   for subj in fits[group1] 
                   if task in fits[group1][subj] and kinematic in fits[group1][subj][task] 
                   and direction in fits[group1][subj][task][kinematic]]
        
        g2_fits = [fits[group2][subj][task][kinematic][direction] 
                   for subj in fits[group2] 
                   if task in fits[group2][subj] and kinematic in fits[group2][subj][task] 
                   and direction in fits[group2][subj][task][kinematic]]
        
        if len(g1_fits) < 3 or len(g2_fits) < 3:
            continue
            
        # Test normality
        _, p1 = stats.shapiro(g1_fits)
        _, p2 = stats.shapiro(g2_fits)
        
        # Choose appropriate test
        if p1 > 0.05 and p2 > 0.05:
            # Both normal - use t-test
            stat, p = stats.ttest_ind(g1_fits, g2_fits, alternative='less')  # Test if g1 < g2
            test_used = 'ttest'
        else:
            # Non-normal - use Mann-Whitney U
            stat, p = stats.mannwhitneyu(g1_fits, g2_fits, alternative='less')
            test_used = 'mannwhitney'
        
        results[(task, kinematic, direction)] = {
            'group1_mean': np.mean(g1_fits),
            'group2_mean': np.mean(g2_fits),
            'statistic': stat,
            'p_value': np.round(p, 3),
            'test_used': test_used,
            'n1': len(g1_fits),
            'n2': len(g2_fits)
        }
    
    return pd.DataFrame(results)

def calculate_modularity_scores(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities):
    """Calculate modularity scores for group communities and individual subjects."""
    group_modularity = {}
    subject_modularity = {}
    
    for group in group_communities:
        group_modularity[group] = {}
        subject_modularity[group] = {}
        
        for task in group_communities[group]:
            group_modularity[group][task] = {}
            subject_modularity[group][task] = {}
            
            for kinematic in group_communities[group][task]:
                group_modularity[group][task][kinematic] = {}
                subject_modularity[group][task][kinematic] = {}
                
                for direction in group_communities[group][task][kinematic]:
                    communities = group_communities[group][task][kinematic][direction]
                    
                    # Group modularity using group allegiance matrix
                    group_matrix = average_group_allegiance_matrices[group][task][kinematic][direction]
                    G_group = nx.from_numpy_array(group_matrix)
                    group_modularity[group][task][kinematic][direction] = nx.community.modularity(G_group, communities, weight='weight')
                    
                    # Subject modularity scores
                    subject_modularity[group][task][kinematic][direction] = {}
                    for subject in avg_subject_allegiance_matrices[group]:
                        if task in avg_subject_allegiance_matrices[group][subject] and \
                           kinematic in avg_subject_allegiance_matrices[group][subject][task] and \
                           direction in avg_subject_allegiance_matrices[group][subject][task][kinematic]:
                            
                            subj_matrix = avg_subject_allegiance_matrices[group][subject][task][kinematic][direction]
                            if subj_matrix is None:
                                continue 
                            elif len(communities) <= 1:
                                subject_modularity[group][task][kinematic][direction][subject] = np.nan
                            else:    
                                G_subj = nx.from_numpy_array(subj_matrix)
                                subject_modularity[group][task][kinematic][direction][subject] = nx.community.modularity(G_subj, communities, weight='weight')
          
    return group_modularity, subject_modularity


def compare_modularity_between_groups(subject_modularity):
    """Compare modularity scores between groups using appropriate statistical test."""
    results = {}
    
    group1 = list(subject_modularity.keys())[0]
    group2 = list(subject_modularity.keys())[1]

    # Get all combinations
    all_combinations = set()
    for group in subject_modularity:
        for task in subject_modularity[group]:
            for kinematic in subject_modularity[group][task]:
                for direction in subject_modularity[group][task][kinematic]:
                    all_combinations.add((task, kinematic, direction))
    
    for task, kinematic, direction in all_combinations:
        # Extract modularity scores
        g1_scores = list(subject_modularity[group1][task][kinematic][direction].values()) if \
                   task in subject_modularity[group1] and kinematic in subject_modularity[group1][task] and \
                   direction in subject_modularity[group1][task][kinematic] else []
        
        g2_scores = list(subject_modularity[group2][task][kinematic][direction].values()) if \
                   task in subject_modularity[group2] and kinematic in subject_modularity[group2][task] and \
                   direction in subject_modularity[group2][task][kinematic] else []
        
        if len(g1_scores) < 3 or len(g2_scores) < 3:
            continue
        
        # Test normality and choose appropriate test
        _, p1 = stats.shapiro(g1_scores)
        _, p2 = stats.shapiro(g2_scores)
        
        if p1 > 0.05 and p2 > 0.05:
            stat, p = stats.ttest_ind(g1_scores, g2_scores)
            test_used = 'ttest'
        else:
            stat, p = stats.mannwhitneyu(g1_scores, g2_scores)
            test_used = 'mannwhitney'
        
        results[(task, kinematic, direction)] = {
            'group1_mean': np.mean(g1_scores),
            'group2_mean': np.mean(g2_scores),
            'group1_std': np.std(g1_scores),
            'group2_std': np.std(g2_scores),
            'statistic': stat,
            'p_value': np.round(p, 3),
            'test_used': test_used,
            'n1': len(g1_scores),
            'n2': len(g2_scores)
        }
    
    return pd.DataFrame(results)

def calculate_within_community_density(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities):
    """Calculate within-community density for each community with community identification."""
    group_densities = {}
    subject_densities = {}
    
    for group in group_communities:
        group_densities[group] = {}
        subject_densities[group] = {}
        
        for task in group_communities[group]:
            group_densities[group][task] = {}
            subject_densities[group][task] = {}
            
            for kinematic in group_communities[group][task]:
                group_densities[group][task][kinematic] = {}
                subject_densities[group][task][kinematic] = {}
                
                for direction in group_communities[group][task][kinematic]:
                    communities = group_communities[group][task][kinematic][direction]
                    
                    # Group within-community densities
                    group_matrix = average_group_allegiance_matrices[group][task][kinematic][direction]
                    group_densities[group][task][kinematic][direction] = {}
                    
                    for i, community in enumerate(communities):
                        nodes = list(community)
                        community_key = f"community_{i}_nodes_{sorted(nodes)}"
                        
                        if len(nodes) > 1:
                            submatrix = group_matrix[np.ix_(nodes, nodes)]
                            mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
                            density = np.mean(submatrix[mask]) if mask.sum() > 0 else 0
                        else:
                            density = 0
                        group_densities[group][task][kinematic][direction][community_key] = density
                    
                    # Subject within-community densities
                    subject_densities[group][task][kinematic][direction] = {}
                    for subject in avg_subject_allegiance_matrices[group]:
                        if task in avg_subject_allegiance_matrices[group][subject] and \
                           kinematic in avg_subject_allegiance_matrices[group][subject][task] and \
                           direction in avg_subject_allegiance_matrices[group][subject][task][kinematic]:
                            
                            subj_matrix = avg_subject_allegiance_matrices[group][subject][task][kinematic][direction]

                            if subj_matrix is None:
                                continue
                            else:
                                subject_densities[group][task][kinematic][direction][subject] = {}
                            
                            for i, community in enumerate(communities):
                                nodes = list(community)
                                community_key = f"community_{i}_nodes_{sorted(nodes)}"
                                
                                if len(nodes) > 1:
                                    submatrix = subj_matrix[np.ix_(nodes, nodes)]
                                    mask = np.triu(np.ones_like(submatrix, dtype=bool), k=1)
                                    density = np.mean(submatrix[mask]) if mask.sum() > 0 else 0
                                else:
                                    density = 0
                                subject_densities[group][task][kinematic][direction][subject][community_key] = density
    
    return group_densities, subject_densities

def modularity_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, correlation_method, threshold):
    
    
    avg_subject_allegiance_matrices, std_subject_allegience_matrices = load_allegiance_matrices(diagnosis, kinematics_list, task_names, 
                                                                                tracking_systems, runs, pd_on, base_path,
                                                                                marker_list, result_base_path, full, correlation_method)
    
    subject_communities = calc_subject_communities(avg_subject_allegiance_matrices, threshold)
    
    average_group_allegiance_matrices = calculate_avg_allg_mtrx(avg_subject_allegiance_matrices, full)
    std_group_allegiance_matrices = calculate_avg_allg_mtrx(std_subject_allegience_matrices, full)
    
    group_communities = calc_group_communities(average_group_allegiance_matrices, threshold)

    # comment out once all the plots are generated
    # plot_all_allegiance_matrices_with_communities(average_group_allegiance_matrices, group_communities, marker_list, result_base_path, correlation_method, full)

    # check the how well the individual community structure fits the group community structure (using adjusted rand index)
    community_fit = calc_community_fit(subject_communities, group_communities)    
    community_fit_stats = calc_community_fit_stats(community_fit)
    
    # check the modularity (strength of the connections) in the communities
    group_modularity, subject_modularity = calculate_modularity_scores(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities)
    modularity_stats = compare_modularity_between_groups(subject_modularity)

    # Get within-community densities
    group_densities, subject_densities = calculate_within_community_density(avg_subject_allegiance_matrices, average_group_allegiance_matrices, group_communities)
    
    
    task ='walkSlow'
    matrix_type ='allegiance_avg'
    kinematic = 'acc'
    direction = 'V'
    
    matrix1 = average_group_allegiance_matrices['Parkinson'][task][kinematic][direction]
    matrix2 = average_group_allegiance_matrices['Control'][task][kinematic][direction]


    permutation.permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method, n_iter = 5000)
    return None