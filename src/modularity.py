import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
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

def build_graph(kinectome, marker_list, bound_value=0.6):

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
    #     # Compute averages for each task, kinematic, direction combination
    #     for task in all_tasks:
    #         for kinematic in all_kinematics.get(task, {}):
    #             group_avg_matrices[group][task][kinematic] = {}
                
    #             for direction in all_kinematics[task][kinematic]:
    #                 # Collect matrices for this combination
    #                 valid_matrices = []
                    
    #                 for participant_id, participant_data in participants.items():
    #                     if (task in participant_data and 
    #                         kinematic in participant_data[task] and 
    #                         direction in participant_data[task][kinematic]):
    #                         # Get the matrix
    #                         matrix = participant_data[task][kinematic][direction]
                            
    #                         # Only include non-None matrices with actual content and correct shape
    #                         if matrix is not None and hasattr(matrix, 'shape') and matrix.shape == (22, 22):
    #                             valid_matrices.append(matrix)
                    
    #                 # Compute average if we have valid matrices
    #                 if valid_matrices:
    #                     # All matrices should be numpy arrays with shape (22, 22)
    #                     avg_matrix = np.mean(valid_matrices, axis=0)
    #                     group_avg_matrices[group][task][kinematic][direction] = avg_matrix
    
    # return group_avg_matrices

def plot_all_allegiance_matrices(allegiance_matrices, marker_list, result_base_path, correlation_method, full):
    """ visualise and save all group allegiance matrices as .png
    """
    
    for group in allegiance_matrices.keys():
        for task in allegiance_matrices[group].keys():
            for kinematic in allegiance_matrices[group][task].keys():
                for direction in allegiance_matrices[group][task][kinematic].keys():
                    matrix = allegiance_matrices[group][task][kinematic][direction]
                    plotting.visualise_allegiance_matrix(matrix, marker_list, group, task, kinematic, direction, result_base_path, correlation_method, full)



def modularity_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, correlation_method):
    
    
    avg_subject_allegience_matrices, std_subject_allegience_matrices = load_allegiance_matrices(diagnosis, kinematics_list, task_names, 
                                                                                tracking_systems, runs, pd_on, base_path,
                                                                                marker_list, result_base_path, full, correlation_method)
    

    average_group_allegiance_matrices = calculate_avg_allg_mtrx(avg_subject_allegience_matrices, full)
    std_group_allegiance_matrices = calculate_avg_allg_mtrx(std_subject_allegience_matrices, full)
    
    # comment out once all the plots are generated
    # plot_all_allegiance_matrices(average_group_allegiance_matrices, marker_list, result_base_path, correlation_method, full)

    task ='walkSlow'
    matrix_type ='allegiance_avg'
    kinematic = 'acc'
    direction = 'V'
    
    matrix1 = average_group_allegiance_matrices['Parkinson'][task][kinematic][direction]
    matrix2 = average_group_allegiance_matrices['Control'][task][kinematic][direction]


    permutation.permute(matrix1, matrix2, marker_list, task, matrix_type, kinematic, direction, result_base_path, correlation_method)
    return None