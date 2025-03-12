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
from pathlib import Path

def visualise_allegiance_matrix(allegiance_matrix, marker_list, sub_id, task_name, direction, figname):
    # Define marker ordering
    left_markers = [m for m in marker_list if m.startswith('l_')]
    right_markers = [m for m in marker_list if m.startswith('r_')]
    middle_markers = ['head', 'ster']
    ordered_marker_list = middle_markers + left_markers + right_markers
    
    # Get new indices based on the ordered list
    index_map = {marker: i for i, marker in enumerate(marker_list)}
    new_order = [index_map[m] for m in ordered_marker_list]

    # Reorder rows and columns
    reordered_matrix = allegiance_matrix[np.ix_(new_order, new_order)]
    
    # visualise
    plt.figure(figsize=(14, 11))
    sns.heatmap(reordered_matrix, cmap="coolwarm", xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)  
    plt.title(f"Allegiance Matrix of {sub_id} during {task_name} in {direction} direction")
    os.chdir('C:/Users/Karolina/Desktop/pykinectome/pykinectome/src/preprocessing')
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(figname, dpi=600)


def save_variability_to_csv(variability_scores, kinematics):
    """
    Saves variability scores to a CSV file.

    Parameters:
    - variability_scores (dict): Dictionary with variability scores structured as:
        variability_scores[group][sub_id][task_name][direction]
    - kinematics (str): The kinematics type (e.g., 'position', 'velocity', 'acceleration').
    - output_dir (str): Directory where CSV files will be saved.
    """
    os.chdir('C:\\Users\\Karolina\\Desktop\\pykinectome\\results')
    # with open('variability_scores_velocity.pickle', 'rb') as handle:
    #     variability_scores = pickle.load(handle)
    output_dir = 'C:\\Users\\Karolina\\Desktop\\pykinectome\\results'
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Define the output file name
    output_file = os.path.join(output_dir, f"variability_{kinematics}.csv")

    # Define CSV column names
    columns = ["group", "subject_id", 
               "pref_AP", "pref_ML", "pref_V",
               "fast_AP", "fast_ML", "fast_V",
               "slow_AP", "slow_ML", "slow_V"]

    # Open the CSV file for writing
    with open(output_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(columns)  # Write header

        # Iterate through groups (PD and Control)
        for group, subjects in variability_scores.items():
            for sub_id, tasks in subjects.items():
                # Extract variability scores or set to None if missing
                row = [
                    group,
                    sub_id,
                    tasks.get("walkPreferred", {}).get("AP", None),
                    tasks.get("walkPreferred", {}).get("ML", None),
                    tasks.get("walkPreferred", {}).get("V", None),
                    tasks.get("walkFast", {}).get("AP", None),
                    tasks.get("walkFast", {}).get("ML", None),
                    tasks.get("walkFast", {}).get("V", None),
                    tasks.get("walkSlow", {}).get("AP", None),
                    tasks.get("walkSlow", {}).get("ML", None),
                    tasks.get("walkSlow", {}).get("V", None),
                ]
                writer.writerow(row)

def build_graph(kinectome, marker_list):
    """Builds weighted graphs for AP, ML, V directions while preserving meaningful negative correlations."""
    graphs = []
    for direction in range(kinectome.shape[2]):
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
                if not np.isnan(weight):
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

def draw_graph_with_weights(G):
    """Visualizes the graph with edge weights."""
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Graph Representation of Kinectome")
    os.chdir('C:/Users/Karolina/Desktop/pykinectome/pykinectome/src/preprocessing')
    plt.savefig('test_plot_graph.png', dpi=600)


def all_allegiance_matrices_for_subject(kinectomes, marker_list):
    """ A function which saves allegiance matrices built from the kinectomes

    note:
        it is not computed per group, so all allegiance matrices (from one subject per trial and per direction) are put into all_allegiance_matrices dict

    """
    all_allegiance_matrices = {"AP": [], "ML": [], "V": []}
    
    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list)
        for idx, direction in enumerate(["AP", "ML", "V"]):
            G = graphs[idx]
            partitions = run_louvain(G, num_iterations=100)
            allegiance_matrix = compute_allegiance_matrix(partitions, marker_list, num_nodes=G.number_of_nodes())
            all_allegiance_matrices[direction].append(np.array(allegiance_matrix))
            
            # Visualize one of the graphs (for debugging purposes)
            # draw_graph_with_weights(G)
    
    return all_allegiance_matrices



def modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path):

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
                        
                        kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics)

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
    avg_save_path = result_folder / "avg_allegiance_matrices.pkl"
    std_save_path = result_folder / "std_allegiance_matrices.pkl"

    # Save dictionaries as pickle files
    with open(avg_save_path, "wb") as f:
        pickle.dump(all_avg_allegiance, f)

    with open(std_save_path, "wb") as f:
        pickle.dump(all_std_allegiance, f)

def load_allegiance_matrices(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path):
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

    avg_save_path = result_folder / "avg_allegiance_matrices.pkl"
    std_save_path = result_folder / "std_allegiance_matrices.pkl"
    
    # if allegiance matrices are not calculated
    if not avg_save_path.exists() and not std_save_path.exists():
        modularity_analysis(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path)
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



def modularity_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_path):
    
    
    avg_allegience_matrices, std_allegience_matrices = load_allegiance_matrices(diagnosis, kinematics_list, task_names, 
                                                                                tracking_systems, runs, pd_on, base_path,
                                                                                marker_list, result_path)


    return None