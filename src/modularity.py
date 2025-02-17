
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
import community as community_louvain
from collections import defaultdict
import re
from pathlib import Path

def extract_onset_indices(filename):
    """Extract numerical onset indices from the kinectome filename."""

    match = re.search(r'kinct(\d+)-(\d+)', filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    
    return None, None


def get_gait_cycle_label(onset, events):
    """Determine if the onset corresponds to a left or right gait cycle."""

    for i in range(len(events) - 1):
        if events.iloc[i]['onset'] <= onset < events.iloc[i + 1]['onset']:
            return 'left' if 'left' in events.iloc[i]['event_type'] else 'right'
        
    return 'unknown'


def load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics):
    """Loads kinectome files and sorts them by onset indices."""
    os.chdir(f'{base_path}/derived_data/sub-{sub_id}/kinectomes')
    file_list = os.listdir()
    
    # run 'on' or 'off' only exists in the file names of pwPD 
    if run:
        relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, run, kinematics])]
    else:
        relevant_files = [file for file in file_list if all(x in file for x in [task_name, tracksys, kinematics])]
    
    sorted_files = sorted(relevant_files, key=lambda file: extract_onset_indices(file)[0])
    
    return [np.load(file) for file in sorted_files]

def build_graph(kinectome):
    """Builds weighted graphs for AP, ML, V directions while preserving meaningful negative correlations."""
    graphs = []
    for direction in range(kinectome.shape[2]):
        G = nx.Graph()
        num_nodes = kinectome.shape[0]
        min_weight = np.min(kinectome[:, :, direction])
        shift = -min_weight if min_weight < 0 else 0  # Shift weights to be non-negative
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = kinectome[i, j, direction] + shift  # Apply shift
                if not np.isnan(weight):
                    G.add_edge(i, j, weight=weight)
        graphs.append(G)
    return graphs

def run_louvain(G, num_iterations=100):
    """Runs Louvain community detection multiple times and returns all partitions."""
    partitions= []
    partitions_part = []

    for _ in range(num_iterations):
        partition= nx.community.louvain_communities(G, weight='weight')
        partitions.append(partition)

        # for checking the difference between louvain communities and partitions
        partition_part = nx.community.louvain_partitions(G, weight='weight')
        partitions_part.append(partition_part)

    return partitions

def compute_allegiance_matrix(partitions, num_nodes):
    """Constructs an allegiance matrix from Louvain community partitions."""
    allegiance_matrix = np.zeros((num_nodes, num_nodes))

    for partition in partitions:
        node_to_community = {}
        for comm_idx, community in enumerate(partition):
            for node in community:
                node_to_community[node] = comm_idx  # Map each node to its community index

        for i in range(num_nodes):
            for j in range(num_nodes):
                if node_to_community.get(i) == node_to_community.get(j):  # Check community membership
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


def modularity_analysis(base_path, sub_id, task_name, tracksys, run, kinematics):
    """Main function for modularity analysis across gait cycles and speeds.
    
    note:
        it is not computed per group, so all allegiance matrices (from one subject per trial and per direction) are put into all_allegiance_matrices dict

    the structure is as follows:
        all_allegiance_matrices = {"AP": [], "ML": [], "V": []}, where the lists contain 33x33 allegiance matrices

        the length of this list depends on the number of gait cycles - there are as many as complete gait cycles in that trial
    
    
    """
    kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics)
    all_allegiance_matrices = {"AP": [], "ML": [], "V": []}
    
    for kinectome in kinectomes:
        graphs = build_graph(kinectome)
        for idx, direction in enumerate(["AP", "ML", "V"]):
            G = graphs[idx]
            partitions = run_louvain(G, num_iterations=100)
            allegiance_matrix = compute_allegiance_matrix(partitions, num_nodes=G.number_of_nodes())
            all_allegiance_matrices[direction].append(allegiance_matrix)
            
            # Visualize one of the graphs (for debugging purposes)
            # draw_graph_with_weights(G)
    
    return all_allegiance_matrices
