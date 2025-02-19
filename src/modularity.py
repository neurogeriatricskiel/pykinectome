import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils.data_loader import load_kinectomes


# def get_gait_cycle_label(onset, events):
#     """Determine if the onset corresponds to a left or right gait cycle."""

#     for i in range(len(events) - 1):
#         if events.iloc[i]['onset'] <= onset < events.iloc[i + 1]['onset']:
#             return 'left' if 'left' in events.iloc[i]['event_type'] else 'right'
        
#     return 'unknown'


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

# def compute_allegiance_matrix(partitions, num_nodes):
#     """Constructs an allegiance matrix from Louvain community partitions."""
#     allegiance_matrix = np.zeros((num_nodes, num_nodes))

#     for partition in partitions:
#         node_to_community = {}
#         for comm_idx, community in enumerate(partition):
#             for node in community:
#                 node_to_community[node] = comm_idx  # Map each node to its community index

#         for i in range(num_nodes):
#             for j in range(num_nodes):
#                 if node_to_community.get(i) == node_to_community.get(j):  # Check community membership
#                     allegiance_matrix[i, j] += 1

#     allegiance_matrix /= len(partitions)  # Normalize by number of iterations
#     return allegiance_matrix

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


def modularity_analysis(base_path, sub_id, task_name, tracksys, run, kinematics, marker_list):
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
        graphs = build_graph(kinectome, marker_list)
        for idx, direction in enumerate(["AP", "ML", "V"]):
            G = graphs[idx]
            partitions = run_louvain(G, num_iterations=100)
            allegiance_matrix = compute_allegiance_matrix(partitions, marker_list, num_nodes=G.number_of_nodes())
            all_allegiance_matrices[direction].append(allegiance_matrix)
            
            # Visualize one of the graphs (for debugging purposes)
            # draw_graph_with_weights(G)
    
    return all_allegiance_matrices
