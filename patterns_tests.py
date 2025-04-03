
# Description: This script is used to test pattern extraction in the kinectome data
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import numpy as np
import networkx as nx
from src.data_utils import data_loader, groups
from src.kinectome import *
from src.preprocessing import preprocessing
from src.graph_utils.kinectome2graph import build_graph, clustering_coef, cc_connected_components
from src.graph_utils.kinectome2pattern import full_subgraph, path_subgraph, cycle_subgraph
from tqdm import tqdm
from pathlib import Path
import csv
from collections import OrderedDict, defaultdict


 ##
# LOADING CONFIG CONSTANTS
##
# DATA-PATH
DATA_PATH = Path("Z:\\Keep Control\\Data\\lab dataset")
# RESULT_PATH
RESULT_PATH = Path("C:\\Users\\Karolina\\Desktop\\pykinectome\\results")
# Seleted tasks
TASK_NAMES = [
"walkSlow", "walkPreferred", "walkFast"
]
TRACKING_SYSTEMS = ["omc"]
# values to be extracted 
KINEMATICS = ['acc']
# ordered list of markers
MARKER_LIST = ['head', 'ster', 'l_sho', 'r_sho',  
            'l_elbl', 'r_elbl','l_wrist', 'r_wrist', 'l_hand', 'r_hand', 
            'l_asis', 'l_psis', 'r_asis', 'r_psis', 
            'l_th', 'r_th', 'l_sk', 'r_sk', 
            'l_ank', 'r_ank', 'l_toe', 'r_toe']

# MARKER_SUB_LIST = ['ster', 'l_psis', 'l_asis', 'l_th', 'l_sk', 'l_ank', 'l_toe']
# MARKER_SUB_LIST = ['ster', 'l_sho', 'l_elbl', 'l_wrist', 'l_hand']
MARKER_SUB_LIST = ['head', 'ster', '']
RUN = 'on' # dont have id "run" in sample data that I have 
FS = 200 # sampling rate
FULL= False
CORRELATION = 'pears'
DIAGNOSIS = ['diagnosis_parkinson']
sub_ids = ["pp008"] 
disease_sub_ids, matched_control_sub_ids = groups.define_groups(DIAGNOSIS)

# for sub_id in sub_ids:
#     for kinematics in KINEMATICS:
#         for task_name in TASK_NAMES:
#             for tracksys in TRACKING_SYSTEMS:
#                 kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics, FULL, CORRELATION)   
#                 for k in kinectomes:
#                     G_directions = build_graph(k, MARKER_LIST) # list of graphs
#                     directions_iterator = tqdm(G_directions, desc=f"---Subject: {sub_id}, Task: {task_name}---")
#                     for i,g in enumerate(directions_iterator):
#                         # full subgraph
#                         # subgraph = full_subgraph(g, MARKER_SUB_LIST)
#                         # path subgraph
#                         subgraph = path_subgraph(g, MARKER_SUB_LIST)
#                         # cycle subgraph
#                         # subgraph = cycle_subgraph(g, MARKER_SUB_LIST)

def calc_subgraph_weights(diagnosis, kinematics_list, task_names, tracking_systems, run, data_path, full, correlation_method, full_marker_list, sub_marker_list):
    # Dictionary to store edge weights
    subgraph_weights = {
        'Control': {},
        f"{diagnosis[0][10:].capitalize()}": {}  # Modify this key dynamically based on diagnosis if needed
    }

    for sub_id in disease_sub_ids + matched_control_sub_ids:
    # for sub_id in ['pp085']:
        group = 'Control' if sub_id in matched_control_sub_ids else f"{diagnosis[0][10:].capitalize()}"
        subgraph_weights[group][sub_id] = {}

        for kinematics in kinematics_list:
            subgraph_weights[group][sub_id][kinematics] = {}

            for task_name in task_names:
                subgraph_weights[group][sub_id][kinematics][task_name] = {"AP": {}, "ML": {}, "V": {}}

                for tracksys in tracking_systems:
                    # run is only needed for subjects with Parkinson's disease
                    run = run if sub_id in disease_sub_ids and f"{diagnosis[0][10:].capitalize()}" == 'Parkinson' else None
                
                    kinectomes = data_loader.load_kinectomes(
                        data_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method
                    )

                    if kinectomes is None:
                        continue

                    for k in kinectomes:
                        G_directions = build_graph(k, full_marker_list)
                        directions_iterator = tqdm(G_directions, desc=f"---Subject: {sub_id}, Task: {task_name}---")

                        for i, g in enumerate(directions_iterator):  # i = 0 (AP), 1 (ML), 2 (V)
                            subgraph = path_subgraph(g, sub_marker_list)  # Or use full_subgraph, cycle_subgraph

                            # Extract edge weights
                            edge_weights = {(u, v): d["weight"] for u, v, d in subgraph.edges(data=True)}

                            # Ensure data gets stored properly
                            direction = ["AP", "ML", "V"][i]

                            print(f"Before updating: {subgraph_weights['Control'].get(sub_id, 'Key does not exist')}")
                            subgraph_weights[group][sub_id][kinematics][task_name][direction].update(edge_weights) 

    return subgraph_weights


def save_weights_to_csv(subgraph_weights, direction, body_path):
    """ saves the weights between the body segments as a csv file.

    input:
    subgraph_weights (dict): contains the weights for each group per subject, kinematics, task and direction
    """
    for kinematics in KINEMATICS:
        csv_filename = f"{body_path}_{direction}_{kinematics}_dcor.csv"
        csv_path = Path(DATA_PATH) / csv_filename

        with open(csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            
                        # **Step 1: Collect all unique edges in the original order they appear**
            unique_edges = OrderedDict()
            for group in subgraph_weights:
                for sub_id in subgraph_weights[group]:
                    for task_name in TASK_NAMES:
                        edges = subgraph_weights[group][sub_id].get(kinematics, {}).get(task_name, {}).get(direction, {})
                        for edge in edges.keys():
                            if edge not in unique_edges:
                                unique_edges[edge] = None  # Maintain order

            # **Step 2: Create column headers**
            headers = ["Group", "Subject_ID"] + [f"{u}-{v}_{task}" for (u, v) in unique_edges for task in TASK_NAMES]
            writer.writerow(headers)

            # **Step 3: Write data rows**
            for group, subjects in subgraph_weights.items():
                for sub_id, sub_data in subjects.items():
                    row = [group, sub_id]
                    for edge in unique_edges:
                        for task_name in TASK_NAMES:
                            weight = sub_data.get(kinematics, {}).get(task_name, {}).get(direction, {}).get(edge, np.nan)
                            row.append(np.round(weight, 2) if not np.isnan(weight) else "nan")
                    writer.writerow(row)



def calculate_group_averages(subgraph_weights):
    # Initialize a nested defaultdict to store the sums and counts
    sums = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(float)))))
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int)))))
    
    # Gather all possible tuples, kinematics, tasks, and directions
    all_tuples = set()
    all_kinematics = set()
    all_tasks = set()
    all_directions = set()
    
    # First pass: collect all unique keys and accumulate sums
    for group in subgraph_weights:
        for participant in subgraph_weights[group]:
            for kinematic in subgraph_weights[group][participant]:
                all_kinematics.add(kinematic)
                for task in subgraph_weights[group][participant][kinematic]:
                    all_tasks.add(task)
                    for direction in subgraph_weights[group][participant][kinematic][task]:
                        all_directions.add(direction)
                        for tuple_key, value in subgraph_weights[group][participant][kinematic][task][direction].items():
                            all_tuples.add(tuple_key)
                            # Convert numpy float to Python float if needed
                            if hasattr(value, 'item'):
                                value = value.item()
                            sums[group][kinematic][task][direction][tuple_key] += value
                            counts[group][kinematic][task][direction][tuple_key] += 1
    
    # Calculate averages
    averages = {}
    for group in sums:
        averages[group] = {}
        for kinematic in sums[group]:
            averages[group][kinematic] = {}
            for task in sums[group][kinematic]:
                averages[group][kinematic][task] = {}
                for direction in sums[group][kinematic][task]:
                    averages[group][kinematic][task][direction] = {}
                    for tuple_key in sums[group][kinematic][task][direction]:
                        if counts[group][kinematic][task][direction][tuple_key] > 0:
                            averages[group][kinematic][task][direction][tuple_key] = (
                                sums[group][kinematic][task][direction][tuple_key] / 
                                counts[group][kinematic][task][direction][tuple_key]
                            )
    
    return averages


def plot_anatomical_subgraphs(group_averages, result_folder, body_path):
    """
    Plot anatomical subgraphs with nodes arranged in a more anatomically correct sequence.
    
    Parameters:
    -----------
    group_averages : dict
        Nested dictionary with structure: group -> kinematic -> task -> direction -> tuple -> weight
    output_folder : str
        Folder to save the plots to
    """

    output_folder = Path(result_folder, 'subgraphs'
    '')
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Iterate through all combinations
    for group in group_averages:
        for kinematic in group_averages[group]:
            for task in group_averages[group][kinematic]:
                for direction in group_averages[group][kinematic][task]:
                    # Get the subgraph weights
                    weights = group_averages[group][kinematic][task][direction]
                    
                    # Create a graph
                    G = nx.Graph()
                    
                    # Add nodes and edges with weights
                    for (node1, node2), weight in weights.items():
                        if node1 not in G:
                            G.add_node(node1)
                        if node2 not in G:
                            G.add_node(node2)
                        G.add_edge(node1, node2, weight=weight)
                    
                    # Determine node order (this is a crucial step)
                    # Either derive the order from the edges or define a custom order
                    node_order = determine_node_order(weights)
                    
                    # Create plot
                    plt.figure(figsize=(12, 8))
                    
                    # Create custom positions for a linear arrangement
                    pos = create_linear_positions(G, node_order)
                    
                    # Draw nodes
                    nx.draw_networkx_nodes(G, pos, node_size=1950, node_color='royalblue',  alpha=1)
                    
                    # Draw edges with weights as labels
                    edge_weights = {(u, v): f"{d['weight']:.2f}" for u, v, d in G.edges(data=True)}
                    nx.draw_networkx_edges(G, pos, width=2, alpha=0.7, edge_color='gray')
                    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights, font_color='darkorange', font_size=16)
                    
                    # Draw node labels
                    nx.draw_networkx_labels(G, pos, font_size=12, font_weight="bold")
                    
                    # Add title
                    plt.title(f"{group} - {kinematic} - {task} - {direction}", fontsize=14)
                    
                    # Remove axis
                    plt.axis('off')
                    
                    # Save plot
                    filename = f"{group}_{kinematic}_{task}_{direction}_{body_path}.png"
                    plt.savefig(os.path.join(output_folder, filename), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"Saved plot: {filename}")

def determine_node_order(weights):
    """
    Determine the order of nodes based on their connections.
    
    Parameters:
    -----------
    weights : dict
        Dictionary with tuple keys (node1, node2) and weight values
    
    Returns:
    --------
    list
        Ordered list of nodes
    """
    # Create a temporary graph to help determine the order
    temp_graph = nx.Graph()
    
    for (node1, node2), _ in weights.items():
        temp_graph.add_edge(node1, node2)
    
    # Find all possible paths and select the longest one
    # This assumes the graph is a simple path or close to it
    all_paths = []
    for source in temp_graph.nodes():
        for target in temp_graph.nodes():
            if source != target:
                try:
                    paths = list(nx.all_simple_paths(temp_graph, source, target))
                    all_paths.extend(paths)
                except nx.NetworkXNoPath:
                    pass
    
    # Sort paths by length and take the longest
    if all_paths:
        longest_path = sorted(all_paths, key=len, reverse=True)[0]
        return longest_path
    
    # Fallback if we couldn't find a good path
    return list(temp_graph.nodes())

def create_linear_positions(G, node_order):
    """
    Create positions for nodes in a linear arrangement based on the specified order.
    
    Parameters:
    -----------
    G : networkx.Graph
        Graph with nodes
    node_order : list
        Ordered list of nodes
    
    Returns:
    --------
    dict
        Dictionary mapping nodes to positions
    """
    pos = {}
    
    # Calculate positions along a diagonal line
    n = len(node_order)
    for i, node in enumerate(node_order):
        # Normalize position from 0 to 1
        t = i / max(1, n - 1)
        # Create a diagonal line from top-left to bottom-right
        pos[node] = (t, 1 - t)
    
    # For any nodes not in the order, place them randomly
    for node in G.nodes():
        if node not in pos:
            pos[node] = (np.random.random(), np.random.random())
    
    return pos

def main() -> None:
   
    subgraph_weights = calc_subgraph_weights(DIAGNOSIS, KINEMATICS, TASK_NAMES, TRACKING_SYSTEMS, RUN, DATA_PATH, FULL, CORRELATION, MARKER_LIST, MARKER_SUB_LIST)

    # do only once (saves as a csv file)
    for direction in ['AP', 'ML', 'V']:
        save_weights_to_csv(subgraph_weights, direction, body_path = 'left_arm') # body path depends on the marker_sub_list defined above 

    group_averages = calculate_group_averages(subgraph_weights)

    plot_anatomical_subgraphs(group_averages, RESULT_PATH, body_path = 'left_arm')


if __name__ == "__main__":
    main()
