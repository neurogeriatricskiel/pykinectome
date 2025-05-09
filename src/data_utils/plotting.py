import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
import networkx as nx
import os
from src.data_utils.permutation import expand_marker_list


def plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value):
    " Plots the average or std of the kinectomes based on task and direction"

    # Define marker ordering
    left_markers = [m for m in marker_list if m.startswith('l_')]
    right_markers = [m for m in marker_list if m.startswith('r_')]
    middle_markers = ['head', 'ster']
    ordered_marker_list = middle_markers + left_markers + right_markers

    # Get new indices based on the ordered list
    index_map = {marker: i for i, marker in enumerate(marker_list)}
    new_order = [index_map[m] for m in ordered_marker_list]

    # Reorder matrices
    reordered_group1 = avg_group1[np.ix_(new_order, new_order)]
    reordered_group2 = avg_group2[np.ix_(new_order, new_order)]

    # Find global min/max for consistent color scaling
    global_min = min(np.min(reordered_group1), np.min(reordered_group2))
    global_max = max(np.max(reordered_group1), np.max(reordered_group2))

    # define the limits of the colour bars
    norm = plt.Normalize(vmin=0, vmax=global_max)

    # Set up figure
    fig, axes = plt.subplots(1, 2, figsize=(12,6))

    # Plot Parkinson's group
    ax1 = axes[0]
    sns.heatmap(reordered_group2, cmap='coolwarm', norm = norm, center=0, xticklabels=ordered_marker_list, yticklabels=ordered_marker_list, ax=ax1)
    ax1.set_title(f'{group2} {matrix_type} matrix\nTask: {task}, Direction: {direction}')
    ax1.set_xticklabels(ordered_marker_list, rotation=90)
    ax1.set_yticklabels(ordered_marker_list, rotation=0)

    # Plot Control group
    ax2 = axes[1]
    sns.heatmap(reordered_group1, cmap='coolwarm', norm = norm, center=0, xticklabels=ordered_marker_list, yticklabels=ordered_marker_list, ax=ax2)
    ax2.set_title(f'{group1} {matrix_type} matrix\nTask: {task}, Direction: {direction}')
    ax2.set_xticklabels(ordered_marker_list, rotation=90)
    ax2.set_yticklabels(ordered_marker_list, rotation=0)

    plt.tight_layout()
    plt.suptitle(f"Spearman's rho = {np.round(rho, 3)}, p_value = {np.round(p_value, 3)}")

    # Define result path
    result_folder = Path(result_base_path) / "avg_std_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"avg_matrices_{task}_{direction}_{matrix_type}.png"

    # Save the figure
    plt.savefig(save_path, dpi=600, bbox_inches='tight')



def visualise_allegiance_matrix(allegiance_matrix, marker_list, group, task_name, kinematic, direction, result_base_path, correlation_method, full):
    """
    Plot allegiance matrix with hierarchical clustering to reorder markers based on correlation values.
    
    Parameters:
    -----------
    allegiance_matrix : numpy.ndarray
        The allegiance matrix to visualize
    marker_list : list
        List of marker names corresponding to the rows/columns of the allegiance matrix
    group : str
        Group name (e.g., 'Parkinson', 'Control')
    task_name : str
        Task name (e.g., 'walkPreferred')
    kinematic : str
        Kinematic type (e.g., 'acc', 'vel')
    direction : str
        Direction (e.g., 'AP', 'ML', 'V')
    result_base_path : str or Path
        Base path for saving results
    """
       
    ma_markers = [m for m in marker_list if m.endswith('_la')]
    la_markers = [m for m in marker_list if m.endswith('_ma')]
    middle_markers = ['head', 'ster']
    ordered_marker_list = middle_markers + ma_markers + la_markers

    # relabel the markers if analysing the full matrix (66x66)
    if allegiance_matrix.shape != (len(marker_list),len(marker_list)):
        marker_list = expand_marker_list(marker_list)
        ordered_marker_list = expand_marker_list(ordered_marker_list)   
 

    # Get new indices based on the ordered list
    index_map = {marker: i for i, marker in enumerate(marker_list)}
    new_order = [index_map[m] for m in ordered_marker_list] 

    # Reorder rows and columns
    reordered_matrix = allegiance_matrix[np.ix_(new_order, new_order)]

    # Visualize
    plt.figure(figsize=(15, 12))
    sns.heatmap(reordered_matrix, cmap="viridis", xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)
    plt.title(f"Allegiance Matrix of {group} group during {task_name} in {direction} direction ({kinematic} data)",
              fontsize=18, y=1.05)

    # Define result path
    result_folder = Path(result_base_path) / "allegiance_matrices"
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define save path for the figure
    save_path = result_folder / f"avg_allegiance_matrices_{group}_{task_name}_{kinematic}_{correlation_method}{'_full' if full else direction}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
 

def plot_correlation_heatmap(corr_matrix, marker_list, title='Maximum Cross-Correlation', result_base_path = 'C:/Users/Karolina/Desktop/pykinectome/results'):
    """Plot a heatmap of the correlation matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
    xticklabels=marker_list, yticklabels=marker_list)
    plt.title(title)
    plt.tight_layout()

    result_folder = Path(result_base_path) / "cross_corr_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"crosscorr_heatmap.png"

    plt.savefig(save_path, dpi = 600)

def plot_lag_heatmap(lag_matrix, markers_list, title='Time Lag at Maximum Correlation', result_base_path = 'C:/Users/Karolina/Desktop/pykinectome/results'):
    """Plot a heatmap of the lag matrix"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(lag_matrix, annot=True, cmap='coolwarm', 
                xticklabels=markers_list, yticklabels=markers_list)
    plt.title(title)
    plt.tight_layout()
        # Define result path
    result_folder = Path(result_base_path) / "cross_corr_matrices"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"lag_heatmap.png"

    plt.savefig(save_path, dpi = 600)

def draw_graph_with_weights(G, result_base_path = 'C:/Users/Karolina/Desktop/pykinectome/results'):
    """Visualizes the graph with edge weights."""
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Graph Representation of Kinectome")
    result_folder = Path(result_base_path) / "graphs"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"graph_all_weights.png"

def draw_graph_with_selected_weights(G, selected_edges=None, result_base_path = 'C:/Users/Karolina/Desktop/pykinectome/results'):
    """
    Visualizes the graph with edge weights for specified edges only.
    
    Parameters:
    G (networkx.Graph): The graph to visualize
    selected_edges (list): List of tuples (node1, node2) for which to display weights.
                          If None, displays all weights.
    """
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    
    # Draw all nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', 
            edge_color='gray', node_size=500, font_size=10)
    
    # If no edges are specified, show all weights
    if selected_edges is None:
        edge_labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges()}
    else:
        # Filter for only the specified edges, ensuring they exist in the graph
        edge_labels = {}
        for node1, node2 in selected_edges:
            # Check if edge exists (in either direction for undirected graphs)
            if G.has_edge(node1, node2):
                edge_labels[(node1, node2)] = f"{G[node1][node2]['weight']:.2f}"
            elif G.has_edge(node2, node1):  # For undirected graphs
                edge_labels[(node2, node1)] = f"{G[node2][node1]['weight']:.2f}"
    
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    plt.title("Graph Representation of Kinectome")

    result_folder = Path(result_base_path) / "graphs"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f"graph_weights.png"

def visualise_kinectome(kinectome, figname, marker_list, sub_id, task_name, kinematics, result_base_path):
    """
    Plots the kinectomes in AP, ML, and V directions with marker names as labels.
    """

    # Split left and right markers
    left_markers = [m for m in marker_list if m.startswith('l_')]
    right_markers = [m for m in marker_list if m.startswith('r_')]
    middle_markers = ['head', 'ster']
    
    # Combine them in the desired order (left-side first, then right-side)
    ordered_marker_list = middle_markers + left_markers + right_markers

    plt.figure(figsize=(15, 5))
    # Reorder the kinectome numpy array accordingly
    marker_indices = [marker_list.index(m) for m in ordered_marker_list]
    reordered_kinectome = kinectome[np.ix_(marker_indices, marker_indices, [0, 1, 2])]
    
    # Define vmin/vmax for each direction
    min_value = np.min(reordered_kinectome)
    
    if min_value < 0:
        scales = [(0.5, 1) if kinematics == 'pos' else (-1, 1), (-1, 1), (-1, 1)]
   
    else:
        scales = [(0.5, 1) if kinematics == 'pos' else (0, 1), (0, 1), (0, 1)]

    for i, matrix in enumerate(reordered_kinectome.transpose(2, 0, 1)):  # Iterate over 3 matrices
        plt.subplot(1, 3, i + 1)  # Create subplot
        sns.heatmap(matrix, cmap="coolwarm", vmin=scales[i][0], vmax=scales[i][1], square=True, cbar=True,
                    xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)  # Add labels
        
        plt.title(f"Correlation Matrix {['Anteroposterior', 'Mediolateral', 'Vertical'][i]}")
    
    plt.suptitle(f"{kinematics.upper()} kinectomes of {sub_id} during {task_name}")

    result_folder = Path(result_base_path) / "kinectomes"

    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define the save path for the figure
    save_path = result_folder / f'{figname}'

    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.savefig(save_path, dpi=600)

def plot_difference_matrix(diff_mtrx_sorted, reordered_markers, task, kin, direction, group1_name, group2_name, result_base_path, figname):
    """ plots the difference matrix sorted according to the highest differences"""
    plt.figure(figsize=(10, 8))
    vmax = np.percentile(np.abs(diff_mtrx_sorted), 98)  # symmetric range around 0
    sns.heatmap(diff_mtrx_sorted,
                xticklabels=reordered_markers,
                yticklabels=reordered_markers,
                cmap="coolwarm",
                center=0,
                square=True,
                cbar_kws={"label": f"{group1_name} - {group2_name} (Correlation Difference)"},
                vmin=-vmax, vmax=vmax)

    plt.title(f'{task} | {kin} | {direction} ')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Define the save path for the figure
    result_folder = Path(result_base_path) / "difference_matrices"
    result_folder.mkdir(parents=True, exist_ok=True)

    save_path = result_folder / f'{figname}'
    plt.savefig(save_path, dpi=600)