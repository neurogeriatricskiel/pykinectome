import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist
import networkx as nx
import os
from tqdm import tqdm

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



def visualise_allegiance_matrix(allegiance_matrix, marker_list, group, task_name, kinematic, direction, result_base_path):
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

    # Pool correlations from 0.5 to 1 together
    # pooled_matrix = allegiance_matrix.astype(np.float64)  # Keep original values

    # # Ensure no NaNs (replace with 0)
    # pooled_matrix = np.nan_to_num(pooled_matrix, nan=0.0)

    # # Convert to a proper distance matrix (avoid negatives)
    # distance_matrix = 1 - pooled_matrix
    # distance_matrix[distance_matrix < 0] = 0
    # np.fill_diagonal(distance_matrix, 0)

    # # Convert to condensed form for hierarchical clustering
    # condensed_distances = squareform(distance_matrix, checks=False)

    # # Perform hierarchical clustering
    # linkage = sch.linkage(condensed_distances, method="ward")

    # # Get dendrogram order
    # dendro_order = sch.leaves_list(linkage)

    # # Reorder markers
    # ordered_marker_list = [marker_list[i] for i in dendro_order]

    # Reorder allegiance matrix
    # reordered_matrix = pooled_matrix[np.ix_(dendro_order, dendro_order)]
    
     # Define marker ordering (based on the results of Lopez et al. 2022)
    if direction == 'AP':
        ordered_marker_list = ['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'r_asis', 'l_psis', 'r_psis', 
                               'l_elbl', 'l_wrist', 'l_hand', 'r_th', 'r_sk', 'r_ank', 'r_toe', 
                               'r_elbl', 'r_wrist', 'r_hand', 'l_th', 'l_sk', 'l_ank', 'l_toe',]
        
    elif direction == 'ML':
        ordered_marker_list = ['ster', 'l_sho', 'r_sho', 
                               'head', 'l_asis', 'r_asis', 'l_psis','r_psis', 'l_elbl', 'l_wrist', 'l_hand', 'r_elbl', 'r_wrist', 'r_hand',
                                'l_th', 'l_sk', 'l_ank', 'l_toe', 'r_th', 'r_sk', 'r_ank', 'r_toe'
        ]
    
    elif direction == 'V':
        ordered_marker_list = ['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'l_psis', 'r_asis', 'r_psis', 'l_th', 'l_sk', 'r_th', 'r_sk',
                               'l_elbl','l_wrist', 'l_hand', 'r_elbl', 'r_wrist', 'r_hand',
                               'l_ank', 'l_toe', 'r_ank', 'r_toe']
        

    # left_markers = [m for m in marker_list if m.startswith('l_')]
    # right_markers = [m for m in marker_list if m.startswith('r_')]
    # middle_markers = ['head', 'ster']
    # ordered_marker_list = middle_markers + left_markers + right_markers   
 

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
    save_path = result_folder / f"avg_allegiance_matrices_{group}_{task_name}_{kinematic}_{direction}.png"
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



def plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8]):
    from src.data_utils import data_loader
    from src.graph_utils.kinectome2graph import build_graph, clustering_coef

    fig, axs = plt.subplots(3,len(threshold_list), figsize=(15, 15))
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    print(f"{kinematics},{task_name}: \n\n Number of events is {len(kinectomes)}")

    for i, limit in enumerate(threshold_list):
        # direction dict, as order of build_graph
        directions_dict = {"AP": [], "ML": [], "V":[]} 
        for k in kinectomes:
            graphs = build_graph(k,MARKER_LIST,limit)
            for idx, direction in enumerate(["AP", "ML", "V"]):
                G = graphs[idx]
                # calculate the clustering coef 
                cc_dict = clustering_coef(G)
                directions_dict[direction].append(cc_dict)

        for j,idx in enumerate(["AP", "ML", "V"]):
            merged_ = data_loader.merge_dicts(directions_dict[idx])
            axs[j,i].boxplot(merged_.values(), vert=False, showfliers=False)
            axs[j,i].set_yticklabels(merged_.keys())
            axs[j,i].set_xlabel("clustering coefficient")
            axs[j,i].set_ylabel("Markers")
            axs[j,i].set_title(f"{idx}_{limit}") 
    for ax in axs.flat:
        ax.label_outer()
    save_path = f"{DATA_PATH}/plots"
    # Ensure directory exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    fig.suptitle(f"{kinematics} {task_name} {sub_id} ") 

    fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{task_name}-cc.png")



def event_plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8],direction=0):
    from src.data_utils import data_loader
    from src.graph_utils.kinectome2graph import build_graph, clustering_coef

    if direction == 0:
        idx = "AP"
    elif direction == 1:
        idx = "ML"
    elif direction == 2:
        idx = "V"
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    if not kinectomes:
        print(f"Warning: No kinectome found for subject {sub_id}, task {task_name} during run-{run}. Skipping...")
    else:
        fig = plt.figure(figsize=(8, 8))
        cmap = mpl.cm.get_cmap("Spectral")
        events_iterator = tqdm(threshold_list, desc=f"---Subject: {kinematics}, Direction: {idx}, Task: {task_name}---")
        ax = None
        for n, limit in enumerate(events_iterator):
            ax = plt.subplot(1,len(threshold_list),n+1, frameon=False, sharex=ax)
            directions_dict = {idx: []}
            for k in kinectomes:
                graphs = build_graph(k,MARKER_LIST,limit)
                G = graphs[direction]
                # calculate the clustering coef 
                cc_dict = clustering_coef(G)
                directions_dict[idx].append(cc_dict)
            
            merged_ = data_loader.merge_dicts(directions_dict[idx])
            for i, k in enumerate(merged_.keys()):
                Y = np.array(merged_[k])
                X = np.arange(len(Y))
                ax.plot(X,Y+i,color="k",zorder=100-i)
                color = cmap(i / 22)
                ax.fill_between(X,Y + i, i, color=color, zorder=100 - i)
            
            if n == 0:
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_yticks(np.arange(len(merged_.keys())))
                ax.set_yticklabels([f"{k}" for k in merged_.keys()],verticalalignment="bottom")
            else:
                ax.yaxis.set_tick_params(labelleft=False)

            ax.text(
            0.0,
            1.0,
            f"Threshold {limit}",
            ha="left",
            va="top",
            weight="bold",
            transform=ax.transAxes,
            )
        plt.tight_layout()
        save_path = f"{DATA_PATH}/plots"
        # Ensure directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{idx}_{task_name}-curve_event.png")


def event_plot_components(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list=[0.2,0.4,0.6,0.8],direction=1):
    from src.data_utils import data_loader
    from src.graph_utils.kinectome2graph import build_graph, clustering_coef, cc_connected_components

    if direction == 0:
        idx = "AP"
    elif direction == 1:
        idx = "ML"
    elif direction == 2:
        idx = "V"
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics)
    if not kinectomes:
        print(f"Warning: No kinectome found for subject {sub_id}, task {task_name} during run-{run}. Skipping...")
    else:
        fig = plt.figure(figsize=(8, 8))
        cmap = mpl.cm.get_cmap("Spectral")
        events_iterator = tqdm(kinectomes, desc=f"---Subject: {kinematics}, Direction: {idx}, Task: {task_name}---")
        ax = None
        for n, k in enumerate(events_iterator):
            ax = plt.subplot(1,len(kinectomes),n+1, frameon=False, sharex=ax)
            directions_dict = {idx: []}
            for j, limit in enumerate(threshold_list):
                graphs = build_graph(k,MARKER_LIST,limit)
                G = graphs[direction]
                # calculate the clustering coef 
                cc = cc_connected_components(G)
                directions_dict[idx].append(len(cc))
            
            # merged_ = data_loader.merge_dicts(directions_dict[idx])
            for i, k in enumerate(directions_dict.keys()):
                Y = np.array(directions_dict[k])
                X = np.array(threshold_list)
                ax.plot(X,Y+i,color="k",zorder=100-i)
                color = cmap(i / 22)
                ax.fill_between(X,Y + i, i, color=color, zorder=100 - i)
            
            if n == 0:
                ax.yaxis.set_tick_params(labelleft=True)
                # ax.set_yticks(np.arange(len(kinectomes)))
                ax.set_yticks(np.arange(10))
                # ax.set_yticklabels([f"Event {n}" for n in range(1,len(kinectomes) + 1 )],verticalalignment="bottom")
            else:
                ax.yaxis.set_tick_params(labelleft=False)

            ax.text(
            0.0,
            1.0,
            f"Event {n}",
            ha="left",
            va="top",
            weight="bold",
            transform=ax.transAxes,
            )
        plt.tight_layout()
        save_path = f"{DATA_PATH}/plots"
        # Ensure directory exists
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        fig.savefig(f"{save_path}/{sub_id}_{kinematics}_{idx}_{task_name}-connected-components.png")