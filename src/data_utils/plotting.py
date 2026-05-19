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
from src.data_utils.permutation import expand_marker_list
from matplotlib.patches import Rectangle
from tqdm import tqdm
import matplotlib.gridspec as gridspec
from src import kinectome_characteristics


def plot_avg_matrices(avg_group1, avg_group2, group1, group2, marker_list, task, direction, matrix_type, result_base_path, rho, p_value, suptitle, figname):
    
    """Plots the average or std of the kinectomes based on task and direction"""
    
    # Define marker ordering
    ordered_marker_list = marker_list
    index_map = {marker: i for i, marker in enumerate(marker_list)}
    new_order = [index_map[m] for m in ordered_marker_list]
    
    # Reorder matrices
    reordered_group1 = avg_group1[np.ix_(new_order, new_order)]
    reordered_group2 = avg_group2[np.ix_(new_order, new_order)]
    
    # Create triangular masks (keep only lower triangle)
    mask = np.triu(np.ones_like(reordered_group1, dtype=bool), k=1)
    
    # Find global min/max for consistent color scaling
    global_min = min(np.min(reordered_group1), np.min(reordered_group2))
    global_max = max(np.max(reordered_group1), np.max(reordered_group2))
    
    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)
    
    # Create axes for the two heatmaps and colorbar
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])
    
    # Plot Control group (left)
    im1 = sns.heatmap(reordered_group2, 
                      mask=mask,
                      cmap='coolwarm', 
                      vmin=global_min, 
                      vmax=global_max,
                      center=0,
                      xticklabels=ordered_marker_list, 
                      yticklabels=ordered_marker_list, 
                      ax=ax1,
                      cbar=False)
    ax1.set_title(f'{"Controls" if group2 else ""}')
    ax1.set_xticklabels(ordered_marker_list, rotation=90)
    ax1.set_yticklabels(ordered_marker_list, rotation=0)
    
    # Plot Parkinson's group (right)
    im2 = sns.heatmap(reordered_group1, 
                      mask=mask,
                      cmap='coolwarm', 
                      vmin=global_min, 
                      vmax=global_max,
                      center=0,
                      xticklabels=ordered_marker_list, 
                      yticklabels=False,  # Remove y-labels from right plot
                      ax=ax2,
                      cbar=False)
    ax2.set_title(f'{"PD" if group1 else ""}')
    ax2.set_xticklabels(ordered_marker_list, rotation=90)
    
    # Add single colorbar using the actual heatmap data
    cbar = fig.colorbar(im1.get_children()[0], cax=cbar_ax)
    
    # Add suptitle with correlation info and more top space
    plt.suptitle(f"{suptitle}\nSpearman's rho = {rho:.3f}, p_value = {p_value:.3f}", y=1.05)
    
    # Create result folder and save
    result_folder = Path(result_base_path) / "avg_std_matrices"
    result_folder.mkdir(parents=True, exist_ok=True)
    
    save_path = result_folder / figname
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

def visualise_allegiance_matrix_with_communities(allegiance_matrix, communities, marker_list, group, task_name, kinematic, direction, result_base_path, correlation_method, full):
    """
    Plot allegiance matrix with community-based reordering and visual community separation.
   
    Parameters:
    -----------
    allegiance_matrix : numpy.ndarray
        The allegiance matrix to visualize
    communities : list of sets
        List of communities, each containing node indices
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
    
    # Expand marker list if needed for full matrix
    current_marker_list = marker_list.copy()
    if allegiance_matrix.shape != (len(marker_list), len(marker_list)):
        current_marker_list = expand_marker_list(marker_list)
    
    # Create community-based ordering
    community_order = []
    community_boundaries = [0]  # Track boundaries for visual separation
    
    for community in communities:
        community_nodes = sorted(list(community))  # Sort within community for consistency
        community_order.extend(community_nodes)
        community_boundaries.append(len(community_order))
    
    # Reorder matrix and labels
    reordered_matrix = allegiance_matrix[np.ix_(community_order, community_order)]
    reordered_labels = [current_marker_list[i] for i in community_order]
    
    # Create the plot
    plt.figure(figsize=(15, 12))
    
    # Create heatmap
    ax = sns.heatmap(reordered_matrix, 
                     cmap="viridis", 
                     xticklabels=reordered_labels, 
                     yticklabels=reordered_labels,
                     cbar_kws={'label': 'Allegiance Probability'})
    
    # # Add community separation lines
    # for boundary in community_boundaries[1:-1]:  # Skip first (0) and last boundary
    #     ax.axhline(y=boundary, color='red', linewidth=2)
    #     ax.axvline(x=boundary, color='red', linewidth=2)

    # # Add boxes around each community
    for i in range(len(community_boundaries) - 1):
        # Get the start and end index for the current community
        start_idx = community_boundaries[i]
        end_idx = community_boundaries[i+1]

        # Calculate the size of the square
        size = end_idx - start_idx

        # Create a Rectangle patch
        rect = Rectangle(
            (start_idx, start_idx),  # (x,y) bottom-left corner
            size,                    # width
            size,                    # height
            linewidth=2,
            edgecolor='red',
            facecolor='none'         # Make the rectangle transparent
        )

        # Add the rectangle to the plot
        ax.add_patch(rect)
    
    # Add community labels
    for i, community in enumerate(communities):
        start_idx = community_boundaries[i]
        end_idx = community_boundaries[i + 1]
        mid_point = (start_idx + end_idx) / 2
        
        # Add text annotation for community
        ax.text(mid_point, -1, f'Community{i+1}', ha='center', va='top', fontweight='bold', fontsize=12)
        # ax.text(-1, mid_point, f'Community{i+1}', ha='right', va='center', fontweight='bold', fontsize=12, rotation=90)
    
    plt.title(f"Allegiance Matrix - {group} group during {task_name} in {direction} direction ({kinematic} data)\n"
              f"Communities: {len(communities)} detected", fontsize=16, y=1.05)
    
    # Rotate x-axis labels for better readability
    plt.xticks(fontsize = 17, rotation=45, ha='right')
    plt.yticks(fontsize = 17, rotation=0)
    
    # Define result path
    result_folder = Path(result_base_path) / "allegiance_matrices_with_communities"
    result_folder.mkdir(parents=True, exist_ok=True)
    
    # Define save path for the figure
    save_path = result_folder / f"community_ordered_allegiance_{group}_{task_name}_{kinematic}_{correlation_method}{'_full' if full else direction}.png"
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()  # Close figure to free memory
    
    print(f"Saved community-ordered allegiance matrix: {save_path}")

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
    sns.heatmap(np.round(lag_matrix, 1), annot=True, cmap='coolwarm', 
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

    plt.savefig(save_path, dpi=300, bbox_inches='tight')  

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

    plt.savefig(save_path, dpi=600)

def visualise_kinectome(kinectome_data, figname, marker_list, sub_id, task_name, kinematics, result_base_path, what_to_plot):
    """
    Plots the average (or std) kinectomes in AP, ML, and V directions with marker names as labels.
    input:
    kinectome contains the avg adn std kinectome in AP, ML, and V directions nested as a dict
    """
    kinectome_matrices = [
        kinectome_data['AP'][what_to_plot],
        kinectome_data['ML'][what_to_plot],
        kinectome_data['V'][what_to_plot]
    ]
    # Calculate global scales across all matrices and extend to theoretical bounds
    all_values = np.concatenate([matrix.flatten() for matrix in kinectome_matrices])
    data_vmin = all_values.min()
    data_vmax = all_values.max()
   
    # Extend to theoretical bounds (e.g., -1 to 1 for correlations, 0 to 1 for other methods)
    if data_vmin < 0:
        global_vmin = -1  # For correlation methods that can go negative
        global_vmax = 1
    else:
        global_vmin = 0   # For methods that are only positive
        global_vmax = 1
   
    scales = [(global_vmin, global_vmax), (global_vmin, global_vmax), (global_vmin, global_vmax)]
    
    # Create figure with gridspec for better control
    fig = plt.figure(figsize=(18, 6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.3)
    
    # Create axes for the three heatmaps
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    
    # Create the heatmaps
    for i, matrix in enumerate(kinectome_matrices):  # Iterate over 3 matrices
        ax = axes[i]
        
        sns.heatmap(matrix, cmap="coolwarm", vmin=scales[i][0], vmax=scales[i][1], square=True,
                    cbar=False, ax=ax,
                    xticklabels=marker_list, yticklabels=marker_list)  # Add labels
        ax.set_title(f"{['Anteroposterior direction', 'Mediolateral direction', 'Vertical direction'][i]}")
    
    # Add a single colorbar for all three plots
    cbar_ax = fig.add_subplot(gs[0, 3])
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
       
    plt.suptitle(f"{kinematics.upper()} kinectomes of {sub_id} during {task_name}", fontsize='xx-large')
    
    result_folder = Path(result_base_path) / "kinectomes"
    # Create the folder if it does not exist
    result_folder.mkdir(parents=True, exist_ok=True)
    # Define the save path for the figure
    save_path = result_folder / f'{figname}'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close()  # Close the figure to free memory

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

def event_plot_cc(DATA_PATH,sub_id,task_name,tracksys,run,kinematics,MARKER_LIST,threshold_list,direction, full, correlation_method):
    from src.data_utils import data_loader
    from src.graph_utils.kinectome2graph import build_graph, clustering_coef

    if direction == 0:
        idx = "AP"
    elif direction == 1:
        idx = "ML"
    elif direction == 2:
        idx = "V"
    elif direction == 'full':
        idx = 'full'
    # load the kinectomes
    kinectomes = data_loader.load_kinectomes(DATA_PATH, sub_id, task_name,tracksys,run,kinematics, full, correlation_method)
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
                G = graphs[direction] if isinstance(direction, int) else graphs[0] # full kinectome has only one element in graphs
                # calculate the clustering coef 
                cc_dict = clustering_coef(G)
                directions_dict[idx].append(cc_dict)
            
            merged_ = data_loader.merge_dicts(directions_dict[idx])
            for i, k in enumerate(merged_.keys()):
                Y = np.array(merged_[k])
                X = np.arange(len(Y))
                ax.plot(X,Y+i,color="k",zorder=100-i)
                # color = cmap(i / 22)
                color=cmap(cc_dict[k]) # colour coding the plot based on connectivity
                ax.fill_between(X,Y + i, i, color=color, zorder=100 - i)
            
            if n == 0:
                ax.yaxis.set_tick_params(labelleft=True)
                ax.set_yticks(np.arange(len(merged_.keys())) + 0.5)
                ax.set_yticklabels([f"{k}" for k in merged_.keys()],verticalalignment="center")
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

        fig.savefig(f"{save_path}/{sub_id}_run-{run}_{kinematics}_{idx}_{task_name}-curve_event.png")

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

        fig.savefig(f"{save_path}/{sub_id}_run{run}_{kinematics}_{idx}_{task_name}-connected-components.png")

def plot_region_difference_matrix(region_diff_matrix, region_names, task, kin, direction, 
                                 group1, group2, result_base_path, figname):
    """
    Plots the difference matrix between anatomical regions.
    
    Parameters:
    -----------
    region_diff_matrix : numpy.ndarray
        Matrix of average differences between regions
    region_names : list
        List of region names
    task : str
        Name of the task (walking speed)
    kin : str
        Kinematic parameter
    direction : str
        Direction (AP, ML, V)
    group1 : str
        Name of first group (typically "Parkinson")
    group2 : str
        Name of second group (typically "Control")
    result_base_path : str
        Path to save the figure
    figname : str
        Filename for the figure
    
    Returns:
    --------
    None (creates and saves the plot)
    """
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Determine the maximum absolute value for symmetrical color scaling
    max_abs_val = np.max(np.abs(region_diff_matrix))
    vmin, vmax = -max_abs_val, max_abs_val
    
    # Create heatmap
    cax = ax.matshow(region_diff_matrix, cmap='coolwarm', vmin=vmin, vmax=vmax)
    
    # Add colorbar
    cbar = fig.colorbar(cax, label=f'{group1} - {group2} (Correlation Difference)')
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(region_names)))
    ax.set_yticks(np.arange(len(region_names)))
    ax.set_xticklabels(region_names, rotation=45, ha='left')
    ax.set_yticklabels(region_names)
    
    # Add value annotations
    for i in range(len(region_names)):
        for j in range(len(region_names)):
            value = region_diff_matrix[i, j]
            text_color = 'white' if abs(value) > max_abs_val/2 else 'black'
            ax.text(j, i, f'{value:.3f}', ha='center', va='center', color=text_color)
    
    # Add grid lines
    ax.set_xticks(np.arange(-.5, len(region_names), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(region_names), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)
    
    # Set title
    ax.set_title(f'Regional Correlation Differences\n{task} - {kin} - {direction}')
    
    # Set margins and layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join(result_base_path, figname), dpi=300, bbox_inches='tight')
    plt.close()

def plot_difference_distributions(avg_matrices, tasks, kinematics, directions):
    """
    Creates histograms showing the distribution of correlation differences
    for each condition (speed/direction).
    
    Parameters:
    -----------
    avg_matrices : dict
        Dictionary containing the averaged correlation matrices for each group
    tasks : list
        List of tasks (walking speeds)
    kinematics : list
        List of kinematic variables
    directions : list
        List of directions (AP, ML, V)
        
    Returns:
    --------
    None (creates and saves plots)
    """
    groups = list(avg_matrices.keys())
    
    # Create a figure with subplots arranged by tasks and directions
    fig, axes = plt.subplots(len(tasks), len(directions), figsize=(15, 10))
    if len(tasks) == 1 and len(directions) == 1:
        axes = np.array([[axes]])
    elif len(tasks) == 1:
        axes = np.array([axes])
    elif len(directions) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, task in enumerate(tasks):
        for j, direction in enumerate(directions):
            diff_values = []
            for kin in kinematics:
                # Get matrices
                mat_group1 = avg_matrices[groups[0]][task][kin][direction]
                mat_group2 = avg_matrices[groups[1]][task][kin][direction]
                
                # Calculate differences
                diff_mat = mat_group1 - mat_group2
                
                # Extract the upper triangular part (excluding diagonal)
                mask = np.triu_indices_from(diff_mat, k=1)
                diff_values.extend(diff_mat[mask])
            
            # Plot histogram
            axes[i, j].hist(diff_values, bins=20, alpha=0.75)
            axes[i, j].set_title(f"{task} - {direction}")
            axes[i, j].set_xlabel("Correlation Difference (PD - Control)")
            axes[i, j].set_ylabel("Frequency")
            
            # Add vertical line at zero
            axes[i, j].axvline(x=0, color='r', linestyle='--')
            
            # Add mean value
            mean_diff = np.mean(diff_values)
            axes[i, j].axvline(x=mean_diff, color='g', linestyle='-')
            axes[i, j].text(0.05, 0.95, f"Mean: {mean_diff:.3f}", 
                           transform=axes[i, j].transAxes, 
                           verticalalignment='top')
    
    plt.tight_layout()
    return fig

#used in centrality analysis
def plot_community_nodal_strength(df, consensus_communities, results, save_dir="./plots"):
    """
    Create plots showing nodal strength by community, with separate subplots for each segment
    within the community, showing all speeds and groups, including statistical significance.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with centrality data
    consensus_communities : list of sets
        List of sets containing body segments for each community
    results : dict
        Dictionary containing statistical test results
    save_dir : str
        Directory to save plots
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import os
    
    def get_significance_symbol(p_value):
        """Convert p-value to significance symbol"""
        if p_value < 0.001:
            return '***'
        elif p_value < 0.01:
            return '**'
        elif p_value < 0.05:
            return '*'
        else:
            return ''
    
    def add_significance_line(ax, x1, x2, y, symbol, offset=0.5):
        """Add significance line and symbol between two positions"""
        line_height = y + offset
        ax.plot([x1, x2], [line_height, line_height], 'k-', linewidth=1)
        ax.plot([x1, x1], [y, line_height], 'k-', linewidth=1)
        ax.plot([x2, x2], [y, line_height], 'k-', linewidth=1)
        ax.text((x1 + x2) / 2, line_height + 0.3, symbol, ha='center', va='bottom', fontsize=17, fontweight='bold')
        return line_height + 2.5  # Return next available height
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    speeds = ['pref', 'fast', 'slow']
    directions = ['AP', 'ML', 'V']
    groups = df['group'].unique()

    consensus_communities = [
        ['head', 'sternum', 'shoulder_las', 'shoulder_mas', 'asis_las', 'asis_mas', 'psis_las', 'psis_mas'],
        ['elbow_las', 'wrist_las', 'hand_las', 'thigh_mas', 'shank_mas', 'ankle_mas', 'toe_mas'],
        ['elbow_mas', 'wrist_mas', 'hand_mas', 'thigh_las', 'shank_las', 'ankle_las', 'toe_las']
    ] # make the consensus communities into a list so the order for subplots remains as indicated
    
    # Set up colors - different colors for each speed within each group
    # PD: red/yellow tones, Controls: blue/green tones
    speed_colors = {
        'Parkinson': {'pref': '#FF6B6B', 'slow': '#FFE66D', 'fast': '#FF8E53'},  # red/yellow tones
        'Control': {'pref': '#4ECDC4', 'slow': '#45B7D1', 'fast': '#96CEB4'}     # blue/green tones
    }
    
    # Fixed y-axis limits for all plots
    fixed_y_min = 0
    fixed_y_max = 35
    y_range = fixed_y_max - fixed_y_min
    
    for community_idx, community in enumerate(consensus_communities):
        community_segments = community 
        
        for direction in directions:
            # Create figure with subplots for each segment in the community
            n_segments = len(community_segments)
            n_cols = min(3, n_segments)  # Max 3 columns
            n_rows = (n_segments + n_cols - 1) // n_cols  # Calculate needed rows
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            if n_segments == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes
            else:
                axes = axes.flatten()
            
            # fig.suptitle(f'Community {community_idx + 1} - {direction} Direction Nodal Strength', 
            #             fontsize=16, fontweight='bold')
            
            for seg_idx, segment in enumerate(community_segments):
                ax = axes[seg_idx] if n_segments > 1 else axes[0]
                
                # Only show y-label on leftmost subplot of each row
                row = seg_idx // n_cols
                col = seg_idx % n_cols
                if col == 0:
                    ax.set_ylabel('Nodal Strength')
                else:
                    ax.set_ylabel('')
                
                # Prepare data for this segment
                plot_data = []
                
                for group in groups:
                    group_df = df[df['group'] == group]
                    
                    for speed in speeds:
                        col_name = f"{segment}_{speed}_{direction}"
                        
                        if col_name in df.columns:
                            values = group_df[col_name].dropna()
                            
                            for value in values:
                                plot_data.append({
                                    'Group': group,
                                    'Speed': speed,
                                    'Value': value,
                                    'Group_Speed': f"{group}_{speed}"
                                })
                
                if plot_data:
                    plot_df = pd.DataFrame(plot_data)
                    
                    # Create box plot
                    box_positions = []
                    box_data = []
                    box_colors = []
                    labels = []
                    
                    # Create positions with spacing between groups
                    pos = 0
                    group_start_positions = {}
                    
                    for group in groups:
                        group_start_positions[group] = pos
                        for speed in speeds:
                            group_speed_data = plot_df[
                                (plot_df['Group'] == group) & 
                                (plot_df['Speed'] == speed)
                            ]['Value']
                            
                            if len(group_speed_data) > 0:
                                box_data.append(group_speed_data)
                                box_positions.append(pos)
                                box_colors.append(speed_colors.get(group, {}).get(speed, 'gray'))
                                labels.append("")  # Empty labels for x-axis
                            pos += 1.2
                        
                        # Add space between groups
                        pos += 0.3  # Increased spacing between groups
                    
                    if box_data:
                        # Create the box plot
                        bp = ax.boxplot(box_data, positions=box_positions, patch_artist=True,
                                       labels=labels, widths=0.9)        

                        # Color the boxes and add speed labels
                        for i, (patch, color) in enumerate(zip(bp['boxes'], box_colors)):
                            patch.set_facecolor(color)
                            patch.set_alpha(0.8)
                            # Add speed label in the center of the box (vertically)
                            speed_for_this_box = speeds[i % len(speeds)]
                            # Get the box's vertical center by averaging the 25th and 75th percentiles
                            box_path = patch.get_path()
                            box_vertices = box_path.vertices
                            box_center_y = (max(box_vertices[:, 1]) + min(box_vertices[:, 1])) / 2
                            ax.text(box_positions[i], box_center_y, 
                                speed_for_this_box, ha='center', va='center',
                                fontweight='bold', fontsize=15, color='black')
                            
                        # Get y positions for significance bars
                        # Start from a fixed position relative to the max visible data
                        max_data_y = max([max(data) for data in box_data])
                        current_sig_height = max(max_data_y + 2, fixed_y_max * 0.7)  # Start at 70% of y_max or above data
                        
                        # Add between-group significance (above boxes)
                        if (community_idx in results.get('between_groups_by_community', {}) and
                            segment in results['between_groups_by_community'][community_idx]):
                            
                            segment_results = results['between_groups_by_community'][community_idx][segment]
                            
                            for speed in speeds:
                                if (speed in segment_results and 
                                    direction in segment_results[speed] and
                                    segment_results[speed][direction].get('significant_fdr_bh', False)):
                                    
                                    p_val = segment_results[speed][direction]['p_corrected_fdr_bh']
                                    symbol = get_significance_symbol(p_val)
                                    
                                    if symbol:
                                        # Find positions for this speed in both groups
                                        speed_positions = []
                                        speed_idx = speeds.index(speed)
                                        
                                        for group_idx, group in enumerate(groups):
                                            # Calculate position: group start + speed index
                                            pos_in_boxes = group_idx * len(speeds) + speed_idx
                                            if pos_in_boxes < len(box_positions):
                                                speed_positions.append(box_positions[pos_in_boxes])
                                        
                                        if len(speed_positions) >= 2:
                                            # Get the max value for this speed across groups for positioning
                                            speed_data_max = max([max(box_data[group_idx * len(speeds) + speed_idx]) 
                                                                for group_idx in range(len(groups)) 
                                                                if group_idx * len(speeds) + speed_idx < len(box_data)])
                                            sig_y_start = speed_data_max + 1.5
                                            current_sig_height = add_significance_line(
                                                ax, speed_positions[0], speed_positions[1], 
                                                sig_y_start, symbol, offset=2
                                            )
                        
                        # Add within-group significance (below boxes)
                        if (community_idx in results.get('within_groups_by_community', {})):
                            within_community = results['within_groups_by_community'][community_idx]
                            
                            for group_idx, group in enumerate(groups):
                                if (group in within_community and
                                    segment in within_community[group] and
                                    direction in within_community[group][segment]):
                                    
                                    within_result = within_community[group][segment][direction]
                                    
                                    # Check if there's an overall significant effect
                                    if within_result.get('significant_fdr_bh', False):
                                        # Get positions for this group's speeds
                                        group_positions_for_sig = []
                                        speed_to_position = {}
                                        
                                        for speed_idx, speed in enumerate(speeds):
                                            box_idx = group_idx * len(speeds) + speed_idx
                                            if box_idx < len(box_positions):
                                                pos = box_positions[box_idx]
                                                group_positions_for_sig.append(pos)
                                                speed_to_position[speed] = pos
                                        
                                        # Get minimum data value for this group
                                        group_box_indices = [group_idx * len(speeds) + i for i in range(len(speeds)) 
                                                           if group_idx * len(speeds) + i < len(box_data)]
                                        if group_box_indices:
                                            min_group_data = min([min(box_data[i]) for i in group_box_indices])
                                            # Start below the minimum data
                                            current_within_height = min_group_data - 1.5
                                        else:
                                            current_within_height = fixed_y_min + 2
                                        
                                        # Check if pairwise results are available
                                        if 'posthoc_results' in within_result:
                                            # Draw individual pairwise comparison lines
                                            for posthoc in within_result['posthoc_results']:
                                                if posthoc.get('significant', False):
                                                    pair_p_val = posthoc['p_corrected']
                                                    pair_symbol = get_significance_symbol(pair_p_val)
                                                    
                                                    if pair_symbol:
                                                        speed1 = posthoc['level1']
                                                        speed2 = posthoc['level2']
                                                        
                                                        if speed1 in speed_to_position and speed2 in speed_to_position:
                                                            pos1 = speed_to_position[speed1]
                                                            pos2 = speed_to_position[speed2]
                                                            
                                                            # Get the data values for positioning
                                                            box_idx1 = group_idx * len(speeds) + speeds.index(speed1)
                                                            box_idx2 = group_idx * len(speeds) + speeds.index(speed2)
                                                            data_min = min(min(box_data[box_idx1]), min(box_data[box_idx2]))
                                                            
                                                            # Draw bracket going upward
                                                            line_y = current_within_height
                                                            ax.plot([pos1, pos2], [line_y, line_y], 'k-', linewidth=1)
                                                            ax.plot([pos1, pos1], [line_y, line_y + 0.5], 'k-', linewidth=1)
                                                            ax.plot([pos2, pos2], [line_y, line_y + 0.5], 'k-', linewidth=1)
                                                            ax.text((pos1 + pos2) / 2, line_y - 0.6, 
                                                                   pair_symbol, ha='center', va='top', fontsize=15, fontweight='bold')
                                                            
                                                            current_within_height = line_y - 2.5  # Stack multiple comparisons with more space
                                        
                                        else:
                                            # Fallback: if no pairwise results, draw overall significance line
                                            if len(group_positions_for_sig) >= 2:
                                                p_val = within_result['p_corrected_fdr_bh']
                                                symbol = get_significance_symbol(p_val)
                                                
                                                line_start = min(group_positions_for_sig)
                                                line_end = max(group_positions_for_sig)
                                                
                                                ax.plot([line_start, line_end], 
                                                       [current_within_height, current_within_height], 
                                                       'k-', linewidth=1)
                                                ax.plot([line_start, line_start], 
                                                       [current_within_height, current_within_height + 0.5], 
                                                       'k-', linewidth=1)
                                                ax.plot([line_end, line_end], 
                                                       [current_within_height, current_within_height + 0.5], 
                                                       'k-', linewidth=1)
                                                ax.text((line_start + line_end) / 2, current_within_height - 0.5, 
                                                       symbol, ha='center', va='top', fontsize=15, fontweight='bold')
                        
                        # Set fixed y-axis limits for all subplots
                        ax.set_ylim(fixed_y_min, fixed_y_max)
                        
                        # Calculate center position for each group based on actual box positions
                        group1_positions = [box_positions[i] for i in range(len(speeds))]
                        group2_positions = [box_positions[i] for i in range(len(speeds), len(speeds)*2)]

                        if group1_positions:
                            group1_center = np.mean(group1_positions)
                            ax.text(group1_center, -1.5, 'Parkinson', ha='center', va='top', 
                                    transform=ax.transData, fontweight='bold', fontsize=12)
                        if group2_positions:
                            group2_center = np.mean(group2_positions)
                            ax.text(group2_center, -1.5, 'Control', ha='center', va='top', 
                                    transform=ax.transData, fontweight='bold', fontsize=12)
                                       
                ax.set_title(f'{segment}', fontsize = 15, fontweight='bold')
                ax.grid(True, alpha=0.2, axis='y')  # Only horizontal grid lines
                ax.tick_params(axis='x', which='both', length=4, width=1, direction='out')
                ax.set_xticks(box_positions, labels=[])  # Show ticks at box positions but no labels
            
            # Hide empty subplots
            for seg_idx in range(len(community_segments), len(axes)):
                axes[seg_idx].set_visible(False)
            
            plt.tight_layout()
            
            # Save the plot
            filename = f"community_{community_idx + 1}_{direction}_nodal_strength.png"
            filepath = os.path.join(save_dir, filename)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Saved plot: {filepath}")

def plot_permutation_histogram(rhos, true_rho, perm_p, results_path, task, kinematic, direction, matrix_type, correlation_method):
    f,ax = plt.subplots()
    plt.hist(rhos,bins=20)
    ax.axvline(true_rho,  color = 'r', linestyle='--')
    ax.set(title=f"Permuted matrix difference p: {perm_p:.3f}", ylabel="counts", xlabel="rho")

    if matrix_type == 'allegiance' or matrix_type == 'allegiance_std':
        os.chdir(Path(results_path, "allegiance_matrices"))
    else:
        os.chdir(Path(results_path, "avg_std_matrices"))

    plt.savefig(f'permutation_{task}_{kinematic}_{direction}_{matrix_type}_{correlation_method}.png', dpi=600)

def create_bootstrap_plots(bootstrap_results, observed_rhos, task_names, kinematics_list, matrix_type):
    """
    Create 3x3 subplot showing bootstrap distributions with observed values marked.
    """
    
    # Create output directory
    output_dir = r"C:\Users\Karolina\Desktop\pykinectome\results\avg_std_matrices\bootstrapping"
    os.makedirs(output_dir, exist_ok=True)
    
    directions = ['AP', 'ML', 'V']
    
    for kinematic in kinematics_list:
        fig, axes = plt.subplots(3, 3, figsize=(15, 13))
        # fig.suptitle(f'Bootstrap Distribution of Correlations - {kinematic} ({matrix_type})', fontsize=16)
        fig.suptitle(f'Bootstrap Distribution of Correlations - Average Kinectomes', fontsize=16)

        x_min, x_max = 0.8, 1.0  # Adjust these values based on your data range

        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes[i, j]
                
                # Get bootstrap results and observed value
                bootstrap_rhos = bootstrap_results[task][kinematic][direction]
                observed_rho = observed_rhos[task][kinematic][direction]
                
                if len(bootstrap_rhos) > 0 and observed_rho is not None:
                    # Plot histogram of bootstrap correlations
                    ax.hist(bootstrap_rhos, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    
                    # Mark observed correlation with red line
                    ax.axvline(observed_rho, color='red', linestyle='--', linewidth=2, 
                              label=f'Observed ρ = {observed_rho:.3f}')
                    
                    # Add statistics
                    bootstrap_mean = np.mean(bootstrap_rhos)
                    bootstrap_std = np.std(bootstrap_rhos)
                    ax.axvline(bootstrap_mean, color='orange', linestyle=':', linewidth=2,
                              label=f'Bootstrap μ = {bootstrap_mean:.3f}')
                    
                    ax.legend(fontsize=8)
                    ax.set_xlabel('Correlation (ρ)', fontsize=10)
                    ax.set_ylabel('Frequency', fontsize=10)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=11)
                ax.grid(True, alpha=0.3)
                ax.set_xlim(x_min, x_max)
        
        plt.tight_layout()
        
        # Save plot
        filename = f'bootstrap_correlations_{kinematic}_{matrix_type}.png'
        filepath = os.path.join(output_dir, filename)
        plt.savefig(filepath, dpi=600, bbox_inches='tight')
        plt.show()
        
        print(f"Saved bootstrap plot: {filepath}")

def create_sample_size_plots(sample_size_results, observed_rhos, task_names, kinematics_list, 
                           matrix_type, subset_fractions, group1, group2):
    """
    Create comprehensive plots showing how correlation stability changes with sample size.
    """
    
    # Create output directory
    output_dir = r"C:\Users\Karolina\Desktop\pykinectome\results\avg_std_matrices\bootstrapping\sample_size_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    directions = ['AP', 'ML', 'V']
    
    for kinematic in kinematics_list:
        # Plot 1: Mean and variability vs sample size (3x3 subplots)
        fig1, axes1 = plt.subplots(3, 3, figsize=(18, 12))
        fig1.suptitle(f'Correlation Stability vs Sample Size - {kinematic} ({matrix_type})', fontsize=16)
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes1[i, j]
                
                means = []
                stds = []
                sample_percentages = [int(f * 100) for f in subset_fractions]
                observed_rho = observed_rhos[task][kinematic][direction]
                
                for frac in subset_fractions:
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        means.append(np.mean(bootstrap_rhos))
                        stds.append(np.std(bootstrap_rhos))
                    else:
                        means.append(np.nan)
                        stds.append(np.nan)
                
                # Plot mean with error bars
                means = np.array(means)
                stds = np.array(stds)
                
                # Remove NaN values for plotting
                valid_idx = ~np.isnan(means)
                if np.any(valid_idx):
                    ax.errorbar(np.array(sample_percentages)[valid_idx], means[valid_idx], 
                              yerr=stds[valid_idx], marker='o', linestyle='-', linewidth=2, 
                              markersize=6, capsize=5, label='Bootstrap Mean ± SD')
                    
                    # Add observed value as horizontal line
                    if observed_rho is not None:
                        ax.axhline(observed_rho, color='red', linestyle='--', linewidth=2, 
                                  label=f'Observed ρ = {observed_rho:.3f}')
                    
                    ax.legend(fontsize=9)
                    ax.set_xlabel('Sample Size (%)', fontsize=10)
                    ax.set_ylabel('Correlation (ρ)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim(0, 1.0)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename1 = f'sample_size_stability_{kinematic}_{matrix_type}.png'
        filepath1 = os.path.join(output_dir, filename1)
        plt.savefig(filepath1, dpi=600, bbox_inches='tight')
        plt.show()
        
        # Plot 2: Coefficient of variation (CV) vs sample size
        fig2, axes2 = plt.subplots(3, 3, figsize=(18, 12))
        fig2.suptitle(f'Coefficient of Variation vs Sample Size - {kinematic} ({matrix_type})', fontsize=16)
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes2[i, j]
                
                cvs = []
                sample_percentages = [int(f * 100) for f in subset_fractions]
                
                for frac in subset_fractions:
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        mean_rho = np.mean(bootstrap_rhos)
                        std_rho = np.std(bootstrap_rhos)
                        cv = (std_rho / mean_rho) * 100 if mean_rho != 0 else np.nan
                        cvs.append(cv)
                    else:
                        cvs.append(np.nan)
                
                cvs = np.array(cvs)
                valid_idx = ~np.isnan(cvs)
                
                if np.any(valid_idx):
                    ax.plot(np.array(sample_percentages)[valid_idx], cvs[valid_idx], 
                           marker='o', linestyle='-', linewidth=2, markersize=6, color='purple')
                    ax.set_xlabel('Sample Size (%)', fontsize=10)
                    ax.set_ylabel('Coefficient of Variation (%)', fontsize=10)
                    ax.grid(True, alpha=0.3)
                    
                    # Add horizontal line at 5% CV (commonly used threshold for good stability)
                    ax.axhline(5, color='green', linestyle=':', linewidth=2, alpha=0.7, 
                              label='5% CV threshold')
                    ax.legend(fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename2 = f'coefficient_variation_{kinematic}_{matrix_type}.png'
        filepath2 = os.path.join(output_dir, filename2)
        plt.savefig(filepath2, dpi=600, bbox_inches='tight')
        plt.show()
        
        # Plot 3: Distribution comparison at key sample sizes (10%, 50%, 90%)
        key_fractions = [0.1, 0.5, 0.9]
        fig3, axes3 = plt.subplots(3, 3, figsize=(18, 12))
        fig3.suptitle(f'Bootstrap Distributions at Key Sample Sizes - {kinematic} ({matrix_type})', fontsize=16)
        
        colors = ['lightcoral', 'skyblue', 'lightgreen']
        alphas = [0.6, 0.7, 0.8]
        
        for i, task in enumerate(task_names):
            for j, direction in enumerate(directions):
                ax = axes3[i, j]
                
                observed_rho = observed_rhos[task][kinematic][direction]
                
                for k, frac in enumerate(key_fractions):
                    bootstrap_rhos = sample_size_results[frac][task][kinematic][direction]
                    if len(bootstrap_rhos) > 0:
                        ax.hist(bootstrap_rhos, bins=20, alpha=alphas[k], color=colors[k], 
                               edgecolor='black', linewidth=0.5, 
                               label=f'{int(frac*100)}% sample (μ={np.mean(bootstrap_rhos):.3f})')
                
                # Add observed value
                if observed_rho is not None:
                    ax.axvline(observed_rho, color='red', linestyle='--', linewidth=2, 
                              label=f'Observed ρ = {observed_rho:.3f}')
                
                ax.legend(fontsize=8)
                ax.set_xlabel('Correlation (ρ)', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Set title
                task_label = task.replace('walk', '').replace('Preferred', 'Preferred').replace('Fast', 'Fast').replace('Slow', 'Slow')
                ax.set_title(f'{task_label} - {direction}', fontsize=12)
        
        plt.tight_layout()
        filename3 = f'distribution_comparison_{kinematic}_{matrix_type}.png'
        filepath3 = os.path.join(output_dir, filename3)
        plt.savefig(filepath3, dpi=600, bbox_inches='tight')
        plt.show()
        
        print(f"Saved sample size analysis plots for {kinematic}:")
        print(f"  - Stability plot: {filepath1}")
        print(f"  - CV plot: {filepath2}")
        print(f"  - Distribution comparison: {filepath3}")
        
        # Create summary statistics table
        kinectome_characteristics.create_summary_table(sample_size_results, observed_rhos, task_names, directions, 
                           kinematic, matrix_type, subset_fractions, output_dir)
