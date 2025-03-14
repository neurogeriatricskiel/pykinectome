import numpy as np
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform, pdist

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
    norm = plt.Normalize(vmin=global_min, vmax=global_max)

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
    pooled_matrix = np.where(allegiance_matrix >= 0.5, 1, allegiance_matrix).astype(np.float64)

    # Ensure no NaNs (replace with 0)
    pooled_matrix = np.nan_to_num(pooled_matrix, nan=0.0)

    # Convert allegiance matrix to a distance matrix (1 - correlation)
    distance_matrix = 1 - pooled_matrix

    # Convert to condensed form for hierarchical clustering
    condensed_distances = squareform(distance_matrix, checks=False)

    # Perform hierarchical clustering
    linkage = sch.linkage(condensed_distances, method="ward")

    # Get dendrogram order
    dendro_order = sch.leaves_list(linkage)

    # Reorder markers
    ordered_marker_list = [marker_list[i] for i in dendro_order]

    # Reorder allegiance matrix
    reordered_matrix = pooled_matrix[np.ix_(dendro_order, dendro_order)]

    # Visualize
    plt.figure(figsize=(15, 12))
    sns.heatmap(reordered_matrix, cmap="viridis", xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)
    plt.title(f"Allegiance Matrix of {group} group during {task_name} in {direction} direction ({kinematic} data)",
              fontsize=18, y=1.05)

    # Define result path
    result_folder = Path(result_base_path) / "allegiance_matrices"
    result_folder.mkdir(parents=True, exist_ok=True)

    # Define save path for the figure
    save_path = result_folder / f"avg_allegiancematrices_{group}_{task_name}_{kinematic}_{direction}.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
 









    # # Define marker ordering
    # if direction == 'AP': 
    #     ordered_marker_list = [
    #                             'head', 'ster', 'l_asis', 'l_psis', 'r_asis', 'r_psis', 
    #                            'l_sho', 'l_elbl', 'l_wrist', 'l_hand', 'r_th', 'r_sk', 'r_ank', 'r_toe',
    #                            'r_sho', 'r_elbl', 'r_wrist', 'r_hand', 'l_th', 'l_sk', 'l_ank', 'l_toe',
    #                            ]   

    # elif direction == 'ML':
    #     ordered_marker_list = [
    #                             'head', 'ster', 'l_sho', 'r_sho', 'l_elbl', 'r_elbl', 'l_asis', 'r_asis',  'l_psis', 'r_psis', 
    #                             'l_wrist', 'l_hand', 'l_th', 'l_sk', 'l_ank', 'l_toe',
    #                             'r_wrist', 'r_hand', 'r_th', 'r_sk', 'r_ank', 'r_toe'
    #                             ]

    # elif direction == 'V':
    #     ordered_marker_list = ['head', 'ster', 'l_sho', 'r_sho','l_asis', 'l_psis', 'r_asis', 'r_psis', 'l_elbl', 'l_th', 'l_sk', 'l_ank', 'r_elbl', 'r_th', 'r_sk', 'r_ank',
    #                            'l_wrist', 'l_hand', 'l_toe', 'r_wrist', 'r_hand', 'r_toe'
    #                             ]

    
    # # Get new indices based on the ordered list
    # index_map = {marker: i for i, marker in enumerate(marker_list)}
    # new_order = [index_map[m] for m in ordered_marker_list]

    # # Reorder rows and columns
    # reordered_matrix = allegiance_matrix[np.ix_(new_order, new_order)]
    
    # # visualise
    # plt.figure(figsize=(15, 12))
    # sns.heatmap(reordered_matrix, cmap="viridis", xticklabels=ordered_marker_list, yticklabels=ordered_marker_list)  
    # plt.title(f"Allegiance Matrix of {group} group during {task_name} in {direction} direction ({kinematic} data)", 
    #              fontsize=18,
    #              y=1.05)
    
    # # Define result path
    # result_folder = Path(result_base_path) / "allegiance_matrices"

    # # Create the folder if it does not exist
    # result_folder.mkdir(parents=True, exist_ok=True)


    # # Define the save path for the figure
    # save_path = result_folder / f"avg_allegiance_matrices_{group}_{task_name}_{kinematic}_{direction}.png"

    # plt.tight_layout()  # Adjust layout to prevent overlap
    # plt.savefig(save_path, dpi=600)