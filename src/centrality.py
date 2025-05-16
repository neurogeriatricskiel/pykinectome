
import numpy as np
from src.modularity import build_graph
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups

def weighted_degree_centrality(G):
    """Calculates weighted degree centrality for each node in the graph."""
    ## TODO: note that this may not be the best parameter. as weight can be negative and even it out. 
    ## TODO: THis is just as in the paper defined. We could think of different meaniful parameters. 
    
    return {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}

def all_graphs_for_subject(kinectomes, marker_list):
    all_graphs = {"AP": [], "ML": [], "V": []}
    
    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list)

        keys = list(all_graphs.keys())
        for i, key in enumerate(keys):
            all_graphs[key].append(graphs[i])
     
    return all_graphs

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def create_comparison_boxplot(group_average_weights, task_name, direction, kinematics):
    # Get the groups (e.g., "Parkinson" and "Control")
    groups = list(group_average_weights.keys())
    
    # Get all unique markers across both groups
    all_markers = set()
    for group in groups:
        all_markers.update(group_average_weights[group][task_name][direction].keys())
    all_markers = sorted(list(all_markers))
    
    # Create figure with adjusted height based on number of markers
    fig, ax = plt.subplots(figsize=(10, len(all_markers)*0.4))
    
    # Calculate positions for the boxplots
    positions = []
    labels = []
    colors = ['#E69F00', '#56B4E9']  # Orange for Parkinson, Blue for Control
    
    # Prepare data for plotting
    for i, marker in enumerate(all_markers):
        for j, group in enumerate(groups):
            # Calculate position for this marker-group combination
            pos = i + j/3  # Offset each group slightly
            positions.append(pos)
            labels.append(f"{marker}" if j == 0 else "")  # Only label once per marker
            
    # Create boxplot data and colors
    boxplot_data = []
    box_colors = []
    
    for i, marker in enumerate(all_markers):
        for j, group in enumerate(groups):
            # Get the value if it exists
            if marker in group_average_weights[group][task_name][direction]:
                value = group_average_weights[group][task_name][direction][marker]
                # For boxplot, we need a list of values, even if just one
                boxplot_data.append([value])
                box_colors.append(colors[j])
            else:
                boxplot_data.append([0])  # Placeholder if no data
                box_colors.append('white')
    
    # Create the boxplot
    bplot = ax.boxplot(boxplot_data, positions=positions, vert=False, 
                      patch_artist=True, widths=0.2, showfliers=False)
    
    # Color the boxes based on group
    for patch, color in zip(bplot['boxes'], box_colors):
        patch.set_facecolor(color)
    
    # Set y-axis ticks and labels
    ax.set_yticks([i + 0.15 for i in range(len(all_markers))])
    ax.set_yticklabels(all_markers)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=group) for i, group in enumerate(groups)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set other plot properties
    ax.set_xlabel("Centrality")
    ax.set_ylabel("Markers")
    ax.set_title(f"{kinematics} {task_name} {direction}: Centrality")
    
    # Set y-axis limits with some padding
    ax.set_ylim(-0.5, len(all_markers) - 0.5)
    
    # Save the figure
    save_path = Path("C:\\Users\\Karolina\\Desktop\\pykinectome\\pykinectome\\src\\preprocessing")
    plt.savefig(save_path / f"{kinematics}_{task_name}_{direction}_centrality_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()


# def centrality_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path):
#     disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)
#     debug_ids = ['pp021', 'pp006']
    
#     # Update the data structure to include task_name
#     group_average_weights = {f"{diagnosis[0][10:].capitalize()}": {}, "Control": {}}
    
#     for kinematics in kinematics_list:
#         for sub_id in disease_sub_ids + matched_control_sub_ids:
#         # for sub_id in debug_ids:    
#             group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            
#             for tracksys in tracking_systems:
#                 for task_name in task_names:
#                     for run in runs:
#                         if sub_id in pd_on:
#                             run = 'on'
#                         elif sub_id not in disease_sub_ids:
#                             run = None
#                         else:
#                             run = run
                            
#                         kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics)
#                         if kinectomes is None:
#                             continue
                            
#                         graphs = all_graphs_for_subject(kinectomes, marker_list)
#                         subject_average_weights = {}
                        
#                         for direction in ['AP', 'ML', 'V']:
#                             direction_graphs = graphs[direction]
#                             total_weights = []
#                             for current_graph in direction_graphs:
#                                 weights = weighted_degree_centrality(current_graph)
#                                 total_weights.append(weights)
#                             average_weights = {node: np.mean([weights[node] for weights in total_weights]) for node in total_weights[0]}
#                             subject_average_weights[direction] = average_weights
                        
#                         # Initialize group structure if needed
#                         if group not in group_average_weights:
#                             group_average_weights[group] = {}
                        
#                         # Initialize task structure if needed
#                         if task_name not in group_average_weights[group]:
#                             group_average_weights[group][task_name] = {}
                        
#                         # Initialize direction structure if needed
#                         for direction in ['AP', 'ML', 'V']:
#                             if direction not in group_average_weights[group][task_name]:
#                                 group_average_weights[group][task_name][direction] = {}
                            
#                             # Add subject data to the appropriate task
#                             for node in subject_average_weights[direction]:
#                                 if node not in group_average_weights[group][task_name][direction]:
#                                     group_average_weights[group][task_name][direction][node] = [subject_average_weights[direction][node]]
#                                 else:
#                                     group_average_weights[group][task_name][direction][node].append(subject_average_weights[direction][node])
    
#     # Calculate averages for each group, task, direction, and node
#     for group in group_average_weights:
#         for task_name in group_average_weights[group]:
#             for direction in group_average_weights[group][task_name]:
#                 for node in group_average_weights[group][task_name][direction]:
#                     group_average_weights[group][task_name][direction][node] = np.round(
#                         np.mean(group_average_weights[group][task_name][direction][node]), 2)
    
    # print(group_average_weights)

    # kinematics = "acc"
    # task_name = "walkPreferred"
    # direction = "AP"

    # create_comparison_boxplot(group_average_weights, task_name, direction, kinematics)
    # return group_average_weights

def create_comparison_boxplot(group_centrality_data, task_name, direction, kinematics, node_order=None):
    # Get the groups (e.g., "Parkinson" and "Control")
    groups = list(group_centrality_data.keys())
    
    # Get all unique markers across both groups
    all_markers = set()
    for group in groups:
        if task_name in group_centrality_data[group] and direction in group_centrality_data[group][task_name]:
            all_markers.update(group_centrality_data[group][task_name][direction].keys())
    
    # If a custom order is provided, use it
    if node_order:
        # Filter the order list to only include markers that exist in the data
        all_markers = [marker for marker in node_order if marker in all_markers]
    else:
        # Otherwise, use alphabetical order
        all_markers = sorted(list(all_markers))
    
    # Create figure with adjusted height based on number of markers
    fig, ax = plt.subplots(figsize=(10, len(all_markers)*0.4))
    
    # Positions for each marker
    positions = np.arange(len(all_markers))
    
    # Colors for each group
    colors = ['#E69F00', '#56B4E9']  # Orange for Parkinson, Blue for Control
    
    # Width of each box
    width = 0.4
    
    # Plot each group
    for i, group in enumerate(groups):
        group_positions = positions + width * (i - 0.5)
        group_data = []
        
        for marker in all_markers:
            if (task_name in group_centrality_data[group] and 
                direction in group_centrality_data[group][task_name] and 
                marker in group_centrality_data[group][task_name][direction]):
                group_data.append(group_centrality_data[group][task_name][direction][marker])
            else:
                group_data.append([])
        
        # Create the boxplot for this group
        bplot = ax.boxplot(group_data, positions=group_positions, 
                          patch_artist=True, widths=width*0.8, 
                          showfliers=False, vert=False)
        
        # Set colors for this group
        for patch in bplot['boxes']:
            patch.set_facecolor(colors[i])
            patch.set_alpha(0.7)
        
        for median in bplot['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
    
    # Set y-axis ticks and labels
    ax.set_yticks(positions)
    ax.set_yticklabels(all_markers)
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=group) for i, group in enumerate(groups)]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Set other plot properties
    ax.set_xlabel("Centrality")
    ax.set_ylabel("Markers")
    ax.set_title(f"{kinematics} {task_name} {direction}: Centrality")
    
    # Set y-axis limits with some padding
    ax.set_ylim(-0.5, len(all_markers) - 0.5)
    
    # Save the figure
    save_path = Path("C:\\Users\\Karolina\\Desktop\\pykinectome\\pykinectome\\src\\preprocessing")
    plt.savefig(save_path / f"{kinematics}_{task_name}_{direction}_centrality_comparison.png", 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    return fig

def centrality_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, correlation_method):
    disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)
    debug_ids = ['pp021', 'pp006']
    
    # Change this to store all subject values instead of just averages
    group_centrality_data = {f"{diagnosis[0][10:].capitalize()}": {}, "Control": {}}
    
    for kinematics in kinematics_list:
        for sub_id in disease_sub_ids + matched_control_sub_ids:
            group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
            
            for tracksys in tracking_systems:
                for task_name in task_names:
                    for run in runs:
                        if sub_id in pd_on:
                            run = 'on'
                        elif sub_id not in disease_sub_ids:
                            run = None
                        else:
                            run = run
                            
                        kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method)
                        if kinectomes is None:
                            continue
                            
                        graphs = all_graphs_for_subject(kinectomes, marker_list)
                        subject_average_weights = {}
                        
                        for direction in ['AP', 'ML', 'V']:
                            direction_graphs = graphs[direction]
                            total_weights = []
                            for current_graph in direction_graphs:
                                weights = weighted_degree_centrality(current_graph)
                                total_weights.append(weights)
                            average_weights = {node: np.mean([weights[node] for weights in total_weights]) for node in total_weights[0]}
                            subject_average_weights[direction] = average_weights
                        
                        # Initialize group structure if needed
                        if group not in group_centrality_data:
                            group_centrality_data[group] = {}
                        
                        # Initialize task structure if needed
                        if task_name not in group_centrality_data[group]:
                            group_centrality_data[group][task_name] = {}
                        
                        # Initialize direction structure if needed
                        for direction in ['AP', 'ML', 'V']:
                            if direction not in group_centrality_data[group][task_name]:
                                group_centrality_data[group][task_name][direction] = {}
                            
                            # Add subject data for each node
                            for node in subject_average_weights[direction]:
                                if node not in group_centrality_data[group][task_name][direction]:
                                    group_centrality_data[group][task_name][direction][node] = []
                                
                                # Store individual subject data point instead of averaging
                                group_centrality_data[group][task_name][direction][node].append(
                                    subject_average_weights[direction][node])
    
    
    custom_node_order = reversed(['head', 'ster', 'l_sho', 'r_sho', 'l_asis', 'l_psis', 'r_asis', 'r_psis', # core nodes
                        'l_elbl','l_wrist','l_hand', # left arm
                        'l_th', 'l_sk','l_ank','l_toe', # left leg
                        'r_elbl','r_wrist','r_hand', # right arm
                        'r_th', 'r_sk', 'r_ank', 'r_toe' # right leg
                        ])
    
    # Create comparison boxplot
    kinematics = "acc"
    task_name = "walkPreferred"
    direction = "AP"
    fig = create_comparison_boxplot(group_centrality_data, task_name, direction, kinematics, node_order=custom_node_order)
    
    return group_centrality_data



