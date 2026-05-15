import numpy as np
import pandas as pd
from src.data_utils.data_loader import load_kinectomes
from src.data_utils import groups
import pickle
from pathlib import Path
from src.data_utils.plotting import plot_community_nodal_strength
from src.data_utils.statistics import analyze_centrality_statistics_by_community
from src.graph_utils.graphs import all_graphs_for_subject


#used
def weighted_degree_centrality(G):
    """Calculates weighted degree centrality for each node in the graph."""
    ## TODO: note that this may not be the best parameter. as weight can be negative and even it out. 
    ## TODO: THis is just as in the paper defined. We could think of different meaniful parameters. 
    
    return {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}

#used
def export_centrality_to_csv(group_centrality_data, save_path=None):
    """
    Export centrality data to a single CSV file with all body segments.
    Each column is named as: {segment}_{speed}_{direction}
    e.g., head_pref_AP, head_pref_ML, head_pref_V, head_slow_AP, etc.
    """
    if save_path is None:
        save_path = Path(r"C:\Users\Karolina\Desktop\pykinectome\results\centrality\csv")
   
    # Create directory if it doesn't exist
    save_path.mkdir(parents=True, exist_ok=True)
   
    # Get all unique body segments (markers/nodes) across all groups, subjects, tasks, and directions
    all_segments = set()
    for group in group_centrality_data:
        for sub_id in group_centrality_data[group]:
            for task_name in group_centrality_data[group][sub_id]:
                for direction in group_centrality_data[group][sub_id][task_name]:
                    all_segments.update(group_centrality_data[group][sub_id][task_name][direction].keys())
   
    # Define the task name mappings
    task_mapping = {
        'walkPreferred': 'pref',
        'walkSlow': 'slow',
        'walkFast': 'fast'
    }
    
    # Sort segments for consistent ordering
    all_segments = sorted(all_segments)
   
    rows = []
    
    # Iterate through each group and subject
    for group in group_centrality_data:
        for sub_id in group_centrality_data[group]:
            row = {'group': group, 'subject_id': sub_id}
           
            # For each segment, task, and direction combination
            for segment in all_segments:
                for task_name, task_prefix in task_mapping.items():
                    for direction in ['AP', 'ML', 'V']:
                        col_name = f"{segment}_{task_prefix}_{direction}"
                       
                        # Get the value if it exists, otherwise NaN
                        if (task_name in group_centrality_data[group][sub_id] and
                            direction in group_centrality_data[group][sub_id][task_name] and
                            segment in group_centrality_data[group][sub_id][task_name][direction]):
                           
                            value = group_centrality_data[group][sub_id][task_name][direction][segment]
                            row[col_name] = value
                        else:
                            row[col_name] = np.nan
           
            rows.append(row)
   
    # Create DataFrame
    df = pd.DataFrame(rows)
   
    # Order columns: group, subject_id, then all segment_task_direction combinations
    column_order = ['group', 'subject_id']
    for segment in all_segments:
        for task_prefix in ['pref', 'slow', 'fast']:
            for direction in ['AP', 'ML', 'V']:
                column_order.append(f"{segment}_{task_prefix}_{direction}")
   
    df = df[column_order]
   
    # Save to CSV
    csv_filename = "all_segments_centrality.csv"
    df.to_csv(save_path / csv_filename, index=False)
    print(f"Saved {csv_filename}")
   
    print(f"CSV file saved to: {save_path}")
    return save_path, df

#used
def community_weighted_degree_centrality(G, consensus_communities):
    """
    Calculates weighted degree centrality for each node considering only edges within their community.
    
    Parameters:
    G: NetworkX graph
    consensus_communities: List of sets, where each set contains nodes belonging to the same community
    
    Returns:
    Dictionary with node centrality values based on intra-community connections only
    """
    # Create a mapping from node to its community
    node_to_community = {}
    for i, community in enumerate(consensus_communities):
        for node in community:
            node_to_community[node] = i
    
    centrality = {}
    
    for node in G.nodes():
        if node not in node_to_community:
            # If node is not in any community, its centrality is 0
            centrality[node] = 0
            continue
            
        node_community = node_to_community[node]
        community_weight_sum = 0
        
        # Sum weights only for edges to nodes in the same community
        for _, neighbor, weight in G.edges(node, data='weight'):
            if neighbor in node_to_community and node_to_community[neighbor] == node_community:
                community_weight_sum += weight
                
        centrality[node] = community_weight_sum
    
    return centrality


def centrality_main(diagnosis, kinematics_list, task_names, tracking_systems, runs, pd_on, base_path, marker_list, result_base_path, full, 
                    correlation_method, interpol, consensus_communities, community_centrality = False):
    
    # Modify pickle path to distinguish between regular and community centrality
    centrality_type = 'community' if community_centrality else 'regular'
    centrality_pickle_path = Path(result_base_path, 'centrality', f'centrality_data_{correlation_method}_{centrality_type}.pkl')


    if not centrality_pickle_path.exists():

        #TO DO: put the below in a separate function
        disease_sub_ids, matched_control_sub_ids = groups.define_groups(diagnosis)
        debug_ids = ['pp021', 'pp006']
        
        # Initialize the data structure with subject IDs
        group_centrality_data = {f"{diagnosis[0][10:].capitalize()}": {}, "Control": {}}
        
        # Initialize all subjects in both groups
        for sub_id in disease_sub_ids:
            group_centrality_data[f"{diagnosis[0][10:].capitalize()}"][sub_id] = {}
        
        for sub_id in matched_control_sub_ids:
            group_centrality_data["Control"][sub_id] = {}
        
        for kinematics in kinematics_list:
            for sub_id in disease_sub_ids + matched_control_sub_ids:
                group = f"{diagnosis[0][10:].capitalize()}" if sub_id in disease_sub_ids else "Control"
                
                for tracksys in tracking_systems:
                    for task_name in task_names:
                        # Initialize task structure for this subject
                        if task_name not in group_centrality_data[group][sub_id]:
                            group_centrality_data[group][sub_id][task_name] = {}
                        
                        for run in runs:
                            if sub_id in pd_on:
                                run = 'on'
                            elif sub_id not in disease_sub_ids:
                                run = None
                            else:
                                run = run
                                
                            kinectomes = load_kinectomes(base_path, sub_id, task_name, tracksys, run, kinematics, full, correlation_method, interpol)
                            
                            if kinectomes is None:
                                # Store NaN values for missing data
                                for direction in ['AP', 'ML', 'V']:
                                    if direction not in group_centrality_data[group][sub_id][task_name]:
                                        group_centrality_data[group][sub_id][task_name][direction] = {}
                                    
                                    # You'll need to know what nodes should be here - this assumes marker_list contains the node names
                                    for node in marker_list:
                                        if node not in group_centrality_data[group][sub_id][task_name][direction]:
                                            group_centrality_data[group][sub_id][task_name][direction][node] = np.nan
                                continue
                                
                            graphs = all_graphs_for_subject(kinectomes, marker_list, bound_value = None) # add bound_value if of interest
                            subject_average_weights = {}
                            
                            for direction in ['AP', 'ML', 'V']:
                                direction_graphs = graphs[direction]
                                total_weights = []
                                for current_graph in direction_graphs:
                                    # Choose centrality calculation method based on community_centrality flag
                                    if community_centrality:
                                        weights = community_weighted_degree_centrality(current_graph, consensus_communities)
                                    else:
                                        weights = weighted_degree_centrality(current_graph)
                                    total_weights.append(weights)
                                average_weights = {node: np.mean([weights[node] for weights in total_weights]) for node in total_weights[0]}
                                subject_average_weights[direction] = average_weights
                            
                            # Initialize direction structure for this subject and task
                            for direction in ['AP', 'ML', 'V']:
                                if direction not in group_centrality_data[group][sub_id][task_name]:
                                    group_centrality_data[group][sub_id][task_name][direction] = {}
                                
                                # Store individual subject data for each node
                                for node in subject_average_weights[direction]:
                                    group_centrality_data[group][sub_id][task_name][direction][node] = subject_average_weights[direction][node]

        with open (centrality_pickle_path, 'wb') as centrality_file:
            pickle.dump(group_centrality_data, centrality_file)

    else:
        with open (centrality_pickle_path, 'rb') as centrality_file:
            group_centrality_data = pickle.load(centrality_file)
    
    csv_save_path, centrality_df = export_centrality_to_csv(group_centrality_data)
    
    # Run analysis
    results = analyze_centrality_statistics_by_community(
        centrality_df, consensus_communities, alpha=0.05, 
        correction_methods=['fdr_bh'], n_bootstrap=1000, use_bootstrap=False
    )

    # TO DO: save dir as global variable
    # Create plots
    plot_community_nodal_strength(centrality_df, consensus_communities, results, save_dir="C:/Users/Karolina/Desktop/pykinectome/results/community_plots")

    return 



