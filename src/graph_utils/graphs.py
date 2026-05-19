import numpy as np 
import networkx as nx


def build_graph(kinectome, marker_list, bound_value=None):

    """Builds weighted graphs for AP, ML, V directions if ndim==2, 
    else builds one graph for the full kinectome (containins all directions)     
    while preserving meaningful negative correlations."""
    
    # np.expand_dims
    # kinectome = kinectome[..., None] if kinectome.ndim == 2 else kinectome
    directions = ['AP', 'ML', 'V']
    marker_list = (
                    [f"{m}_{d}" for m in marker_list for d in directions] if kinectome.ndim == 2 else marker_list
                    )
    kinectome = np.expand_dims(kinectome, axis=-1) if kinectome.ndim == 2 else kinectome
     
    
    graphs = []
    for direction in range(kinectome.shape[-1]):
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
                if bound_value is None:
                    if not np.isnan(weight):
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
                else: # Apply threshold
                    if not np.isnan(weight) and weight >= bound_value:
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
        
        graphs.append(G)
    
    return graphs

#used
def all_graphs_for_subject(kinectomes, marker_list, bound_value):
    all_graphs = {"AP": [], "ML": [], "V": []}
    
    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list, bound_value)

        keys = list(all_graphs.keys())
        for i, key in enumerate(keys):
            all_graphs[key].append(graphs[i])
     
    return all_graphs