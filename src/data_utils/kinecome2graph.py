import numpy as np 
import networkx as nx


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


def weighted_degree_centrality(G):
    """Calculates weighted degree centrality for each node in the graph."""
    ## TODO: note that this may not be the best parameter. as weight can be negative and even it out. 
    ## TODO: THis is just as in the paper defined. We could think of different meaniful parameters. 
    return {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}