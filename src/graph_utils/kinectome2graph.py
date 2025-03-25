import numpy as np 
import networkx as nx


def build_graph(kinectome, marker_list, bound_value=None,flag_shift = False, full_graph = False):
    """Builds weighted graphs for AP, ML, V directions while preserving meaningful negative correlations.
    Parameters:
    ----------
    kinectome : np.ndarray
        A 3D numpy array containing the kinectome data.
    marker_list : list
        A list of marker names.
    bound_value : float, optional
        The threshold value for edge weights. If None, all edges are included.
    flag_shift : bool, optional
        Flag to indicate whether to shift weights to be non-negative.
    full_graph : bool, optional
        Flag to indicate whether to build a full graph.I.e, A graph with all the directions combined.
    Returns:    
    -------
    graphs : list
        A list of weighted graphs for each direction (AP, ML, V), if full_graph is False.
    G : nx.Graph
        A weighted graph with all directions combined, if full_graph is True.

    """
    if full_graph:
        G = nx.Graph()
        num_nodes = kinectome.shape[0]
        min_weight = np.min(kinectome)
        shift = -min_weight if min_weight < 0 else 0  # Shift weights to be non-negative
        
        # Add nodes with marker labels
        for i in range(num_nodes):
            G.add_node(marker_list[i])  # Assign marker name as node label

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if flag_shift:
                    weight = kinectome[i, j] + shift  # Apply shift
                else:
                    weight = kinectome[i, j]
                if bound_value is None:
                    if not np.isnan(weight):
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
                else: # Apply threshold
                    if not np.isnan(weight) and weight >= bound_value:
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
        
        return G
    else:
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
                    if flag_shift:
                        weight = kinectome[i, j, direction] + shift  # Apply shift
                    else:
                        weight = kinectome[i, j, direction]
                    if bound_value is None:
                        if not np.isnan(weight):
                            G.add_edge(marker_list[i], marker_list[j], weight=weight)
                    else: # Apply threshold
                        if not np.isnan(weight) and weight >= bound_value:
                            G.add_edge(marker_list[i], marker_list[j], weight=weight)
            
            graphs.append(G)
        
        return graphs


def weighted_degree_centrality(G):
    """Calculates weighted degree centrality for each node in the graph.""" 
    return {node: sum(weight for _, _, weight in G.edges(node, data='weight')) for node in G.nodes()}

def cc_connected_components(G):
    """Calculates connected components in the filtered graph sorted by connected components"""
    components= sorted(nx.connected_components(G),key=len,reverse=True)
    return components

def clustering_coef(G):
    """Calculates clustering coefficients for each node in the graph."""
    return nx.clustering(G)