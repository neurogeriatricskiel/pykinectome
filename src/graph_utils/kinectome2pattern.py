import numpy as np 
import networkx as nx 
from src.graph_utils.kinectome2graph import build_graph
from itertools import chain, tee


__all__ = ['full_subgraph', 'path_subgraph', 'cycle_subgraph']

def full_subgraph(G, marker_sublist:list):
    """Returns  the full subgraph of G that includes nodes in the given list. Ie., the complete subgraph on n nodes, where 
    n is the length of the marker_sublist.
    Parameters:
    ----------
    G : nx.Graph
        The input graph.
    marker_sublist : list
        A list of marker names.
    Returns: 
    -------
    subgraph : nx.Graph
        The subgraph of G that includes the nodes in marker_sublist.       
    
    """
    subgraph = G.__class__()

    subgraph.add_nodes_from((n,G.nodes[n]) for n in marker_sublist)

    subgraph.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if u in marker_sublist and v in marker_sublist])
    print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    print(f"The edge data is: {subgraph.edges(data=True)}.")
    return subgraph


def path_subgraph(G,marker_sublist: list):
    """Returns a the path subgraph of G that corresponds to nodes in the given list, with 
    respect to the listed order. .e., the path  n_0 -> n_1 -> ... -> n_k 
    Parameters:
    ----------
    G : nx.Graph
        The input graph.
    marker_sublist : list
        A list of marker names.
    Returns: 
    -------
    subgraph : nx.Graph
        The subgraph of G that includes the nodes in marker_sublist.       
    
    """

    subgraph = G.__class__()


    subgraph.add_nodes_from((n,G.nodes[n]) for n in marker_sublist)
    pairwise_list = zip(marker_sublist[0::1],marker_sublist[1::1])
    for pair in pairwise_list:
        subgraph.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if u in pair and v in pair])
    print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    print(f"The edge data is: {subgraph.edges(data=True)}.")
    return subgraph

def cycle_subgraph(G,marker_sublist: list):
    """Returns a the cycle subgraph of G that corresponds to nodes in the given list, with 
    respect to the listed order. I.e., the closed cycle n_0 -> n_1 -> ... -> n_k -> n_0.
    Parameters:
    ----------
    G : nx.Graph
        The input graph.
    marker_sublist : list
        A list of marker names.
    Returns: 
    -------
    subgraph : nx.Graph
        The subgraph of G that includes the nodes in marker_sublist.       
    
    """

    subgraph = G.__class__()


    subgraph.add_nodes_from((n,G.nodes[n]) for n in marker_sublist)
    nodes_1,nodes_2 = tee(marker_sublist)
    first_node = next(nodes_2,None)
    pairwise_list = zip(nodes_1,chain(nodes_2,(first_node,)))
    for pair in pairwise_list:
        subgraph.add_edges_from([(u, v, d) for u, v, d in G.edges(data=True) if u in pair and v in pair])
    print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    print(f"The edge data is: {subgraph.edges(data=True)}.")
    return subgraph

def min_pattern_subgraph(G, length: int, start_node):
    """
    Returns a path subgraph of G of the specified length, starting from a given node,
    walking through the graph via edges with the lowest weight and without repeating nodes.

    Parameters
    ----------
    G : nx.Graph
        The input graph.
    length : int
        The number of nodes to include in the path.
    start_node : node
        The node to start the path from.

    Returns
    -------
    subgraph : nx.Graph
        A subgraph of G that forms a path of the given length.
    """

    if start_node not in G:
        raise ValueError(f"Start node {start_node} is not in the graph.")
    print(f"Start node is {start_node}.")
    if length < 2:
        raise ValueError("Length must be at least 2 to form a path.")

    if G.number_of_nodes() < length:
        raise ValueError("Graph does not have enough nodes to form the path of the given length.")

    subgraph = G.__class__()
    visited = [start_node]
    current_node = start_node

    while len(visited) < length:
        # Get unvisited neighbors with weights
        neighbors = [
            (v, G[current_node][v]['weight']) 
            for v in G.neighbors(current_node) 
            if v not in visited and 'weight' in G[current_node][v]
        ]

        if not neighbors:
            raise ValueError(f"No path of the required length found from node {start_node}: dead end at {current_node}.")

        next_node = min(neighbors, key=lambda x: x[1])[0]
        visited.append(next_node)
        current_node = next_node

    # Build subgraph
    subgraph.add_nodes_from((n, G.nodes[n]) for n in visited)
    for u, v in zip(visited, visited[1:]):
        subgraph.add_edge(u, v, **G.get_edge_data(u, v))

    print(f"Subgraph has {subgraph.number_of_nodes()} nodes and {subgraph.number_of_edges()} edges.")
    print(f"The edge data is: {subgraph.edges(data=True)}.")
    return subgraph
