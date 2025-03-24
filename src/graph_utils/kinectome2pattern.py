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