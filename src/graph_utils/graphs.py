import numpy as np
import networkx as nx


def build_graph(kinectome, marker_list, bound_value=None):
    """Build weighted NetworkX graphs from a kinectome matrix.

    Constructs one graph per movement direction (AP, ML, V) when the kinectome
    is 3-dimensional, or a single graph when it is 2-dimensional (full kinectome
    combining all directions).  Edge weights are shifted so that all values are
    non-negative, preserving the relative magnitude of meaningful negative
    correlations.

    Parameters
    ----------
    kinectome : np.ndarray
        Correlation matrix of shape (n_markers, n_markers) for a full kinectome,
        or (n_markers, n_markers, n_directions) for direction-specific kinectomes.
    marker_list : list[str]
        Ordered list of marker names corresponding to the matrix rows/columns.
    bound_value : float or None, optional
        Minimum edge weight threshold.  Edges below this value are excluded.
        If None (default), all non-NaN edges are included.

    Returns
    -------
    list[nx.Graph]
        A list of weighted undirected graphs, one per direction
        ([AP, ML, V] or [full]).
    """
    directions = ['AP', 'ML', 'V']
    marker_list = (
        [f"{m}_{d}" for m in marker_list for d in directions]
        if kinectome.ndim == 2 else marker_list
    )
    kinectome = np.expand_dims(kinectome, axis=-1) if kinectome.ndim == 2 else kinectome

    graphs = []
    for direction in range(kinectome.shape[-1]):
        G = nx.Graph()
        num_nodes = kinectome.shape[0]
        min_weight = np.min(kinectome[:, :, direction])
        shift = -min_weight if min_weight < 0 else 0

        for i in range(num_nodes):
            G.add_node(marker_list[i])

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                weight = kinectome[i, j, direction] + shift
                if bound_value is None:
                    if not np.isnan(weight):
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)
                else:
                    if not np.isnan(weight) and weight >= bound_value:
                        G.add_edge(marker_list[i], marker_list[j], weight=weight)

        graphs.append(G)

    return graphs


def all_graphs_for_subject(kinectomes, marker_list, bound_value):
    """Build graphs for all gait cycles of a single subject.

    Parameters
    ----------
    kinectomes : list[np.ndarray]
        List of kinectome matrices, one per gait cycle.
    marker_list : list[str]
        Ordered list of marker names.
    bound_value : float or None
        Minimum edge weight threshold passed to :func:`build_graph`.

    Returns
    -------
    dict[str, list[nx.Graph]]
        Dictionary with keys ``'AP'``, ``'ML'``, ``'V'``, each containing a
        list of graphs — one graph per gait cycle.
    """
    all_graphs = {"AP": [], "ML": [], "V": []}

    for kinectome in kinectomes:
        graphs = build_graph(kinectome, marker_list, bound_value)
        keys = list(all_graphs.keys())
        for i, key in enumerate(keys):
            all_graphs[key].append(graphs[i])

    return all_graphs


def weighted_degree_centrality(G):
    """Calculate weighted degree centrality for each node.

    Weighted degree centrality is defined as the sum of edge weights
    connected to a node.

    Parameters
    ----------
    G : nx.Graph
        A weighted undirected graph.

    Returns
    -------
    dict[str, float]
        Mapping of node name to its weighted degree centrality.
    """
    return {
        node: sum(weight for _, _, weight in G.edges(node, data='weight'))
        for node in G.nodes()
    }


def cc_connected_components(G):
    """Return connected components of a graph, sorted by size (largest first).

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.

    Returns
    -------
    list[set]
        Connected components sorted in descending order of size.
    """
    return sorted(nx.connected_components(G), key=len, reverse=True)


def clustering_coef(G):
    """Calculate the clustering coefficient for each node.

    Parameters
    ----------
    G : nx.Graph
        An undirected graph.

    Returns
    -------
    dict[str, float]
        Mapping of node name to its clustering coefficient.
    """
    return nx.clustering(G)


def jaccard_complete_communities(comm1, comm2):
    """Calculate Jaccard similarity between two community node sets.

    Parameters
    ----------
    comm1 : set or list
        Nodes in the first community.
    comm2 : set or list
        Nodes in the second community.

    Returns
    -------
    float
        Jaccard similarity score in [0, 1].  Returns 0 if both sets are empty.
    """
    intersection = len(set(comm1) & set(comm2))
    union = len(set(comm1) | set(comm2))
    return intersection / union if union > 0 else 0
