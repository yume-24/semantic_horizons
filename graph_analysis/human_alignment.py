import networkx as nx

def compare_with_human_map(auto_graph, human_graph):
    # Compare overlap in nodes and edges
    auto_nodes = set(auto_graph.nodes)
    human_nodes = set(human_graph.nodes)

    node_overlap = len(auto_nodes & human_nodes) / max(len(human_nodes), 1)

    auto_edges = set(auto_graph.edges)
    human_edges = set(human_graph.edges)

    edge_overlap = len(auto_edges & human_edges) / max(len(human_edges), 1)

    return {
        "node_overlap": node_overlap,
        "edge_overlap": edge_overlap
    }
