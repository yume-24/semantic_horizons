import networkx as nx
import numpy as np


def compute_basic_graph_stats(graph: nx.Graph) -> dict:
    stats = {
        "Num Nodes": graph.number_of_nodes(),
        "Num Edges": graph.number_of_edges(),
        "Average Degree": float(np.mean([d for _, d in graph.degree()])) if graph.number_of_nodes() > 0 else 0,
        "Density": nx.density(graph),
        "Is Connected": nx.is_connected(graph) if isinstance(graph, nx.Graph) and graph.number_of_nodes() > 0 else False,
        "Num Connected Components": nx.number_connected_components(graph),
        "Average Clustering Coefficient": nx.average_clustering(graph) if graph.number_of_nodes() > 0 else 0,
        "Transitivity": nx.transitivity(graph),
    }
    return stats


def compute_graph_centrality_stats(graph: nx.Graph) -> dict:
    if graph.number_of_nodes() == 0:
        return {
            "Average Betweenness Centrality": 0,
            "Average Closeness Centrality": 0,
            "Average Degree Centrality": 0
        }

    betweenness = nx.betweenness_centrality(graph)
    closeness = nx.closeness_centrality(graph)
    degree = nx.degree_centrality(graph)

    stats = {
        "Average Betweenness Centrality": float(np.mean(list(betweenness.values()))),
        "Average Closeness Centrality": float(np.mean(list(closeness.values()))),
        "Average Degree Centrality": float(np.mean(list(degree.values())))
    }
    return stats


def summarize_graph(graph: nx.Graph, name: str = "") -> dict:
    summary = {"Graph Name": name}
    summary.update(compute_basic_graph_stats(graph))
    summary.update(compute_graph_centrality_stats(graph))
    return summary
