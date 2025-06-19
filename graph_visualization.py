# 📊 graph_visualization.py
import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph, title="Graph Visualization", layout="spring", node_color='lightblue'):
    plt.figure(figsize=(10, 8))

    if layout == "spring":
        pos = nx.spring_layout(graph)
    elif layout == "kamada":
        pos = nx.kamada_kawai_layout(graph)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph)  # default

    nx.draw_networkx(
        graph, pos,
        with_labels=True,
        node_color=node_color,
        edge_color='gray',
        font_size=10,
        node_size=800
    )
    edge_labels = nx.get_edge_attributes(graph, 'label')
    if edge_labels:
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red')

    plt.title(title)
    plt.axis('off')
    plt.show()

# Example usage (uncomment for testing):
G = nx.read_gml("outputs/graphs/creativity_expert_concept.gml")
visualize_graph(G, title="Creativity - Expert Concept Map")
