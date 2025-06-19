def clean_graph_data(G):
    # Remove isolated nodes or fix malformed edge labels
    isolated = list(nx.isolates(G))
    G.remove_nodes_from(isolated)
    return G
