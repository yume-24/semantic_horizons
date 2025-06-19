import networkx as nx

def build_discourse_graph(sentences, relations=None):
    G = nx.DiGraph()
    for idx, sentence in enumerate(sentences):
        G.add_node(idx, text=sentence)

    if relations:
        for (src, tgt, rel) in relations:
            G.add_edge(src, tgt, label=rel)

    return G
