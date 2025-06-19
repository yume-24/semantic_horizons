# graph_analysis/graph_visualizer.py

import os
import webbrowser
from pyvis.network import Network
import networkx as nx


def visualize_graph_pyvis(graph, title="Graph Visualization", output_file="graph.html"):
    if graph is None or len(graph.nodes) == 0:
        print(f"⚠️ Skipping empty graph for {output_file}")
        return None

    try:
        net = Network(height='750px', width='100%', directed=True, notebook=False)
        net.force_atlas_2based()

        for node in graph.nodes():
            net.add_node(node, label=str(node))

        for source, target, data in graph.edges(data=True):
            label = data.get('label', '')
            net.add_edge(source, target, title=label)

        net.show_buttons(filter_=['physics'])
        net.set_options("""
        var options = {
          "nodes": {
            "shape": "dot",
            "font": { "size": 14 }
          },
          "edges": {
            "font": { "size": 12, "align": "middle" },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.5 } }
          },
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.01,
              "springLength": 150,
              "springConstant": 0.05
            },
            "minVelocity": 0.75,
            "solver": "forceAtlas2Based"
          }
        }""")

        net.save_graph(output_file)
        print(f"✅ Saved: {output_file}")
        webbrowser.open('file://' + os.path.realpath(output_file))  # Automatically open
        return net

    except Exception as e:
        print(f"❌ Error rendering {output_file}: {e}")
        return None


def visualize_graph_pair(expert_graph, novice_graph, topic, output_dir="outputs/visualizations"):
    os.makedirs(output_dir, exist_ok=True)

    expert_path = os.path.join(output_dir, f"{topic}_expert.html")
    novice_path = os.path.join(output_dir, f"{topic}_novice.html")

    visualize_graph_pyvis(expert_graph, title=f"{topic.title()} - Expert", output_file=expert_path)
    visualize_graph_pyvis(novice_graph, title=f"{topic.title()} - Novice", output_file=novice_path)
