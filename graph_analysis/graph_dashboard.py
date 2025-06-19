# graph_analysis/graph_dashboard.py
import os
import glob

def build_dashboard_html(graph_dir="outputs/visualizations", dashboard_path="outputs/visualizations/all_graphs_dashboard.html"):
    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
    graph_files = sorted(glob.glob(os.path.join(graph_dir, "*.html")))

    # Filter out the dashboard itself if it exists already
    graph_files = [f for f in graph_files if not f.endswith("all_graphs_dashboard.html")]

    html_blocks = []
    for file in graph_files:
        title = os.path.basename(file).replace(".html", "").replace("_", " ").title()
        block = f"""
        <div style="width: 48%; display: inline-block; margin: 1%;">
            <h3 style="text-align:center;">{title}</h3>
            <iframe src="{os.path.basename(file)}" width="100%" height="400px" frameborder="0"></iframe>
        </div>
        """
        html_blocks.append(block)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>All Graph Visualizations</title>
        <meta charset="utf-8">
    </head>
    <body>
        <h1 style="text-align:center;">Concept & Discourse Graphs Dashboard</h1>
        {''.join(html_blocks)}
    </body>
    </html>
    """

    with open(dashboard_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"✅ Dashboard created: {dashboard_path}")
