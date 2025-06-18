
import os
import pandas as pd
from compute_stats import collect_and_append_stats

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

print("Current Working Directory:", os.getcwd())
print("Files in data/:", os.listdir("data"))
print("Expert file exists:", os.path.exists("data/great_decision_expert_full.txt"))
print("Novice file exists:", os.path.exists("data/great_decision_novice_full.txt"))

stats_columns = [
    "Centroid Distance (Euclidean)",
    "Centroid Distance (Cosine)",
    "Avg Cosine Similarity (Novice)",
    "Avg Cosine Similarity (Expert)",
    "Avg Dist to Centroid (Novice)",
    "Avg Dist to Centroid (Expert)",
    "PCA Explained Variance Ratio (1st)",
    "PCA Components > 95% Variance",
    "Convex Hull Area (Novice)",
    "Convex Hull Area (Expert)",
    "Intra-cluster Distance Mean (Novice)",
    "Intra-cluster Distance Std (Novice)",
    "Intra-cluster Distance Mean (Expert)",
    "Intra-cluster Distance Std (Expert)",
    "Inter-cluster Distance Mean",
    "KL Divergence (Novice || Expert)",
    "Spectral Entropy",
    "KDE Mean Log Density (Novice)",
    "KDE Mean Log Density (Expert)",
    "Topic"
]
stats_df = pd.DataFrame(columns=stats_columns)

# Topic list: (expert_file, novice_file, topic_name)
topic_inputs = [
    ("data/great_decision_expert_full.txt", "data/great_decision_novice_full.txt", "decision making"),
    # Add more topics below as needed
    # ("data/vision_expert.txt", "data/vision_novice.txt", "vision"),
    # ("data/ai_expert.txt", "data/ai_novice.txt", "AI"),
]

# Initialize DataFrame


# Process each topic
for expert_path, novice_path, topic in topic_inputs:
    print(f"Processing topic: {topic}")
    collect_and_append_stats(expert_path, novice_path, topic, stats_df)

# Save output
output_path = "outputs/results.csv"
stats_df.to_csv(output_path, index=False)
print(f"\nAll topics complete. Results saved to: {output_path}")
