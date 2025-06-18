
import os
import pandas as pd
from compute_stats import collect_and_append_stats

# Ensure output folder exists
os.makedirs("outputs", exist_ok=True)

print("Current Working Directory:", os.getcwd())
print("Files in data/:", os.listdir("data"))
print("Expert file exists:", os.path.exists("data/vision_expert_full.txt"))
print("Novice file exists:", os.path.exists("data/ai_expert.txt"))

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
    ("data/vision_expert_full.txt", "data/vision_novice_full.txt", "vision"),
    ("data/ai_expert_refined.txt", "data/ai_novice_refined.txt", "AI refined"),
    ("data/arts_expert.txt", "data/arts_novice.txt", "arts"),
    ("data/language_expert.txt", "data/language_novice.txt", "language"),
    ("data/memory_expert.txt", "data/memory_novice.txt", "memory"),
    ("data/mental_expert.txt", "data/mental_novice.txt", "mental health"),
    ("data/mind_expert.txt", "data/mind_novice.txt", "mind"),
    ("data/moral_expert.txt", "data/moral_novice.txt", "morals"),
    ("data/music_expert.txt", "data/music_novice.txt", "robotics"),
    ("data/time_expert.txt", "data/time_novice.txt", "time"),
    ("data/good_creativity_expert_expanded.txt", "data/good_creativity_novice_expanded.txt", "creativity"),
    ("data/good_language_acquisition_expert_freespeech.txt", "data/good_language_acquisition_novice_freespeech.txt", "language acquisition"),
    ("data/good_nutrition_expert_expanded.txt", "data/good_nutrition_novice_expanded.txt", "nutrition"),
    ("data/bioacoustics_expert.txt", "data/bioacoustics_novice.txt", "bioacoustics"),
    ("data/olfactory_expert.txt", "data/olfactory_novice.txt", "olfactory memory"),
    ("data/speechlike_coral_expert.txt", "data/speechlike_coral_novice.txt", "coral"),
    ("data/speechlike_crypto_expert.txt", "data/speechlike_crypto_novice.txt", "time"),
    ("data/speechlike_dreams_expert.txt", "data/speechlike_dreams_novice.txt", "dreams")
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
