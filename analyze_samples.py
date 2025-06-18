import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA

from compute_stats import embed_sentences_from_txt, compute_extended_stats, sanitize_stats

# Define paths
expert_path = "data/expert_generated.txt"
novice_path = "data/novice_generated.txt"
model_path = "models/stat_classifier_model.joblib"  # Update if stored elsewhere

# Load trained classifier
model = load(model_path)

# Define features used by the classifier
features = [
    "Avg Cosine Similarity (Expert)",
    "Avg Cosine Similarity (Novice)",
    "Avg Dist to Centroid (Expert)",
    "Avg Dist to Centroid (Novice)",
    "Convex Hull Area (Expert)",
    "Convex Hull Area (Novice)",
    "Intra-cluster Distance Mean (Expert)",
    "Intra-cluster Distance Mean (Novice)",
    "Intra-cluster Distance Std (Expert)",
    "Intra-cluster Distance Std (Novice)",
    "KDE Mean Log Density (Expert)",
    "KDE Mean Log Density (Novice)"
]

# Process sample and compute features
def process_text(path):
    _, emb = embed_sentences_from_txt(path)
    return sanitize_stats(compute_extended_stats(emb, emb))

novice_stats = process_text(novice_path)
expert_stats = process_text(expert_path)

# Build DataFrame for prediction
df = pd.DataFrame([expert_stats, novice_stats])
X = df[features]
labels = ["Expert", "Novice"]
preds = model.predict(X)

# Print prediction results
for true_label, pred in zip(labels, preds):
    print(f"{true_label} sample predicted as: {pred}")

# PCA for visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8, 6))
colors = ["blue" if p == "Expert" else "orange" for p in preds]
sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=preds, palette={"Expert": "blue", "Novice": "orange"}, s=100)

for i, label in enumerate(labels):
    plt.text(X_2d[i, 0] + 0.01, X_2d[i, 1], f"{label} (→ {preds[i]})", fontsize=9)

plt.title("PCA of Sample Statistics with Predicted Labels")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.tight_layout()
plt.show()

