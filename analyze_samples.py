import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import sent_tokenize
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer





# Load files
with open("data/expert_generated.txt", "r", encoding="utf-8") as f:
    expert_text = f.read()
with open("data/novice_generated.txt", "r", encoding="utf-8") as f:
    novice_text = f.read()

# Sentence tokenization
expert_sents = [s.strip() for s in sent_tokenize(expert_text) if len(s.strip().split()) > 1]
novice_sents = [s.strip() for s in sent_tokenize(novice_text) if len(s.strip().split()) > 1]

# Embed
model = SentenceTransformer("all-MiniLM-L6-v2")
expert_emb = model.encode(expert_sents)
novice_emb = model.encode(novice_sents)

# PCA
combined = np.vstack([expert_emb, novice_emb])
pca = PCA(n_components=2)
combined_2d = pca.fit_transform(combined)
expert_2d = combined_2d[:len(expert_emb)]
novice_2d = combined_2d[len(expert_emb):]

# Plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x=expert_2d[:, 0], y=expert_2d[:, 1], label="Expert", s=70, marker="o")
sns.scatterplot(x=novice_2d[:, 0], y=novice_2d[:, 1], label="Novice", s=70, marker="s")
plt.title("2D PCA Projection of Expert vs Novice Embeddings")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# analyze_samples.py
import pandas as pd
import numpy as np
from compute_stats import embed_sentences_from_txt, compute_extended_stats, sanitize_stats
from joblib import load
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the trained classifier
model = load("models/stat_classifier_model.joblib")

# Features used in the model
features = [
    "Avg Cosine Similarity",
    "Avg Dist to Centroid",
    "Convex Hull Area",
    "Intra-cluster Distance Mean",
    "Intra-cluster Distance Std",
    "KDE Mean Log Density"
]

# Paths to samples
novice_path = "data/novice_generated.txt"
expert_path = "data/expert_generated.txt"

# Process each
def get_stat_vector(path):
    _, emb = embed_sentences_from_txt(path)
    stats = compute_extended_stats(emb, emb)
    return sanitize_stats(stats)

novice_stats = get_stat_vector(novice_path)
expert_stats = get_stat_vector(expert_path)

# Assemble into a DataFrame
df = pd.DataFrame([novice_stats, expert_stats])
X = df[features]
labels = ["Novice", "Expert"]

# Prediction
preds = model.predict(X)

# PCA Visualization
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Plotting
plt.figure(figsize=(6, 6))
colors = ["orange" if p == "Novice" else "blue" for p in preds]
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=colors, label="Predicted", s=80)
for i, label in enumerate(labels):
    plt.annotate(f"{label} ({preds[i]})", (X_2d[i, 0] + 0.01, X_2d[i, 1]), fontsize=9)

plt.title("PCA of Sample Statistics")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.grid(True)
plt.tight_layout()
plt.show()
