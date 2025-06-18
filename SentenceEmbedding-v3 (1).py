#%%
!pip install sentence-transformers matplotlib scikit-learn
!pip install spacy gensim nltk pyLDAvis
!pip install "numpy<2.0" --upgrade --force-reinstall
!pip uninstall -y nltk
!pip install nltk --upgrade --no-cache-dir

#%%
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize 
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import string
#%%
# Load a sentence embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # small but high-quality model
#%% md
# ## acutual polished pipeline 
#%%
try:
    from umap import UMAP
    has_umap = True
except ImportError:
    has_umap = False
#%%
import seaborn as sns
import plotly.graph_objects as go
#takes in filepath, creates embeddings based on specified model 
def embed_sentences_from_txt(file_path, model=None):
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip().split()) > 1]
    embeddings = model.encode(sentences)
    return sentences, embeddings


def plot_2d_scatter(embeddings, labels, method="PCA", title=""):
    if method == "PCA":
        reducer = PCA(n_components=2)
    elif method == "TSNE":
        reducer = TSNE(n_components=2, perplexity=10, random_state=42)
    else:
        raise ValueError("Unsupported method")

    reduced = reducer.fit_transform(embeddings)

    fig = go.Figure()

    for group, color in zip(['Novice', 'Expert'], ['Blues', 'Reds']):
        idxs = [i for i, l in enumerate(labels) if l == group]
        x = reduced[idxs, 0]
        y = reduced[idxs, 1]
        fig.add_trace(go.Scatter(
            x=x,
            y=y,
            mode='markers+lines',
            line=dict(width=1, color='lightgrey'),
            marker=dict(
                size=8,
                color=np.linspace(0.3, 1.0, len(x)),
                colorscale=color,
                showscale=False
            ),
            name=group,
            hovertext=[f"{group} - Step {i}" for i in range(len(x))],
            hoverinfo="text"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=f"{method} Dimension 1",
        yaxis_title=f"{method} Dimension 2",
        width=1000,
        height=700
    )
    fig.show()

def plot_cosine_heatmap(embeddings, sentences):
    sim_matrix = cosine_similarity(embeddings)
    plt.figure(figsize=(10, 8))
    sns.heatmap(sim_matrix, cmap="viridis")
    plt.title("Cosine Similarity Heatmap")
    plt.xlabel("Sentence Index")
    plt.ylabel("Sentence Index")
    plt.show()


#a is novice, b is expert
def compute_stats(emb_a, emb_b):
    centroid_a = np.mean(emb_a, axis=0)
    centroid_b = np.mean(emb_b, axis=0)
    dist_euclidean = np.linalg.norm(centroid_a - centroid_b)
    dist_cosine = 1 - cosine_similarity([centroid_a], [centroid_b])[0][0]

    stats = {
        "Centroid Distance (Euclidean)": dist_euclidean,
        "Centroid Distance (Cosine)": dist_cosine
    }

    # Average pairwise cosine similarity
    def avg_pairwise_cosine(emb):
        if len(emb) < 2:
            return "Insufficient data"
        sim_matrix = cosine_similarity(emb)
        n = len(emb)
        upper_triangle = sim_matrix[np.triu_indices(n, k=1)]
        return np.mean(upper_triangle)

    stats["Avg Cosine Similarity (Novice)"] = avg_pairwise_cosine(emb_a)
    stats["Avg Cosine Similarity (Expert)"] = avg_pairwise_cosine(emb_b)

    # Average distance to centroid
    stats["Avg Dist to Centroid (Novice)"] = np.mean([np.linalg.norm(vec - centroid_a) for vec in emb_a])
    stats["Avg Dist to Centroid (Expert)"] = np.mean([np.linalg.norm(vec - centroid_b) for vec in emb_b])

    # Convex hull area in 2D PCA space
    try:
        if len(emb_a) >= 3:
            hull_a = ConvexHull(PCA(n_components=2).fit_transform(emb_a))
            stats["Convex Hull Area (Novice)"] = hull_a.volume
        else:
            stats["Convex Hull Area (Novice)"] = "Insufficient data"

        if len(emb_b) >= 3:
            hull_b = ConvexHull(PCA(n_components=2).fit_transform(emb_b))
            stats["Convex Hull Area (Expert)"] = hull_b.volume
        else:
            stats["Convex Hull Area (Expert)"] = "Insufficient data"
    except:
        stats["Convex Hull Error"] = "Unable to compute"

    return stats



import numbers

def sanitize_stats(stats_dict):
    cleaned_stats = {}
    for k, v in stats_dict.items():
        # Handle arrays or 0-D tensors
        if isinstance(v, (np.ndarray, list)) and np.array(v).size == 1:
            v = np.array(v).item()
        # Replace anything that's not a scalar (like arrays) with string fallback
        if isinstance(v, numbers.Number):
            if np.isnan(v) or np.isinf(v):
                cleaned_stats[k] = np.nan
            else:
                cleaned_stats[k] = v
        else:
            try:
                cleaned_stats[k] = float(v)
            except:
                cleaned_stats[k] = np.nan
    return cleaned_stats


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import entropy
from sklearn.mixture import GaussianMixture


# a is novice b is expert
def compute_extended_stats(emb_a, emb_b):
    stats = {}

    # Combine and prepare
    all_embeddings = np.vstack([emb_a, emb_b])
    labels = ['Novice'] * len(emb_a) + ['Expert'] * len(emb_b)

    # Centroids and distances
    centroid_a = np.mean(emb_a, axis=0)
    centroid_b = np.mean(emb_b, axis=0)
    stats["Centroid Distance (Euclidean)"] = np.linalg.norm(centroid_a - centroid_b)
    stats["Centroid Distance (Cosine)"] = 1 - cosine_similarity([centroid_a], [centroid_b])[0][0]

    # Average pairwise cosine similarity
    def avg_pairwise_cosine(emb):
        if len(emb) < 2:
            return np.nan
        sim_matrix = cosine_similarity(emb)
        return np.mean(sim_matrix[np.triu_indices(len(emb), k=1)])

    stats["Avg Cosine Similarity (Novice)"] = avg_pairwise_cosine(emb_a)
    stats["Avg Cosine Similarity (Expert)"] = avg_pairwise_cosine(emb_b)

    # Avg distance to centroid
    stats["Avg Dist to Centroid (Novice)"] = np.mean(np.linalg.norm(emb_a - centroid_a, axis=1))
    stats["Avg Dist to Centroid (Expert)"] = np.mean(np.linalg.norm(emb_b - centroid_b, axis=1))

    # PCA Explained Variance
    pca_full = PCA()
    pca_full.fit(all_embeddings)
    stats["PCA Explained Variance Ratio (1st)"] = pca_full.explained_variance_ratio_[0]
    stats["PCA Components > 95% Variance"] = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1

    # Convex Hull in PCA 2D
    try:
        pca_2d = PCA(n_components=2)
        emb_a_2d = pca_2d.fit_transform(emb_a)
        emb_b_2d = pca_2d.fit_transform(emb_b)
        stats["Convex Hull Area (Novice)"] = ConvexHull(emb_a_2d).volume if len(emb_a) >= 3 else np.nan
        stats["Convex Hull Area (Expert)"] = ConvexHull(emb_b_2d).volume if len(emb_b) >= 3 else np.nan
    except:
        stats["Convex Hull Area (Novice)"] = np.nan
        stats["Convex Hull Area (Expert)"] = np.nan

    # Pairwise intra-cluster distances
    def intra_cluster_dists(emb):
        dists = distance_matrix(emb, emb)
        triu_vals = dists[np.triu_indices(len(emb), k=1)]
        return np.mean(triu_vals), np.std(triu_vals)

    intra_nov_mean, intra_nov_std = intra_cluster_dists(emb_a)
    intra_exp_mean, intra_exp_std = intra_cluster_dists(emb_b)

    stats["Intra-cluster Distance Mean (Novice)"] = intra_nov_mean
    stats["Intra-cluster Distance Std (Novice)"] = intra_nov_std
    stats["Intra-cluster Distance Mean (Expert)"] = intra_exp_mean
    stats["Intra-cluster Distance Std (Expert)"] = intra_exp_std

    # Inter-cluster distance
    inter_dists = distance_matrix(emb_a, emb_b)
    stats["Inter-cluster Distance Mean"] = np.mean(inter_dists)

    # KL Divergence using Gaussian Mixture Approximation
    try:
        gmm_nov = GaussianMixture(n_components=1, covariance_type='full').fit(emb_a)
        gmm_exp = GaussianMixture(n_components=1, covariance_type='full').fit(emb_b)
        sample_nov = gmm_nov.sample(1000)[0]
        log_p = gmm_nov.score_samples(sample_nov)
        log_q = gmm_exp.score_samples(sample_nov)
        log_p = np.clip(log_p, -700, 700)
        log_q = np.clip(log_q, -700, 700)
        p = np.exp(log_p)
        q = np.exp(log_q)
        p /= np.sum(p)
        q /= np.sum(q)
        stats["KL Divergence (Novice || Expert)"] = entropy(p, q)


        
    except:
        stats["KL Divergence (Novice || Expert)"] = np.nan

    # Spectral Entropy of distance matrix
    try:
        dist_mat = distance_matrix(all_embeddings, all_embeddings)
        eigvals = np.linalg.eigvalsh(dist_mat)
        norm_eigvals = np.abs(eigvals / np.sum(np.abs(eigvals)))
        stats["Spectral Entropy"] = entropy(norm_eigvals)
    except:
        stats["Spectral Entropy"] = np.nan

    # Density Estimate via KDE
    for group_name, emb in zip(['Novice', 'Expert'], [emb_a, emb_b]):
        try:
            kde = KernelDensity(kernel='gaussian').fit(emb)
            log_density = kde.score_samples(emb)
            stats[f"KDE Mean Log Density ({group_name})"] = np.mean(log_density)
        except:
            stats[f"KDE Mean Log Density ({group_name})"] = np.nan

    

    return sanitize_stats(stats)





def analyze_text_embedding(novice_path, expert_path, topic):
    sentences_nov, emb_nov = embed_sentences_from_txt(novice_path)
    sentences_exp, emb_exp = embed_sentences_from_txt(expert_path)

    all_embeddings = np.vstack([emb_nov, emb_exp])
    all_sentences = sentences_nov + sentences_exp
    labels = ['Novice'] * len(emb_nov) + ['Expert'] * len(emb_exp)

    # Plots
    plot_2d_scatter(all_embeddings, labels, method="PCA", title=f"PCA: Expert vs Novice - {topic}")
    plot_2d_scatter(all_embeddings, labels, method="TSNE", title=f"t-SNE: Expert vs Novice - {topic}")
    plot_cosine_heatmap(all_embeddings, all_sentences)

    # Stats
    stats = compute_stats(emb_nov, emb_exp)
    print("\n--- Statistical Summary ---")
    for k, v in stats.items():
        print(f"{k}: {v}")
#%%
analyze_text_embedding("expert_generated.txt",
                       "novice_generated.txt", "car")

#%%
# Global dataframe to collect all stats
stats_df = pd.DataFrame()

def collect_and_append_stats(expert_path, novice_path, topic):
    global stats_df

    # Embed and compute stats
    sentences_nov, emb_nov = embed_sentences_from_txt(novice_path)
    sentences_exp, emb_exp = embed_sentences_from_txt(expert_path)
    stats = compute_extended_stats(emb_nov, emb_exp)

    # Add topic and append to global dataframe
    stats["Topic"] = topic
    stats_df = pd.concat([stats_df, pd.DataFrame([stats])], ignore_index=True)

#%%
collect_and_append_stats("great_decision_expert_full.txt",
                       "great_decision_novice_full.txt", "decision making")

collect_and_append_stats("good_creativity_expert_expanded.txt",
                       "good_creativity_novice_expanded.txt", "creativity")

collect_and_append_stats("good_nutrition_expert_expanded.txt",
                       "good_nutrition_novice_expanded.txt", "nutrition")

collect_and_append_stats("good_language_acquisition_novice_freespeech.txt",
                       "good_language_acquisition_expert_freespeech.txt", "language acquisition")

collect_and_append_stats("climate_expert_full.txt",
                       "climate_novice_full.txt", "climate")

collect_and_append_stats("vision_expert_full.txt",
                       "vision_novice_full.txt", "vision")
#%%
stats_df
#%%
collect_and_append_stats("ai_expert_refined.txt",
                       "ai_novice_refined.txt", "AI")

collect_and_append_stats("memory_expert.txt",
                       "memory_novice.txt", "memory")

collect_and_append_stats("time_expert.txt",
                       "time_novice.txt", "time")


#%%
stats_df
#%%
collect_and_append_stats("robotics_expert.txt",
                       "robotics_novice.txt", "robotics")

collect_and_append_stats("mental_expert.txt",
                       "mental_novice.txt", "mental health")

collect_and_append_stats("mind_expert.txt",
                       "mind_novice.txt", "mind")

collect_and_append_stats("moral_expert.txt",
                       "moral_novice.txt", "morals")

collect_and_append_stats("language_expert.txt",
                       "language_novice.txt", "language")


#%%
stats_df
#%%
