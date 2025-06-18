# semantic_horizons/compute_stats.py

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import entropy
from sentence_transformers import SentenceTransformer
import os
import numbers
import gc
from nltk.tokenize import sent_tokenize



def embed_sentences_from_txt(file_path):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in sent_tokenize(text) if len(s.strip().split()) > 1]
    embeddings = model.encode(sentences)
    return sentences, embeddings

def sanitize_stats(stats_dict):
    cleaned_stats = {}
    for k, v in stats_dict.items():
        if isinstance(v, (np.ndarray, list)) and np.array(v).size == 1:
            v = np.array(v).item()
        if isinstance(v, numbers.Number):
            cleaned_stats[k] = np.nan if (np.isnan(v) or np.isinf(v)) else v
        else:
            try:
                cleaned_stats[k] = float(v)
            except:
                cleaned_stats[k] = np.nan
    return cleaned_stats


#a for novice, b for expert
def compute_extended_stats(emb_a, emb_b):
    stats = {}
    all_embeddings = np.vstack([emb_a, emb_b])

    centroid_a = np.mean(emb_a, axis=0)
    centroid_b = np.mean(emb_b, axis=0)
    stats["Centroid Distance (Euclidean)"] = np.linalg.norm(centroid_a - centroid_b)
    stats["Centroid Distance (Cosine)"] = 1 - cosine_similarity([centroid_a], [centroid_b])[0][0]

    def avg_pairwise_cosine(emb):
        if len(emb) < 2: return np.nan
        sim_matrix = cosine_similarity(emb)
        return np.mean(sim_matrix[np.triu_indices(len(emb), k=1)])

    stats["Avg Cosine Similarity (Novice)"] = avg_pairwise_cosine(emb_a)
    stats["Avg Cosine Similarity (Expert)"] = avg_pairwise_cosine(emb_b)

    stats["Avg Dist to Centroid (Novice)"] = np.mean(np.linalg.norm(emb_a - centroid_a, axis=1))
    stats["Avg Dist to Centroid (Expert)"] = np.mean(np.linalg.norm(emb_b - centroid_b, axis=1))

    pca_full = PCA()
    pca_full.fit(all_embeddings)
    stats["PCA Explained Variance Ratio (1st)"] = pca_full.explained_variance_ratio_[0]
    stats["PCA Components > 95% Variance"] = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.95) + 1

    try:
        pca_2d = PCA(n_components=2)
        emb_a_2d = pca_2d.fit_transform(emb_a)
        emb_b_2d = pca_2d.fit_transform(emb_b)
        stats["Convex Hull Area (Novice)"] = ConvexHull(emb_a_2d).volume if len(emb_a) >= 3 else np.nan
        stats["Convex Hull Area (Expert)"] = ConvexHull(emb_b_2d).volume if len(emb_b) >= 3 else np.nan
    except:
        stats["Convex Hull Area (Novice)"] = np.nan
        stats["Convex Hull Area (Expert)"] = np.nan

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

    inter_dists = distance_matrix(emb_a, emb_b)
    stats["Inter-cluster Distance Mean"] = np.mean(inter_dists)

    try:
        gmm_nov = GaussianMixture(n_components=1).fit(emb_a)
        gmm_exp = GaussianMixture(n_components=1).fit(emb_b)
        sample_nov = gmm_nov.sample(1000)[0]
        log_p = gmm_nov.score_samples(sample_nov)
        log_q = gmm_exp.score_samples(sample_nov)
        log_p, log_q = np.clip(log_p, -700, 700), np.clip(log_q, -700, 700)
        p, q = np.exp(log_p), np.exp(log_q)
        p, q = p/np.sum(p), q/np.sum(q)
        stats["KL Divergence (Novice || Expert)"] = entropy(p, q)
    except:
        stats["KL Divergence (Novice || Expert)"] = np.nan

    try:
        dist_mat = distance_matrix(all_embeddings, all_embeddings)
        eigvals = np.linalg.eigvalsh(dist_mat)
        norm_eigvals = np.abs(eigvals / np.sum(np.abs(eigvals)))
        stats["Spectral Entropy"] = entropy(norm_eigvals)
    except:
        stats["Spectral Entropy"] = np.nan

    for label, emb in zip(["Novice", "Expert"], [emb_a, emb_b]):
        try:
            kde = KernelDensity().fit(emb)
            log_density = kde.score_samples(emb)
            stats[f"KDE Mean Log Density ({label})"] = np.mean(log_density)
        except:
            stats[f"KDE Mean Log Density ({label})"] = np.nan

    return sanitize_stats(stats)

def collect_and_append_stats(expert_path, novice_path, topic, stats_df):
    sentences_nov, emb_nov = embed_sentences_from_txt(novice_path)
    sentences_exp, emb_exp = embed_sentences_from_txt(expert_path)
    stats = compute_extended_stats(emb_nov, emb_exp)
    stats["Topic"] = topic
    stats_df.loc[len(stats_df)] = stats
    del emb_nov, emb_exp
    gc.collect()
