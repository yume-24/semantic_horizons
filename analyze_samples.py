import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import load
from sklearn.decomposition import PCA

# Local import
from compute_stats import embed_sentences_from_txt, compute_extended_stats, sanitize_stats

# Define paths
expert_path = "data/expert_generated.txt"
novice_path = "data/novice_generated.txt"

# Features used in models
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

# Load all models
model_paths = {
    "Random Forest": "models/random_forest_model.joblib",
    "Logistic Regression": "models/logistic_regression_model.joblib",
    "SVM (RBF)": "models/svm_rbf_model.joblib",
    "MLP (Neural Net)": "models/mlp_neural_net_model.joblib",
    "XGBoost": "models/xgboost_model.joblib"
}
models = {name: load(path) for name, path in model_paths.items()}

# Helper: process a sample file
def process_text(path):
    _, emb = embed_sentences_from_txt(path)
    return sanitize_stats(compute_extended_stats(emb, emb))

# Generate features for both samples
expert_stats = process_text(expert_path)
novice_stats = process_text(novice_path)
df = pd.DataFrame([expert_stats, novice_stats])
X = df[features]
labels = ["Expert", "Novice"]

# Evaluation
def evaluate_model(model, name, threshold=0.65):
    print(f"\n📊 Model: {name}")
    probs = model.predict_proba(X)

    # Get class labels
    if hasattr(model, "classes_"):
        classes = model.classes_
    elif hasattr(model.named_steps["clf"], "classes_"):
        classes = model.named_steps["clf"].classes_
    else:
        raise ValueError("Could not extract class labels from model.")

    # Normalize to string class labels
    class_map = {}
    if set(classes) == set([0, 1]):
        class_map = {0: "Novice", 1: "Expert"}
    elif set(classes) == set(["Novice", "Expert"]):
        class_map = {"Novice": "Novice", "Expert": "Expert"}
    else:
        raise ValueError(f"Unexpected class labels: {classes}")

    inverse_map = {v: k for k, v in class_map.items()}
    expert_index = inverse_map["Expert"]

    preds = ["Expert" if p[expert_index] > threshold else "Novice" for p in probs]

    for i, label in enumerate(labels):
        expert_prob = probs[i][expert_index]
        print(f"{label} sample → Predicted: {preds[i]} | P(Expert) = {expert_prob:.2f}")

    # Feature importance or coefficients
    try:
        clf = model.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            print("\n🌲 Random Forest Feature Importances:")
            for feat, imp in sorted(zip(features, clf.feature_importances_), key=lambda x: -x[1]):
                print(f"{feat}: {imp:.3f}")
        elif hasattr(clf, "coef_"):
            print("\n📈 Logistic Regression Coefficients:")
            for feat, coef in zip(features, clf.coef_[0]):
                print(f"{feat}: {coef:.3f}")
    except Exception as e:
        print(f"[Warning] Could not extract feature importances: {e}")

    return preds


# PCA plot
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

plt.figure(figsize=(15, 8))
for i, (name, model) in enumerate(models.items()):
    preds = evaluate_model(model, name)
    plt.subplot(2, 3, i + 1)
    palette = {"Expert": "blue", "Novice": "orange"}
    sns.scatterplot(x=X_2d[:, 0], y=X_2d[:, 1], hue=preds, palette=palette, s=100)
    for j, label in enumerate(labels):
        plt.text(X_2d[j, 0] + 0.01, X_2d[j, 1], f"{label} (→ {preds[j]})", fontsize=9)
    plt.title(name)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.grid(True)

plt.tight_layout()
plt.show()
