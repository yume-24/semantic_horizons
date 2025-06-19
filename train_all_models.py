# train_all_models.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score
from xgboost import XGBClassifier
from joblib import dump
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Load the cleaned statistics dataset
df = pd.read_csv("outputs/results.csv")  # ← Change this if needed

# Define expert and novice samples by comparing cosine similarity
df["Label"] = df.apply(
    lambda row: "Expert" if row["Avg Cosine Similarity (Expert)"] > row["Avg Cosine Similarity (Novice)"] else "Novice",
    axis=1
)

# Features to use
features = [
    "Avg Cosine Similarity (Expert)", "Avg Cosine Similarity (Novice)",
    "Avg Dist to Centroid (Expert)", "Avg Dist to Centroid (Novice)",
    "Convex Hull Area (Expert)", "Convex Hull Area (Novice)",
    "Intra-cluster Distance Mean (Expert)", "Intra-cluster Distance Mean (Novice)",
    "Intra-cluster Distance Std (Expert)", "Intra-cluster Distance Std (Novice)",
    "KDE Mean Log Density (Expert)", "KDE Mean Log Density (Novice)"
]

X = df[features]
y = df["Label"]

# Binary encoding: Expert → 1, Novice → 0
le = LabelEncoder()
y_binary = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, stratify=y_binary, random_state=42, test_size=0.2)

# Define model configs
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM (RBF)": SVC(probability=True, kernel='rbf', random_state=42),
    "MLP (Neural Net)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", model)
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline.named_steps["clf"], "predict_proba") else None

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    if y_proba is not None:
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.3f}")
    else:
        print("ROC AUC: Not available (no probability estimates)")

    # Save the model
    safe_name = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    dump(pipeline, f"models/{safe_name}_model.joblib")
    print(f"Saved {name} to models/{safe_name}_model.joblib")
