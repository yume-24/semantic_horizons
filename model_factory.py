from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# Create and return all classifier pipelines
def create_models():
    models = {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ]),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),

        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", probability=True, random_state=42))
        ]),

        "MLP (Neural Net)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, random_state=42))
        ]),

        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))
        ])
    }
    return models

# Example usage:
# models = create_models()
# for name, pipeline in models.items():
#     pipeline.fit(X_train, y_train)
#     preds = pipeline.predict(X_test)


# model_factory.py

import os
from joblib import load

# Central registry of saved models
MODEL_PATHS = {
    "random_forest": "models/random_forest_model.joblib",
    "logistic_regression": "models/logistic_regression_model.joblib",
    "svm_rbf": "models/svm_rbf_model.joblib",
    "mlp_neural_net": "models/mlp_neural_net_model.joblib",
    "xgboost": "models/xgboost_model.joblib"
}

# Shared feature names
FEATURES = [
    "Avg Cosine Similarity (Expert)", "Avg Cosine Similarity (Novice)",
    "Avg Dist to Centroid (Expert)", "Avg Dist to Centroid (Novice)",
    "Convex Hull Area (Expert)", "Convex Hull Area (Novice)",
    "Intra-cluster Distance Mean (Expert)", "Intra-cluster Distance Mean (Novice)",
    "Intra-cluster Distance Std (Expert)", "Intra-cluster Distance Std (Novice)",
    "KDE Mean Log Density (Expert)", "KDE Mean Log Density (Novice)"
]


def load_model(model_name: str):
    """
    Load a saved classifier model by name.
    :param model_name: One of 'random_forest', 'logistic_regression', 'svm_rbf', 'mlp_neural_net', 'xgboost'
    :return: loaded model or raises KeyError
    """
    model_name = model_name.lower()
    if model_name not in MODEL_PATHS:
        raise KeyError(f"Unknown model '{model_name}'. Available: {list(MODEL_PATHS.keys())}")

    path = MODEL_PATHS[model_name]
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")

    return load(path)


def get_feature_names():
    return FEATURES


def list_available_models():
    return list(MODEL_PATHS.keys())
