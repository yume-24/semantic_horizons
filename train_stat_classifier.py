import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from joblib import dump

# Load the cleaned statistics dataset
df = pd.read_csv("outputs/results.csv")

# Define expert and novice samples by comparing cosine similarity
# Assume: higher "Avg Cosine Similarity (Expert)" means expert
df["Label"] = df.apply(lambda row: "Expert" if row["Avg Cosine Similarity (Expert)"] > row["Avg Cosine Similarity (Novice)"] else "Novice", axis=1)

# Features to use
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

X = df[features]
y = df["Label"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
#uses random forest for classifier
# Pipeline: Standardization + Classifier
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
print("Classification Report on Test Set:")
print(classification_report(y_test, pipeline.predict(X_test)))

# Cross-validation score
cv_score = cross_val_score(pipeline, X, y, cv=5)
print(f"Mean CV Accuracy: {cv_score.mean():.3f} ± {cv_score.std():.3f}")

# Save model
dump(pipeline, "models/stat_classifier_model.joblib")
