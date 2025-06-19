import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_curve, auc
from joblib import dump

# Load dataset
df = pd.read_csv("outputs/results.csv")

# Define labels
df["Label"] = df.apply(
    lambda row: "Expert" if row["Avg Cosine Similarity (Expert)"] > row["Avg Cosine Similarity (Novice)"] else "Novice",
    axis=1
)

# Features
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

# Binary encoding for ROC
lb = LabelBinarizer()
y_binary = lb.fit_transform(y).ravel()  # Expert = 1, Novice = 0

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_binary, stratify=y_binary, random_state=42, test_size=0.2)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Report
y_pred = pipeline.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Novice", "Expert"]))

# Cross-validation score
cv_score = cross_val_score(pipeline, X, y_binary, cv=5)
print(f"Mean CV Accuracy: {cv_score.mean():.3f} ± {cv_score.std():.3f}")

# ROC and AUC
y_prob = pipeline.predict_proba(X_test)[:, 1]  # probability of Expert (label=1)
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Logistic Regression Classifier")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

# Save model
dump(pipeline, "models/logreg_classifier_model.joblib")
