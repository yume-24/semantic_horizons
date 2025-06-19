import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import r2_score, mean_squared_error
from joblib import dump

# Load dataset
df = pd.read_csv("outputs/labeled_results.csv")

# Features and target
features = [
    "Avg Cosine Similarity (Expert)", "Avg Cosine Similarity (Novice)",
    "Avg Dist to Centroid (Expert)", "Avg Dist to Centroid (Novice)",
    "Convex Hull Area (Expert)", "Convex Hull Area (Novice)",
    "Intra-cluster Distance Mean (Expert)", "Intra-cluster Distance Mean (Novice)",
    "Intra-cluster Distance Std (Expert)", "Intra-cluster Distance Std (Novice)",
    "KDE Mean Log Density (Expert)", "KDE Mean Log Density (Novice)"
]

X = df[features]
y = LabelBinarizer().fit_transform(df["Label"]).ravel()  # Expert = 1, Novice = 0

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.2)

# Build pipeline
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("ridge", Ridge(alpha=1.0))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("R^2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Extract and print equation
coefs = pipeline.named_steps["ridge"].coef_
intercept = pipeline.named_steps["ridge"].intercept_

print("\n📈 Ridge Regression Equation:")
equation = f"y = {intercept:.3f}"
for feat, coef in zip(features, coefs):
    sign = "+" if coef >= 0 else "-"
    equation += f" {sign} {abs(coef):.3f} * ({feat})"
print(equation)

# Save model
dump(pipeline, "models/ridge_regression_model.joblib")
