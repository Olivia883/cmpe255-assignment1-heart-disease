"""
Heart Disease Dataset
Model Evaluation
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

# Find the project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "heart.csv"

# Figure folder
FIGURE_DIR = BASE_DIR / "figures"

FIGURE_DIR.mkdir(
    exist_ok=True
)

# Load the data
df = pd.read_csv(DATA_PATH)

# Remove duplicates
df = df.drop_duplicates().reset_index(drop=True)

# Separate features and target
X = df.drop(columns=["target"])
y = df["target"]

# Numerical features
NUMERICAL_FEATURES = [
    "age",
    "trestbps",
    "chol",
    "thalach",
    "oldpeak"
]

# Categorical features
CATEGORICAL_FEATURES = [
    "sex",
    "cp",
    "fbs",
    "restecg",
    "exang",
    "slope",
    "ca",
    "thal"
]

# Create preprocessing pipeline
preprocessor = ColumnTransformer([
    (
        "numerical",
        StandardScaler(),
        NUMERICAL_FEATURES
    ),
    (
        "categorical",
        OneHotEncoder(handle_unknown="ignore"),
        CATEGORICAL_FEATURES
    )
])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

# Create final Logistic Regression pipeline
model = Pipeline([
    (
        "preprocessing",
        preprocessor
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=2000,
            random_state=42
        )
    )
])

# Train the model
model.fit(
    X_train,
    y_train
)

# Make predictions
y_pred = model.predict(X_test)

y_probability = model.predict_proba(
    X_test
)[:, 1]

# Calculate metrics
accuracy = accuracy_score(
    y_test,
    y_pred
)

balanced_accuracy = balanced_accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred
)

recall = recall_score(
    y_test,
    y_pred
)

f1 = f1_score(
    y_test,
    y_pred
)

roc_auc = roc_auc_score(
    y_test,
    y_probability
)

# Display metrics
print("\nFinal Model Evaluation")
print("=" * 50)
print(f"Accuracy:          {accuracy:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy:.4f}")
print(f"Precision:         {precision:.4f}")
print(f"Recall:            {recall:.4f}")
print(f"F1 Score:          {f1:.4f}")
print(f"ROC-AUC:           {roc_auc:.4f}")


# Classification report
print("\nClassification Report")
print("=" * 50)

print(
    classification_report(
        y_test,
        y_pred
    )
)

# Create confusion matrix
cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No Disease", "Disease"],
    yticklabels=["No Disease", "Disease"]
)

plt.title(
    "Logistic Regression Confusion Matrix"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "confusion_matrix.png",
    dpi=300
)

plt.close()

# Create ROC curve
fig, ax = plt.subplots(
    figsize=(7, 5)
)

RocCurveDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title(
    "Logistic Regression ROC Curve"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "roc_curve.png",
    dpi=300
)

plt.close()

# Create precision-recall curve
fig, ax = plt.subplots(
    figsize=(7, 5)
)

PrecisionRecallDisplay.from_predictions(
    y_test,
    y_probability,
    ax=ax
)

ax.set_title(
    "Logistic Regression Precision-Recall Curve"
)

plt.tight_layout()

plt.savefig(
    FIGURE_DIR / "precision_recall_curve.png",
    dpi=300
)

plt.close()

# Save final metrics
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)

metrics = pd.DataFrame({
    "Metric": [
        "Accuracy",
        "Balanced Accuracy",
        "Precision",
        "Recall",
        "F1 Score",
        "ROC-AUC"
    ],
    "Value": [
        accuracy,
        balanced_accuracy,
        precision,
        recall,
        f1,
        roc_auc
    ]
})

metrics.to_csv(
    RESULTS_DIR / "final_model_metrics.csv",
    index=False
)

print("\nEvaluation figures saved to the figures folder.")
print("Evaluation results saved to the results folder.")
