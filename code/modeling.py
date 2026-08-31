"""
Heart Disease Dataset
Modeling
"""

from pathlib import Path
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate
)

# Find the project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "heart.csv"

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

# Preprocessing pipeline
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

# Define classification models
models = {

    "Majority Baseline":
        DummyClassifier(
            strategy="most_frequent"
        ),

    "Logistic Regression":
        LogisticRegression(
            max_iter=2000,
            random_state=42
        ),

    "KNN":
        KNeighborsClassifier(
            n_neighbors=5
        ),

    "SVM":
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        ),

    "Decision Tree":
        DecisionTreeClassifier(
            max_depth=5,
            random_state=42
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ),

    "Extra Trees":
        ExtraTreesClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingClassifier(
            random_state=42
        )
}

# Use stratified 10-fold cross-validation
cv = StratifiedKFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

# Metrics to calculate
scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc"
}

# Store model results
results = []

# Test each model
for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Balanced Accuracy":
            scores["test_balanced_accuracy"].mean(),
        "Precision":
            scores["test_precision"].mean(),
        "Recall":
            scores["test_recall"].mean(),
        "F1": scores["test_f1"].mean(),
        "ROC-AUC":
            scores["test_roc_auc"].mean()
    })


# Create results table
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="ROC-AUC",
    ascending=False
)


# Display results
print("\nModel Comparison")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# Save results
RESULTS_DIR = BASE_DIR / "results"

RESULTS_DIR.mkdir(
    exist_ok=True
)

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False
)

print("\nResults saved to results/model_comparison.csv")
