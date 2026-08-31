"""
Heart Disease Dataset
Data Preparation
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder


# Find the project folder
BASE_DIR = Path(__file__).resolve().parents[1]

# Dataset location
DATA_PATH = BASE_DIR / "data" / "heart.csv"


# Load the dataset
def load_data():
    return pd.read_csv(DATA_PATH)


# Check the dataset
def inspect_data(df):
    print("\nDataset shape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nTarget distribution:")
    print(df["target"].value_counts())


# Remove duplicate rows
def remove_duplicates(df):
    df = df.drop_duplicates().reset_index(drop=True)
    return df


# Separate features and target
def split_features_target(df):
    X = df.drop(columns=["target"])
    y = df["target"]

    return X, y


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
def create_preprocessor():

    numerical_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            NUMERICAL_FEATURES
        ),
        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_FEATURES
        )
    ])

    return preprocessor


# Split the data
def create_train_test_split(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    return X_train, X_test, y_train, y_test


# Run the preparation steps
if __name__ == "__main__":

    df = load_data()

    inspect_data(df)

    df = remove_duplicates(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = create_train_test_split(
        X,
        y
    )

    preprocessor = create_preprocessor()

    print("\nData preparation completed.")
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
