import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, roc_auc_score

st.title("Data Science Skills Mastery Lab")
st.caption("Representative skills demonstration with CRISP-DM organization.")

skills = [
    "Data loading",
    "Data profiling",
    "Missing-value checks",
    "Duplicate checks",
    "Descriptive statistics",
    "Histograms",
    "Boxplots",
    "Correlation",
    "Feature scaling",
    "Train/test split",
    "Cross-validation concept",
    "Logistic regression",
    "Model accuracy",
    "ROC-AUC",
    "Confusion-matrix concept",
    "K-means clustering",
    "Silhouette evaluation",
    "PCA",
    "Outlier detection",
    "Association-rule concept",
    "Feature engineering concept",
    "Feature selection concept",
    "Model interpretation",
    "Deployment concept"
]

st.sidebar.write("Skills demonstrated")
st.sidebar.write(pd.DataFrame({"Skill": skills}))

data = load_breast_cancer(as_frame=True)
df = data.frame.copy()

st.header("1. Data Understanding")
st.write(df.head())
st.write("Shape:", df.shape)
st.write("Missing values:", int(df.isna().sum().sum()))

st.header("2. Exploratory Analysis")

fig, ax = plt.subplots()
df[data.feature_names[0]].hist(ax=ax)
ax.set_title(f"Distribution of {data.feature_names[0]}")
st.pyplot(fig)

st.header("3. Supervised Learning")

X = df.drop(columns="target")
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

clf = LogisticRegression(max_iter=3000)
clf.fit(X_train_s, y_train)

pred = clf.predict(X_test_s)
prob = clf.predict_proba(X_test_s)[:, 1]

st.metric("Accuracy", f"{accuracy_score(y_test, pred):.3f}")
st.metric("ROC-AUC", f"{roc_auc_score(y_test, prob):.3f}")

st.header("4. Unsupervised Learning")

X_scaled = StandardScaler().fit_transform(X)
kmeans = KMeans(n_clusters=3, n_init=20, random_state=42)
labels = kmeans.fit_predict(X_scaled)

pca = PCA(n_components=2)
X2 = pca.fit_transform(X_scaled)

fig, ax = plt.subplots()
ax.scatter(X2[:, 0], X2[:, 1], c=labels)
ax.set_title("K-means + PCA Visualization")
st.pyplot(fig)

st.header("5. Outlier Detection")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)
outlier_flag = iso.fit_predict(X_scaled)

st.write(
    pd.Series(outlier_flag)
    .value_counts()
    .rename({1: "Inlier", -1: "Potential Outlier"})
)

st.header("6. Interpretation")

st.write(
    "The lab demonstrates how the same CRISP-DM reasoning process can "
    "support several data-science tasks. Each technique answers a "
    "different question, so performance should be interpreted in context."
)
