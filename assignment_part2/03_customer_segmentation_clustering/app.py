from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.title("Customer Intelligence & Segmentation")

path = Path("data/Mall_Customers.csv")

if not path.exists():
    st.error("Place Mall_Customers.csv in data/.")
    st.stop()

df = pd.read_csv(path)

st.subheader("Data Understanding")
st.write(df.head())
st.write("Shape:", df.shape)

features = [
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

X = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

scores = {}
for k in range(2, 7):
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    labels = km.fit_predict(X_scaled)
    scores[k] = silhouette_score(X_scaled, labels)

best_k = max(scores, key=scores.get)

st.subheader("Cluster Evaluation")
st.write("Silhouette scores:", scores)
st.success(f"Selected K = {best_k}")

model = KMeans(n_clusters=best_k, n_init=20, random_state=42)
labels = model.fit_predict(X_scaled)

pca = PCA(n_components=2)
X_2d = pca.fit_transform(X_scaled)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(X_2d[:, 0], X_2d[:, 1], c=labels)
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.set_title("Customer Clusters")
st.pyplot(fig)

df_result = df.loc[X.index].copy()
df_result["cluster"] = labels

st.subheader("Clustered Customers")
st.dataframe(df_result)
