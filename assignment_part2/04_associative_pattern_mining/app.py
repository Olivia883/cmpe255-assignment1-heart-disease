from pathlib import Path
import pandas as pd
import streamlit as st

from mlxtend.frequent_patterns import apriori, association_rules

st.title("Market Basket Pattern Mining")

path = Path("data/transactions.csv")

if not path.exists():
    st.error("Place transactions.csv in data/.")
    st.stop()

df = pd.read_csv(path)
transactions = df["items"].fillna("").str.split(",")

items = sorted({
    item.strip()
    for row in transactions
    for item in row
    if item.strip()
})

matrix = pd.DataFrame(False, index=range(len(transactions)), columns=items)

for i, row in enumerate(transactions):
    for item in row:
        item = item.strip()
        if item in matrix.columns:
            matrix.loc[i, item] = True

min_support = st.slider("Minimum support", 0.01, 0.50, 0.05, 0.01)

frequent = apriori(
    matrix,
    min_support=min_support,
    use_colnames=True
)

st.subheader("Frequent Itemsets")
st.dataframe(frequent.sort_values("support", ascending=False))

if len(frequent):
    rules = association_rules(
        frequent,
        metric="lift",
        min_threshold=1.0
    )

    if len(rules):
        rules = rules.sort_values(
            ["lift", "confidence"],
            ascending=False
        )

        st.subheader("Association Rules")
        st.dataframe(
            rules[
                ["antecedents", "consequents",
                 "support", "confidence", "lift"]
            ]
        )
    else:
        st.info("No rules met the selected threshold.")
