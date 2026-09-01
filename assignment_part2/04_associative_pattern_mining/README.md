# 04 — Associative Pattern Mining

## Goal
Reproduce the professor's market-basket association-rule experiment
using a popular transaction dataset.

## Input
Place a transaction dataset in `data/transactions.csv` with one
transaction per row and an `items` column containing comma-separated items.

Example:
```text
items
bread,milk,eggs
bread,coffee
milk,eggs,cereal
```

## Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app calculates support, confidence, lift, and frequent itemsets.
