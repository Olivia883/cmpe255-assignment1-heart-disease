import json
import pandas as pd
import streamlit as st

st.title("Nano LLM Training Dashboard")

with open("metrics.json", "r", encoding="utf-8") as f:
    metrics = json.load(f)

df = pd.DataFrame(metrics)

st.metric("Training Steps", int(df["step"].max()) if len(df) else 0)

if not df.empty:
    st.line_chart(df.set_index("step")["train_loss"])

st.write("This dashboard exposes simple training telemetry for the small model.")
