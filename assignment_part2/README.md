# 02 — Nano LLM Transformer

## Goal
Build a small autoregressive transformer/chatbot that can train on a
laptop-friendly corpus and expose training metrics.

## Run
```bash
pip install -r requirements.txt
python train.py
python chat.py
```

Optional:
```bash
streamlit run dashboard.py
```

The model is deliberately tiny so the experiment remains practical in a
limited-compute environment.