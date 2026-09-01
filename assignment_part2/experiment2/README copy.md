# 01 — NYC Taxi Trip Prediction

## Goal
Reproduce the end-to-end taxi prediction experiment with CRISP-DM,
training, deployment, and an interactive trip-estimation interface.

## Input
Place the Kaggle NYC Taxi dataset in `data/train.csv`.

Expected columns include:
- pickup_datetime
- passenger_count
- pickup_longitude
- pickup_latitude
- dropoff_longitude
- dropoff_latitude
- trip_duration

## Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The dashboard shows CRISP-DM findings, model metrics, and trip
estimates. The implementation is intentionally compute-friendly.
