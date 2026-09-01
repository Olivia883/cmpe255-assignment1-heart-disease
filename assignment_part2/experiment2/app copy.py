import joblib
import numpy as np
import pandas as pd
import streamlit as st

bundle = joblib.load("model.joblib")
model = bundle["model"]
features = bundle["features"]

st.title("NYC Taxi Trip Duration Predictor")
st.caption("Student replication of the professor-provided NYC taxi prompt")

st.subheader("Trip Estimation")

passengers = st.number_input("Passengers", 1, 8, 1)
pickup_lat = st.number_input("Pickup Latitude", value=40.75)
pickup_lon = st.number_input("Pickup Longitude", value=-73.98)
drop_lat = st.number_input("Dropoff Latitude", value=40.76)
drop_lon = st.number_input("Dropoff Longitude", value=-73.97)
hour = st.slider("Pickup Hour", 0, 23, 12)
weekday = st.slider("Weekday", 0, 6, 2)
month = st.slider("Month", 1, 12, 6)

distance = np.sqrt(
    (drop_lon - pickup_lon) ** 2 +
    (drop_lat - pickup_lat) ** 2
)

row = pd.DataFrame([{
    "passenger_count": passengers,
    "pickup_longitude": pickup_lon,
    "pickup_latitude": pickup_lat,
    "dropoff_longitude": drop_lon,
    "dropoff_latitude": drop_lat,
    "hour": hour,
    "weekday": weekday,
    "month": month,
    "distance": distance
}])[features]

if st.button("Estimate Trip"):
    duration_seconds = float(np.expm1(model.predict(row)[0]))
    st.metric("Estimated Duration (minutes)",
              f"{duration_seconds / 60:.1f}")

st.subheader("Simple Map")
st.map(pd.DataFrame({
    "lat": [pickup_lat, drop_lat],
    "lon": [pickup_lon, drop_lon]
}))
