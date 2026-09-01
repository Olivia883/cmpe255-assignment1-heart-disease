from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

DATA = Path("data/train.csv")
MODEL_PATH = Path("model.joblib")

df = pd.read_csv(DATA)

df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
df = df.dropna(subset=[
    "pickup_datetime",
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "trip_duration"
])

df["hour"] = df["pickup_datetime"].dt.hour
df["weekday"] = df["pickup_datetime"].dt.weekday
df["month"] = df["pickup_datetime"].dt.month

df["distance"] = np.sqrt(
    (df["dropoff_longitude"] - df["pickup_longitude"]) ** 2 +
    (df["dropoff_latitude"] - df["pickup_latitude"]) ** 2
)

features = [
    "passenger_count",
    "pickup_longitude",
    "pickup_latitude",
    "dropoff_longitude",
    "dropoff_latitude",
    "hour",
    "weekday",
    "month",
    "distance"
]

# Limit rows for laptop-friendly training.
df = df.sample(min(len(df), 100000), random_state=42)

X = df[features]
y = np.log1p(df["trip_duration"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

model = RandomForestRegressor(
    n_estimators=150,
    max_depth=18,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)
pred_log = model.predict(X_test)
pred = np.expm1(pred_log)
actual = np.expm1(y_test)

print("MAE (seconds):", mean_absolute_error(actual, pred))
print("R2:", r2_score(actual, pred))

joblib.dump(
    {"model": model, "features": features},
    MODEL_PATH
)

print("Saved:", MODEL_PATH)
