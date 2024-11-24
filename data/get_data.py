from pathlib import Path
from .db import DatabaseConnection
import pandas as pd
import pickle
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
from dotenv import load_dotenv
import datetime
import torch

TIMEZONE = datetime.datetime.now().astimezone().tzinfo

load_dotenv(dotenv_path=".env")
db = DatabaseConnection()

models = {}

def get_data_frame(library_id: int) -> pd.DataFrame:
    db = DatabaseConnection()
    utilizations = db.get_utilizations_by_library(library_id)
    data = pd.DataFrame([utilization.__dict__ for utilization in utilizations])
    if data.empty:
        return
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    db.close()
    data['user_count'] = data['user_count'].fillna(0)
    return data


def get_max_user_count(library_id: int) -> int:
    db = DatabaseConnection()
    max_count = db.get_max_count_for_library(library_id)
    db.close()
    return max_count


def predict_one_day(model, df, start_timestamp) -> list:
    """
    Predicts the user count for a complete day (96 15-minute intervals) starting from the given timestamp.

    Parameters:
    - model: The trained DeepAREstimator model.
    - df: The DataFrame containing the historical data.
    - start_timestamp: The starting timestamp for the prediction.

    Returns:
    - A list of dictionaries with predicted values and their corresponding timestamps for the next 24 hours (96 timestamps).
    """
    prediction_length = 96  
    freq = "15min" 

    timestamp = datetime.datetime.fromtimestamp(start_timestamp, TIMEZONE).isoformat()

    # Prepare the input data for prediction
    input_data = ListDataset(
        [{"target": df['user_count'].values, "start": pd.Period(timestamp, freq=freq)}],
        freq=freq
    )

    forecasts = list(model.predict(input_data))

    forecast_entry = forecasts[0]
    predicted_values = forecast_entry.mean[:prediction_length]  

    # Generate timestamps for the predicted values
    timestamps = pd.date_range(start=timestamp, periods=prediction_length, freq=freq)

    # Combine timestamps and predicted values into a list of dictionaries
    predictions_with_timestamps = [{"timestamp": timestamp, "predicted_user_count": value} for timestamp, value in zip(timestamps, predicted_values)]

    return predictions_with_timestamps

def get_model(library_id: int):
    if not library_id in models:
        models[library_id] = Predictor.deserialize(Path(f"./models/{library_id}"), device="cpu")

    return models[library_id]

def load_model_and_get_prediction2(start_timestamp: int, library_id: int):
    last_week_timestamp = datetime.datetime.now().replace(hour=0, minute=0, second=0) - datetime.timedelta(days=14)

    data = db.get_one_day(library_id, int(last_week_timestamp.timestamp()))

    data = map(lambda d: {'predicted_user_count': d[0]}, data)

    return list(data)
    

def load_model_and_get_prediction(start_timestamp: int, library_id: int):
    model = get_model(library_id)
    data = get_data_frame(library_id)  
    predictions = predict_one_day(model, data, start_timestamp)

    return predictions


def get_count_from_last_week(start_timestamp: int, library_id: int):
    db = DatabaseConnection()
    last_week_timestamp = datetime.datetime.now().replace(hour=0, minute=0, second=0) - datetime.timedelta(days=7)

    util = db.get_user_count_with_timestamp(library_id, int(last_week_timestamp.timestamp()))
    print(util)
    db.close()
    return util[0]