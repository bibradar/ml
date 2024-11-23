from pathlib import Path
from .db import DatabaseConnection
import pandas as pd
import pickle
from gluonts.dataset.common import ListDataset
from gluonts.model.predictor import Predictor
import datetime


def get_data_frame(library_id: int) -> pd.DataFrame:
    db = DatabaseConnection()
    utilizations = db.get_utilizations_by_library(library_id)
    data = pd.DataFrame([utilization.__dict__ for utilization in utilizations])
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    db.close()
    data['user_count'] = data['user_count'].fillna(0)
    return data


def get_max_user_count(aggregated_user_data: pd.DataFrame) -> int:
    sorted_user_counts = aggregated_user_data['user_count'].sort_values(ascending=False)
    top_5_percent_count = int(len(sorted_user_counts) * 0.05)
    top_5_percent_values = sorted_user_counts.head(top_5_percent_count)
    median_value = top_5_percent_values.median()

    return median_value

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

    # Prepare the input data for prediction
    input_data = ListDataset(
        [{"target": df['user_count'].values, "start": pd.Period(start_timestamp, freq=freq)}],
        freq=freq
    )

    forecasts = list(model.predict(input_data))

    forecast_entry = forecasts[0]
    predicted_values = forecast_entry.mean[:prediction_length]  

    # Generate timestamps for the predicted values
    timestamps = pd.date_range(start=start_timestamp, periods=prediction_length, freq=freq)

    # Combine timestamps and predicted values into a list of dictionaries
    predictions_with_timestamps = [{"timestamp": timestamp, "predicted_user_count": value} for timestamp, value in zip(timestamps, predicted_values)]

    return predictions_with_timestamps

def load_model_and_get_prediction(timestamp: str) -> float:
    data = get_data_frame(1)
    pred = Predictor.deserialize(Path("./models"))
    predictions = predict_one_day(pred, data, timestamp)
    for prediction in predictions:
        print(prediction)
        if prediction['timestamp'] == pd.Timestamp(timestamp):
            return prediction['predicted_user_count']

