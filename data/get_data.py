from .db import DatabaseConnection
import pandas as pd


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

def get_library_ids():
    db = DatabaseConnection()
    libraries = db.get_libraries()
    library_ids = [library.id for library in libraries]
    db.close()
    return library_ids