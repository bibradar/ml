from ml.data.db import DatabaseConnection
import pandas as pd


def get_data_frame(db: DatabaseConnection, library_id: int) -> pd.DataFrame:
    utilizations = db.get_utilizations_by_library(library_id)
    data = pd.DataFrame([utilization.__dict__ for utilization in utilizations])
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    return data