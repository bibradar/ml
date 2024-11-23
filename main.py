from dotenv import load_dotenv, dotenv_values

from ml.data.db import DatabaseConnection
from ml.data.get_data import get_data_frame 

load_dotenv(dotenv_path='ml/.env')


db = DatabaseConnection()
print(db.get_libraries())
print(get_data_frame(1))


