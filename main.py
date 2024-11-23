from dotenv import load_dotenv, dotenv_values

from data.db import DatabaseConnection
from data.get_data import get_data_frame 

load_dotenv()

db = DatabaseConnection()
print(db.get_libraries())
print(get_data_frame(db, 1))


