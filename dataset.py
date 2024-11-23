from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
import pandas as pd

# db_config = load from env file  

connection_string = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"

engine = create_engine(connection_string)

query = """
SELECT *
FROM 
"""

data = pd.read_sql(query, engine)