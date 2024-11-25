import pandas as pd
from sqlalchemy import create_engine
from config.config import DATABASE_URI

# Create a SQLAlchemy engine
engine = create_engine(DATABASE_URI)


def load_data(query):
    """
    Execute a SQL query and return the results as a pandas DataFrame.

    :param query: The SQL query to execute.
    :return: A pandas DataFrame containing the query results.
    """
    try:
        # Execute query using SQLAlchemy engine
        with engine.connect() as connection:
            data = pd.read_sql_query(query, connection)
        print("Data loaded successfully.")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


# Read SQL query from an external file
def load_query(file_path):
    """
    Load an SQL query from a file.

    :param file_path: Path to the SQL file.
    :return: The SQL query as a string.
    """
    try:
        with open(file_path, 'r') as file:
            query = file.read()
        return query
    except Exception as e:
        print(f"Error reading SQL file: {e}")
        return None