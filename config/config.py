import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

DATABASE_URI = os.getenv("DATABASE_URI", "postgresql://dillon:@localhost:5434/furniture")
MODEL_PATH = os.getenv("MODEL_PATH", "data/model.pkl")