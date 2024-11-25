import joblib
import pandas as pd
from data.database_connection import load_data, load_query



def load_model(model_path):
    """Load the trained model from disk."""
    return joblib.load(model_path)

def make_forecast(model, data):
    """Make predictions using the trained model."""
    return model.predict(data)


data = pd.read_csv('sales_data.csv')

kvp = {"LCH": 'Living Room Chair', "DCH": 'Dining Chair', "DTB": 'Dining Table', "OCC": 'Occasional Table',
           "BDR": 'Bedroom', "SOF": 'Sofa', "SEC": 'Sectional', "DEC": 'Decoration'}

data['category'] = data['product'].apply(lambda product: kvp.get(product[:3], 'Unknown'))

data.to_csv('sales_data.csv')