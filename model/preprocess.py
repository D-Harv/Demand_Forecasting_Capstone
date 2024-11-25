from data.database_connection import load_data
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def clean_and_preprocess_data(query):
    sales_data = load_data(query)

    kvp = {"LCH": 'Living Room Chair', "DCH": 'Dining Chair', "DTB": 'Dining Table', "OCC": 'Occasional Table',
           "BDR": 'Bedroom', "SOF": 'Sofa', "SEC": 'Sectional', "DEC": 'Decoration'}

    # Map categories
    sales_data['category'] = sales_data['product'].apply(lambda product: kvp.get(product[:3], 'Unknown'))

    # Label encode 'product'
    sales_data = label_encode_data('product', 'product', sales_data)

    # One-hot encode 'category'
    sales_data = one_hot_encode_data('category', 'cat', sales_data)

    sales_data = label_encode_data('isonsale', 'isonsale', sales_data)

    sales_data = label_encode_data('replenish', 'replenish', sales_data)

    sales_data = label_encode_data('delivered', 'delivered', sales_data)

    sales_data['saledate'] = pd.to_datetime(sales_data['saledate']).dt.date

    sales_data = sales_data.drop('category', axis=1)

    sales_data = sales_data.drop('id', axis=1)

    sales_data = sales_data.drop('inventoryatpurchase', axis=1)

    sales_data['realizedprofit'] = sales_data['realizedprofit'] * sales_data['quantity']

    sales_data.rename(columns={'potentialprofit': 'price'}, inplace=True)

    sales_data.rename(columns={'realizedprofit': 'revenue'}, inplace=True)

    sales_data.index.name = 'salenum'

    sales_data['year'] = sales_data['saledate'].apply(lambda x: x.year)
    sales_data['month'] = sales_data['saledate'].apply(lambda x: x.month)
    sales_data['day'] = sales_data['saledate'].apply(lambda x: x.day)

    # Drop the original 'saledate' column
    sales_data.drop(columns=['saledate'], inplace=True)

    sales_data['po'] = sales_data['po'].fillna(0)
    
    return sales_data


def label_encode_data(column, enc_column, sales_data):
    le = LabelEncoder()

    sales_data[enc_column] = le.fit_transform(sales_data[column])

    return sales_data



def one_hot_encode_data(column, prefix, sales_data):
    """
    One-hot encodes a categorical column in the DataFrame.

    :param column: Name of the column to encode.
    :param prefix: Prefix for the new one-hot encoded columns.
    :param sales_data: DataFrame containing the data.
    :return: DataFrame with the one-hot encoded columns added.
    """
    # Create OneHotEncoder instance
    oh = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Perform one-hot encoding
    encoded = oh.fit_transform(sales_data[[column]])

    # Generate new column names
    column_names = [f"{prefix}_{category}" for category in oh.categories_[0]]

    # Create a new DataFrame with encoded values as integers
    encoded_df = pd.DataFrame(encoded.astype(int), columns=column_names, index=sales_data.index)

    # Concatenate the original DataFrame with the encoded DataFrame
    sales_data = pd.concat([sales_data, encoded_df], axis=1)

    return sales_data

