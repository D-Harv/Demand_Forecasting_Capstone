from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from data.database_connection import load_query
from model.preprocess import clean_and_preprocess_data
import joblib


def train_model(model_path, query, test_query):
    """
    Train a Random Forest model if no model is saved. Use the model for predictions if it exists.

    :param model_path: Path to the saved model file.
    :param query: SQL query to load training data.
    :param test_query: SQL query to load testing data.
    :return: X_train, X_test, y_train, y_test, y_pred
    """
    # Load and preprocess the data
    train = load_query(query)
    test_data = load_query(test_query)
    data = clean_and_preprocess_data(train)
    test = clean_and_preprocess_data(test_data)

    # Define features and target for training and testing
    X_train = data.drop(columns=['quantity'])
    y_train = data['quantity']
    X_test = test.drop(columns=['quantity'])
    y_test = test['quantity']

    try:
        # Try loading an existing model
        generated_model = joblib.load(model_path)
        print(f"Loaded model from {model_path}")
    except FileNotFoundError:
        # Train and save the model if not found
        print("No model found. Training a new model...")
        model = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=35)
        model.fit(X_train, y_train)
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")
        generated_model = model  # Use the newly trained model

    # Make predictions on the test set
    y_pred = generated_model.predict(X_test)
    return X_train, X_test, y_train, y_test, y_pred, generated_model




def model_tester(model, query):
    data = load_query(query)
    test = clean_and_preprocess_data(data)
    X_test = test.drop(columns=['quantity'])
    y_test = test['quantity']
    # Predict on the test set
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"Mean Absolute Error (MAE): {mae}")
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    return mae, mse, r2
