# Furniture Sales Forecasting

This project predicts future furniture sales using machine learning models.

## Setup Instructions

1. Install Python 3.10.
2. Create a virtual environment:
    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate
    ```
3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4. Run the Flask app:
    ```bash
    python run.py
    ```
5. Run tests:
    ```bash
    pytest
    ```

## Docker Setup

1. Build the Docker container:
    ```bash
    docker-compose build
    ```
2. Start the services:
    ```bash
    docker-compose up
    ```