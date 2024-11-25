from flask import Flask, Blueprint, render_template
from data.visualization import DataVisualizer, ModelVisualizer
from model.train_model import train_model
import pandas as pd
from model.preprocess import label_encode_data

bp = Blueprint('main', __name__)


@bp.route('/')
def index():
    return render_template('index.html')

@bp.route('/quantity-distribution')
def quantity_distribution():
    data = pd.read_csv('./data.csv')
    visualizer = DataVisualizer(data)
    plot_html = visualizer.plot_quantity_distribution()
    return render_template('plot.html', plot_html=plot_html)

@bp.route('/model-performance')
def model_performance():
    # Load data, predictions, and model
    # Example placeholders:
    X_train, X_test, y_train, y_test, y_pred, model = train_model('./model/model.pkl', './data/query.sql', './data/test.sql')
    visualizer = ModelVisualizer(model, X_train, X_test, y_train, y_test, y_pred)
    plot_html = visualizer.model_performance()
    return render_template('plot.html', plot_html=plot_html)

@bp.route('/historical-data')
def historical_data():
    # Load data
    data = pd.read_csv('./model/sales_data.csv')
    sales_data = pd.read_csv('./model/sales_data.csv')
    data.drop(columns=['id'], inplace=True)
    data.index.name = 'id'# Replace with your dataset
    visualizer = DataVisualizer(data)
    sales_data.drop(columns=['id'], inplace=True)
    sales_data.index.name = 'id'  # Replace with your dataset
    sales_data['saledate'] = pd.to_datetime(sales_data['saledate']).dt.date
    sales_data['year'] = sales_data['saledate'].apply(lambda x: x.year)
    sales_data['month'] = sales_data['saledate'].apply(lambda x: x.month)
    sales_data['day'] = sales_data['saledate'].apply(lambda x: x.day)
    sales_data = label_encode_data('product', 'product', sales_data)
    sales_data = label_encode_data('isonsale', 'isonsale', sales_data)
    sales_data = label_encode_data('replenish', 'replenish', sales_data)
    sales_data = label_encode_data('delivered', 'delivered', sales_data)
    sales_data = label_encode_data('category', 'category', sales_data)
    sales_data.drop(columns=['saledate'], inplace=True)
    numbers_only_visualizer = DataVisualizer(sales_data)

    # Generate plots
    plots = {
        'quantity_distribution': {
            'title': 'Quantity Distribution',
            'description': 'This graph shows the distribution of quantity sold, highlighting common ranges and outliers.',
            'html': visualizer.plot_quantity_distribution()
        },
        'sales_over_time': {
            'title': 'Sales Over Time',
            'description': 'This graph shows how sales quantities have changed over time, highlighting trends and seasonality.',
            'html': visualizer.sales_over_time()
        },
        'categories_vs_sales': {
            'title': 'Category Sales',
            'description': 'This graph compares the total sales of each product category, showing which categories perform best.',
            'html': visualizer.categories_vs_sales()
        },
        'correlation_heatmap': {
            'title': 'Correlation Heatmap',
            'description': 'This heatmap shows the correlation between numerical features in the dataset.',
            'html': numbers_only_visualizer.correlation_heatmap()
        }
    }

    return render_template('historical-data.html', plots=plots)

