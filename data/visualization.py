import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DataVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_quantity_distribution(self):
        fig = px.histogram(
            self.data,
            x='quantity',
            nbins=30,
            title='Distribution of Quantity',
            labels={'quantity': 'Quantity'},
            template='plotly_dark'
        )
        fig.update_layout(xaxis_title='Quantity', yaxis_title='Frequency')
        return fig.to_html(full_html=False)

    def sales_over_time(self):
        self.data['saledate'] = pd.to_datetime(self.data['saledate'])
        daily_sales = self.data.groupby('saledate')['quantity'].sum().reset_index()

        fig = px.line(
            daily_sales,
            x='saledate',
            y='quantity',
            title='Daily Sales Over Time',
            labels={'saledate': 'Date', 'quantity': 'Total Quantity Sold'},
            template='plotly_dark'
        )
        return fig.to_html(full_html=False)

    def categories_vs_sales(self):
        category_sales = self.data.groupby('category')['quantity'].sum().reset_index()

        fig = px.bar(
            category_sales,
            x='quantity',
            y='category',
            title='Total Sales by Product Category',
            labels={'quantity': 'Total Quantity Sold', 'category': 'Category'},
            template='plotly_dark',
            orientation='h'
        )
        return fig.to_html(full_html=False)

    def correlation_heatmap(self):
        corr = self.data.corr()
        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.columns,
                colorscale='Viridis',
                zmin=-1,
                zmax=1
            )
        )
        fig.update_layout(title='Feature Correlation Heatmap', template='plotly_dark')
        return fig.to_html(full_html=False)

    def price_vs_quantity(self):
        fig = px.scatter(
            self.data,
            x='price',
            y='quantity',
            title='Price vs Quantity',
            labels={'price': 'Price', 'quantity': 'Quantity Sold'},
            template='plotly_dark'
        )
        return fig.to_html(full_html=False)


class ModelVisualizer:
    def __init__(self, model, X_train, X_test, y_train, y_test, y_pred):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.y_pred = y_pred

    def model_performance(self):
        fig = px.scatter(
            x=self.y_test,
            y=self.y_pred,
            title='Predicted vs Actual',
            labels={'x': 'Actual Quantity', 'y': 'Predicted Quantity'},
            template='plotly_dark'
        )
        fig.add_shape(
            type="line",
            x0=min(self.y_test),
            y0=min(self.y_test),
            x1=max(self.y_test),
            y1=max(self.y_test),
            line=dict(color="Red", dash="dash"),
        )
        return fig.to_html(full_html=False)

    def residuals(self):
        residuals = self.y_test - self.y_pred
        fig = px.histogram(
            x=residuals,
            nbins=30,
            title='Residual Distribution',
            labels={'x': 'Residuals'},
            template='plotly_dark'
        )
        fig.update_layout(xaxis_title='Residuals', yaxis_title='Frequency')
        return fig.to_html(full_html=False)

    def feature_importance(self):
        feature_importance = pd.DataFrame({
            'Feature': self.X_train.columns,
            'Importance': self.model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            title='Feature Importance',
            labels={'Importance': 'Importance Score', 'Feature': 'Feature'},
            template='plotly_dark',
            orientation='h'
        )
        return fig.to_html(full_html=False)