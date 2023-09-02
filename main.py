import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import IsolationForest
from __future__ import annotations
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

logging.basicConfig(level=logging.INFO)

class AdvancedDataAnalysisPipeline:
    """
    An advanced data analysis pipeline for visualizing and analyzing trading volume data.

    Attributes:
    file_path (str): Path to the CSV file containing the trading volume data.
    df (pd.DataFrame): DataFrame holding the trading volume data.
    outliers (pd.DataFrame): DataFrame holding the identified outlier data points.
    """

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.df: pd.DataFrame = None
        self.outliers: pd.DataFrame = None
        self.main()

    def load_data(self) -> None:
        """Load the trading volume data from the CSV file."""

        try:
            self.df = pd.read_csv(self.file_path)
            self.df["snapped_at"] = pd.to_datetime(self.df["snapped_at"])
            self.df["volume"] = self.df["volume"].astype(int)
            self.df["date"] = self.df["snapped_at"].dt.date
            self.df["month"] = self.df["snapped_at"].dt.month
            self.df["day_name"] = self.df["snapped_at"].dt.day_name()
            self.df["is_weekend"] = self.df["day_name"].isin(["Saturday", "Sunday"])
            self.df = self.df[["date", "month", "day_name", "is_weekend", "volume"]]
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")

    def feature_engineering(self) -> None:
        """Perform feature engineering on the loaded data, like lag and rolling (moving) average."""

        try:
            # Introduce lag feature
            self.df['lag_volume'] = self.df['volume'].shift(1)
            # Rolling window feature: 7-day rolling average
            self.df['rolling_avg'] = self.df['volume'].rolling(window=7).mean()
        except Exception as e:
            logging.error(f"Error in feature engineering: {str(e)}")

    def plot_data(self, title: str) -> None:
        """
        Plot the trading volume data.

        Args:
        title (str): The title of the plot.
        """

        fig = px.line(self.df, x='date', y='volume', title=title)
        fig.show()

    def detect_outliers(self) -> None:
        """Detect and remove outliers in the trading volume data using Isolation Forest."""
      
        iso = IsolationForest(contamination=0.05)
        self.df['outlier'] = iso.fit_predict(self.df[['volume']].values)
        self.outliers = self.df[self.df['outlier'] == -1]
        self.df = self.df[self.df['outlier'] != -1]
        logging.info(f"Identified and removed {len(self.outliers)} outliers.")
    
    def heatmap_volume(self) -> None:
        """Generate a heatmap visualizing average volume by day and month."""

        try:
            heatmap_data = self.df.groupby(['day_name', 'month'])['volume'].mean().unstack()
            days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            heatmap_data = heatmap_data.reindex(days_order)
            fig = px.imshow(heatmap_data, title='Average Volume by Day and Month')
            fig.show()
        except Exception as e:
            logging.error(f"Error in heatmap volume: {str(e)}")

    def plot_trend(self) -> None:
        """Plot the trading volume along with its trend over time."""
      
        # Using lowess to smooth the curve
        smoothed = lowess(self.df['volume'], np.arange(len(self.df['volume'])), frac=0.1)
        self.df['trend'] = smoothed[:, 1]
        fig = px.line(self.df, x='date', y=['volume', 'trend'], title='Volume with Trend over Time')
        fig.show()

    def plot_volume_distribution(self) -> None:
        """Plot the distribution of trading volumes with percentiles and statistics."""

        # Compute mean, median, and percentiles
        mean_volume = self.df["volume"].mean()
        median_volume = self.df["volume"].median()
        p_05 = self.df["volume"].quantile(0.05)
        p_95 = self.df["volume"].quantile(0.95)
        
        fig = px.histogram(self.df, 
                          x="volume", 
                          nbins=50, 
                          title='Volume Distribution',
                          color="is_weekend",
                          marginal="violin", # can also use 'box', 'rug'
                          hover_data=self.df.columns) # displays all columns when hovering over a bar
        
        # Add vertical lines for mean and median
        fig.add_vline(x=mean_volume, line_dash="dash", line_color="blue", name="Mean")
        fig.add_vline(x=median_volume, line_dash="dash", line_color="green", name="Median")
        
        # Shade potential outliers
        fig.add_vrect(x0=p_05, x1=p_95, fillcolor='rgba(0,0,0,0)', line=dict(color="red", dash="dash"), name="5th-95th Percentile")



        # Annotations for mean, median, and percentiles
        fig.add_annotation(x=mean_volume, y=0.9, yref='paper', text="Mean", showarrow=False)
        fig.add_annotation(x=median_volume, y=0.8, yref='paper', text="Median", showarrow=False)
        fig.add_annotation(x=p_05, y=0.7, yref='paper', text="5th Percentile", showarrow=False)
        fig.add_annotation(x=p_95, y=0.6, yref='paper', text="95th Percentile", showarrow=False)

        fig.show()

    def _set_day_order(self) -> None:
        """Set the order of days for plotting purposes."""

        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        self.df['day_name'] = pd.Categorical(self.df['day_name'], categories=day_order, ordered=True)

    def _set_month_order(self) -> None:
        """Set the order of months for plotting purposes."""


        month_order = ['January', 'February', 'March', 'April',
                      'May', 'June', 'July', 'August',
                      'September', 'October', 'November', 'December']
        month_name_map = dict(enumerate(month_order, 1))
        self.df['month_name'] = self.df['month'].map(month_name_map)
        self.df['month_name'] = pd.Categorical(self.df['month_name'], categories=month_order, ordered=True)


    def plot_daywise_distribution(self) -> None:
        """Plot the distribution of trading volumes for each day of the week."""
      
        self._set_day_order()
        fig = px.box(self.df, x='day_name', y='volume', title='Volume Distribution by Day of Week', points="all")
        fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig.show()

    def plot_daywise_summary(self) -> None:
        """Plot the summary of average trading volumes for each day of the week."""

        self._set_day_order()
        daywise_volume = self.df.groupby('day_name')['volume'].mean().reset_index()
        fig = px.bar(daywise_volume, x='day_name', y='volume', title='Daywise Trading Volume Summary')
        fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
        fig.show()

    def plot_monthwise_distribution(self) -> None:
        """Plot the distribution of trading volumes for each month."""

        self._set_month_order()
        fig = px.box(self.df, x='month_name', y='volume', title='Volume Distribution by Month', points="all")
        fig.update_xaxes(categoryorder='array', categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        fig.show()

    def plot_monthwise_summary(self) -> None:
        """Plot the summary of average trading volumes for each month."""

        self._set_month_order()
        monthly_volume = self.df.groupby('month_name')['volume'].mean().reset_index()
        fig = px.bar(monthly_volume, x='month_name', y='volume', title='Monthly Trading Volume Summary')
        fig.update_xaxes(categoryorder='array', categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
        fig.show()

    def perform_clustering(self) -> None:
        """Perform clustering on the trading volume data and visualize the results."""

        self.df['date_numeric'] = (self.df['date'] - self.df['date'].min()).dt.days
        volume_data = self.df[['date_numeric', 'volume']]
        
        optimal_clusters = 3

        kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
        self.df['cluster'] = kmeans.fit_predict(volume_data)
        
        # Custom color map for clusters
        cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
        
        # Create an empty figure
        fig = go.Figure()

        # Add a scatter plot for each cluster with its respective color
        for i in range(optimal_clusters):
            cluster_data = self.df[self.df['cluster'] == i]
            fig.add_trace(
                go.Scatter(x=cluster_data['date'], y=cluster_data['volume'], 
                          mode='markers', name=f'Cluster {i}', 
                          marker=dict(color=cluster_colors[i]))
            )
            
            if len(cluster_data) >= 3:  # Need at least 3 points to create a convex hull
                hull = ConvexHull(cluster_data[['date_numeric', 'volume']])
                hull_points = hull.vertices.tolist()
                hull_points.append(hull_points[0])  # Close the loop
                fig.add_trace(
                    go.Scatter(x=cluster_data.iloc[hull_points]['date'], y=cluster_data.iloc[hull_points]['volume'], 
                              mode='lines', showlegend=False, 
                              line=dict(color=cluster_colors[i]))
                )

        fig.update_layout(title='Clusters of Volume', xaxis_title='Date', yaxis_title='Volume')
        fig.show()

        # Descriptive statistics for each cluster
        cluster_summaries = self.df.groupby('cluster')['volume'].agg(['mean', 'median', 'std', 'count'])
        
        # Daywise distribution of clusters
        daywise_clusters = self.df.groupby(['day_name', 'cluster']).size().unstack().fillna(0)
        fig_daywise = px.bar(daywise_clusters, title='Daywise Distribution of Clusters', labels={'value': 'Count'})
        # Apply custom colors for bar chart
        for i, color in cluster_colors.items():
            fig_daywise.update_traces(selector=dict(name=str(i)), marker_color=color)
        fig_daywise.show()
        
        # Monthwise distribution of clusters
        monthwise_clusters = self.df.groupby(['month', 'cluster']).size().unstack().fillna(0)
        fig_monthwise = px.bar(monthwise_clusters, title='Monthwise Distribution of Clusters', labels={'value': 'Count'})
        # Apply custom colors for bar chart
        for i, color in cluster_colors.items():
            fig_monthwise.update_traces(selector=dict(name=str(i)), marker_color=color)
        fig_monthwise.show()


    def main(self) -> None:
        """Main execution flow of the pipeline."""

        self.load_data()
        self.feature_engineering()
        self.plot_data('Volume over Time')
        self.detect_outliers()
        self.plot_data('Volume over Time (After Removing Outliers)')
        self.plot_trend()
        self.plot_volume_distribution()
        self.plot_daywise_summary()
        self.plot_daywise_distribution()
        self.plot_monthwise_summary()
        self.plot_monthwise_distribution()
        self.heatmap_volume()
        self.perform_clustering()

if __name__ == "__main__":
    pipeline = AdvancedDataAnalysisPipeline("bitlo-trading-volume-1-year.csv")
