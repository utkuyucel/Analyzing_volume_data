from __future__ import annotations
import pandas as pd
import numpy as np
import requests
import logging
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from datetime import datetime

logging.basicConfig(level=logging.INFO)


class DataExtractor:
  def __init__(self):
    self.raw_data = None


  def get_raw_data_from_api(self, endpoint: str) -> pd.DataFrame:
    # API data with timestamps and volumes
    self.endpoint = endpoint

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(self.endpoint, headers=headers)
    raw_data = response.json()["volumes"]

    self.raw_data = raw_data
    return raw_data



class DataTransformer:
  def __init__(self, data: pd.DataFrame):
    self.df = data
    self.outliers = None

  def transform_timestamps(self):
    # Transforming timestamps into dates
    processed_data = [(self._timestamp_to_date(item[0]), item[1].split(".")[0]) for item in self.df]
    self.df = pd.DataFrame(processed_data, columns = ["snapped_at", "volume"])
    return self.df


  def _timestamp_to_date(self, timestamp) -> dt.Datetime:
    # Convert milliseconds to seconds
    timestamp_in_seconds = timestamp / 1000
    dt_object = datetime.utcfromtimestamp(timestamp_in_seconds)
    date_str = dt_object.strftime("%Y-%m-%d")
    return date_str

  def extract_data(self) -> None:
    """Load the trading volume data from the CSV file."""

    try:
        self.df["snapped_at"] = pd.to_datetime(self.df["snapped_at"])
        self.df["volume"] = self.df["volume"].astype(int)
        self.df["date"] = self.df["snapped_at"].dt.date
        self.df["month"] = self.df["snapped_at"].dt.month
        self.df["day_name"] = self.df["snapped_at"].dt.day_name()
        self.df["is_weekend"] = self.df["day_name"].isin(["Saturday", "Sunday"])
        self.df = self.df[["date", "month", "day_name", "is_weekend", "volume"]]
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")

  def detect_outliers(self) -> None:
    """Detect and remove outliers in the trading volume data using Isolation Forest."""

    iso = IsolationForest(contamination=0.05)
    self.df['outlier'] = iso.fit_predict(self.df[['volume']].values)
    self.outliers = self.df[self.df['outlier'] == -1]
    self.df = self.df[self.df['outlier'] != -1]
    logging.info(f"Identified and removed {len(self.outliers)} outliers.")

  def transform(self) -> None:
    self.transform_timestamps()
    self.extract_data()
    self.detect_outliers()
    return self.df

class DataLoader:
  def __init__(self, data : pd.DataFrame):
    self.df = data

  def save_to_csv(self, file_path: str) -> None:
    self.df.to_csv(file_path)


class DataAnalyzer:
  def __init__(self, data : pd.DataFrame):
    self.df = data
  
  def plot_data(self, title: str) -> None:
    """
    Plot the trading volume data.

    Args:
    title (str): The title of the plot.
    """

    fig = px.line(self.df, x='date', y='volume', title=title)
    fig.show()

  def heatmap_volume(self) -> None:
      """Generate a heatmap visualizing average volume by day and month."""

      heatmap_data = self.df.groupby(['day_name', 'month'])['volume'].mean().unstack()
      days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
      heatmap_data = heatmap_data.reindex(days_order)
      
      # Create the heatmap with annotations
      fig = go.Figure(go.Heatmap(
          z=heatmap_data.values,
          x=heatmap_data.columns,
          y=heatmap_data.index,
          colorscale='Viridis',
          hoverongaps=False,
          hoverinfo='z',
          zauto=False,
          zmax=heatmap_data.values.max(),
          zmin=heatmap_data.values.min(),
          showscale=True
      ))

      # Add annotations with a '$' symbol at the end
      for i, row in enumerate(heatmap_data.values):
          for j, value in enumerate(row):
              fig.add_annotation(go.layout.Annotation(
                  text=f"{round(value, 2)}$", 
                  x=heatmap_data.columns[j], 
                  y=days_order[i], 
                  xref='x1', 
                  yref='y1', 
                  showarrow=False,
                  font=dict(color="white" if value > (heatmap_data.values.max() / 2) else "black")
              ))
      
      fig.update_layout(title='Average Volume by Day and Month', xaxis=dict(title='Month'), yaxis=dict(title='Day'))
      fig.show()

    
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
                      marginal="violin", 
                      hover_data=self.df.columns) 
    
    fig.add_vline(x=mean_volume, line_dash="dash", line_color="blue", name="Mean")
    fig.add_vline(x=median_volume, line_dash="dash", line_color="green", name="Median")
    
    # Shading Outliers
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
    
    # Custom color map for clusters (Every cluster must be different color in a plot but each one must be same color in the whole analysis process.)
    cluster_colors = {0: 'red', 1: 'blue', 2: 'green'}
    
    # Create an empty figure
    fig = go.Figure()

    # Add a scatter plot for each cluster with its color
    for i in range(optimal_clusters):
        cluster_data = self.df[self.df['cluster'] == i]
        fig.add_trace(
            go.Scatter(x=cluster_data['date'], y=cluster_data['volume'], 
                      mode='markers', name=f'Cluster {i}', 
                      marker=dict(color=cluster_colors[i]))
        )
        
        if len(cluster_data) >= 3:  # Convex Hull -> Minimum 3 points
            hull = ConvexHull(cluster_data[['date_numeric', 'volume']])
            hull_points = hull.vertices.tolist()
            hull_points.append(hull_points[0])  
            fig.add_trace(
                go.Scatter(x=cluster_data.iloc[hull_points]['date'], y=cluster_data.iloc[hull_points]['volume'], 
                          mode='lines', showlegend=False, 
                          line=dict(color=cluster_colors[i]))
            )

    fig.update_layout(title='Clusters of Volume', xaxis_title='Date', yaxis_title='Volume')
    fig.show()

    cluster_summaries = self.df.groupby('cluster')['volume'].agg(['mean', 'median', 'std', 'count'])
    
    # Daywise distribution of clusters
    daywise_clusters = self.df.groupby(['day_name', 'cluster']).size().unstack().fillna(0)
    daywise_total = daywise_clusters.sum(axis=1)
    daywise_percentage = (daywise_clusters.divide(daywise_total, axis=0) * 100).round(2).astype(str) + '%'

    fig_daywise = go.Figure()
    for i in range(optimal_clusters):
        y_cumulative = daywise_clusters.iloc[:, :i].sum(axis=1)  # cumulative height up to the cluster of interest
        fig_daywise.add_trace(
            go.Bar(
                x=daywise_clusters.index, 
                y=daywise_clusters[i],
                name=f'Cluster {i}',
                marker_color=cluster_colors[i],
                text=daywise_percentage[i],
                textposition='inside',
                insidetextanchor='middle',
            )
        )
    fig_daywise.update_layout(title='Daywise Distribution of Clusters', xaxis_title='Day', yaxis_title='Count')
    fig_daywise.show()

    # Monthwise distribution of clusters
    monthwise_clusters = self.df.groupby(['month', 'cluster']).size().unstack().fillna(0)
    monthwise_total = monthwise_clusters.sum(axis=1)
    monthwise_percentage = (monthwise_clusters.divide(monthwise_total, axis=0) * 100).round(2).astype(str) + '%'

    fig_monthwise = go.Figure()
    for i in range(optimal_clusters):
        y_cumulative = monthwise_clusters.iloc[:, :i].sum(axis=1)
        fig_monthwise.add_trace(
            go.Bar(
                x=monthwise_clusters.index, 
                y=monthwise_clusters[i],
                name=f'Cluster {i}',
                marker_color=cluster_colors[i],
                text=monthwise_percentage[i],
                textposition='inside',
                insidetextanchor='middle',
            )
        )
    fig_monthwise.update_layout(title='Monthwise Distribution of Clusters', xaxis_title='Month', yaxis_title='Count')
    fig_monthwise.show()
  
  def plot_weekday_vs_weekend_monthly_averages(self) -> None:
      """Plot the average trading volumes for weekdays vs weekends for each month with percentage annotations."""
      
      # Determine if it's a weekend
      self.df['is_weekend'] = self.df['day_name'].isin(['Saturday', 'Sunday'])
      
      # Calculate monthly average trading volume
      monthly_averages = self.df.groupby(['month', 'is_weekend'])['volume'].mean().reset_index()
      
      # Add month names
      month_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
      month_name_map = dict(enumerate(month_order, 1))
      monthly_averages['month_name'] = monthly_averages['month'].map(month_name_map)
      
      # Calculate percentage values
      total_volumes = monthly_averages.groupby('month_name')['volume'].sum()
      monthly_averages['percentage'] = monthly_averages.apply(lambda row: (row['volume'] / total_volumes[row['month_name']]) * 100, axis=1)
      monthly_averages['text'] = monthly_averages['percentage'].round(2).astype(str) + '%'
      
      # Visualize the averages
      fig = px.bar(monthly_averages, 
                  x='month_name', 
                  y='volume', 
                  color='is_weekend',
                  title='Average Trading Volume by Month (Weekday vs Weekend)',
                  labels={'is_weekend': 'Is Weekend?', 'volume': 'Average Volume'},
                  text='text'  # Add the percentage values
                  )
      fig.update_xaxes(categoryorder='array', categoryarray=month_order)
      fig.update_traces(textposition='inside')
      fig.show()



  def perform_eda(self) -> None:
    self.plot_data("Volume over Time (Removed The Outliers)")
    self.plot_trend()
    self.plot_volume_distribution()
    self.plot_daywise_summary()
    self.plot_daywise_distribution()
    self.plot_monthwise_summary()
    self.plot_weekday_vs_weekend_monthly_averages()
    self.plot_monthwise_distribution()
    self.heatmap_volume()
    self.perform_clustering()

  
  def save_to_csv(self, file_path: str) -> None:
    self.df.to_csv(file_path)


if __name__ == "__main__":
  ENDPOINT = "https://www.coingecko.com/exchanges/968/usd/1_year.json?locale=en"
  raw_data = DataExtractor().get_raw_data_from_api(ENDPOINT)
  transformed_data = DataTransformer(raw_data).transform()
  loader = DataLoader(transformed_data).save_to_csv("raw_data.csv")
  analyzer = DataAnalyzer(transformed_data)
  analyzer.perform_eda()
  analyzer.save_to_csv("analyzed.csv")
