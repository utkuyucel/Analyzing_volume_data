from __future__ import annotations
from typing import Tuple, List
from scipy.stats import pearsonr
import statsmodels.api as sm
import pandas as pd
import yfinance as yf
import numpy as np
import json
import requests
import logging
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from prophet import Prophet
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from plotly.subplots import make_subplots
from scipy.spatial import ConvexHull
from datetime import datetime

logging.basicConfig(filename = "pipeline.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class DataExtractor:
  def __init__(self):
    self.raw_data = None


  def get_raw_data_from_coingecko(self, endpoint: str) -> List[List[Union[int, str]]]:
    # API data with timestamps and volumes
    self.endpoint = endpoint

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }

    response = requests.get(self.endpoint, headers=headers)
    raw_data = response.json()["volumes"]

    self.raw_data = raw_data
    return self.raw_data

  def get_manual_from_coingecko_as_json(self, json_file: str) -> List[List[Union[int, str]]]:
    with open (json_file, "r") as file:
      data = json.load(file)["volumes"]

    self.raw_data = data
    return self.raw_data

class DataTransformer:
  def __init__(self, data: pd.DataFrame):
    self.df = data
    self.outliers = None
    self.btc_data = None
    self.start_date = None
    self.start_date = None
    self.vix_data = None


  def _get_btc_data(self):
    self.start_date = self.df["date"].min()
    self.end_date = self.df["date"].max()

    ticker = yf.Ticker("BTC-USD")
    historical_data = ticker.history(start = self.start_date, end = self.end_date).reset_index()
    historical_data.rename(columns = {"Date": "date", "Close": "btc_price"}, inplace = True)
    historical_data['date'] = historical_data['date'].dt.tz_localize(None)
    return historical_data[["date", "btc_price"]]

  def _get_vix_data(self):
    # 1. Fetch VIX data for the desired time range.
    ticker = yf.Ticker("^VIX")
    vix_data = ticker.history(start=self.start_date, end=self.end_date).reset_index()
    vix_data.rename(columns={"Date": "date", "Close": "vix_close"}, inplace=True)
    vix_data['date'] = vix_data['date'].dt.tz_localize(None)
    
    # Create a DataFrame with all dates including weekends
    all_dates_df = pd.DataFrame({
        'date': pd.date_range(start=self.start_date, end=self.end_date)
    })

    # Left merge to ensure all dates are present
    vix_data = pd.merge(all_dates_df, vix_data, on='date', how='left')
    
    # Backward fill for missing values
    vix_data['vix_close'].fillna(method='ffill', inplace=True)
    vix_data['vix_close'].fillna(method='bfill', inplace=True)
    
    return vix_data[["date", "vix_close"]]


  def transform_timestamps_from_coingecko(self):
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

  def transform_dataframe_from_coingecko(self) -> None:
    """Load the trading volume data from the CSV file."""

    try:
        self.df["snapped_at"] = pd.to_datetime(self.df["snapped_at"])
        self.df["volume"] = self.df["volume"].astype(int)
        self.df["date"] = self.df["snapped_at"].dt.date
        self.df["month"] = self.df["snapped_at"].dt.month
        self.df["day_name"] = self.df["snapped_at"].dt.day_name()
        self.df["is_weekend"] = self.df["day_name"].isin(["Saturday", "Sunday"])
        self.df = self.df[["date", "month", "day_name", "is_weekend", "volume"]]
        # Ensure 'snapped_at' column is of type datetime64[ns] right after creation
        self.df['date'] = pd.to_datetime(self.df['date'])
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")

  def detect_outliers(self) -> None:
    """Detect and remove outliers in the trading volume data using Prophet."""


    df_prophet = self.df[['date', 'volume']]
    df_prophet.columns = ['ds', 'y']


    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)


    forecast = model.predict(df_prophet)

    residuals = self.df['volume'] - forecast['yhat']
    threshold = 3 * residuals.std()
    outliers = (residuals.abs() > threshold).values

    self.outliers = self.df[outliers]
    self.df = self.df[~outliers]
    logging.info(f"Identified and removed {len(self.outliers)} outliers.")



  def transform(self) -> None:
    self.transform_timestamps_from_coingecko()
    self.transform_dataframe_from_coingecko()
    self.detect_outliers()

    self.btc_data = self._get_btc_data()
    self.vix_data = self._get_vix_data()  # Fetch VIX data

    # Merge BTC, VIX, and Df
    self.merged_data = pd.merge(self.df, self.btc_data, on="date", how="inner")
    self.merged_data = pd.merge(self.merged_data, self.vix_data, on="date", how="left")  # Merge with VIX

    return self.merged_data


class DataLoader:
  def __init__(self, data : pd.DataFrame):
    self.df = data

  def save_to_csv(self, file_path: str) -> None:
    self.df.to_csv(file_path)


class DataAnalyzer:
  def __init__(self):
    self.df = None


  def perform_regression(self,data: pd.DataFrame):
    """
    Perform a simple linear regression with BTC Price as the independent variable
    and Volume as the dependent variable, and visualize the result.
    """
    Y = data['volume']
    X = data['btc_price']
    X = sm.add_constant(X)  # Adding a constant term for intercept

    # Fitting the regression model
    model = sm.OLS(Y, X).fit()

    # Predicted values
    data['predicted_volume'] = model.predict(X)

    # Calculating the Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(data['btc_price'], data['volume'])

    # Plotting
    title_text = (f'BTC Price vs Volume<br>'
                  f'Correlation: {corr_coef:.2f}, p-value: {p_value:.4f}')

    fig = px.scatter(data, x='btc_price', y='volume', title=title_text)
    fig.add_scatter(x=data['btc_price'], y=data['predicted_volume'], mode='lines', name='Regression Line')

    fig.show()

  def perform_regression_on_vix(self,data: pd.DataFrame):
    """
    Perform a simple linear regression with BTC Price as the independent variable
    and Volume as the dependent variable, and visualize the result.
    """
    Y = data['volume']
    X = data['vix_close']
    X = sm.add_constant(X)  # Adding a constant term for intercept

    # Fitting the regression model
    model = sm.OLS(Y, X).fit()

    # Predicted values
    data['predicted_vix'] = model.predict(X)

    # Calculating the Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(data['vix_close'], data['volume'])

    # Plotting
    title_text = (f'Vix vs Volume<br>'
                  f'Correlation: {corr_coef:.2f}, p-value: {p_value:.4f}')

    fig = px.scatter(data, x='vix_close', y='volume', title=title_text)
    fig.add_scatter(x=data['vix_close'], y=data['predicted_vix'], mode='lines', name='Regression Line')

    fig.show()


  def plot_data(self, title: str) -> None:
    """
    Plot the trading volume data.

    Args:
    title (str): The title of the plot.
    """

    fig = px.line(self.df, x='date', y='volume', title=title)
    fig.show()


  def plot_trend(self) -> None:
    """Plot the trading volume along with its trend over time."""

    # Using lowess to smooth the curve
    # (Computationally Expensive)
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

      # Format the volume as integer and add a dollar sign in front
      formatted_volume = '$' + daywise_volume['volume'].round(0).astype(int).astype(str)

      fig = px.bar(daywise_volume,
                  x='day_name',
                  y='volume',
                  title='Daywise Trading Volume Summary',
                  text=formatted_volume  # Add formatted volume values
                  )

      fig.update_xaxes(categoryorder='array', categoryarray=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

      # Update the position of the text to be inside the bars
      fig.update_traces(textposition='inside')

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

    formatted_volume = '$' + monthly_volume['volume'].round(0).astype(int).astype(str)

    fig = px.bar(monthly_volume,
                x='month_name',
                y='volume',
                title='Monthly Trading Volume Summary',
                text=formatted_volume
                )

    fig.update_xaxes(categoryorder='array', categoryarray=['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'])
    fig.update_traces(textposition='inside')

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

    self._plot_clustered_daywise_distribution(cluster_colors)
    self._plot_clustered_monthwise_distribution(cluster_colors)

  def _plot_clustered_daywise_distribution(self, cluster_colors) -> None:
    """Plot the daywise distribution of clusters."""

    optimal_clusters = 3  # or however you determine this in your context

    daywise_clusters = self.df.groupby(['day_name', 'cluster']).size().unstack().fillna(0)
    daywise_total = daywise_clusters.sum(axis=1)
    daywise_percentage = (daywise_clusters.divide(daywise_total, axis=0) * 100).round(2).astype(str) + '%'

    fig_daywise = go.Figure()
    for i in range(optimal_clusters):
        y_cumulative = daywise_clusters.iloc[:, :i].sum(axis=1)
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

  def _plot_clustered_monthwise_distribution(self, cluster_colors) -> None:
    """Plot the monthwise distribution of clusters."""

    optimal_clusters = 3  # or however you determine this in your context

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

    # Calculate monthly total trading volume for weekdays and weekends
    monthly_totals = self.df.groupby(['month', 'is_weekend'])['volume'].sum().reset_index()

    # Calculate total days in each month
    monthly_days = self.df.groupby('month').size().reset_index(name='days')

    # Merge totals with days
    monthly_totals = monthly_totals.merge(monthly_days, on=['month'])

    # Calculate monthly average trading volume for weekdays and weekends based on total days
    monthly_totals['average_volume'] = monthly_totals['volume'] / monthly_totals['days']

    # Add month names
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                  'July', 'August', 'September', 'October', 'November', 'December']
    month_name_map = dict(enumerate(month_order, 1))
    monthly_totals['month_name'] = monthly_totals['month'].map(month_name_map)

    # Calculate percentage values
    total_volumes = monthly_totals.groupby('month_name')['average_volume'].sum()
    monthly_totals['percentage'] = monthly_totals.apply(lambda row: (row['average_volume'] / total_volumes[row['month_name']]) * 100, axis=1)
    monthly_totals['text'] = monthly_totals['percentage'].round(2).astype(str) + '%'

    # Visualize the averages
    fig = px.bar(monthly_totals,
                x='month_name',
                y='average_volume',
                color='is_weekend',
                title='Average Trading Volume by Month (Weekday vs Weekend)',
                labels={'is_weekend': 'Is Weekend?', 'average_volume': 'Average Volume'},
                text='text'  # Add the percentage values
                )
    fig.update_xaxes(categoryorder='array', categoryarray=month_order)
    fig.update_traces(textposition='inside')
    fig.show()




  def perform_eda(self, data : pd.DataFrame) -> None:
    self.df = data
    self.plot_data("Volume over Time (Removed The Outliers)")
    self.plot_trend()
    self.plot_volume_distribution()
    self.plot_daywise_summary()
    self.plot_daywise_distribution()
    self.plot_monthwise_summary()
    self.plot_weekday_vs_weekend_monthly_averages()
    self.plot_monthwise_distribution()
    self.perform_clustering()


class DataValidator:
    def __init__(self, data):
        if isinstance(data, list):
            self.df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("Unsupported data type. Please provide a list or a pandas DataFrame.")

    def check_missing_values(self) -> bool:
        missing_data = self.df.isnull().sum()
        if missing_data.any():
            logging.warning(f"Found missing data:\n{missing_data}")
            return True
        return False

    def check_duplicates(self) -> bool:
        if self.df.duplicated().any():
            logging.warning("Found duplicate rows in the data!")
            return True
        return False

    def validate(self) -> bool:
        """Run all validation checks on the data.

        Returns:
            bool: True if data passes all checks, False otherwise.
        """
        if self.check_missing_values() or self.check_duplicates():
            logging.error("Data validation failed!")
            return False
        logging.info("Data validation passed!")
        return True

if __name__ == "__main__":

    ENDPOINT = "https://www.coingecko.com/exchanges/968/usd/1_year.json?locale=en"
    data_extractor = DataExtractor()
    # raw_data = data_extractor.get_raw_data_from_coingecko(ENDPOINT)
    raw_data = data_extractor.get_manual_from_coingecko_as_json("response.json")
    print(raw_data)

    data_validator = DataValidator(raw_data)
    if not data_validator.validate():
        raise ValueError("Data validation failed! Check logs for more details.")
    else:
        transformer = DataTransformer(raw_data)
        transformed_data = transformer.transform()

        DataLoader(transformed_data).save_to_csv("raw_data.csv")
        analyzer = DataAnalyzer()
        analyzer.perform_eda(transformed_data)
        analyzer.perform_regression(analyzer.df)
        analyzer.perform_regression_on_vix(analyzer.df)
        DataLoader(analyzer.df).save_to_csv("analyzed_data.csv")
