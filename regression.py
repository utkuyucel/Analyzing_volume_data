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
np.random.seed(42)


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

  def get_manual_from_google_analytics(self, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    return df

  # It is exactly similar with "get_manual_from_google_analytics" method but will be changed in future.
  def get_manual_from_similarweb(self, file_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_name)
    return df

class DataTransformer:
  def __init__(self, data: pd.DataFrame):
    self.df = data
    self.grouped_df = None
    self.outliers = None
    self.android_data = None
    self.ios_data = None
    self.google_analytics_data = None
    self.similarweb_data = None

  # TODO: Will be added to "DataExtractor" class in Future
  def _get_btc_data(self):
    self.start_date = self.df["date"].min()
    self.end_date = self.df["date"].max()

    ticker = yf.Ticker("BTC-USD")
    historical_data = ticker.history(start = self.start_date, end = self.end_date).reset_index()
    historical_data.rename(columns = {"Date": "date", "Close": "btc_price"}, inplace = True)
    historical_data['date'] = historical_data['date'].dt.tz_localize(None)
    return historical_data[["date", "btc_price"]]


  def transform_google_analytics_data(self, data: pd.DataFrame) -> pd.DataFrame:
    data["date"] = pd.to_datetime(data["date"], format="%d.%m.%Y")
    data["view_count"] = data["view_count"].replace(regex = "\\.", value = "").astype(int)

    self.google_analytics_data = data
    return self.google_analytics_data


  def transform_similarweb_data(self, data: pd.DataFrame) -> pd.DataFrame:
    data["date"] = pd.to_datetime(data["date"], format= "%d-%m-%Y")
    data["similarweb_count"] = data["similarweb_count"].replace(regex = "\\,", value = ".").astype(float)

    self.similarweb_data = data
    return self.similarweb_data

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


  def _remove_outliers_from_mobile_data(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
    df_prophet = data[['date', column]]
    df_prophet.columns = ['ds', 'y']

    model = Prophet(yearly_seasonality=False, daily_seasonality=False)
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.fit(df_prophet)

    forecast = model.predict(df_prophet)

    residuals = data[column] - forecast['yhat']
    threshold = 3 * residuals.std()
    outliers = (residuals.abs() > threshold).values

    data = data[~outliers]
    logging.info(f"Identified and removed outliers from {column}.")

    return data

  def aggregate_monthly_volume(self) -> pd.DataFrame:
    """Aggregate the daily volume data to monthly data."""
    monthly_df = self.df.groupby(self.df['date'].dt.to_period("M")).agg({'volume': 'sum'}).reset_index()
    monthly_df['date'] = monthly_df['date'].dt.to_timestamp()  # Convert period back to datetime
    print(monthly_df)
    return monthly_df

  def merge_with_ga_data(self) -> pd.DataFrame:
      """Merge the aggregated monthly data with Google Analytics data."""
      monthly_data = self.aggregate_monthly_volume()
      merged_ga_data = pd.merge(monthly_data, self.google_analytics_data, on="date", how="inner")
      self.google_analytics_data = merged_ga_data
      return self.google_analytics_data


  def transform(self) -> pd.DataFrame:
    self.transform_timestamps_from_coingecko()
    self.transform_dataframe_from_coingecko()
    self.detect_outliers()

    self.btc_data = self._get_btc_data()

    self.google_analytics_data = DataExtractor().get_manual_from_google_analytics("google_analytics_data.csv")
    self.google_analytics_data = self.transform_google_analytics_data(self.google_analytics_data)
    self.merge_with_ga_data()  # Calling the merge function

    print(self.google_analytics_data)

    # self.similarweb_data = DataExtractor().get_manual_from_similarweb("bitlo_similarweb.csv")
    # self.similarweb_data = self.transform_similarweb_data(self.similarweb_data)
    # self.similarweb_data = pd.merge(self.similarweb_data, self.df, on = "date", how = "inner")

    self.merged_data = pd.merge(self.df, self.btc_data, on="date", how="inner")
    return self.merged_data



class DataLoader:
  def __init__(self, data : pd.DataFrame):
    self.df = data

  def save_to_csv(self, file_path: str) -> None:
    self.df.to_csv(file_path)


class DataAnalyzer:
  def __init__(self):
    self.df = None


  def perform_regression_on_volume(self,data: pd.DataFrame):
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


  def perform_regression_on_similarweb_data(self,data: pd.DataFrame):

    Y = data['volume']
    X = data['similarweb_count']
    X = sm.add_constant(X)  # Adding a constant term for intercept

    # Fitting the regression model
    model = sm.OLS(Y, X).fit()

    # Predicted values
    data['predicted_similarweb_count'] = model.predict(X)

    # Calculating the Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(data['similarweb_count'], data['volume'])

    # Plotting
    title_text = (f'similarweb_count vs Volume<br>'
                  f'Correlation: {corr_coef:.2f}, p-value: {p_value:.4f}')

    fig = px.scatter(data, x='similarweb_count', y='volume', title=title_text)
    fig.add_scatter(x=data['similarweb_count'], y=data['predicted_similarweb_count'], mode='lines', name='Regression Line')

    fig.show()



  def perform_regression_on_ga_data(self,data: pd.DataFrame):

    Y = data['volume']
    X = data['view_count']
    X = sm.add_constant(X)  # Adding a constant term for intercept

    # Fitting the regression model
    model = sm.OLS(Y, X).fit()

    # Predicted values
    data['predicted_view_count'] = model.predict(X)

    # Calculating the Pearson correlation coefficient and p-value
    corr_coef, p_value = pearsonr(data['view_count'], data['volume'])

    # Plotting
    title_text = (f'View Count vs Volume<br>'
                  f'Correlation: {corr_coef:.2f}, p-value: {p_value:.4f}')

    fig = px.scatter(data, x='view_count', y='volume', title=title_text)
    fig.add_scatter(x=data['view_count'], y=data['predicted_view_count'], mode='lines', name='Regression Line')

    fig.show()

if __name__ == "__main__":

    ENDPOINT = "https://www.coingecko.com/exchanges/968/usd/1_year.json?locale=en"
    data_extractor = DataExtractor()
    raw_data = data_extractor.get_raw_data_from_coingecko(ENDPOINT)
    # raw_data = data_extractor.get_manual_from_coingecko_as_json("response.json")

    transformer = DataTransformer(raw_data)
    transformed_data = transformer.transform()
    DataLoader(transformed_data).save_to_csv("raw_data.csv") #Raw data
    DataLoader(transformer.google_analytics_data).save_to_csv("ad.csv") #Raw data

    analyzer = DataAnalyzer()
    analyzer.perform_regression_on_volume(transformed_data)
    analyzer.perform_regression_on_ga_data(transformer.google_analytics_data)
    # analyzer.perform_regression_on_similarweb_data(transformer.similarweb_data)
    DataLoader(transformed_data).save_to_csv("analyzed_data.csv") # Saving transformed & analyzed data
