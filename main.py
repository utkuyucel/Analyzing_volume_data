import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalysisPipeline:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = None
        self.outliers = None
        self.main()

    def load_data(self):
        self.df = pd.read_csv(self.file_path)
        self.df["snapped_at"] = pd.to_datetime(self.df["snapped_at"])
        self.df["volume"] = self.df["volume"].astype(int)
        self.df["date"] = self.df["snapped_at"].dt.date
        self.df["month"] = self.df["snapped_at"].dt.month
        self.df["day_name"] = self.df["snapped_at"].dt.day_name()
        self.df["is_weekend"] = self.df["day_name"].isin(["Saturday", "Sunday"])
        self.df = self.df[["date", "month", "day_name", "is_weekend", "volume"]]

    def plot_data(self, title):
        plt.figure(figsize=(20, 8))
        plt.plot(self.df["date"], self.df["volume"])
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Volume')
        plt.show()

    def find_outliers_iqr(self):
        Q1 = self.df["volume"].quantile(0.25)
        Q3 = self.df["volume"].quantile(0.75)
        IQR = Q3 - Q1
        self.outliers = self.df[((self.df["volume"] < (Q1 - 1.5 * IQR)) | (self.df["volume"] > (Q3 + 1.5 * IQR)))]

    def remove_outliers(self):
        self.df = self.df[~self.df.index.isin(self.outliers.index)]

    def analyze_volume_by_day(self):
        volume_by_day = self.df.groupby(["day_name"]).agg({"volume": ["mean"]})
        volume_by_day = volume_by_day.reset_index()
        volume_by_day.columns = volume_by_day.columns.droplevel(level=1)

        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        volume_by_day['day_name'] = pd.Categorical(volume_by_day['day_name'], categories=days_order, ordered=True)
        volume_by_day = volume_by_day.sort_values('day_name')

        plt.figure(figsize=(12, 6))
        sns.barplot(x='day_name', y='volume', data=volume_by_day, palette='viridis')

        plt.xlabel('Day of the Week', fontsize=12)
        plt.ylabel('Average Trading Volume', fontsize=12)
        plt.title('Average Trading Volume by Day of the Week', fontsize=16)

        for i in range(volume_by_day.shape[0]):
            plt.text(x=i, y=volume_by_day.iloc[i]['volume'],
                     s=f'{volume_by_day.iloc[i]["volume"]:.0f}',
                     ha='center', fontsize=12)

        plt.show()

        return volume_by_day

    def analyze_volume_by_month(self):
        volume_by_month = self.df.groupby(["month"]).agg({"volume": ["mean"]})
        volume_by_month = volume_by_month.reset_index()
        volume_by_month.columns = volume_by_month.columns.droplevel(level=1)

        plt.figure(figsize=(12, 6))
        sns.barplot(x='month', y='volume', data=volume_by_month, palette='viridis')

        plt.xlabel('Month', fontsize=12)
        plt.ylabel('Average Trading Volume', fontsize=12)
        plt.title('Average Trading Volume by Month', fontsize=16)

        for i in range(volume_by_month.shape[0]):
            plt.text(x=i, y=volume_by_month.iloc[i]['volume'],
                     s=f'{volume_by_month.iloc[i]["volume"]:.0f}',
                     ha='center', fontsize=12)

        plt.show()

        return volume_by_month

    def analyze_volume_by_weekend(self):
        volume_by_weekend = self.df.groupby(["is_weekend"]).agg({"volume": ["mean"]})
        volume_by_weekend = volume_by_weekend.reset_index()
        volume_by_weekend.columns = volume_by_weekend.columns.droplevel(level=1)
        volume_by_weekend['is_weekend'] = volume_by_weekend['is_weekend'].replace({False: 'Weekday', True: 'Weekend'})

        plt.figure(figsize=(8, 6))
        sns.barplot(x='is_weekend', y='volume', data=volume_by_weekend, palette='viridis')

        plt.xlabel('Type of Day', fontsize=12)
        plt.ylabel('Average Trading Volume', fontsize=12)
        plt.title('Average Trading Volume by Type of Day (Weekday vs Weekend)', fontsize=16)

        for i in range(volume_by_weekend.shape[0]):
            plt.text(x=i, y=volume_by_weekend.iloc[i]['volume'],
                     s=f'{volume_by_weekend.iloc[i]["volume"]:.0f}',
                     ha='center', fontsize=12)

        plt.show()

        return volume_by_weekend

    def main(self):
        self.load_data()
        self.plot_data('Volume over Time')
        self.find_outliers_iqr()
        self.remove_outliers()
        self.plot_data('Volume over Time (After Removing Outliers)')
        print(self.analyze_volume_by_day())
        print(self.analyze_volume_by_month())
        print(self.analyze_volume_by_weekend())


if __name__ == "__main__":
  pipeline = DataAnalysisPipeline("x-trading-volume-1-year.csv")
