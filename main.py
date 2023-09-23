from __future__ import annotations
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler


class DataCleaner:
    """A class used for cleaning the DataFrame."""

    @staticmethod
    def _handle_thousand_and_decimal_separators(value) -> float:
        """Converts the given value to float and handles thousand and decimal separators."""
        if isinstance(value, str):
            return float(value.replace('.', '').replace(',', '.'))
        return value

    @staticmethod
    def clean_data(data: pd.DataFrame) -> pd.DataFrame:
        """Cleans the given DataFrame."""
        decimal_point_columns = [
            'cmc_score', 'android_rate', 'ios_rate', 'sikayetvar_rate'
        ]
        numerical_columns = [
            '3m_avg_visitors', 'twitter_followers', 'instagram_followers',
            'eksisozluk_comments', 'instagram_post_count'
        ] + decimal_point_columns

        data[numerical_columns] = data[numerical_columns].applymap(DataCleaner._handle_thousand_and_decimal_separators)
        return data


class DataProcessor:
    """A class used for processing the DataFrame."""

    @staticmethod
    def compute_weighted_ratings(data: pd.DataFrame) -> pd.DataFrame:
        """Computes and returns DataFrame with weighted ratings."""
        data['weighted_android_rate'] = data['android_rate'] * data['android_comments']
        data['weighted_ios_rate'] = data['ios_rate'] * data['ios_comments']
        data['weighted_sikayetvar_rate'] = data['sikayetvar_rate'] * data['sikayetvar_tickets']
        data.drop(columns=['android_rate', 'ios_rate', 'sikayetvar_rate'], inplace=True)
        return data


class ClusteringHandler:
    """Handles clustering related operations on DataFrame."""

    def __init__(self, n_clusters: int = 5, random_state: int = 34):
        self.n_clusters = n_clusters
        self.random_state = random_state

    def cluster_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Performs clustering on the given DataFrame."""
        features = data.drop(columns=['exchange_name']).dropna()
        imputed_data = KNNImputer(n_neighbors=3).fit_transform(features)
        scaled_data = StandardScaler().fit_transform(imputed_data)

        clusters = KMeans(n_clusters=self.n_clusters, random_state=self.random_state).fit_predict(scaled_data)
        data.loc[features.index, 'Cluster'] = clusters
        return data

    @staticmethod
    def compute_feature_importances(data, clusters):
        """Computes and returns sorted feature importances using Random Forest."""
        clf = RandomForestClassifier(n_estimators=100, random_state=42).fit(data, clusters)
        feature_importances = dict(zip(list(data.columns), clf.feature_importances_))
        return sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def sort_clusters(data: pd.DataFrame, sort_by: str) -> pd.DataFrame:
        """Sorts and returns clusters in DataFrame based on mean values of the specified column."""
        cluster_means = data.groupby('Cluster')[sort_by].mean().sort_values()
        sorted_cluster_mapping = {old_cluster: new_cluster for new_cluster, old_cluster in enumerate(cluster_means.index)}
        data['Cluster'] = data['Cluster'].map(sorted_cluster_mapping)
        return data


class Reporting:
    """Handles the reporting of cluster information."""

    @staticmethod
    def display_cluster_averages(cluster_averages: pd.DataFrame):
        """Displays mean values for each cluster."""
        for cluster_num in cluster_averages.index:
            print(f"------- Cluster {cluster_num} Mean Values -------")
            for column in cluster_averages.columns:
                mean_value = cluster_averages.loc[cluster_num, column]
                print(f"{column}: {mean_value:.2f}" if isinstance(mean_value, float) else f"{column}: {mean_value}")
            print("\n")


def main():
    data = pd.read_csv('exchanges_data.csv')
    data = DataCleaner.clean_data(data)
    data = DataProcessor.compute_weighted_ratings(data)

    clustering_handler = ClusteringHandler()
    data = clustering_handler.cluster_data(data)
    data = ClusteringHandler.sort_clusters(data, '3m_avg_visitors')

    cluster_averages = data.groupby('Cluster').mean()
    Reporting.display_cluster_averages(cluster_averages)

    for cluster_num in cluster_averages.index:
        exchanges_in_cluster = data[data['Cluster'] == cluster_num]['exchange_name'].tolist()
        print(f"Cluster {cluster_num}:\n", ', '.join(exchanges_in_cluster), "\n")

    features = data.drop(columns=['exchange_name', 'Cluster']).dropna()
    clusters = data.loc[features.index, 'Cluster']
    feature_importances = ClusteringHandler.compute_feature_importances(features, clusters)

    print("\nFeature Importances:")
    for feature, importance in feature_importances:
        print(f"{feature}: %{(importance * 100):.2f}")


if __name__ == "__main__":
    main()
