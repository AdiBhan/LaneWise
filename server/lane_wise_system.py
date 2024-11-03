import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
from datetime import datetime
import logging
import os
from pathlib import Path
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns


class LaneWiseSystem:
    '''Traffic analysis system that uses clustering to identify congestion patterns and recommend optimal lanes'''

    def __init__(self):
         # Set up system logging configuration
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.data = None
        self.logger = self.setup_logger()

    def setup_logger(self):
         # Sets up system logging configuration
        logger = logging.getLogger('LaneWise')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_and_preprocess_data(self, data_input, sample_size=None):
           # Loads traffic data and aggregates into 5-minute windows
        if isinstance(data_input, str):
            self.data = pd.read_csv(data_input, nrows=sample_size)
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise ValueError("data_input must be either a file path or DataFrame")

        if 'Global_Time' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['Global_Time'], unit='ms')
        else:
            self.data['timestamp'] = pd.Timestamp.now()

        group_cols = ['Lane_ID' if 'Lane_ID' in self.data.columns else 'lane_id',
                     pd.Grouper(key='timestamp', freq='5min')]

        metrics_cols = {
            'Vehicle_ID' if 'Vehicle_ID' in self.data.columns else 'vehicle_count': 'count',
            'v_Vel' if 'v_Vel' in self.data.columns else 'avg_speed': 'mean',
            'Space_Headway' if 'Space_Headway' in self.data.columns else 'avg_space': 'mean',
            'Time_Headway' if 'Time_Headway' in self.data.columns else 'avg_time': 'mean'
        }

        lane_metrics = self.data.groupby(group_cols).agg(metrics_cols).reset_index()
        lane_metrics.columns = ['lane_id', 'timestamp', 'vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        
        return lane_metrics

    def train_clustering_model(self, lane_metrics):
          # Trains K-means model to classify lanes by congestion level
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        X = lane_metrics[features]
        X_scaled = self.scaler.fit_transform(X)
        self.clustering_model.fit(X_scaled)
        lane_metrics['congestion_cluster'] = self.clustering_model.labels_
        
        cluster_means = pd.DataFrame(lane_metrics.groupby('congestion_cluster')[features].mean())
        cluster_mappings = self.determine_cluster_mappings(cluster_means)
        lane_metrics['congestion_level'] = lane_metrics['congestion_cluster'].map(cluster_mappings)
        
        return lane_metrics

    def get_lane_recommendations(self, current_conditions):
         # Analyzes current traffic conditions and generates lane recommendations
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        conditions_scaled = self.scaler.transform(current_conditions[features])
        clusters = self.clustering_model.predict(conditions_scaled)
        
        recommendations = current_conditions.copy()
        recommendations['congestion_cluster'] = clusters
        recommendations['lane_score'] = (
            normalize(-recommendations['vehicle_count']) * 0.6 + 
            normalize(recommendations['avg_speed']) * 0.2 +
            normalize(recommendations['avg_space']) * 0.2
        )
        
        cluster_means = pd.DataFrame({
            'avg_speed': recommendations.groupby('congestion_cluster')['avg_speed'].mean(),
            'vehicle_count': recommendations.groupby('congestion_cluster')['vehicle_count'].mean(),
            'avg_space': recommendations.groupby('congestion_cluster')['avg_space'].mean()
        })

        cluster_mappings = self.determine_cluster_mappings(cluster_means)
        recommendations['congestion_level'] = recommendations['congestion_cluster'].map(cluster_mappings)
        recommendations = recommendations.sort_values('lane_score', ascending=False)
        
        return self.format_recommendations(recommendations)

    def determine_cluster_mappings(self, cluster_means):
          # Maps numeric clusters to congestion levels (low, medium, high)
        cluster_scores = pd.DataFrame()
        cluster_scores['score'] = (
            normalize(cluster_means['avg_speed']) * 0.5 +
            normalize(-cluster_means['vehicle_count']) * 0.3 +
            normalize(cluster_means['avg_space']) * 0.2
        )
        
        sorted_clusters = cluster_scores.sort_values('score', ascending=False)
        mappings = {}
        num_clusters = len(sorted_clusters)
        thresholds = [int(num_clusters / 3), int(2 * num_clusters / 3)]

        for i, (cluster, _) in enumerate(sorted_clusters.iterrows()):
            if i < thresholds[0]:
                mappings[cluster] = 'low'
            elif i < thresholds[1]:
                mappings[cluster] = 'medium'
            else:
                mappings[cluster] = 'high'

        return mappings

    def format_recommendations(self, recommendations):
         # Formats lane recommendations into dictionaries
        output = []
        for _, row in recommendations.iterrows():
            output.append({
                'lane_id': int(row['lane_id']),
                'congestion_level': row['congestion_level'],
                'score': round(row['lane_score'], 2),
                'metrics': {
                    'vehicle_count': int(row['vehicle_count']),
                    'average_speed': round(row['avg_speed'], 1),
                    'space_headway': round(row['avg_space'], 1)
                },
                'recommendation': 'Recommended' if row['lane_score'] > 0.7 else
                                'Acceptable' if row['lane_score'] > 0.4 else
                                'Not Recommended'
            })
        return output

    def save_model(self, path="models/"):
        joblib.dump(self.clustering_model, f"{path}clustering_model.joblib")
        joblib.dump(self.scaler, f"{path}scaler.joblib")

    def load_model(self, path="models/"):
        self.clustering_model = joblib.load(f"{path}clustering_model.joblib")
        self.scaler = joblib.load(f"{path}scaler.joblib")

    def evaluate_clustering(self, data):
         # Evaluates clustering quality using silhouette score
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        X_scaled = self.scaler.transform(data[features])
        labels = self.clustering_model.predict(X_scaled)
        return silhouette_score(X_scaled, labels)

    def visualize_clustering(self, lane_metrics):
          # Generates visualizations of clustering results and traffic patterns
        photos_dir = Path("photos")
        photos_dir.mkdir(exist_ok=True)

        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=lane_metrics, x='avg_speed', y='vehicle_count',
                       hue='congestion_level', palette='viridis', 
                       style='congestion_cluster', s=100)
        plt.title("Lane Clustering by Speed and Vehicle Count")
        plt.xlabel("Average Speed (mph)")
        plt.ylabel("Vehicle Count")
        plt.grid(True)
        plt.savefig(photos_dir / "clustering_scatter_plot.png")
        plt.close()

        plt.figure(figsize=(10, 5))
        sns.countplot(data=lane_metrics, x='congestion_level', palette='pastel')
        plt.title("Lane Distribution by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(photos_dir / "congestion_level_histogram.png")
        plt.close()

        plt.figure(figsize=(12, 6))
        sns.boxplot(data=lane_metrics, x='congestion_level', y='avg_speed', palette='Set2')
        plt.title("Speed Distribution by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Average Speed (mph)")
        plt.grid(True)
        plt.savefig(photos_dir / "average_speed_boxplot.png")
        plt.close()

        plt.figure(figsize=(10, 8))
        correlation_matrix = lane_metrics[['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                   cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Metric Correlations")
        plt.savefig(photos_dir / "correlation_heatmap.png")
        plt.close()


def normalize(series):
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val > min_val else series


if __name__ == '__main__':
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)
    
    lanewise = LaneWiseSystem()
    data_path = "data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"
    Path("data").mkdir(exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    lane_metrics = lanewise.load_and_preprocess_data(data_path)
    lane_metrics = lanewise.train_clustering_model(lane_metrics)
    lanewise.visualize_clustering(lane_metrics)
    lanewise.save_model()
    
    current_conditions = lane_metrics.iloc[-4:]
    recommendations = lanewise.get_lane_recommendations(current_conditions)
    
    print("\nLane Recommendations:")
    for rec in recommendations:
        print(f"\nLane {rec['lane_id']}:")
        print(f"Congestion Level: {rec['congestion_level']}")
        print(f"Score: {rec['score']}")
        print(f"Status: {rec['recommendation']}")
        print("Metrics:", rec['metrics'])
    
    print(f"\nSilhouette Score: {lanewise.evaluate_clustering(lane_metrics)}")