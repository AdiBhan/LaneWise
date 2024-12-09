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
        self.clustering_model = KMeans(
            n_clusters=3, 
            init='k-means++', 
            n_init=10,         
            random_state=42
        )
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
        if isinstance(data_input, str):
            self.data = pd.read_csv(data_input, nrows=sample_size, low_memory=False)
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise ValueError("data_input must be either a file path or DataFrame")

        # Clean numeric columns
        numeric_cols = ['v_Vel', 'Space_Headway', 'Time_Headway']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col].astype(str).str.replace(',', ''), errors='coerce')

        # Handle timestamp
        if 'Global_Time' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(self.data['Global_Time'], unit='ms')
        else:
            self.data['timestamp'] = pd.Timestamp.now()

        group_cols = ['Lane_ID' if 'Lane_ID' in self.data.columns else 'lane_id', pd.Grouper(key='timestamp', freq='5min')]

        lane_metrics = self.data.groupby(group_cols).agg({
            'Vehicle_ID': 'count',
            'v_Vel': 'mean',
            'Space_Headway': 'mean',
            'Time_Headway': 'mean'
        }).reset_index()

        lane_metrics.columns = ['lane_id', 'timestamp', 'vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        lane_metrics = lane_metrics.dropna()

        # Add density and flow
        segment_length_miles = 0.5
        lane_metrics['density'] = lane_metrics['vehicle_count'] / segment_length_miles
        lane_metrics['flow'] = lane_metrics['vehicle_count'] * 12  # 12*5min = 60min

        return lane_metrics


    def train_clustering_model(self, lane_metrics):
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time', 'density', 'flow']
        X = lane_metrics[features]
        X_scaled = self.scaler.fit_transform(X)
        self.clustering_model.fit(X_scaled)
        lane_metrics['congestion_cluster'] = self.clustering_model.labels_

        # Inverse transform cluster centers to get original scale
        cluster_centers_scaled = self.clustering_model.cluster_centers_
        cluster_centers = self.scaler.inverse_transform(cluster_centers_scaled)
        # cluster_centers is now an array of shape (n_clusters, n_features)
        # Create a DataFrame for easier manipulation
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

        cluster_mappings = self.assign_congestion_levels(cluster_centers_df)
        lane_metrics['congestion_level'] = lane_metrics['congestion_cluster'].map(cluster_mappings)
        
        return lane_metrics

    def assign_congestion_levels(self, cluster_centers):
        # cluster_centers: DataFrame with columns ['vehicle_count','avg_speed','avg_space','avg_time','density','flow']
        # Rank clusters by avg_speed (higher speed = lower congestion)
        speed_order = cluster_centers['avg_speed'].sort_values(ascending=False)
        # speed_order is a Series with cluster_id as index, sorted by descending speed
        
        mappings = {}
        # If we have exactly 3 clusters: top speed = low congestion, bottom speed = high congestion, middle = medium
        if len(speed_order) == 3:
            ordered_clusters = speed_order.index.to_list()
            mappings[ordered_clusters[0]] = 'low'
            mappings[ordered_clusters[1]] = 'medium'
            mappings[ordered_clusters[2]] = 'high'
        else:
            # For a different number of clusters, split into thirds
            num_clusters = len(speed_order)
            thresholds = [int(num_clusters/3), int(2*num_clusters/3)]
            for i, cluster_id in enumerate(speed_order.index):
                if i < thresholds[0]:
                    mappings[cluster_id] = 'low'
                elif i < thresholds[1]:
                    mappings[cluster_id] = 'medium'
                else:
                    mappings[cluster_id] = 'high'
        
        return mappings


    def get_lane_recommendations(self, current_conditions):
        # Ensure features are consistent with training
        segment_length_miles = 0.5
        current_conditions['density'] = current_conditions['vehicle_count'] / segment_length_miles
        current_conditions['flow'] = current_conditions['vehicle_count'] * 12

        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time', 'density', 'flow']
        conditions_scaled = self.scaler.transform(current_conditions[features])
        clusters = self.clustering_model.predict(conditions_scaled)
        
        recommendations = current_conditions.copy()
        recommendations['congestion_cluster'] = clusters
        recommendations['lane_score'] = (
            normalize(-recommendations['vehicle_count']) * 0.6 + 
            normalize(recommendations['avg_speed']) * 0.2 +
            normalize(recommendations['avg_space']) * 0.2
        )

        # Use the cluster centers from the model to assign levels (no recalculation needed)
        # Just call assign_congestion_levels with the original cluster centers:
        # However, we need cluster centers again:
        cluster_centers_scaled = self.clustering_model.cluster_centers_
        cluster_centers = self.scaler.inverse_transform(cluster_centers_scaled)
        features = ['vehicle_count','avg_speed','avg_space','avg_time','density','flow']
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
        cluster_mappings = self.assign_congestion_levels(cluster_centers_df)

        recommendations['congestion_level'] = recommendations['congestion_cluster'].map(cluster_mappings)
        recommendations = recommendations.sort_values('lane_score', ascending=False)
        
        return self.format_recommendations(recommendations)


    def determine_cluster_mappings(self, cluster_means):
        """
        Determines congestion level for each cluster based on their relative differences.
        Instead of using hard-coded thresholds, we:
        - Compute a 'congestion_score' for each cluster, where high speed and low vehicle_count
        yield a better score.
        - Sort clusters by this score and label them:
        - top 1/3 as 'low'
        - middle 1/3 as 'medium'
        - bottom 1/3 as 'high'
        
        If we have exactly 3 clusters, it simply assigns:
        - highest scoring cluster: low
        - next: medium
        - lowest: high
        """

        # Features we consider important
        # Speed (higher = better), vehicle_count (lower = better), space (higher = slightly better)
        # We'll normalize each feature within cluster_means to ensure fairness
        features_for_scoring = ['vehicle_count', 'avg_speed', 'avg_space']
        cm = cluster_means[features_for_scoring].copy()

        # Normalize each feature column so each gets equal footing
        for col in cm.columns:
            cm[col] = normalize(cm[col])

        # Compute a score favoring high speed and space, and low vehicle_count
        # For example:
        # score = 0.5*(normalized avg_speed) + 0.3*(normalized avg_space) + 0.2*(1 - normalized vehicle_count)
        # The "1 - normalized(vehicle_count)" inverts it so fewer vehicles = higher score
        cm['inv_vehicle_count'] = 1 - cm['vehicle_count']  # invert vehicle_count because lower is better

        cm['score'] = cm['inv_vehicle_count'] * 0.4 + cm['avg_speed'] * 0.4 + cm['avg_space'] * 0.2

        # Sort clusters by score from highest to lowest
        sorted_clusters = cm['score'].sort_values(ascending=False)

        mappings = {}
        num_clusters = len(sorted_clusters)

        # For convenience, if num_clusters=3:
        # highest = low, middle = medium, lowest = high
        # For a general case with more clusters, use percentiles or splitting into thirds
        if num_clusters == 3:
            # Extract cluster IDs in order of descending score
            cluster_order = sorted_clusters.index.to_list()
            mappings[cluster_order[0]] = 'low'
            mappings[cluster_order[1]] = 'medium'
            mappings[cluster_order[2]] = 'high'
        else:
            # If more or fewer clusters, we can still assign based on thirds:
            # This code gracefully handles other numbers of clusters
            thresholds = [int(num_clusters / 3), int(2 * num_clusters / 3)]
            for i, cluster_id in enumerate(sorted_clusters.index):
                if i < thresholds[0]:
                    mappings[cluster_id] = 'low'
                elif i < thresholds[1]:
                    mappings[cluster_id] = 'medium'
                else:
                    mappings[cluster_id] = 'high'

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

    def save_model(self):
        # Get the absolute path to the current file (lane_wise_system.py)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Define the models directory path
        models_dir = os.path.join(current_dir, "models")
        
        # Ensure the models directory exists
        os.makedirs(models_dir, exist_ok=True)
        
        # Define the full path to save the clustering model
        model_path = os.path.join(models_dir, "clustering_model.joblib")
        
        # Save the model
        joblib.dump(self.clustering_model, model_path)

    def load_model(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        model_path = os.path.join(models_dir, "clustering_model.joblib")
        
        if os.path.exists(model_path):
            self.clustering_model = joblib.load(model_path)
        else:
            # Handle the absence of the model file
            self.clustering_model = None


    def evaluate_clustering(self, data):
        # Ensure that density and flow are computed for the data just like in training
        segment_length_miles = 0.5
        data['density'] = data['vehicle_count'] / segment_length_miles
        data['flow'] = data['vehicle_count'] * 12  # Convert 5-min to hourly
        
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time', 'density', 'flow']
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