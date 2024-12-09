# LaneWise System - Traffic Analysis using K-means Clustering
# Final Project for CS506
# Uses k-means to group traffic patterns and recommend best lanes

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
    """
    LaneWiseSystem() is the main system for analyzing traffic data 
    and recommending optimal lanes using clustering and machine learning.
    """

    def __init__(self):
        # Initialize our ML stuff
        self.scaler = StandardScaler()  # Scales data so clustering works better

        # Using 3 clusters because we want low/medium/high congestion
        self.clustering_model = KMeans(
            n_clusters=3,
            init='k-means++',  # Better than random initialization
            n_init=10,  # Try 10 times to get best clusters
            random_state=42  # For reproducible results
        )
        self.data = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """
        setup_logger() configures logger for LaneWise system. 
        Enables debugging and tracking bugs/errors with formatted log messages.
        """
        # Basic logging setup to debug issues
        logger = logging.getLogger('LaneWise')
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger

    def load_and_preprocess_data(self, data_input, sample_size=None):
        """
        load_and_preprocess_data() loads and cleans traffic data from a CSV file or DataFrame, 
        aggregates it into 5-minute summaries, and computes metrics like density and flow.
        """
        # Load data from file or DataFrame
        if isinstance(data_input, str):
            self.data = pd.read_csv(
                data_input, nrows=sample_size, low_memory=False)
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise ValueError(
                "data_input must be either a file path or DataFrame")

        # Fix numbers that have commas in them
        numeric_cols = ['v_Vel', 'Space_Headway', 'Time_Headway']
        for col in numeric_cols:
            self.data[col] = pd.to_numeric(self.data[col].astype(
                str).str.replace(',', ''), errors='coerce')

        # Convert timestamps or use current time
        if 'Global_Time' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(
                self.data['Global_Time'], unit='ms')
        else:
            self.data['timestamp'] = pd.Timestamp.now()

        # Group data into 5-minute chunks for each lane
        group_cols = ['Lane_ID' if 'Lane_ID' in self.data.columns else 'lane_id', pd.Grouper(
            key='timestamp', freq='5min')]
        lane_metrics = self.data.groupby(group_cols).agg({
            'Vehicle_ID': 'count',  # How many cars
            'v_Vel': 'mean',       # Average speed
            'Space_Headway': 'mean',  # Space between cars
            'Time_Headway': 'mean'    # Time between cars
        }).reset_index()

        # Clean up column names
        lane_metrics.columns = [
            'lane_id', 'timestamp', 'vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        lane_metrics = lane_metrics.dropna()

        # Calculate density (cars per mile) and flow (cars per hour)
        segment_length_miles = 0.5  # Professor said to use 0.5 miles
        lane_metrics['density'] = lane_metrics['vehicle_count'] / \
            segment_length_miles
        lane_metrics['flow'] = lane_metrics['vehicle_count'] * \
            12  # Convert 5-min count to hourly

        return lane_metrics


    def train_clustering_model(self, lane_metrics):
        """
        train_clustering_model() trains a k-means model to identify congestion patterns 
        using features: vehicle count, speed, space, time, density, flow. 
        Maps clusters to congestion levels (low, medium, high).
        """
        # Features we're using to cluster
        features = ['vehicle_count', 'avg_speed',
                    'avg_space', 'avg_time', 'density', 'flow']
        X = lane_metrics[features]

        # Scale features so they're all equally important
        X_scaled = self.scaler.fit_transform(X)
        self.clustering_model.fit(X_scaled)
        lane_metrics['congestion_cluster'] = self.clustering_model.labels_

        # Convert cluster centers back to original scale for interpretability
        cluster_centers_scaled = self.clustering_model.cluster_centers_
        cluster_centers = self.scaler.inverse_transform(cluster_centers_scaled)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)

        # Figure out which clusters mean low/medium/high congestion
        cluster_mappings = self.assign_congestion_levels(cluster_centers_df)
        lane_metrics['congestion_level'] = lane_metrics['congestion_cluster'].map(
            cluster_mappings)

        return lane_metrics

    def assign_congestion_levels(self, cluster_centers):
        """
        assign_congestion_levels() determines congestion levels (low, medium, high) 
        for clusters based on average speed, with faster clusters indicating less congestion.
        """
        # Sort clusters by speed (higher is better)
        speed_order = cluster_centers['avg_speed'].sort_values(ascending=False)

        mappings = {}
        if len(speed_order) == 3:  # If we have exactly 3 clusters
            ordered_clusters = speed_order.index.to_list()
            # Fastest cluster = low congestion
            mappings[ordered_clusters[0]] = 'low'
            mappings[ordered_clusters[1]] = 'medium'  # Middle speed = medium
            # Slowest = high congestion
            mappings[ordered_clusters[2]] = 'high'
        else:  # If we have different number of clusters
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
        """
        get_lane_recommendations() analyzes current traffic data to recommend 
        lanes for merging, using clustering and scoring based on vehicle count, 
        speed, and space metrics.
        """
        # Calculate density and flow like we did in training
        segment_length_miles = 0.5
        current_conditions['density'] = current_conditions['vehicle_count'] / \
            segment_length_miles
        current_conditions['flow'] = current_conditions['vehicle_count'] * 12

        # Use same features as training
        features = ['vehicle_count', 'avg_speed',
                    'avg_space', 'avg_time', 'density', 'flow']
        conditions_scaled = self.scaler.transform(current_conditions[features])
        clusters = self.clustering_model.predict(conditions_scaled)

        recommendations = current_conditions.copy()
        recommendations['congestion_cluster'] = clusters

        # Calculate lane score (higher = better)
        # 60% based on fewer cars
        # 20% based on higher speed
        # 20% based on more space between cars
        recommendations['lane_score'] = (
            normalize(-recommendations['vehicle_count']) * 0.6 +
            normalize(recommendations['avg_speed']) * 0.2 +
            normalize(recommendations['avg_space']) * 0.2
        )

        # Figure out congestion levels
        cluster_centers_scaled = self.clustering_model.cluster_centers_
        cluster_centers = self.scaler.inverse_transform(cluster_centers_scaled)
        cluster_centers_df = pd.DataFrame(cluster_centers, columns=features)
        cluster_mappings = self.assign_congestion_levels(cluster_centers_df)

        recommendations['congestion_level'] = recommendations['congestion_cluster'].map(
            cluster_mappings)
        recommendations = recommendations.sort_values(
            'lane_score', ascending=False)

        return self.format_recommendations(recommendations)

    def format_recommendations(self, recommendations):
        """
        format_recommendations() formats the clustering recommendations 
        for frontend use by adding descriptive labels like 'Recommended', 
        'Acceptable', or 'Not Recommended' based on lane scores.
        """
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
                # Label lanes based on their scores
                'recommendation': 'Recommended' if row['lane_score'] > 0.7 else
                'Acceptable' if row['lane_score'] > 0.4 else
                'Not Recommended'
            })
        return output

    def save_model(self):
        """save_model() saves our trained model so we don't need to retrain in models/ folder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        model_path = os.path.join(models_dir, "clustering_model.joblib")
        joblib.dump(self.clustering_model, model_path)

    def load_model(self):
        """load_model() loads a previously trained model from models/ folder"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, "models")
        model_path = os.path.join(models_dir, "clustering_model.joblib")

        if os.path.exists(model_path):
            self.clustering_model = joblib.load(model_path)
        else:
            self.clustering_model = None

    def evaluate_clustering(self, data):
        """
        evaluate_clustering()  evaluates the quality of lane clustering using silhouette score, 
        where higher values indicate better-defined clusters.
        """
        segment_length_miles = 0.5
        data['density'] = data['vehicle_count'] / segment_length_miles
        data['flow'] = data['vehicle_count'] * 12

        features = ['vehicle_count', 'avg_speed',
                    'avg_space', 'avg_time', 'density', 'flow']
        X_scaled = self.scaler.transform(data[features])
        labels = self.clustering_model.predict(X_scaled)
        return silhouette_score(X_scaled, labels)

    def visualize_clustering(self, lane_metrics):
        """
        visualize_clustering() generates and saves visualizations of lane clustering metrics, including a scatter plot 
        of speed vs. vehicle count, a congestion level histogram, a speed distribution box plot, 
        and a correlation heatmap. 
        """
        photos_dir = Path("photos")
        photos_dir.mkdir(exist_ok=True)

        # Plot 1: Scatter plot of speed vs vehicle count
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

        # Plot 2: Count of each congestion level
        plt.figure(figsize=(10, 5))
        sns.countplot(data=lane_metrics,
                      x='congestion_level', palette='pastel')
        plt.title("Lane Distribution by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Count")
        plt.grid(True)
        plt.savefig(photos_dir / "congestion_level_histogram.png")
        plt.close()

        # Plot 3: Speed distribution for each congestion level
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=lane_metrics, x='congestion_level',
                    y='avg_speed', palette='Set2')
        plt.title("Speed Distribution by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Average Speed (mph)")
        plt.grid(True)
        plt.savefig(photos_dir / "average_speed_boxplot.png")
        plt.close()

        # Plot 4: Correlation heatmap
        plt.figure(figsize=(10, 8))
        correlation_matrix = lane_metrics[[
            'vehicle_count', 'avg_speed', 'avg_space', 'avg_time']].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                    cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Metric Correlations")
        plt.savefig(photos_dir / "correlation_heatmap.png")
        plt.close()


def normalize(series):
    # normalize() scales values between 0 and 1
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val > min_val else series


# Test the system if we run this file directly
if __name__ == '__main__':
    # Make sure we have a models folder
    MODELS_DIR = Path("models")
    MODELS_DIR.mkdir(exist_ok=True)

    # Create system and load data
    lanewise = LaneWiseSystem()
    data_path = "data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"
    Path("data").mkdir(exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at {data_path}")

    # Process data and train model
    lane_metrics = lanewise.load_and_preprocess_data(data_path)
    lane_metrics = lanewise.train_clustering_model(lane_metrics)
    lanewise.visualize_clustering(lane_metrics)
    lanewise.save_model()

    # Test recommendations with last 4 time periods
    current_conditions = lane_metrics.iloc[-4:]
    recommendations = lanewise.get_lane_recommendations(current_conditions)

    # Print results
    print("\nLane Recommendations:")
    for rec in recommendations:
        print(f"\nLane {rec['lane_id']}:")
        print(f"Congestion Level: {rec['congestion_level']}")
        print(f"Score: {rec['score']}")
        print(f"Status: {rec['recommendation']}")
        print("Metrics:", rec['metrics'])

    # Print model quality score
    print(f"\nSilhouette Score: {lanewise.evaluate_clustering(lane_metrics)}")
