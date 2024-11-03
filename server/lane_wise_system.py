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
    
    '''LaneWiseSystem is a traffic management and analysis tool designed to process traffic data, 
    perform clustering for lane congestion analysis, and provide lane recommendations. 
    It leverages machine learning models to classify lanes based on congestion metrics 
    and predicts optimal lanes for smoother traffic flow.'''

    def __init__(self):
        self.scaler = StandardScaler()
        self.clustering_model = KMeans(n_clusters=5, random_state=42)
        self.data = None
        self.logger = self.setup_logger()

    def setup_logger(self):
        """Set up logging configuration"""
        logger = logging.getLogger('LaneWise')
        logger.setLevel(logging.INFO)

        # Create handler if not exists
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def load_and_preprocess_data(self, data_input, sample_size=None):
        """Load and preprocess data from file or DataFrame"""
        self.logger.info("Loading and preprocessing data...")

        # Handle both file paths and DataFrames
        if isinstance(data_input, str):
            self.data = pd.read_csv(data_input, nrows=sample_size)
        elif isinstance(data_input, pd.DataFrame):
            self.data = data_input
        else:
            raise ValueError(
                "data_input must be either a file path or DataFrame")

        # Create time-based features if Global_Time exists
        if 'Global_Time' in self.data.columns:
            self.data['timestamp'] = pd.to_datetime(
                self.data['Global_Time'],
                unit='ms'
            )
        else:
            # Use current time for sample data
            self.data['timestamp'] = pd.Timestamp.now()

        # Calculate metrics per lane
        group_cols = ['Lane_ID' if 'Lane_ID' in self.data.columns else 'lane_id',
                      pd.Grouper(key='timestamp', freq='5T')]

        metrics_cols = {
            'Vehicle_ID' if 'Vehicle_ID' in self.data.columns else 'vehicle_count': 'count',
            'v_Vel' if 'v_Vel' in self.data.columns else 'avg_speed': 'mean',
            'Space_Headway' if 'Space_Headway' in self.data.columns else 'avg_space': 'mean',
            'Time_Headway' if 'Time_Headway' in self.data.columns else 'avg_time': 'mean'
        }

        lane_metrics = self.data.groupby(
            group_cols).agg(metrics_cols).reset_index()

        # Rename columns
        lane_metrics.columns = [
            'lane_id', 'timestamp', 'vehicle_count',
            'avg_speed', 'avg_space', 'avg_time'
        ]

        return lane_metrics

    def train_clustering_model(self, lane_metrics):
        """Train clustering model for lane classification"""
        self.logger.info("Training clustering model...")

        #  features for clustering
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        X = lane_metrics[features]

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train clustering model
        self.clustering_model.fit(X_scaled)

        # Add cluster labels
        lane_metrics['congestion_cluster'] = self.clustering_model.labels_

        # Map numeric clusters to meaningful labels
        cluster_means = pd.DataFrame(
            lane_metrics.groupby('congestion_cluster')[features].mean()
        )

        # Determine cluster meanings based on characteristics
        cluster_mappings = self.determine_cluster_mappings(cluster_means)
        lane_metrics['congestion_level'] = lane_metrics['congestion_cluster'].map(
            cluster_mappings
        )

        return lane_metrics

    def get_lane_recommendations(self, current_conditions):
        """Generate lane recommendations based on current conditions"""
        self.logger.info("Generating lane recommendations...")
        self.logger.debug(f"Current conditions for prediction: {current_conditions}")

        # Scale current conditions
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        conditions_scaled = self.scaler.transform(current_conditions[features])
        self.logger.debug(f"Scaled conditions: {conditions_scaled}")

        # Predict congestion cluster
        try:
            clusters = self.clustering_model.predict(conditions_scaled)
            self.logger.debug(f"Predicted clusters: {clusters}")
        except Exception as e:
            self.logger.error(f"Error during prediction: {str(e)}")
            raise

        # Calculate lane scores
        recommendations = current_conditions.copy()
        recommendations['congestion_cluster'] = clusters

        # Score each lane
        recommendations['lane_score'] = (
            normalize(-recommendations['vehicle_count']) * 0.6 + 
            normalize(recommendations['avg_speed']) * 0.2 +
            normalize(recommendations['avg_space']) * 0.2
        )
        # Map clusters to congestion levels
        cluster_means = pd.DataFrame({
            'avg_speed': recommendations.groupby('congestion_cluster')['avg_speed'].mean(),
            'vehicle_count': recommendations.groupby('congestion_cluster')['vehicle_count'].mean(),
            'avg_space': recommendations.groupby('congestion_cluster')['avg_space'].mean()
        })

        # Determine congestion levels for clusters
        cluster_mappings = self.determine_cluster_mappings(cluster_means)
        recommendations['congestion_level'] = recommendations['congestion_cluster'].map(
            cluster_mappings)

        # Sort lanes by score
        recommendations = recommendations.sort_values(
            'lane_score',
            ascending=False
        )

        return self.format_recommendations(recommendations)

    def determine_cluster_mappings(self, cluster_means):
        """Map cluster numbers to congestion levels"""
        # Calculate scores for each cluster
        cluster_scores = pd.DataFrame()
        cluster_scores['score'] = (
            normalize(cluster_means['avg_speed']) * 0.5 +
            normalize(-cluster_means['vehicle_count']) * 0.3 +
            normalize(cluster_means['avg_space']) * 0.2
        )

        # Sort clusters by score
        sorted_clusters = cluster_scores.sort_values('score', ascending=False)

        # Map to congestion levels based on relative scores
        mappings = {}
        # Assign first 1/3 as "low", next 1/3 as "medium", and last 1/3 as "high"
        num_clusters = len(sorted_clusters)
        levels = ['low', 'medium', 'high']
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
        """Format lane recommendations for output"""
        output = []

        for _, row in recommendations.iterrows():
            recommendation = {
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
            }
            output.append(recommendation)

        return output

    def save_model(self, path="models/"):
        """Save trained model and scaler"""
        joblib.dump(self.clustering_model, f"{path}clustering_model.joblib")
        joblib.dump(self.scaler, f"{path}scaler.joblib")
        self.logger.info(f"Models saved to {path}")


    def load_model(self, path="models/"):
        """Load trained model and scaler"""
        self.clustering_model = joblib.load(f"{path}clustering_model.joblib")
        self.scaler = joblib.load(f"{path}scaler.joblib")
        self.logger.info("Models loaded successfully")
        self.logger.info(f"Clustering model type: {type(self.clustering_model)}")


    def evaluate_clustering(self, data):
        features = ['vehicle_count', 'avg_speed', 'avg_space', 'avg_time']
        X_scaled = self.scaler.transform(data[features])
        labels = self.clustering_model.predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"Silhouette Score: {score}")
        return score

    def visualize_clustering(self, lane_metrics):
        """Visualize clustering results and save plots to photos/ directory"""
        photos_dir = Path("photos")
        photos_dir.mkdir(exist_ok=True)

        # Scatter plot of Average Speed vs Vehicle Count
        plt.figure(figsize=(12, 6))
        sns.scatterplot(data=lane_metrics, x='avg_speed', y='vehicle_count',
                        hue='congestion_level', palette='viridis', style='congestion_cluster', s=100)
        plt.title("Clustering of Lanes based on Speed and Vehicle Count")
        plt.xlabel("Average Speed (mph)")
        plt.ylabel("Vehicle Count")
        plt.legend(title='Congestion Level')
        plt.grid(True)
        scatter_plot_path = photos_dir / "clustering_scatter_plot.png"
        plt.savefig(scatter_plot_path)
        plt.close()

        # Histogram of congestion levels
        plt.figure(figsize=(10, 5))
        sns.countplot(data=lane_metrics,
                      x='congestion_level', palette='pastel')
        plt.title("Count of Lanes by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Count of Lanes")
        plt.grid(True)
        histogram_path = photos_dir / "congestion_level_histogram.png"
        plt.savefig(histogram_path)
        plt.close()

        # Box Plot for Average Speed
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=lane_metrics, x='congestion_level',
                    y='avg_speed', palette='Set2')
        plt.title("Box Plot of Average Speed by Congestion Level")
        plt.xlabel("Congestion Level")
        plt.ylabel("Average Speed (mph)")
        plt.grid(True)
        box_plot_path = photos_dir / "average_speed_boxplot.png"
        plt.savefig(box_plot_path)
        plt.close()

        # Heatmap of Correlation
        plt.figure(figsize=(10, 8))
        correlation_matrix = lane_metrics[[
            'vehicle_count', 'avg_speed', 'avg_space', 'avg_time']].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f",
                    cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("Correlation Heatmap")
        heatmap_path = photos_dir / "correlation_heatmap.png"
        plt.savefig(heatmap_path)
        plt.close()

        self.logger.info(f"Visualizations saved to {photos_dir}")


def normalize(series):
    """Normalize values to 0-1 range"""
    min_val = series.min()
    max_val = series.max()
    return (series - min_val) / (max_val - min_val) if max_val > min_val else series


# Create models directory if it doesn't exist
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# Run to generate models
if __name__ == '__main__':
    lanewise = LaneWiseSystem()

    try:

        data_path = "data/Next_Generation_Simulation__NGSIM__Vehicle_Trajectories_and_Supporting_Data.csv"

        # Create data directory
        Path("data").mkdir(exist_ok=True)

        # Verify data file exists
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {
                                    data_path}. Please check the file path.")

        # Load and preprocess data
        print("Loading data from:", data_path)
        lane_metrics = lanewise.load_and_preprocess_data(
            data_path,
            sample_size=None
        )

        # Train clustering model
        lane_metrics = lanewise.train_clustering_model(lane_metrics)

        # Create Visualizations of dataset

        lanewise.visualize_clustering(lane_metrics)

        # Save trained model
        model_path = Path("models/clustering_model.joblib")
        lanewise.save_model(path=model_path)

        # Generate sample recommendations
        current_conditions = lane_metrics.iloc[-4:]
        recommendations = lanewise.get_lane_recommendations(current_conditions)

        print("\nLane Recommendations:")
        for rec in recommendations:
            print(f"\nLane {rec['lane_id']}:")
            print(f"Congestion Level: {rec['congestion_level']}")
            print(f"Score: {rec['score']}")
            print(f"Status: {rec['recommendation']}")
            print("Metrics:", rec['metrics'])

        print("\n")
        lanewise.evaluate_clustering(lane_metrics)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
    except Exception as e:
        lanewise.logger.error(f"Error: {str(e)}")