
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware  
from pydantic import BaseModel
import pandas as pd  
import uvicorn
from .lane_wise_system import LaneWiseSystem
from pathlib import Path
import traceback
from typing import List
from fastapi import HTTPException
import logging
import os
# Setting up basic logging for debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LaneWiseAPI")

# Starting data shown on frontend
SAMPLE_DATA = [
    {'lane_id': 1, 'congestion_level': 'low', 'score': 0.85, 'metrics': {'vehicle_count': 12, 'average_speed': 65.5, 'space_headway': 120}},
    {'lane_id': 2, 'congestion_level': 'medium', 'score': 0.65, 'metrics': {'vehicle_count': 18, 'average_speed': 55.2, 'space_headway': 90}},
    {'lane_id': 3, 'congestion_level': 'high', 'score': 0.35, 'metrics': {'vehicle_count': 25, 'average_speed': 45.8, 'space_headway': 60}},
    {'lane_id': 4, 'congestion_level': 'low', 'score': 0.78, 'metrics': {'vehicle_count': 15, 'average_speed': 60.5, 'space_headway': 100}}
]

# Create the FastAPI app
app = FastAPI(title="LaneWise API")

# Handles CORS network errors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create LaneWise System for giving lane predictions
lanewise = LaneWiseSystem()

def initialize_model():
    """
    initialize_model() attempts to load a pre-trained model. 
    If no model is found, it trains a new one using sample data.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(current_dir, "models", "clustering_model.joblib")
        if os.path.exists(model_path):
            success = lanewise.load_model()
            if success:
                logger.info(f"Loaded model from {model_path}")
            else:
                logger.warning("Failed to load model, training new one")
                train_new_model()
        else:
            logger.info(f"Model not found at {model_path}. Training new model.")
            train_new_model()
    except Exception as e:
        logger.error(f"Error in initialize_model: {str(e)}")
        logger.error(traceback.format_exc())

        train_new_model()

def train_new_model():
    """
    train_new_model() generates sample traffic data and trains a 
    clustering model to analyze lane congestion.
    """
    num_samples = 1000  # Use 1000 samples to help generate data
    training_data = pd.DataFrame({
         'Lane_ID': list(range(1, 5)) * (num_samples // 4),
         'Vehicle_ID': range(num_samples),
         'Global_Time': pd.date_range(start='now', periods=num_samples, freq='1min'),
         'v_Vel': [65, 55, 45, 35] * (num_samples // 4),  # Speed patterns for each lane
         'Space_Headway': [120, 90, 60, 40] * (num_samples // 4),  # Space between cars
         'Time_Headway': [3.0, 2.5, 2.0, 1.5] * (num_samples // 4)  # Time between cars
    })
    lane_metrics = lanewise.load_and_preprocess_data(training_data)
    lane_metrics = lanewise.train_clustering_model(lane_metrics)
    lanewise.save_model()

# Start up the model when we run the API
initialize_model()

# Base endpoint to make sure site is working
@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to LaneWise API"}

# Using pydantic classes to ensure types are correct
class LaneDataInput(BaseModel):
    lane_id: int
    vehicle_count: int
    avg_speed: float
    avg_space: float
    avg_time: float

# Main endpoint
@app.post("/evaluate")
async def evaluate_lane_data(data: List[LaneDataInput]):
    """
    evaluate_lane_data() processes traffic data from the request and 
    evaluates lane congestion, returning lane recommendations. If an error 
    occurs, dummy sample data is returned for the frontend.
    """
    logger.info("Received data for evaluation.")
    try:
        # Convert the input data into a pandas DataFrame
        data_df = pd.DataFrame([item.dict() for item in data])
        logger.info(f"DataFrame created with columns: {data_df.columns.tolist()}")
        
        logger.debug(f"Data received for evaluation: {data_df}")
        
        # Get recommendations from our ML model
        recommendations = lanewise.get_lane_recommendations(data_df)
        logger.info("Lane recommendations generated successfully.")
        return recommendations
    except Exception as e:
        # If something breaks, use sample data
        logger.error(f"Error generating recommendations: {str(e)}. Loading random sample data...")
        traceback.print_exc()
        return SAMPLE_DATA

# Starts the FastAPI server on Port 8000
if __name__ == "__main__":
    logger.info("Starting LaneWise API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)