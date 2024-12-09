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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("LaneWiseAPI")

# Sample data as a fallback in case of errors
SAMPLE_DATA = [
    {'lane_id': 1, 'congestion_level': 'low', 'score': 0.85, 'metrics': {'vehicle_count': 12, 'average_speed': 65.5, 'space_headway': 120}},
    {'lane_id': 2, 'congestion_level': 'medium', 'score': 0.65, 'metrics': {'vehicle_count': 18, 'average_speed': 55.2, 'space_headway': 90}},
    {'lane_id': 3, 'congestion_level': 'high', 'score': 0.35, 'metrics': {'vehicle_count': 25, 'average_speed': 45.8, 'space_headway': 60}},
    {'lane_id': 4, 'congestion_level': 'low', 'score': 0.78, 'metrics': {'vehicle_count': 15, 'average_speed': 60.5, 'space_headway': 100}}
]

app = FastAPI(title="LaneWise API")

# Enable CORS for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LaneWise system
lanewise = LaneWiseSystem()

def initialize_model():
    """Load the model if it exists or train a new one."""
    model_path = Path("models/clustering_model.joblib")
    if model_path.exists():
        try:
            lanewise.load_model("models/")
            logger.info(f"Loaded model from {model_path}.")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}. Training new model.")
            train_new_model()
    else:
        logger.info(f"Model not found at {model_path}. Training new model.")
        train_new_model()


def train_new_model():
    """Train a new model with sample data."""
    num_samples = 1000
    training_data = pd.DataFrame({
         'Lane_ID': list(range(1, 5)) * (num_samples // 4), 
        'Vehicle_ID': range(num_samples),
        'Global_Time': pd.date_range(start='now', periods=num_samples, freq='1min'),
        'v_Vel': [65, 55, 45, 35] * (num_samples // 4),
        'Space_Headway': [120, 90, 60, 40] * (num_samples // 4),
        'Time_Headway': [3.0, 2.5, 2.0, 1.5] * (num_samples // 4)
    })
    lane_metrics = lanewise.load_and_preprocess_data(training_data)
    lane_metrics = lanewise.train_clustering_model(lane_metrics)
    lanewise.save_model()

# Initialize model 
initialize_model()

@app.get("/")
async def root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to LaneWise API"}

class LaneDataInput(BaseModel):
    lane_id: int
    vehicle_count: int
    avg_speed: float
    avg_space: float
    avg_time: float

@app.post("/evaluate")
async def evaluate_lane_data(data: List[LaneDataInput]):
    logger.info("Received data for evaluation.")
    try:
        data_df = pd.DataFrame([item.dict() for item in data])
        logger.info(f"DataFrame created with columns: {data_df.columns.tolist()}")

        # Log the content of data_df
        logger.debug(f"Data received for evaluation: {data_df}")

        recommendations = lanewise.get_lane_recommendations(data_df)
        logger.info("Lane recommendations generated successfully.")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}. Loading random sample data...")
        traceback.print_exc()
        return SAMPLE_DATA



if __name__ == "__main__":
    logger.info("Starting LaneWise API...")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
