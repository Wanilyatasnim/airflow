"""
Utility Functions Module

This module contains utility functions for the ML pipeline.
"""

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


def load_model_for_api(model_path: str) -> None:
    """
    Load the production model from MLflow and save it for the API service.
    
    Args:
        model_path: Path where the model will be saved for API
    """
    logger.info("Loading production model for API service")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    
    try:
        # Find production model
        registered_models = client.search_registered_models()
        production_model = None
        
        for rm in registered_models:
            for version in rm.latest_versions:
                if version.current_stage == "Production":
                    production_model = version
                    logger.info(f"Found production model: {version.name} v{version.version}")
                    break
            if production_model:
                break
        
        if production_model:
            # Load model from MLflow
            model_uri = f"models:/{production_model.name}/{production_model.current_stage}"
            model = mlflow.sklearn.load_model(model_uri)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save model for API
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            logger.info(f"Model saved to {model_path}")
            logger.info(f"Model ready for API service")
        else:
            logger.warning("No production model found. API will need to load from MLflow directly.")
            
    except Exception as e:
        logger.error(f"Error loading model for API: {str(e)}")
        # Don't raise - API can still load from MLflow directly
        pass

