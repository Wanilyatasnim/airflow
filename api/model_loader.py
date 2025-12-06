"""
Model Loader Module

This module handles loading and caching the production model from MLflow.
"""

import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
import mlflow.sklearn
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


class ModelLoader:
    """
    Model loader that fetches and caches the production model from MLflow.
    """
    
    def __init__(self):
        """Initialize the model loader."""
        self.model = None
        self.model_version = None
        self.model_name = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient(MLFLOW_TRACKING_URI)
        self._load_production_model()
    
    def _load_production_model(self) -> None:
        """
        Load the production model from MLflow registry.
        """
        try:
            logger.info("Loading production model from MLflow...")
            
            # Search for production models
            registered_models = self.client.search_registered_models()
            production_model = None
            
            for rm in registered_models:
                for version in rm.latest_versions:
                    if version.current_stage == "Production":
                        production_model = version
                        self.model_name = version.name
                        self.model_version = f"{version.name} v{version.version}"
                        logger.info(f"Found production model: {self.model_version}")
                        break
                if production_model:
                    break
            
            if production_model:
                # Load model from MLflow
                model_uri = f"models:/{production_model.name}/Production"
                self.model = mlflow.sklearn.load_model(model_uri)
                logger.info(f"Successfully loaded model: {self.model_version}")
            else:
                logger.warning("No production model found in MLflow registry")
                # Try to load from local file as fallback
                local_model_path = "/opt/airflow/data/models/production_model.pkl"
                if os.path.exists(local_model_path):
                    import pickle
                    with open(local_model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    logger.info(f"Loaded model from local file: {local_model_path}")
                else:
                    logger.error("No production model available")
                    
        except Exception as e:
            logger.error(f"Error loading production model: {str(e)}")
            self.model = None
    
    def get_model(self):
        """
        Get the loaded model.
        
        Returns:
            Loaded model or None
        """
        return self.model
    
    def get_model_version(self) -> Optional[str]:
        """
        Get the model version string.
        
        Returns:
            Model version string or None
        """
        return self.model_version
    
    def reload_model(self) -> None:
        """
        Reload the production model from MLflow.
        Useful when a new model is promoted to production.
        """
        logger.info("Reloading production model...")
        self.model = None
        self.model_version = None
        self.model_name = None
        self._load_production_model()

