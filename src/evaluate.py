"""
Model Evaluation and Promotion Module

This module compares new models with production models and promotes better ones.
"""

import pandas as pd
import numpy as np
import os
import logging
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


def evaluate_and_promote(
    test_path: str,
    experiment_name: str = 'titanic-survival-prediction',
    metric_threshold: float = 0.01  # Minimum improvement to promote
) -> None:
    """
    Evaluate latest model and promote to production if better than current production model.
    
    Args:
        test_path: Path to test data
        experiment_name: MLflow experiment name
        metric_threshold: Minimum improvement in F1 score to promote
    """
    logger.info("Starting model evaluation and promotion")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = MlflowClient(MLFLOW_TRACKING_URI)
    
    # Load test data
    test_df = pd.read_csv(test_path)
    X_test = test_df.drop('survived', axis=1)
    y_test = test_df['survived']
    
    logger.info(f"Test set shape: {X_test.shape}")
    
    # Get the latest run from the experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.error(f"Experiment {experiment_name} not found")
            return
        
        # Get latest runs (get last 2 runs - one for each model type)
        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10  # Get last 10 runs to find the best from latest training session
        )
        
        if runs.empty:
            logger.error("No runs found in experiment")
            return
        
        # Find the best run from the latest training session (runs with similar timestamps)
        # For simplicity, we'll take the run with highest F1 score from recent runs
        runs['f1_score'] = runs['metrics.f1_score']
        best_recent_run = runs.nlargest(1, 'f1_score').iloc[0]
        latest_run_id = best_recent_run['run_id']
        latest_f1 = best_recent_run['f1_score']
        
        logger.info(f"Best recent run ID: {latest_run_id}, F1 score: {latest_f1:.4f}")
        
        # Get production model
        production_model = None
        production_f1 = 0.0
        
        try:
            # Try to get production model
            registered_models = client.search_registered_models()
            
            for rm in registered_models:
                for version in rm.latest_versions:
                    if version.current_stage == "Production":
                        production_model = version
                        # Get metrics from production model
                        prod_run = mlflow.get_run(version.run_id)
                        if 'f1_score' in prod_run.data.metrics:
                            production_f1 = prod_run.data.metrics['f1_score']
                        logger.info(f"Found production model: {version.name} v{version.version}, F1: {production_f1:.4f}")
                        break
        except Exception as e:
            logger.warning(f"Could not find production model: {e}. This might be the first run.")
        
        # Compare models
        if production_model is None:
            # No production model exists, promote latest to production
            logger.info("No production model found. Promoting latest model to production.")
            _promote_to_production(client, latest_run_id, experiment_name)
        elif latest_f1 > production_f1 + metric_threshold:
            # New model is better, promote it
            improvement = latest_f1 - production_f1
            logger.info(f"New model is better! Improvement: {improvement:.4f}")
            logger.info(f"Promoting latest model to production...")
            _promote_to_production(client, latest_run_id, experiment_name)
        else:
            logger.info(f"Production model is better or similar. Keeping current production model.")
            logger.info(f"Production F1: {production_f1:.4f}, Latest F1: {latest_f1:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        raise


def _promote_to_production(client: MlflowClient, run_id: str, experiment_name: str) -> None:
    """
    Promote a model version to production stage.
    
    Args:
        client: MLflow client
        run_id: Run ID of the model to promote
        experiment_name: Experiment name
    """
    try:
        # Get the model URI from the run
        run = mlflow.get_run(run_id)
        model_uri = f"runs:/{run_id}/model"
        
        # Find registered model name
        registered_models = client.search_registered_models()
        model_name = None
        
        for rm in registered_models:
            for version in rm.latest_versions:
                if version.run_id == run_id:
                    model_name = version.name
                    break
            if model_name:
                break
        
        if not model_name:
            # Register new model
            model_name = f"titanic_best_model"
            logger.info(f"Registering new model: {model_name}")
            model_version = mlflow.register_model(model_uri, model_name)
        else:
            # Get latest version of the model
            model_versions = client.search_model_versions(f"name='{model_name}'")
            model_version = None
            for mv in model_versions:
                if mv.run_id == run_id:
                    model_version = mv
                    break
        
        if model_version:
            # Transition to Production
            client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production",
                archive_existing_versions=True  # Archive old production versions
            )
            logger.info(f"Successfully promoted {model_name} v{model_version.version} to Production")
        else:
            logger.error("Could not find model version to promote")
            
    except Exception as e:
        logger.error(f"Error promoting model to production: {str(e)}")
        raise

