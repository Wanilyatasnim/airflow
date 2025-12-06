"""
Model Training Module

This module handles model training, logging to MLflow, and model registration.
"""

import pandas as pd
import numpy as np
import os
import logging
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set MLflow tracking URI
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')


def train_models(
    train_path: str,
    test_path: str,
    experiment_name: str = 'titanic-survival-prediction'
) -> str:
    """
    Train multiple models and log to MLflow.
    
    Args:
        train_path: Path to training data
        test_path: Path to test data
        experiment_name: MLflow experiment name
        
    Returns:
        Best model name
    """
    logger.info(f"Starting model training")
    logger.info(f"MLflow tracking URI: {MLFLOW_TRACKING_URI}")
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    # Load data
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    X_train = train_df.drop('survived', axis=1)
    y_train = train_df['survived']
    X_test = test_df.drop('survived', axis=1)
    y_test = test_df['survived']
    
    logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Set or create experiment
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            logger.info(f"Created new experiment: {experiment_name}")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Error getting experiment: {e}. Creating new one.")
        experiment_id = mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    models = {
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    }
    
    best_model_name = None
    best_score = 0
    best_model = None
    best_run_id = None
    
    # Train each model
    for model_name, model in models.items():
        logger.info(f"Training {model_name}...")
        
        with mlflow.start_run(run_name=f"{model_name}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}") as run:
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Log parameters
            if model_name == 'logistic_regression':
                mlflow.log_params({
                    'model_type': 'logistic_regression',
                    'max_iter': 1000,
                    'random_state': 42
                })
            elif model_name == 'random_forest':
                mlflow.log_params({
                    'model_type': 'random_forest',
                    'n_estimators': 100,
                    'max_depth': 10,
                    'random_state': 42
                })
            
            # Log metrics
            mlflow.log_metrics({
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
            # Log confusion matrix as artifact
            cm_path = f'/tmp/confusion_matrix_{model_name}.json'
            with open(cm_path, 'w') as f:
                json.dump(cm.tolist(), f)
            mlflow.log_artifact(cm_path)
            
            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                feature_importance = dict(zip(X_train.columns, model.feature_importances_))
                importance_path = f'/tmp/feature_importance_{model_name}.json'
                with open(importance_path, 'w') as f:
                    json.dump(feature_importance, f)
                mlflow.log_artifact(importance_path)
            
            # Log model
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"titanic_{model_name}"
            )
            
            logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Track best model
            if f1 > best_score:
                best_score = f1
                best_model_name = model_name
                best_model = model
                best_run_id = run.info.run_id
    
    logger.info(f"Best model: {best_model_name} with F1 score: {best_score:.4f}")
    
    # Register best model version (already registered during training, but log it)
    if best_model_name and best_run_id:
        try:
            model_name_registry = f"titanic_{best_model_name}"
            logger.info(f"Best model {model_name_registry} was registered during training with run_id: {best_run_id}")
        except Exception as e:
            logger.warning(f"Could not verify model registration: {e}")
    
    return best_model_name

