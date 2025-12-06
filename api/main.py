"""
FastAPI Inference Service

This service provides REST API endpoints for model inference.
It loads the production model from MLflow and serves predictions.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pandas as pd
import os
import logging
from typing import List
from model_loader import ModelLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="ML Model Inference API",
    description="API for Titanic survival prediction",
    version="1.0.0"
)

# Initialize model loader
model_loader = ModelLoader()

# Request/Response models
class PredictionRequest(BaseModel):
    pclass: int
    sex: str
    age: float
    sibsp: int
    parch: int
    fare: float
    embarked: str

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

class BatchPredictionRequest(BaseModel):
    instances: List[PredictionRequest]

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]


@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        dict: Health status
    """
    try:
        model = model_loader.get_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "model_version": model_loader.get_model_version() if model else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Single prediction endpoint.
    
    Args:
        request: Prediction request with feature values
        
    Returns:
        PredictionResponse: Prediction result with probability
    """
    try:
        model = model_loader.get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features
        features = _prepare_features(request)
        
        # Make prediction
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            model_version=model_loader.get_model_version()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction endpoint.
    
    Args:
        request: Batch prediction request with multiple instances
        
    Returns:
        BatchPredictionResponse: Batch prediction results
    """
    try:
        model = model_loader.get_model()
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Prepare features for all instances
        features_list = [_prepare_features(inst) for inst in request.instances]
        features_df = pd.concat(features_list, ignore_index=True)
        
        # Make predictions
        predictions = model.predict(features_df)
        probabilities = model.predict_proba(features_df)[:, 1]
        
        # Format responses
        results = [
            PredictionResponse(
                prediction=int(pred),
                probability=float(prob),
                model_version=model_loader.get_model_version()
            )
            for pred, prob in zip(predictions, probabilities)
        ]
        
        return BatchPredictionResponse(predictions=results)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


def _prepare_features(request: PredictionRequest) -> pd.DataFrame:
    """
    Prepare features from request for model prediction.
    
    Args:
        request: Prediction request
        
    Returns:
        DataFrame with prepared features
    """
    # Encode categorical variables (matching preprocessing)
    sex_encoded = 1 if request.sex.lower() == 'male' else 0
    embarked_map = {'S': 0, 'C': 1, 'Q': 2}
    embarked_encoded = embarked_map.get(request.embarked.upper(), 0)
    
    # Create feature dictionary
    features = {
        'pclass': [request.pclass],
        'sex': [sex_encoded],
        'age': [request.age],
        'sibsp': [request.sibsp],
        'parch': [request.parch],
        'fare': [request.fare],
        'embarked': [embarked_encoded],
        'family_size': [request.sibsp + request.parch + 1],
        'is_alone': [1 if (request.sibsp + request.parch) == 0 else 0],
    }
    
    # Add age_group (simplified encoding)
    if request.age <= 12:
        age_group = 0  # child
    elif request.age <= 18:
        age_group = 1  # teen
    elif request.age <= 35:
        age_group = 2  # adult
    elif request.age <= 60:
        age_group = 3  # middle_age
    else:
        age_group = 4  # senior
    features['age_group'] = [age_group]
    
    # Add fare_group (simplified encoding)
    if request.fare < 10:
        fare_group = 0  # low
    elif request.fare < 30:
        fare_group = 1  # medium
    elif request.fare < 100:
        fare_group = 2  # high
    else:
        fare_group = 3  # very_high
    features['fare_group'] = [fare_group]
    
    # Create DataFrame
    df = pd.DataFrame(features)
    
    # Ensure column order matches training (add missing columns with 0)
    expected_columns = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
                       'family_size', 'is_alone', 'age_group', 'fare_group']
    
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns
    df = df[expected_columns]
    
    return df


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

