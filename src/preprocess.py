"""
Data Preprocessing Module

This module handles data cleaning, feature engineering, and train/test splitting.
"""

import pandas as pd
import numpy as np
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def preprocess_data(
    raw_data_path: str,
    processed_train_path: str,
    processed_test_path: str
) -> None:
    """
    Preprocess raw data: clean, encode, scale, and split into train/test.
    
    Args:
        raw_data_path: Path to raw data CSV
        processed_train_path: Path to save processed training data
        processed_test_path: Path to save processed test data
    """
    logger.info(f"Starting data preprocessing")
    logger.info(f"Reading data from {raw_data_path}")
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(processed_train_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_test_path), exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(raw_data_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    # Clean data
    df_clean = _clean_data(df)
    logger.info(f"After cleaning shape: {df_clean.shape}")
    
    # Feature engineering
    df_features = _engineer_features(df_clean)
    logger.info(f"After feature engineering shape: {df_features.shape}")
    
    # Encode categorical variables
    df_encoded, encoders = _encode_features(df_features)
    logger.info(f"After encoding shape: {df_encoded.shape}")
    
    # Split features and target
    X = df_encoded.drop('survived', axis=1)
    y = df_encoded['survived']
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_df = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_df = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Combine features and target
    train_df = pd.concat([X_train_df, y_train], axis=1)
    test_df = pd.concat([X_test_df, y_test], axis=1)
    
    # Save processed data
    train_df.to_csv(processed_train_path, index=False)
    test_df.to_csv(processed_test_path, index=False)
    
    # Save scaler and encoders for inference
    model_dir = os.path.join(os.path.dirname(processed_train_path), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(os.path.join(model_dir, 'encoders.pkl'), 'wb') as f:
        pickle.dump(encoders, f)
    
    logger.info(f"Preprocessing complete. Train saved to {processed_train_path}")
    logger.info(f"Test saved to {processed_test_path}")


def _clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers.
    
    Args:
        df: Raw dataframe
        
    Returns:
        Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Handle missing values in age (fill with median)
    if 'age' in df_clean.columns:
        df_clean['age'].fillna(df_clean['age'].median(), inplace=True)
    
    # Handle missing values in embarked (fill with mode)
    if 'embarked' in df_clean.columns:
        df_clean['embarked'].fillna(df_clean['embarked'].mode()[0], inplace=True)
    
    # Remove rows with missing target
    if 'survived' in df_clean.columns:
        df_clean = df_clean.dropna(subset=['survived'])
    
    # Clip outliers in fare
    if 'fare' in df_clean.columns:
        q99 = df_clean['fare'].quantile(0.99)
        df_clean['fare'] = df_clean['fare'].clip(upper=q99)
    
    return df_clean


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing ones.
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Dataframe with engineered features
    """
    df_feat = df.copy()
    
    # Create family size feature
    if 'sibsp' in df_feat.columns and 'parch' in df_feat.columns:
        df_feat['family_size'] = df_feat['sibsp'] + df_feat['parch'] + 1
    
    # Create is_alone feature
    if 'family_size' in df_feat.columns:
        df_feat['is_alone'] = (df_feat['family_size'] == 1).astype(int)
    
    # Create age groups
    if 'age' in df_feat.columns:
        df_feat['age_group'] = pd.cut(
            df_feat['age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=['child', 'teen', 'adult', 'middle_age', 'senior']
        )
    
    # Create fare groups
    if 'fare' in df_feat.columns:
        df_feat['fare_group'] = pd.qcut(
            df_feat['fare'],
            q=4,
            labels=['low', 'medium', 'high', 'very_high'],
            duplicates='drop'
        )
    
    return df_feat


def _encode_features(df: pd.DataFrame) -> tuple:
    """
    Encode categorical variables.
    
    Args:
        df: Dataframe with categorical features
        
    Returns:
        Tuple of (encoded dataframe, encoders dict)
    """
    df_encoded = df.copy()
    encoders = {}
    
    # Encode categorical columns
    categorical_cols = ['sex', 'embarked', 'age_group', 'fare_group']
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
            encoders[col] = le
    
    # Select final features
    feature_cols = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
                    'family_size', 'is_alone', 'age_group', 'fare_group']
    
    # Keep only available columns
    available_cols = [col for col in feature_cols if col in df_encoded.columns]
    available_cols.append('survived')  # Add target
    
    df_encoded = df_encoded[available_cols]
    
    return df_encoded, encoders

