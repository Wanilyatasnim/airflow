"""
Data Extraction Module

This module handles data extraction from various sources.
For this example, we'll use the Titanic dataset from seaborn.
"""

import pandas as pd
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_data(raw_data_path: str) -> None:
    """
    Extract data from source and save to raw data directory.
    
    Args:
        raw_data_path: Path where raw data will be saved
    """
    logger.info(f"Starting data extraction to {raw_data_path}")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
    
    try:
        # Try to load from seaborn (if available)
        try:
            import seaborn as sns
            df = sns.load_dataset('titanic')
            logger.info("Loaded Titanic dataset from seaborn")
        except ImportError:
            # Fallback: Create synthetic Titanic-like dataset
            logger.warning("Seaborn not available, creating synthetic dataset")
            df = _create_synthetic_titanic_data()
        
        # Save to CSV
        df.to_csv(raw_data_path, index=False)
        logger.info(f"Data extracted successfully. Shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
    except Exception as e:
        logger.error(f"Error during data extraction: {str(e)}")
        raise


def _create_synthetic_titanic_data():
    """
    Create a synthetic Titanic-like dataset for demonstration.
    
    Returns:
        pandas.DataFrame: Synthetic Titanic dataset
    """
    import numpy as np
    
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic data
    data = {
        'survived': np.random.choice([0, 1], n_samples, p=[0.6, 0.4]),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'age': np.random.normal(30, 15, n_samples).clip(0, 80),
        'sibsp': np.random.poisson(0.5, n_samples).clip(0, 5),
        'parch': np.random.poisson(0.4, n_samples).clip(0, 4),
        'fare': np.random.lognormal(3, 1.5, n_samples).clip(0, 500),
        'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1]),
    }
    
    df = pd.DataFrame(data)
    
    # Make survival more realistic based on features
    df.loc[(df['sex'] == 'female') & (df['pclass'] <= 2), 'survived'] = np.random.choice([0, 1], 
        size=len(df[(df['sex'] == 'female') & (df['pclass'] <= 2)]), p=[0.1, 0.9])
    df.loc[(df['sex'] == 'male') & (df['age'] > 50), 'survived'] = np.random.choice([0, 1], 
        size=len(df[(df['sex'] == 'male') & (df['age'] > 50)]), p=[0.8, 0.2])
    
    return df


if __name__ == "__main__":
    extract_data("../../data/raw/titanic_data.csv")

