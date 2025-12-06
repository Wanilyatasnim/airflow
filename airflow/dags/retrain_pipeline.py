"""
ML Retraining Pipeline DAG

This DAG orchestrates a complete ML retraining pipeline:
1. Extract data from source
2. Preprocess and prepare data
3. Train multiple models
4. Evaluate and compare with production model
5. Promote best model to production
6. Load model for inference service
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.utils.dates import days_ago
import sys
import os

# Add src directory to path
sys.path.insert(0, '/opt/airflow/src')

from extract import extract_data
from preprocess import preprocess_data
from train import train_models
from evaluate import evaluate_and_promote
from utils import load_model_for_api

# Default arguments for the DAG
default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Define the DAG
dag = DAG(
    'ml_retraining_pipeline',
    default_args=default_args,
    description='Daily ML model retraining pipeline',
    schedule_interval='@daily',  # Run daily
    start_date=days_ago(1),
    catchup=False,
    tags=['mlops', 'retraining', 'mlflow'],
)

# Task 1: Extract data
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    op_kwargs={
        'raw_data_path': '/opt/airflow/data/raw/titanic_data.csv'
    },
    dag=dag,
)

# Task 2: Preprocess data
preprocess_task = PythonOperator(
    task_id='preprocess',
    python_callable=preprocess_data,
    op_kwargs={
        'raw_data_path': '/opt/airflow/data/raw/titanic_data.csv',
        'processed_train_path': '/opt/airflow/data/processed/train.csv',
        'processed_test_path': '/opt/airflow/data/processed/test.csv',
    },
    dag=dag,
)

# Task 3: Train models
train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_models,
    op_kwargs={
        'train_path': '/opt/airflow/data/processed/train.csv',
        'test_path': '/opt/airflow/data/processed/test.csv',
        'experiment_name': 'titanic-survival-prediction',
    },
    dag=dag,
)

# Task 4: Evaluate and promote model
evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_and_promote,
    op_kwargs={
        'test_path': '/opt/airflow/data/processed/test.csv',
        'experiment_name': 'titanic-survival-prediction',
    },
    dag=dag,
)

# Task 5: Load model for API
load_model_task = PythonOperator(
    task_id='load_model',
    python_callable=load_model_for_api,
    op_kwargs={
        'model_path': '/opt/airflow/data/models/production_model.pkl',
    },
    dag=dag,
)

# Define task dependencies
extract_task >> preprocess_task >> train_task >> evaluate_task >> load_model_task

