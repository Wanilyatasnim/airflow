# Project Structure

```
mlops-retraining/
│
├── airflow/                          # Apache Airflow Configuration
│   ├── dags/
│   │   └── retrain_pipeline.py      # Main Airflow DAG for ML pipeline
│   ├── Dockerfile                    # Airflow container image
│   └── requirements.txt              # Python dependencies for Airflow
│
├── mlflow/                           # MLflow Server Configuration
│   ├── Dockerfile                    # MLflow container image
│   ├── artifacts/                    # MLflow artifacts storage (volume)
│   └── backend_store/                 # MLflow backend store (volume)
│
├── api/                              # FastAPI Inference Service
│   ├── main.py                       # FastAPI application with endpoints
│   ├── model_loader.py               # Model loading from MLflow
│   ├── Dockerfile                    # FastAPI container image
│   └── requirements.txt              # Python dependencies for API
│
├── src/                              # ML Pipeline Source Code
│   ├── extract.py                    # Data extraction module
│   ├── preprocess.py                 # Data preprocessing module
│   ├── train.py                      # Model training module
│   ├── evaluate.py                   # Model evaluation & promotion
│   └── utils.py                      # Utility functions
│
├── data/                             # Data Storage (volumes)
│   ├── raw/                          # Raw data storage
│   ├── processed/                    # Processed data storage
│   └── models/                       # Saved models for API
│
├── docker-compose.yml                # Docker Compose orchestration
├── README.md                         # Main project documentation
├── QUICKSTART.md                     # Quick start guide
├── .gitignore                        # Git ignore rules
└── .dockerignore                     # Docker ignore rules
```

## File Descriptions

### Airflow
- **retrain_pipeline.py**: Defines the DAG with 5 tasks: extract → preprocess → train → evaluate → load_model
- **Dockerfile**: Extends Apache Airflow base image, installs dependencies
- **requirements.txt**: Python packages needed for Airflow tasks

### MLflow
- **Dockerfile**: Sets up MLflow server with file-based backend
- **artifacts/**: Stores model artifacts (mounted as volume)
- **backend_store/**: Stores MLflow metadata (mounted as volume)

### API (FastAPI)
- **main.py**: REST API with `/health`, `/predict`, `/predict/batch` endpoints
- **model_loader.py**: Loads production model from MLflow registry
- **Dockerfile**: FastAPI service container
- **requirements.txt**: FastAPI and ML dependencies

### Source Code (src/)
- **extract.py**: Loads Titanic dataset (or creates synthetic data)
- **preprocess.py**: Cleans data, engineers features, splits train/test
- **train.py**: Trains Logistic Regression and Random Forest, logs to MLflow
- **evaluate.py**: Compares models, promotes best to production
- **utils.py**: Helper functions for model loading

### Data Directories
- **raw/**: Raw input data
- **processed/**: Preprocessed train/test splits
- **models/**: Saved models for API service

## Service Ports

- **Airflow UI**: 8080
- **MLflow UI**: 5000
- **FastAPI**: 8000
- **PostgreSQL**: 5432 (internal)

## Data Flow

1. **Extract**: Raw data → `data/raw/`
2. **Preprocess**: Raw data → Processed data → `data/processed/`
3. **Train**: Processed data → Models → MLflow registry
4. **Evaluate**: Compare models → Promote best → Production stage
5. **Load**: Production model → `data/models/` → FastAPI service

## Volume Mounts

- `./airflow/dags` → `/opt/airflow/dags` (DAG files)
- `./src` → `/opt/airflow/src` (Source code)
- `./data` → `/opt/airflow/data` (Data storage)
- MLflow artifacts and backend store are Docker volumes

