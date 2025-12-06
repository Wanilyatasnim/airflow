# MLOps Retraining Pipeline

A complete production-ready MLOps project featuring automated ML model retraining using Apache Airflow, MLflow, and FastAPI.

## 🏗️ Architecture

- **Apache Airflow**: Orchestrates the daily ML retraining pipeline
- **MLflow**: Tracks experiments, versions models, and manages model registry
- **FastAPI**: Serves model predictions via REST API
- **PostgreSQL**: Backend database for Airflow
- **Docker Compose**: Orchestrates all services

## 📁 Project Structure

```
mlops-retraining/
├── airflow/
│   ├── dags/
│   │   └── retrain_pipeline.py    # Main Airflow DAG
│   ├── Dockerfile
│   └── requirements.txt
├── mlflow/
│   ├── Dockerfile
│   ├── artifacts/                 # MLflow artifacts
│   └── backend_store/             # MLflow backend store
├── api/
│   ├── main.py                    # FastAPI application
│   ├── model_loader.py            # Model loading from MLflow
│   ├── Dockerfile
│   └── requirements.txt
├── src/
│   ├── extract.py                 # Data extraction
│   ├── preprocess.py              # Data preprocessing
│   ├── train.py                   # Model training
│   ├── evaluate.py                # Model evaluation & promotion
│   └── utils.py                   # Utility functions
├── data/
│   ├── raw/                       # Raw data
│   ├── processed/                 # Processed data
│   └── models/                    # Saved models
└── docker-compose.yml
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose installed
- At least 4GB RAM available
- Ports 8080, 5000, and 8000 available

### Running the Project

1. **Start all services:**
   ```bash
   docker-compose up --build
   ```

2. **Access services:**
   - **Airflow UI**: http://localhost:8080
     - Username: `airflow`
     - Password: `airflow`
   - **MLflow UI**: http://localhost:5000
   - **FastAPI**: http://localhost:8000
   - **API Docs**: http://localhost:8000/docs

3. **Trigger the pipeline:**
   - Go to Airflow UI (http://localhost:8080)
   - Find the `ml_retraining_pipeline` DAG
   - Toggle it ON and trigger it manually or wait for the daily schedule

## 🔄 Pipeline Workflow

The Airflow DAG runs daily and executes the following tasks:

1. **extract_data**: Downloads/loads the Titanic dataset
2. **preprocess**: Cleans data, engineers features, splits train/test
3. **train_model**: Trains Logistic Regression and Random Forest models
4. **evaluate_model**: Compares new models with production model
5. **load_model**: Loads best model for API service

## 📊 Model Training

The pipeline trains two models:
- **Logistic Regression**: Baseline model
- **Random Forest**: Ensemble model

Both models are:
- Logged to MLflow with metrics (accuracy, precision, recall, F1)
- Registered in MLflow Model Registry
- Compared against the current production model
- Promoted to production if they perform better

## 🎯 API Endpoints

### Health Check
```bash
GET /health
```

### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "pclass": 1,
  "sex": "female",
  "age": 25.0,
  "sibsp": 0,
  "parch": 0,
  "fare": 50.0,
  "embarked": "S"
}
```

### Batch Prediction
```bash
POST /predict/batch
Content-Type: application/json

{
  "instances": [
    {
      "pclass": 1,
      "sex": "female",
      "age": 25.0,
      "sibsp": 0,
      "parch": 0,
      "fare": 50.0,
      "embarked": "S"
    }
  ]
}
```

## 🔧 Configuration

### Environment Variables

- `MLFLOW_TRACKING_URI`: MLflow server URI (default: `http://mlflow:5000`)
- `AIRFLOW__CORE__EXECUTOR`: Airflow executor (default: `LocalExecutor`)

### MLflow

- Tracking URI: `http://localhost:5000`
- Artifacts stored in Docker volume
- Model registry managed through MLflow UI

## 📝 Notes

- The pipeline uses the Titanic dataset (synthetic if seaborn is unavailable)
- Models are automatically promoted to production if F1 score improves by at least 0.01
- The FastAPI service automatically loads the latest production model from MLflow
- All data and models are persisted in Docker volumes

## 🐛 Troubleshooting

1. **Services not starting**: Check Docker logs with `docker-compose logs`
2. **Airflow DAG not appearing**: Wait a few minutes for Airflow to scan DAGs
3. **Model not loading**: Ensure MLflow service is running and a production model exists
4. **Port conflicts**: Modify ports in `docker-compose.yml`

## 📚 Additional Resources

- [Airflow Documentation](https://airflow.apache.org/docs/)
- [MLflow Documentation](https://www.mlflow.org/docs/latest/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 📄 License

This project is provided as-is for educational and demonstration purposes.

