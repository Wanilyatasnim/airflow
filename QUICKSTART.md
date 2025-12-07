# Quick Start Guide

## Prerequisites

- Docker Desktop installed and running
- At least 4GB RAM available
- Ports 8080, 5000, and 8000 available

## STEP 1 — Verify Environment Before Running

**⚠️ IMPORTANT: Complete these checks before starting services!**

### ✔️ 1.1. Docker Desktop is Running

Without Docker daemon running → nothing will start.

**Check Docker status:**
```bash
docker ps
```

If you see an error, start Docker Desktop and wait for it to fully start.

### ✔️ 1.2. Validate Docker Compose Configuration

Run this in the project root to verify your YAML is valid:

```bash
docker-compose config
```

✅ If you see no errors → your YAML is valid  
❌ If you see errors → fix them before proceeding

### ✔️ 1.3. Check Ports are Free

Verify these ports are available:

- **8080** → Airflow UI
- **5000** → MLflow UI  
- **8000** → FastAPI service

**Check ports (Windows PowerShell):**
```powershell
netstat -ano | findstr :8080
netstat -ano | findstr :5000
netstat -ano | findstr :8000
```

**Check ports (Linux/Mac):**
```bash
lsof -i :8080
lsof -i :5000
lsof -i :8000
```

**If ports are in use:**
- Stop the conflicting service, OR
- Modify port mappings in `docker-compose.yml`

---

## Step-by-Step Setup

### 1. Start All Services

```bash
docker-compose up --build
```

This will:
- Build all Docker images
- Start PostgreSQL database
- Start Airflow webserver and scheduler
- Start MLflow server
- Start FastAPI service

**Note:** First startup may take 5-10 minutes as it downloads images and initializes Airflow.

### 2. Access Services

Once all services are running:

- **Airflow UI**: http://localhost:8080
  - Username: `airflow`
  - Password: `airflow`
  
- **MLflow UI**: http://localhost:5000

- **FastAPI**: http://localhost:8000
  - API Docs: http://localhost:8000/docs
  - Health Check: http://localhost:8000/health

### 3. Trigger the Pipeline

#### Option A: Manual Trigger (Recommended for first run)

1. Open Airflow UI: http://localhost:8080
2. Login with credentials (airflow/airflow)
3. Find the `ml_retraining_pipeline` DAG
4. Toggle it ON (switch on the left)
5. Click the "Play" button to trigger manually

#### Option B: Wait for Schedule

The pipeline runs daily at midnight (schedule: `@daily`). You can modify the schedule in `airflow/dags/retrain_pipeline.py`.

### 4. Monitor Pipeline Execution

1. In Airflow UI, click on the DAG name
2. Click on the graph view to see task dependencies
3. Click on individual tasks to see logs
4. Green = success, Red = failed, Yellow = running

### 5. View Results in MLflow

1. Open MLflow UI: http://localhost:5000
2. Click on "Experiments" to see all runs
3. Click on a run to see:
   - Metrics (accuracy, precision, recall, F1)
   - Parameters
   - Artifacts (confusion matrix, feature importance)
   - Model artifacts

### 6. Test the API

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 25.0,
    "sibsp": 0,
    "parch": 0,
    "fare": 50.0,
    "embarked": "S"
  }'
```

#### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "instances": [
      {
        "pclass": 1,
        "sex": "female",
        "age": 25.0,
        "sibsp": 0,
        "parch": 0,
        "fare": 50.0,
        "embarked": "S"
      },
      {
        "pclass": 3,
        "sex": "male",
        "age": 30.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 20.0,
        "embarked": "S"
      }
    ]
  }'
```

## Troubleshooting

### Services Won't Start

1. Check Docker is running: `docker ps`
2. Check logs: `docker-compose logs [service-name]`
3. Check port conflicts: Ensure 8080, 5000, 8000 are free

### Airflow DAG Not Appearing

1. Wait 2-3 minutes for Airflow to scan DAGs
2. Check DAG file syntax: `docker-compose exec airflow-webserver python -m py_compile /opt/airflow/dags/retrain_pipeline.py`
3. Check Airflow logs: `docker-compose logs airflow-scheduler`

### Model Not Loading in API

1. Ensure pipeline has run at least once
2. Check MLflow has a production model: http://localhost:5000
3. Check API logs: `docker-compose logs fastapi`
4. The API will automatically retry loading from MLflow

### Pipeline Tasks Failing

1. Check individual task logs in Airflow UI
2. Common issues:
   - MLflow not accessible: Check MLflow service is running
   - Data path issues: Check volume mounts in docker-compose.yml
   - Memory issues: Increase Docker memory allocation

## Stopping Services

```bash
docker-compose down
```

To remove volumes (clears all data):
```bash
docker-compose down -v
```

## Next Steps

1. **Customize the Dataset**: Modify `src/extract.py` to use your own data source
2. **Add More Models**: Edit `src/train.py` to add additional model types
3. **Modify Schedule**: Change the schedule in `airflow/dags/retrain_pipeline.py`
4. **Add Monitoring**: Integrate with monitoring tools like Prometheus/Grafana
5. **Add Alerting**: Configure email/Slack notifications in Airflow

## Project Structure

```
mlops-retraining/
├── airflow/          # Airflow configuration and DAGs
├── mlflow/           # MLflow server configuration
├── api/              # FastAPI inference service
├── src/              # ML pipeline source code
├── data/             # Data directories (mounted as volumes)
└── docker-compose.yml # Service orchestration
```

## Support

For issues or questions:
1. Check the logs: `docker-compose logs`
2. Review the README.md for detailed documentation
3. Check Airflow/MLflow/FastAPI official documentation


