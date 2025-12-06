FROM python:3.10-slim

WORKDIR /mlflow

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==2.9.2 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.0

# Create directories for artifacts and backend store
RUN mkdir -p /mlflow/artifacts /mlflow/backend_store

EXPOSE 5000

CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "file:///mlflow/backend_store", "--default-artifact-root", "file:///mlflow/artifacts"]

