# Imagen del servidor de MLflow (servicio "mlflow" de docker-compose.yml).
# Va separada de la imagen de Airflow porque el servidor de MLflow 3 necesita
# SQLAlchemy 2, y Airflow 2.10 usa SQLAlchemy 1.4: juntos en una misma imagen
# chocan. La version debe ser la misma que el cliente (requirements-docker.txt).
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow==3.17.0"

ENV MLFLOW_DISABLE_AGENT_HINT=1
RUN mkdir -p /mlflow
WORKDIR /mlflow
EXPOSE 5000
