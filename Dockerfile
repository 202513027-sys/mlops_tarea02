# Imagen del proyecto: Airflow 2.10.5 (Python 3.11) + librerias del AutoML.
# La usan los servicios "airflow" e "init-permisos" de docker-compose.yml.
#
# El codigo del proyecto NO se copia dentro de la imagen: docker-compose lo
# monta como volumen en /opt/proyecto. Asi un cambio en el codigo no obliga a
# reconstruir la imagen (solo se reconstruye si cambia requirements-docker.txt).

FROM apache/airflow:2.10.5-python3.11

# libgomp1: libreria del sistema que necesitan LightGBM y XGBoost (OpenMP)
USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Librerias de Python (se fija apache-airflow para que pip no lo cambie)
USER airflow
COPY requirements-docker.txt /tmp/requirements-docker.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /tmp/requirements-docker.txt

ENV PROYECTO_DIR=/opt/proyecto \
    PYTHONPATH=/opt/proyecto \
    MLFLOW_DISABLE_AGENT_HINT=1
WORKDIR /opt/proyecto
