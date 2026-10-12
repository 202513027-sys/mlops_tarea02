"""
training_dag.py
===============
DAG de Apache Airflow que orquesta el pipeline de entrenamiento del AutoML.

    validar_datos -> preprocesar_datos -> optimizar_hiperparametros ->
    entrenar_modelos -> backtesting -> seleccionar_campeon ->
    registrar_experimentos -> generar_reporte

- Ejecucion MANUAL: schedule=None (boton "Trigger DAG" en la interfaz, o
  `airflow dags test automl_venta_financiera`). La periodicidad que tendria
  en produccion se discute en docs/decisiones_diseno.md (D9).
- Granularidad (D3): una tarea por etapa; cada tarea procesa todos los modelos.
- Reintentos (retries / retry_delay) y demas parametros: config/config.yaml.
- XCom: solo viajan la ruta de la carpeta de la corrida y su identificador.
  Datos y modelos van a disco (data/intermedios/<corrida_id>/) y a MLflow.
- Toda la logica esta en src/pipeline.py (la misma que usa main.py).

Parametros al lanzar ("Trigger DAG w/ config"), todos opcionales (D2):
    {"paralelo": true, "n_jobs": 2, "use_gpu": "auto", "input_path": "data/muestra"}
"""

import os
import sys
from datetime import datetime, timedelta

import yaml
from airflow import DAG
from airflow.exceptions import AirflowFailException
from airflow.models.param import Param
from airflow.operators.python import PythonOperator

# Raiz del proyecto: /opt/proyecto dentro de Docker (variable PROYECTO_DIR)
RAIZ = os.environ.get("PROYECTO_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)
RUTA_CONFIG = os.path.join(RAIZ, "config", "config.yaml")

# Solo se lee el YAML al definir el DAG (rapido). Las librerias pesadas
# (pandas, sklearn, mlflow...) se importan dentro de cada tarea.
with open(RUTA_CONFIG, encoding="utf-8") as f:
    CFG_AIRFLOW = yaml.safe_load(f)["airflow"]

# Mismo orden que src/pipeline.py -> ORDEN_TAREAS (se repite aqui para no
# importar librerias pesadas al leer el DAG)
ORDEN_TAREAS = [
    "validar_datos",
    "preprocesar_datos",
    "optimizar_hiperparametros",
    "entrenar_modelos",
    "backtesting",
    "seleccionar_campeon",
    "registrar_experimentos",
    "generar_reporte",
]


def ejecutar(nombre_tarea: str, **context):
    """Callable comun de todas las tareas: llama a src/pipeline.py."""
    from src import pipeline
    from src.validacion import DatosInvalidosError

    ti = context["ti"]
    dag_run_id = context["run_id"]  # se guarda como tag airflow_dag_run_id en MLflow

    try:
        if nombre_tarea == "validar_datos":
            overrides = {k: v for k, v in (context.get("params") or {}).items() if v is not None}
            salida = pipeline.ejecutar_tarea(nombre_tarea, dag_run_id, ruta_config=RUTA_CONFIG,
                                             overrides=overrides)
        else:
            carpeta = ti.xcom_pull(task_ids="validar_datos", key="carpeta_corrida")
            salida = pipeline.ejecutar_tarea(nombre_tarea, dag_run_id, carpeta_corrida=carpeta)
    except DatosInvalidosError as e:
        # Reintentar no arregla datos malos: se falla de inmediato, sin reintentos
        raise AirflowFailException(str(e)) from e

    # XCom: solo rutas, identificadores y valores livianos
    for clave, valor in salida.items():
        ti.xcom_push(key=clave, value=valor)


default_args = {
    "owner": CFG_AIRFLOW["owner"],
    "depends_on_past": False,
    "retries": CFG_AIRFLOW["retries"],
    "retry_delay": timedelta(minutes=CFG_AIRFLOW["retry_delay_minutes"]),
    "email_on_failure": False,
}

with DAG(
    dag_id=CFG_AIRFLOW["dag_id"],
    description="AutoML propension de venta: validar -> preprocesar -> HPO -> entrenar -> "
                "backtesting -> campeon -> MLflow -> reporte",
    default_args=default_args,
    schedule=None,              # solo ejecucion manual (ver D9)
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,          # dos ejecuciones a la vez no deben pisar la misma corrida
    tags=CFG_AIRFLOW["tags"],
    params={
        "paralelo": Param(None, type=["null", "boolean"],
                          description="true/false: entrenar en paralelo (vacio = config.yaml)"),
        "n_jobs": Param(None, type=["null", "integer", "string"],
                        description='Workers en paralelo: numero o "auto" (vacio = config.yaml)'),
        "use_gpu": Param(None, type=["null", "boolean", "string"],
                         description='"auto", true o false (vacio = config.yaml)'),
        "input_path": Param(None, type=["null", "string"],
                            description="Carpeta de datos, ej. data/muestra (vacio = config.yaml)"),
    },
) as dag:

    tareas = [
        PythonOperator(
            task_id=nombre,
            python_callable=ejecutar,
            op_kwargs={"nombre_tarea": nombre},
        )
        for nombre in ORDEN_TAREAS
    ]

    # Encadenar en orden: t1 >> t2 >> ... >> t8
    for anterior, siguiente in zip(tareas, tareas[1:]):
        anterior >> siguiente
