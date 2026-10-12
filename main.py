"""
main.py
=======
Ejecucion LOCAL del AutoML (sin Docker ni Airflow), igual que en la P1/P2.

Desde el parcial, main.py ejecuta exactamente las mismas tareas que el DAG
de Airflow (src/pipeline.py), una tras otra:

  validar_datos -> preprocesar_datos -> optimizar_hiperparametros ->
  entrenar_modelos -> backtesting -> seleccionar_campeon ->
  registrar_experimentos -> generar_reporte

Comandos:
    python main.py
    python main.py --input data/input --config config/config.yaml
    python main.py --parallel            # fuerza modo paralelo
    python main.py --sequential          # fuerza modo secuencial
    python main.py --input data/muestra  # prueba rapida con una muestra

MLflow: si "docker compose up" esta corriendo, los runs van al mismo servidor
(http://localhost:5000). Si no, se guardan en mlflow_local.db (respaldo).
"""

import argparse
import sys
from datetime import datetime

from src import pipeline
from src.validacion import DatosInvalidosError


def parsear_argumentos():
    """Define y parsea los argumentos de linea de comandos."""
    parser = argparse.ArgumentParser(description="AutoML - Propension de venta financiera (Parcial)")
    parser.add_argument("--input", type=str, default=None,
                        help="Carpeta con los CSV particionados (sobreescribe config.yaml)")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                        help="Ruta al archivo de configuracion YAML")
    modo = parser.add_mutually_exclusive_group()
    modo.add_argument("--parallel", action="store_true", help="Fuerza el entrenamiento en paralelo")
    modo.add_argument("--sequential", action="store_true", help="Fuerza el entrenamiento secuencial")
    parser.add_argument("--n-jobs", type=int, default=None, help="Numero de workers en modo paralelo")
    return parser.parse_args()


def main():
    """Ejecuta todas las tareas del pipeline en orden."""
    args = parsear_argumentos()
    overrides = {
        "input_path": args.input,
        "paralelo": True if args.parallel else (False if args.sequential else None),
        "n_jobs": args.n_jobs,
    }
    dag_run_id = "local__" + datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    carpeta = None
    try:
        for tarea in pipeline.ORDEN_TAREAS:
            salida = pipeline.ejecutar_tarea(tarea, dag_run_id, carpeta_corrida=carpeta,
                                             ruta_config=args.config, overrides=overrides)
            carpeta = salida["carpeta_corrida"]
    except FileNotFoundError as e:
        print(f"\n[ERROR] No se encontro un archivo: {e}")
        print("Revisa que los CSV esten en data/input/ (ver README -> 'Como obtener el dataset').\n")
        sys.exit(1)
    except DatosInvalidosError as e:
        print(f"\n[ERROR] {e}\n")
        sys.exit(1)

    print(f"\nListo. Resultados de la corrida en: {carpeta}\n")


if __name__ == "__main__":
    main()
