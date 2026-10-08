"""
registry.py
===========
Guarda cada modelo entrenado como .pkl y su metadata como .json, y crea el
directorio de la corrida con timestamp (data/output/YYYY-MM-DD_HH-MM-SS/).
 
"""

import json
import os
import platform
import sys
from datetime import datetime

import joblib
import numpy
import pandas
import sklearn
import xgboost
import lightgbm
import catboost


def crear_carpeta_corrida(output_dir: str) -> str:
    """Crea output_dir/YYYY-MM-DD_HH-MM-SS/ y retorna la ruta creada.

    Ese directorio con timestamp es la unidad de experimento registrado:
    cada corrida queda versionada por separado, sin sobrescribir corridas
    anteriores.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    carpeta = os.path.join(output_dir, timestamp)
    os.makedirs(carpeta, exist_ok=True)
    print(f"[registry] Carpeta de corrida creada: {carpeta}")
    return carpeta


def versiones_librerias() -> dict:
    """Retorna la version de cada libreria relevante, para reproducibilidad."""
    return {
        "python": sys.version.split()[0],
        "sistema_operativo": platform.platform(),
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "lightgbm": lightgbm.__version__,
        "catboost": catboost.__version__,
    }


def construir_metadata(nombre_modelo: str, auc_train: float, auc_test: float,
                        auc_val: float, decay_pct: float, training_time_sec: float,
                        hiperparametros: dict) -> dict:
    """Construye el dict de metadata con el formato exacto pedido:
    ml_name, performance, hyperparameters, library_versions, timestamp.
    """
    return {
        "ml_name": nombre_modelo,
        "performance": {
            "auc_train": round(auc_train, 4),
            "auc_test": round(auc_test, 4),
            "auc_val": round(auc_val, 4),
            "decay_pct": round(decay_pct, 2),
            "training_time_sec": round(training_time_sec, 2),
        },
        "hyperparameters": hiperparametros,
        "library_versions": versiones_librerias(),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def guardar_modelo_y_metadata(carpeta_corrida: str, nombre_experimento: str,
                               modelo, metadata: dict) -> None:
    """Guarda <modelo>_model.pkl y <modelo>_metadata.json en la carpeta de la corrida."""
    ruta_pkl = os.path.join(carpeta_corrida, f"{nombre_experimento}_model.pkl")
    ruta_json = os.path.join(carpeta_corrida, f"{nombre_experimento}_metadata.json")

    joblib.dump(modelo, ruta_pkl)

    with open(ruta_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)


def guardar_resumen_corrida(carpeta_corrida: str, resumen: dict) -> None:
    """Guarda results_summary.json con el ranking completo y el modelo campeon.

    (En la P1 este archivo se llamaba resumen_corrida.json; la P2 pide el
    nombre results_summary.json -- se renombra aqui para mantener un solo
    lugar de verdad sobre el nombre del archivo.)
    """
    ruta = os.path.join(carpeta_corrida, "results_summary.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False, default=str)
    print(f"[registry] Resumen de la corrida guardado en: {ruta}")


def guardar_backtesting_results(carpeta_corrida: str, backtesting_dict: dict) -> None:
    """Guarda backtesting_results.json (NUEVO en la P2): las olas walk-forward
    del modelo campeon final, con AUC/decay/n_obs/PSI por mes."""
    ruta = os.path.join(carpeta_corrida, "backtesting_results.json")
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(backtesting_dict, f, indent=2, ensure_ascii=False, default=str)
    print(f"[registry] Resultados de backtesting guardados en: {ruta}")
