"""
parallelism.py
===============
Orquesta el entrenamiento paralelo de TODOS los modelos del diccionario y
mide el speed-up real contra el mismo trabajo corriendo en un for-loop
secuencial estandar.

Backend elegido: joblib.Parallel + delayed, con n_jobs configurable desde
config.yaml (parallelism.n_jobs: -1 usa todos los nucleos disponibles).
Se eligio joblib sobre ProcessPoolExecutor/multiprocessing.Pool porque es
el recomendado por el enunciado para trabajar con scikit-learn, y porque
serializa (pickling) los objetos de scikit-learn/XGBoost/LightGBM/CatBoost
de forma mas robusta que multiprocessing puro.

IMPORTANTE: el paralelismo esta orquestado explicitamente AQUI, a nivel de
"un proceso de Python por experimento". No se usa n_jobs interno de
scikit-learn como sustituto (de hecho, cuando corre en paralelo aqui, cada
modelo individual se fuerza a n_jobs=1 internamente para no competir por
los mismos nucleos -- ver _forzar_n_jobs_interno_a_uno).
"""

import os
import time

from joblib import Parallel, delayed

from src.training import entrenar_experimento, entrenar_todos_secuencial


def _forzar_n_jobs_interno_a_uno(experimento: dict) -> dict:
    """Evita que un modelo con n_jobs=-1 interno (ej. RandomForest) lance a
    su vez N hilos DENTRO de cada worker de joblib, lo cual saturaria los
    nucleos y haria que el paralelismo a nivel de experimento (el que pide
    el enunciado) compita contra el paralelismo interno de sklearn.
    """
    exp = dict(experimento)
    hp = dict(exp["hiperparametros"])
    if "n_jobs" in hp:
        hp["n_jobs"] = 1
    exp["hiperparametros"] = hp
    return exp


def contar_workers(n_jobs: int) -> int:
    """Traduce n_jobs (puede ser -1) al numero real de workers que se usaran."""
    cpus = os.cpu_count() or 1
    if n_jobs is None or n_jobs == -1:
        return cpus
    return max(1, min(n_jobs, cpus))


def entrenar_en_paralelo(experimentos: list, data: dict, n_jobs: int) -> tuple:
    """Entrena todos los experimentos en paralelo con joblib.Parallel.

    Retorna
    -------
    (resultados, tiempo_segundos)
    """
    experimentos_ajustados = [_forzar_n_jobs_interno_a_uno(e) for e in experimentos]

    inicio = time.time()
    resultados = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(entrenar_experimento)(exp, data) for exp in experimentos_ajustados
    )
    tiempo = time.time() - inicio
    return resultados, tiempo


def ejecutar_con_benchmark(experimentos: list, data: dict, config: dict) -> tuple:
    """Entrena todos los experimentos y mide el speed-up real del paralelismo.

    Corre el MISMO conjunto de experimentos dos veces sobre el mismo hardware
    y el mismo dataset: una vez en un for-loop secuencial (T_secuencial) y
    una vez con joblib.Parallel (T_paralelo). Los resultados que se usan
    para el resto del pipeline (metricas, GetBestModel, backtesting) son los
    de la corrida PARALELA -- la corrida secuencial es solo para medir el
    tiempo de referencia (se descartan sus modelos entrenados, con la misma
    semilla ambas corridas deberian dar metricas casi identicas).

    Parametros
    ----------
    config : dict
        Se lee config['parallelism']['n_jobs'].

    Retorna
    -------
    (resultados, info_parallelism) donde info_parallelism es el dict con
    el formato exacto pedido: backend, n_workers, t_sequential_s,
    t_parallel_s, speed_up.
    """
    n_jobs = config.get("parallelism", {}).get("n_jobs", -1)
    n_workers = contar_workers(n_jobs)

    print(f"\n[parallelism] Midiendo T_secuencial (for-loop, {len(experimentos)} experimentos)...")
    _, t_secuencial = _medir_secuencial(experimentos, data)

    print(f"[parallelism] Midiendo T_paralelo (joblib.Parallel, n_jobs={n_jobs} "
          f"-> {n_workers} workers)...")
    resultados, t_paralelo = entrenar_en_paralelo(experimentos, data, n_jobs)

    speed_up = (t_secuencial / t_paralelo) if t_paralelo > 0 else float("nan")

    info_parallelism = {
        "backend": "joblib",
        "n_workers": n_workers,
        "t_sequential_s": round(t_secuencial, 2),
        "t_parallel_s": round(t_paralelo, 2),
        "speed_up": round(speed_up, 4),
    }

    print(f"[parallelism] T_secuencial={t_secuencial:.2f}s | T_paralelo={t_paralelo:.2f}s "
          f"| speed_up={speed_up:.2f}x")
    if speed_up < 1.0:
        print("[parallelism] AVISO: speed_up < 1.0 -- con pocos experimentos o datasets "
              "pequenos, el overhead de serializar los datos hacia cada proceso worker "
              "puede superar la ganancia de paralelizar. Ver README -> analisis de speed-up.")

    return resultados, info_parallelism


def _medir_secuencial(experimentos: list, data: dict) -> tuple:
    inicio = time.time()
    resultados = entrenar_todos_secuencial(experimentos, data)
    tiempo = time.time() - inicio
    return resultados, tiempo
